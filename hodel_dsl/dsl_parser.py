"""Parse, validate and rewrite programs written in the DSL.

``ProgramRewriter.parse`` checks that every name in a program is a DSL
function or constant or an already-defined variable, flattens nested calls into
one call per line, and collapses ``identity(x)`` into ``x``. ``clean`` removes
code that does not affect the returned value (dead-code elimination),
``rewrite`` prints the program in a canonical, topologically sorted form, and
``get_line_count`` gives its length.
This is how edited programs are normalised before they are run and measured.

Written for this project; the DSL itself (``dsl.py``) is adapted from Michael
Hodel's arc-dsl (see ``LICENSE`` in this directory).
"""

import ast
import re
import random
from collections import defaultdict
import textwrap
import networkx as nx
from pathlib import Path

_HERE = Path(__file__).resolve().parent


class ProgramRewriter:
    """
    Parses a Python program written in a specific DSL,
    validates it, and can rewrite it into a canonical string representation.
    """

    def __init__(
        self,
        dsl_symbols_path=_HERE / "dsl.py",
        dsl_constants_path=_HERE / "primitives.py",
    ):
        """
        Initializes the ProgramRewriter.

        Args:
            dsl_symbols_path (str): Path to the DSL definition file.
        """
        self.dsl_functions = set()
        self.dsl_constants = set()
        self.dsl_function_signatures = {}
        self.type_aliases = {}
        # Registry of commutative functions for canonicalization logic
        self.commutative_functions = set()
        self._get_dsl_symbols(dsl_symbols_path)
        if dsl_constants_path:
            self._get_dsl_symbols(dsl_constants_path)
        self.dsl_symbols = self.dsl_functions.union(self.dsl_constants)
        self._reset_state()

    def _reset_state(self):
        """Resets the internal state for parsing a new program."""
        self.ops = {}
        self.op_types = {}  # To store the return type of each operation
        self.return_var = None
        self.temp_var_count = 0
        self.defined_vars = set()
        self.aliases = (
            {}
        )  # Maps variables to their aliased values (constants or other variables)
        self.op_cache = {}  # Cache for common subexpression elimination

    def add_dsl_symbol(self, symbol_name):
        """Adds a symbol to the DSL symbols set."""
        self.dsl_symbols.add(symbol_name)

    def add_commutative_function(self, func_name):
        """
        Marks a DSL function as commutative for canonical functional rewriting.

        Args:
            func_name (str): Name of the DSL function to mark commutative.
        """
        self.commutative_functions.add(func_name)

    def _resolve_alias(self, var_name):
        """
        Recursively resolves a variable to its final aliased value.
        Returns the final resolved value (constant or variable).
        """
        if var_name in self.aliases:
            return self._resolve_alias(self.aliases[var_name])
        return var_name

    def _annotation_to_str(self, node):
        if node is None:
            return "Any"
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            value = self._annotation_to_str(node.value)
            slice_node = node.slice
            # For python < 3.9, node.slice is ast.Index(value=...)
            if isinstance(slice_node, ast.Index):
                slice_node = slice_node.value
            slice_val = self._annotation_to_str(slice_node)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elts = [self._annotation_to_str(elt) for elt in node.elts]
            return ", ".join(elts)
        elif isinstance(node, (ast.Constant)):
            return repr(node.value)
        elif isinstance(node, ast.Str):  # Python < 3.8
            return repr(node.s)

        # Fallback using ast.unparse if available (Python 3.9+)
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(node)
        except ImportError:
            pass

        return "Any"

    def _get_dsl_symbols(self, filepath):
        """
        Reads the DSL file and extracts all top-level function and variable names,
        and their type signatures.
        """
        with open(filepath, "r") as f:
            source = f.read()
        tree = ast.parse(source)

        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    self.dsl_functions.add(node.name)
                    self.dsl_function_signatures[node.name] = {
                        "return": self._annotation_to_str(node.returns),
                        "args": [
                            self._annotation_to_str(arg.annotation)
                            for arg in node.args.args
                        ],
                    }
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper() or target.id in ("T", "F"):
                            self.dsl_constants.add(target.id)
                        # Heuristic for type aliases: starts with uppercase but not all uppercase
                        elif target.id[0].isupper():
                            self.type_aliases[target.id] = self._annotation_to_str(
                                node.value
                            )

    def _node_to_str(self, node):
        """Converts an AST node to its string representation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        # For older Python versions
        elif isinstance(node, ast.Num):
            return str(node.n)
        elif isinstance(node, ast.Str):
            return repr(node.s)
        else:
            raise TypeError(
                f"Unsupported node type for string conversion: {type(node)}"
            )

    def _process_call_recursive(self, call_node):
        """
        Recursively processes an ast.Call node to flatten nested calls
        and build the operations dictionary.
        """
        if not isinstance(call_node.func, ast.Name):
            raise TypeError(f"Unsupported function call type: {type(call_node.func)}")
        func_name = call_node.func.id

        # If the callee is a variable (e.g., a function produced by `branch`),
        # resolve its alias to the underlying temp var so dependencies are tracked.
        # This ensures cleaning keeps the producer op and topological sort orders it correctly.
        if func_name not in self.dsl_symbols:
            # Resolve alias if any (falls through unchanged when none)
            func_name = self._resolve_alias(func_name)

        if func_name not in self.dsl_symbols and func_name not in self.defined_vars:
            raise ValueError(
                f"Function or variable '{func_name}' not defined before use."
            )

        # Collapse identity(x) into an alias: identity is a no-op, so just
        # return the (resolved) argument directly instead of creating an operation.
        if func_name == "identity" and len(call_node.args) == 1:
            arg = call_node.args[0]
            if isinstance(arg, ast.Call):
                return self._process_call_recursive(arg)
            else:
                arg_name = self._node_to_str(arg)
                if isinstance(arg, ast.Name):
                    arg_name = self._resolve_alias(arg_name)
                return arg_name

        arg_names = []
        for arg in call_node.args:
            if isinstance(arg, ast.Call):
                var_name = self._process_call_recursive(arg)
                arg_names.append(var_name)
            else:
                arg_name = self._node_to_str(arg)
                if (
                    isinstance(arg, ast.Name)
                    and arg.id not in self.defined_vars
                    and arg.id not in self.dsl_symbols
                ):
                    raise ValueError(f"Variable '{arg.id}' not defined before use.")
                if isinstance(arg, ast.Name):
                    arg_name = self._resolve_alias(arg_name)
                arg_names.append(arg_name)

        # Canonicalize arguments for cache key
        canonical_args = (
            tuple(sorted(arg_names))
            if func_name in self.commutative_functions
            else tuple(arg_names)
        )
        op_key = (func_name, canonical_args)

        if op_key in self.op_cache:
            return self.op_cache[op_key]

        # Operation not in cache, create a new one
        new_var = f"_tmp{self.temp_var_count}"
        self.temp_var_count += 1
        self.defined_vars.add(new_var)

        self.ops[new_var] = {"func": func_name, "args": arg_names}
        if func_name in self.dsl_function_signatures:
            self.op_types[new_var] = self.dsl_function_signatures[func_name]["return"]
        else:
            self.op_types[new_var] = "Any"  # Fallback

        self.op_cache[op_key] = new_var
        return new_var

    def _process_call(self, target_var, call_node):
        """
        Recursively processes an ast.Call node to flatten nested calls
        and build the operations dictionary.
        """
        call_result_var = self._process_call_recursive(call_node)
        self.aliases[target_var] = call_result_var

    def parse(self, source_code):
        """
        Parses and validates the source code of a DSL program.
        """
        self._reset_state()
        source_code = textwrap.dedent(source_code)

        # Find and register any HOLE placeholders as DSL symbols
        hole_names = re.findall(r"\bHOLE\d+\b", source_code)
        for hole_name in hole_names:
            self.dsl_symbols.add(hole_name)

        tree = ast.parse(source_code)

        # Basic validation
        if not (tree.body and isinstance(tree.body[0], ast.FunctionDef)):
            raise ValueError("Source code must contain a function definition.")

        main_func = tree.body[0]

        self.defined_vars = {arg.arg for arg in main_func.args.args}

        # Validate function body and build operations list
        for node in ast.walk(main_func):
            if isinstance(node, (ast.For, ast.While, ast.If)):
                raise ValueError(
                    f"Invalid statement type found in DSL program: {type(node)}"
                )

        for stmt in main_func.body:
            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    raise ValueError("Assignment must be to a single variable.")
                target_var = stmt.targets[0].id
                self.defined_vars.add(target_var)

                if isinstance(stmt.value, ast.Call):
                    self._process_call(target_var, stmt.value)
                elif isinstance(stmt.value, ast.Name):
                    # Variable assignment: x = y
                    source_var = stmt.value.id
                    if (
                        source_var not in self.defined_vars
                        and source_var not in self.dsl_symbols
                    ):
                        raise ValueError(
                            f"Variable '{source_var}' not defined before use."
                        )
                    self.aliases[target_var] = source_var
                elif isinstance(stmt.value, (ast.Constant, ast.Num, ast.Str)):
                    # Constant assignment: x = CONSTANT
                    constant_value = self._node_to_str(stmt.value)
                    self.aliases[target_var] = constant_value
                    if target_var in self.dsl_constants:
                        self.op_types[target_var] = (
                            "Any"  # Or more specific if possible
                        )
                    else:
                        # Infer type for literals
                        if isinstance(stmt.value, (ast.Num)):
                            self.op_types[target_var] = "Integer"
                        elif isinstance(stmt.value, (ast.Str)):
                            self.op_types[target_var] = "String"
                        else:
                            self.op_types[target_var] = "Any"

                else:
                    raise ValueError(f"Unsupported assignment type: {type(stmt.value)}")
            elif isinstance(stmt, ast.Return):
                if isinstance(stmt.value, ast.Call):
                    # Handle return with function call - assign to temp var first
                    temp_var = f"_tmp{self.temp_var_count}"
                    self.temp_var_count += 1
                    self.defined_vars.add(temp_var)
                    self._process_call(temp_var, stmt.value)
                    self.return_var = temp_var
                else:
                    return_value = self._node_to_str(stmt.value)
                    # Resolve alias if this is a variable
                    if isinstance(stmt.value, ast.Name):
                        return_value = self._resolve_alias(return_value)
                    self.return_var = return_value
            else:
                raise ValueError(
                    f"Invalid statement type in function body: {type(stmt)}"
                )

        return self

    def clean(self):
        """
        Removes redundant or useless pieces of code that are not used for
        computing the output (returned variable). This process is also known
        as dead code elimination.
        """
        if self.return_var is None:
            return self

        self.return_var = self._resolve_alias(self.return_var)

        # The return_var from parse() is already resolved to its canonical name or a constant.
        if not isinstance(self.return_var, str):
            # Return var is a constant, so no ops are needed.
            self.ops = {}
            return self

        essential_vars = set()
        queue = [self.return_var]
        visited = set()

        while queue:
            var = queue.pop(0)
            if var in visited:
                continue
            visited.add(var)

            if var in self.ops:
                essential_vars.add(var)
                op_details = self.ops[var]
                for arg in op_details["args"]:
                    if isinstance(arg, str):
                        queue.append(arg)
                if (
                    isinstance(op_details["func"], str)
                    and op_details["func"] in self.ops
                ):
                    queue.append(op_details["func"])

        self.ops = {var: self.ops[var] for var in essential_vars if var in self.ops}
        return self

    def rewrite(self, randomize=False, program_name="solver"):
        """
        Rewrites the parsed program into a canonical, topologically sorted string format.

        Args:
            randomize (bool): If True, shuffles the order of independent operations.
        Returns:
            str: The rewritten program code.
        """
        dependencies = defaultdict(list)
        dependents = defaultdict(list)

        initial_vars = {
            arg
            for op in self.ops.values()
            for arg in op["args"]
            if isinstance(arg, str) and arg not in self.ops
        }

        for var, op_details in self.ops.items():
            for arg in op_details["args"]:
                if arg in self.ops:
                    dependencies[var].append(arg)
                    dependents[arg].append(var)
            if op_details["func"] in self.ops:
                dependencies[var].append(op_details["func"])
                dependents[op_details["func"]].append(var)

        in_degree = {var: len(deps) for var, deps in dependencies.items()}
        queue = [var for var in self.ops if var not in in_degree]

        if not randomize:
            queue.sort()

        sorted_vars = []
        last_processed = (
            None  # Track the last processed variable for prioritizing usage
        )

        while queue:
            if randomize:
                random.shuffle(queue)
            elif last_processed is not None:
                # Sort queue to prioritize variables that use the last processed variable
                def priority(var):
                    # Check if this variable uses the last processed variable
                    op = self.ops.get(var, {})
                    uses_last = (
                        last_processed in op.get("args", [])
                        or op.get("func") == last_processed
                    )
                    return (not uses_last, var)  # False sorts before True

                queue.sort(key=priority)

            var = queue.pop(0)
            sorted_vars.append(var)
            last_processed = var  # Update the last processed variable

            for dependent in dependents.get(var, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(sorted_vars) != len(self.ops):
            raise ValueError("A cycle was detected in the dependency graph.")

        # Generate code from sorted_vars
        lines = [f"def {program_name}(I):"]
        # Start variables at x1 instead of x0, and use O for the last variable
        num_vars = len(sorted_vars)
        var_map = {}
        for i, var in enumerate(sorted_vars):
            if i == num_vars - 1:
                var_map[var] = "O"
            else:
                var_map[var] = f"x{i + 1}"

        # Ensure input variables are mapped correctly if they are not part of `ops`
        for var in initial_vars:
            if var not in var_map:
                var_map[var] = var

        for old_var in sorted_vars:
            op = self.ops[old_var]
            new_target = var_map[old_var]

            new_args = [var_map.get(arg, arg) for arg in op["args"]]
            args_str = ", ".join(new_args)

            func_name = var_map.get(op["func"], op["func"])
            lines.append(f"    {new_target} = {func_name}({args_str})")

        final_return_var = var_map.get(self.return_var, self.return_var)
        lines.append(f"    return {final_return_var}")

        return "\n".join(lines)

    def rewrite_functional(self, pure_func=False, canonical=False):
        """
        Rewrites the parsed program into a canonical, functional, nested string format.

        Returns:
            str: The rewritten program code in functional format.
        """
        if self.return_var is None:
            if pure_func:
                return ""
            return "def verifier(I):\n    pass"

        self.return_var = self._resolve_alias(self.return_var)

        functional_body = self._build_functional_str(
            self.return_var, canonical=canonical
        )
        if pure_func:
            return functional_body

        return f"def verifier(I):\n    return {functional_body}"

    def parse_functional(self, source_code):
        """
        Parses a functional DSL program.

        Accepts either:
        - A pure functional expression (e.g., "fill(canvas(ZERO, ...), TWO, ...)")
        - A wrapped version identical to what `rewrite_functional` returns:
          "def verifier(I):\n    return <functional expression>"

        This method internally wraps pure expressions in a minimal verifier
        function so they can be parsed by the existing `parse` implementation.

        Args:
            source_code (str): Functional DSL string (pure or wrapped).

        Returns:
            ProgramRewriter: self, after parsing.
        """
        # Find and register any <HOLE...> placeholders as DSL constants
        hole_names = re.findall(r"<([^>]+)>", source_code)
        for hole_name in hole_names:
            # self.dsl_constants.add(hole_name)
            self.dsl_symbols.add(hole_name)

        # Normalize whitespace
        code = textwrap.dedent(source_code).strip()

        # Replace <HOLE_D> with HOLE_D for compatibility with Python's parser.
        code = re.sub(r"<([^>]+)>", r"\1", code)

        # If this already looks like a function definition, defer to the main parser
        if code.startswith("def "):
            return self.parse(code)

        # Otherwise, treat it as a pure functional expression and wrap it
        # in a trivial verifier so we can reuse the existing machinery.
        wrapped = (
            "def verifier(I):\n"
            "    return (\n"
            f"{textwrap.indent(code, ' ' * 8)}\n"
            "    )\n"
        )
        return self.parse(wrapped)

    def _build_functional_str(self, var_name, canonical=False):
        """
        Recursively builds the functional string for a given variable.
        """
        if var_name not in self.ops:
            return str(var_name)

        op = self.ops[var_name]
        func_str = self._build_functional_str(op["func"], canonical=canonical)
        arg_strs = [
            self._build_functional_str(arg, canonical=canonical) for arg in op["args"]
        ]
        # Canonicalize argument order for declared commutative functions (only when function is a name)
        if (
            canonical
            and op["func"] not in self.ops
            and op["func"] in self.commutative_functions
        ):
            arg_strs = sorted(arg_strs)
        return f"{func_str}({', '.join(arg_strs)})"

    def build_tree(self):
        """
        Builds a networkx DiGraph as a tree, representing the functional
        form of the program. The root of the tree is the final operation.
        It also tags nodes with a 'computation_id' to identify identical subtrees.
        """
        g = nx.DiGraph()
        if self.return_var is None:
            return g

        # We need a way to create unique nodes, since the same operation/constant
        # can appear multiple times in the tree. A simple counter will do.
        self._node_counter = 0
        self._computation_id_counter = 0
        self._subtree_to_computation_id = {}

        self._build_tree_recursive(g, self.return_var)

        # Clean up after the build is complete
        del self._node_counter
        del self._computation_id_counter
        del self._subtree_to_computation_id

        return g

    def _get_node_type(self, var_name):
        """Get the type of a variable from op_types, resolving aliases."""
        resolved_var = self._resolve_alias(var_name)
        if resolved_var in self.op_types:
            return self.op_types[resolved_var]

        # Try to find the type of the alias source if it's another variable
        current = var_name
        while current in self.aliases:
            current = self.aliases[current]
            if current in self.op_types:
                return self.op_types[current]

        # Fallback for constants and inputs
        if resolved_var in self.dsl_constants:
            return "Constant"  # Generic constant type
        if resolved_var == "I":  # Assuming 'I' is the input
            return "Grid"

        return "Any"  # Default fallback

    def _build_tree_recursive(
        self, g, current_var, parent_label=None, parent_arg_index=None
    ):
        """
        Recursively builds the tree, adding nodes and edges to the graph `g`.
        Returns a tuple containing the node ID and a canonical representation
        of the subtree.
        """
        node_id = self._node_counter
        self._node_counter += 1

        node_dsl_type = self._get_node_type(current_var)

        # If `current_var` is not the result of an operation, it's a leaf.
        if current_var not in self.ops:
            node_label = str(current_var)

            # Determine node type for leaves based on DSL symbols or literal values.
            if node_label in self.dsl_constants:
                node_type = "constant"
            elif node_label in self.dsl_functions:
                node_type = "leaf_function"  # This is a leaf function
            else:
                node_type = "input"

            subtree_repr = node_label
            # For constants, assign computation_id based on parent context
            # so repeated identical subtrees share IDs across reuse,
            # while multiple constants in distinct argument positions remain distinct.
            if node_type == "constant":
                if parent_label is not None and parent_arg_index is not None:
                    contextual_key = (
                        f"CONST:{node_label}@{parent_label}[{parent_arg_index}]"
                    )
                else:
                    # No reliable parent context; ensure uniqueness
                    contextual_key = f"CONST:{node_label}@<root>#{self._node_counter}"

                if contextual_key not in self._subtree_to_computation_id:
                    self._subtree_to_computation_id[contextual_key] = (
                        self._computation_id_counter
                    )
                    self._computation_id_counter += 1
                computation_id = self._subtree_to_computation_id[contextual_key]
            else:
                if subtree_repr not in self._subtree_to_computation_id:
                    self._subtree_to_computation_id[subtree_repr] = (
                        self._computation_id_counter
                    )
                    self._computation_id_counter += 1
                computation_id = self._subtree_to_computation_id[subtree_repr]
            g.add_node(
                node_id,
                label=node_label,
                node_type=node_type,
                dsl_type=node_dsl_type,
                computation_id=computation_id,
                variable_name=current_var,
            )
            return node_id, subtree_repr

        op_details = self.ops[current_var]
        func_var = op_details["func"]

        # If `func_var` is a variable (i.e., it is in self.ops), we have a
        # higher-order function call. We represent this as a "call" node.
        if func_var in self.ops:
            node_label = "call"
            node_type = "call"

            # The children are the function and its arguments
            children_vars = [func_var] + op_details["args"]
            child_info = []
            for idx, child_var in enumerate(children_vars):
                child_info.append(
                    self._build_tree_recursive(
                        g, child_var, parent_label=node_label, parent_arg_index=idx
                    )
                )
            child_node_ids = [info[0] for info in child_info]
            # Preserve order for higher-order calls; they are not commutative
            child_subtree_reprs = [info[1] for info in child_info]
            subtree_repr = f"{node_label}({','.join(child_subtree_reprs)})"

        else:
            # It's an internal node representing a standard function call.
            node_label = str(func_var)
            node_type = "function"
            child_info = []
            for idx, arg in enumerate(op_details["args"]):
                child_info.append(
                    self._build_tree_recursive(
                        g, arg, parent_label=node_label, parent_arg_index=idx
                    )
                )
            child_node_ids = [info[0] for info in child_info]
            # Only canonicalize by sorting when the function is declared commutative
            if node_label in self.commutative_functions:
                child_subtree_reprs = sorted([info[1] for info in child_info])
            else:
                child_subtree_reprs = [info[1] for info in child_info]
            subtree_repr = f"{node_label}({','.join(child_subtree_reprs)})"

        if subtree_repr not in self._subtree_to_computation_id:
            self._subtree_to_computation_id[subtree_repr] = self._computation_id_counter
            self._computation_id_counter += 1

        computation_id = self._subtree_to_computation_id[subtree_repr]
        g.add_node(
            node_id,
            label=node_label,
            node_type=node_type,
            dsl_type=node_dsl_type,
            computation_id=computation_id,
            variable_name=current_var,
        )

        # Preserve argument order on edges for stable reconstruction
        for idx, child_node_id in enumerate(child_node_ids):
            g.add_edge(node_id, child_node_id, arg_index=idx)

        return node_id, subtree_repr

    def tree_to_functional(self, g, canonical=False):
        """
        Converts a program tree (networkx graph) back into a functional string.
        """
        if not g.nodes:
            return ""

        # Find the root node (node with out-degree 0)
        root_nodes = [node for node, out_degree in g.out_degree() if out_degree == 0]
        if not root_nodes:
            # could be a cycle or empty graph
            return ""  # Or raise an error

        root = root_nodes[0]
        return self._build_str_from_node(g, root, canonical=canonical)

    def _build_str_from_node(self, g, node_id, canonical=False):
        """
        Recursively builds a functional string from a node in the graph.
        """
        node_attrs = g.nodes[node_id]
        label = node_attrs.get("label", str(node_id))

        if g.in_degree(node_id) == 0:
            return label

        # It's a function call
        predecessors = list(g.predecessors(node_id))

        # Get arguments and sort them by 'arg_index'
        args_with_indices = []
        for p in predecessors:
            edge_data = g.get_edge_data(p, node_id)
            arg_index = edge_data.get("arg_index", 0)
            args_with_indices.append((arg_index, p))

        sorted_args = sorted(args_with_indices, key=lambda x: x[0])

        arg_strings = [
            self._build_str_from_node(g, arg_node, canonical=canonical)
            for _, arg_node in sorted_args
        ]

        # If canonical requested and this node represents a commutative function,
        # sort argument strings deterministically. Do not reorder for 'call' nodes.
        node_type = node_attrs.get("node_type", "")
        if (
            canonical
            and node_type == "function"
            and label in self.commutative_functions
        ):
            arg_strings = sorted(arg_strings)

        return f"{label}({', '.join(arg_strings)})"

    def get_line_count(self):
        """
        Returns the number of lines of code in the cleaned program.
        Does not count the function definition or return statement.
        """
        if not self.ops:
            return 0
        return len(self.ops)


if __name__ == "__main__":
    # Example usage with the program provided by the user.
    incorrect_program = """
    def verifier(I):
        x0 = replace(I, THREE, decrement(THREE)) # this is a comment
        x1 = switch(x0, EIGHT, NINE)
        # this too is a comment
        new_var = identity(x1)
        x3 = replace(new_var, ONE, decrement(ONE))
        return x3
    """

    rewriter = ProgramRewriter()

    print("--- Parsing and Rewriting Incorrect Program ---")
    try:
        parsed_program = rewriter.parse(incorrect_program)
        print("Program parsed successfully.")

        print("\n--- Generating a Correct String Representation ---")
        correct_program_1 = parsed_program.rewrite()
        print(correct_program_1)

        print("\n--- Generating another (randomized) Correct String Representation ---")
        correct_program_2 = correct_program_1
        while correct_program_2 == correct_program_1:
            correct_program_2 = parsed_program.rewrite(randomize=True)
        print(correct_program_2)

    except ValueError as e:
        print(f"Error parsing program: {e}")

    # Example with a valid program from verifiers_rearc.py
    print("\n\n--- Parsing a known-good program ---")
    correct_program_example = """
    def verify_0d3d703e(I: Grid) -> Grid:
        x0 = switch(I, THREE, FOUR)
        x1 = switch(x0, EIGHT, NINE)
        x2 = switch(x1, TWO, SIX)
        x3 = switch(x2, ONE, FIVE)
        return x3
    """
    # Quick adaptation to make it work with the parser
    correct_program_example = (
        correct_program_example.replace("verify_0d3d703e", "verifier")
        .replace(": Grid", "")
        .replace(" -> Grid", "")
    )

    try:
        parsed_program = rewriter.parse(correct_program_example)
        print("Program parsed successfully.")
        rewritten = parsed_program.rewrite()
        print("\nRewritten version:")
        print(rewritten)
    except ValueError as e:
        print(f"Error parsing program: {e}")

    # Example with a program that returns a function call directly
    print("\n\n--- Parsing a program that returns a function call ---")
    return_call_program = """
    def verifier(I):
        return canvas(ZERO, THREE_BY_THREE)
    """
    try:
        parsed_program = rewriter.parse(return_call_program)
        print("Program with return call parsed successfully.")
        rewritten = parsed_program.rewrite()
        print("\nRewritten version:")
        print(rewritten)
    except ValueError as e:
        print(f"Error parsing program: {e}")

    # Example with nested function calls in return
    print("\n\n--- Parsing a program that returns nested function calls ---")
    nested_return_program = """
    def verifier(I):
        return switch(replace(I, ONE, TWO), THREE, FOUR)
    """
    try:
        parsed_program = rewriter.parse(nested_return_program)
        print("Program with nested return calls parsed successfully.")
        rewritten = parsed_program.rewrite()
        print("\nRewritten version:")
        print(rewritten)
    except ValueError as e:
        print(f"Error parsing program: {e}")

    # NEW TEST: Example with constant and variable assignments
    print("\n\n--- Parsing a program with constant and variable assignments ---")
    alias_program = """
    def verifier(I):
        x0 = objects(I, T, F, T)
        x1 = THREE  # Constant assignment
        x2 = astuple(x1, x1)
        x16 = x2    # Variable assignment
        result = replace(x16, ONE, TWO)
        return result
    """
    try:
        parsed_program = rewriter.parse(alias_program)
        print("Program with aliases parsed successfully.")
        print(f"Aliases found: {parsed_program.aliases}")
        rewritten = parsed_program.rewrite()
        print("\nRewritten version:")
        print(rewritten)
    except ValueError as e:
        print(f"Error parsing program: {e}")

    # Example with an invalid program (for loop)
    print("\n\n--- Parsing an invalid program with a for-loop ---")
    invalid_program = """
    def verifier(I):
        x = I
        for i in range(5):
            x = replace(x, i, i+1)
        return x
    """
    try:
        rewriter.parse(invalid_program)
    except ValueError as e:
        print(f"Successfully caught error: {e}")

    program_code = textwrap.dedent(
        """
        def verifier(I):
            x0 = objects(I, T, F, T)
            x1 = THREE  # Constant assignment
            x2 = astuple(x1, x1)
            x16 = x2    # Variable assignment
            x5 = add(x1, TWO)
            result = replace(x16, ONE, TWO)
            return result
    """
    )
    print("Original Program:")
    print(program_code)
    print("-" * 20)

    rewriter = ProgramRewriter()

    # Parse, then rewrite without cleaning
    rewriter.parse(program_code)
    rewritten_code = rewriter.rewrite()
    print("Rewritten Program (without cleaning):")
    print(rewritten_code)
    print("-" * 20)

    # Parse again (to reset state), clean, and then rewrite
    rewriter.parse(program_code)
    cleaned_rewritten_code = rewriter.clean().rewrite()
    print("Cleaned and Rewritten Program:")
    print(cleaned_rewritten_code)

    # Example for functional rewrite
    print("\n\n--- Testing functional rewrite ---")
    fill_program = """
    def verifier(I):
        x0 = most_frequent(I)
        x1 = find_indices(I, x0)
        x2 = get_value(x1, ZERO)
        x3 = get_value(x1, ONE)
        x4 = add(x3, ONE)
        x5 = to_slice(x2, x4)
        x6 = replace(I, x0, x5, NEG_ONE)
        return x6
    """
    try:
        rewriter = ProgramRewriter()
        parsed_program = rewriter.parse(fill_program)
        functional_rep = parsed_program.rewrite_functional()
        print("Functional representation:")
        print(functional_rep)
    except Exception as e:
        print(f"Error during functional rewrite test: {e}")

    # Example with computation reuse
    print("\n\n--- Testing computation reuse tagging ---")
    reuse_program = """
    def verifier(I):
        x0 = decrement(ONE)
        x1 = decrement(ONE)
        x2 = astuple(x0, x1)
        return x2
    """
    try:
        rewriter = ProgramRewriter()
        rewriter.add_dsl_symbol("decrement")
        parsed_program = rewriter.parse(reuse_program)
        g = parsed_program.build_tree()
        print("Graph nodes with computation_id:")
        for node, data in g.nodes(data=True):
            print(
                f"  Node {node}: label='{data['label']}', dsl_type='{data.get('dsl_type', 'N/A')}', computation_id={data['computation_id']}"
            )

        # A quick check
        computation_ids = defaultdict(list)
        for node, data in g.nodes(data=True):
            computation_ids[data["computation_id"]].append(data["label"])

        print("\nRepeated computations:")
        for cid, labels in computation_ids.items():
            if len(labels) > 1:
                print(
                    f"  Computation ID {cid} is shared by nodes with labels: {labels}"
                )

    except Exception as e:
        print(f"Error during computation reuse test: {e}")
