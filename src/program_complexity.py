"""Program size measured in AST nodes.

``safe_node_count`` counts the nodes of a DSL program. By default, a subtree
reused through a variable is counted once; with ``dedup_subtrees=False`` every
use is counted.
"""

from __future__ import annotations

from hodel_dsl.dsl_parser import ProgramRewriter


def safe_node_count(source: str | None, dedup_subtrees: bool = True) -> int | None:
    """Parse a DSL program and return its AST node count, or None on failure.

    When ``dedup_subtrees`` is True (default), structurally-identical subtrees
    share a single count via ``computation_id`` — i.e., the DAG node count.
    Otherwise returns the executed-trace count (each variable reference
    expanded into a fresh subtree).
    """
    if not source:
        return None
    try:
        rw = ProgramRewriter()
        rw.parse(source).clean()
        g = rw.build_tree()
        if dedup_subtrees:
            return len({d["computation_id"] for _, d in g.nodes(data=True)})
        return g.number_of_nodes()
    except Exception:
        return None
