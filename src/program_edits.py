"""Line edits to a reference program: the response schema and how edits are applied.

An edit replaces the right-hand side of one existing assignment line.
``get_response_schema`` returns the JSON schema the LLM's structured output
must follow, and ``apply_modifications`` applies a list of edits to a program.
"""

from __future__ import annotations

import re
from typing import Iterable

from pydantic import BaseModel, Field


class ModifyOnlyLineModification(BaseModel):
    """A single line edit: replace the RHS of one variable's assignment (modify-only mode)."""

    variable_name: str = Field(
        description="Name of the variable whose assignment line to replace (e.g. 'x1', 'x2', 'O'). Must match a variable in the original program."
    )
    replacement: str = Field(
        description="The new right-hand side expression only (e.g. 'rotate(I)', 'objects(I, T, F, T)'). Same syntax as DSL: function_name(arg1, arg2, ...). No 'var = ' prefix."
    )


class ModifyOnlyProgramModifications(BaseModel):
    """List of line modifications to apply to a ground-truth program (modify-only mode). At most K modifications."""

    modifications: list[ModifyOnlyLineModification] = Field(
        description="Ordered list of edits. Each edit specifies a variable and the new RHS to set. Duplicate variable names overwrite (last wins).",
        min_length=0,
    )


def get_response_schema(max_modifications: int | None = None) -> dict:
    """JSON schema for the LLM's structured output.

    Args:
        max_modifications: If set, cap the length of the ``modifications``
            array (``maxItems``) so the model cannot exceed the edit budget.
    """
    schema = ModifyOnlyProgramModifications.model_json_schema()
    if max_modifications is not None and "properties" in schema:
        mods = schema["properties"].get("modifications", {})
        if isinstance(mods, dict):
            mods["maxItems"] = max_modifications
    return schema


# -- Assignment-line regex used by apply helpers -------------------------
_ASSIGN_RE = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.*)$")


def apply_modifications(
    program_source: str, modifications: Iterable[ModifyOnlyLineModification]
) -> str:
    """Apply line edits to a solver program's source.

    Each edit replaces the right-hand side of the assignment to
    ``variable_name`` with ``replacement``, keeping the line's indentation.
    If a variable is edited more than once, the last edit wins.  Edits to
    variables the program does not assign are ignored.

    Args:
        program_source: Full source of the solver function (``def solve_xxx(I): ...``).
        modifications: The edits, in the order the model returned them.

    Returns:
        The edited program source.
    """
    to_modify: dict[str, str] = {}
    for m in modifications:
        to_modify[m.variable_name] = m.replacement

    result_lines: list[str] = []
    for line in program_source.splitlines():
        stripped = line.lstrip()
        match = _ASSIGN_RE.match(stripped) if stripped else None
        if match and match.group(1) in to_modify:
            var_name = match.group(1)
            indent = line[: len(line) - len(stripped)]
            result_lines.append(f"{indent}{var_name} = {to_modify[var_name]}")
            continue
        result_lines.append(line)

    return "\n".join(result_lines)
