"""Tests for ``src.program_edits``: applying line edits and the response schema."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.program_edits import (  # noqa: E402
    ModifyOnlyLineModification as Edit,
    apply_modifications,
    get_response_schema,
)

_PROGRAM = """def solve_x(I):
    x1 = vmirror(I)
    x2 = rot90(x1)
    O = hconcat(I, x2)
    return O
"""


def _lines(src: str) -> list:
    """Strip the ``def`` / ``return`` lines so we can compare bodies easily."""
    return [
        line.strip()
        for line in src.splitlines()
        if line.strip() and not line.strip().startswith(("def ", "return "))
    ]


def test_modify_replaces_rhs_only():
    result = apply_modifications(_PROGRAM, [Edit(variable_name="x1", replacement="hmirror(I)")])
    assert _lines(result) == ["x1 = hmirror(I)", "x2 = rot90(x1)", "O = hconcat(I, x2)"]


def test_modify_preserves_indentation():
    result = apply_modifications(_PROGRAM, [Edit(variable_name="x2", replacement="rot180(x1)")])
    body_lines = [line for line in result.splitlines() if "= " in line]
    assert all(line.startswith("    ") for line in body_lines)


def test_last_edit_to_a_variable_wins():
    mods = [
        Edit(variable_name="x1", replacement="hmirror(I)"),
        Edit(variable_name="x1", replacement="rot90(I)"),
    ]
    assert "x1 = rot90(I)" in _lines(apply_modifications(_PROGRAM, mods))


def test_unknown_variable_is_ignored():
    result = apply_modifications(_PROGRAM, [Edit(variable_name="x9", replacement="hmirror(I)")])
    assert _lines(result) == _lines(_PROGRAM)


def test_line_count_never_changes():
    mods = [
        Edit(variable_name="x1", replacement="hmirror(I)"),
        Edit(variable_name="O", replacement="vconcat(I, x2)"),
    ]
    assert len(apply_modifications(_PROGRAM, mods).splitlines()) == len(_PROGRAM.splitlines())


def test_schema_has_no_add_or_delete_fields():
    schema = get_response_schema(max_modifications=3)
    assert schema["properties"]["modifications"]["maxItems"] == 3
    (item,) = schema["$defs"].values()
    assert set(item["properties"]) == {"variable_name", "replacement"}
    assert set(item["required"]) == {"variable_name", "replacement"}
