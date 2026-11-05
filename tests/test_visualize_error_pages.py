"""Unit tests for the pure helpers in ``scripts/visualize_error_pages.py``.

These pin the rule-type / order-tag logic and the RLC definition.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

# The script lives under scripts/ (not an importable package), so load it by path.
_spec = importlib.util.spec_from_file_location(
    "visualize_error_pages", _root / "scripts" / "visualize_error_pages.py"
)
vep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vep)


# ---------------------------------------------------------------------------
# rule_type
# ---------------------------------------------------------------------------


def test_rule_type_alternative_all_match():
    assert vep.rule_type([True, True, True]) == ("alternative", None)


def test_rule_type_neither_no_match():
    assert vep.rule_type([False, False, False]) == ("neither", None)


def test_rule_type_unknown_empty_or_none():
    assert vep.rule_type([]) == ("unknown", None)
    assert vep.rule_type(None) == ("unknown", None)


def test_rule_type_partial_first_k():
    assert vep.rule_type([True, False, False]) == ("partial", "first-k")
    assert vep.rule_type([True, True, False, False]) == ("partial", "first-k")


def test_rule_type_partial_last_k():
    assert vep.rule_type([False, False, True]) == ("partial", "last-k")
    assert vep.rule_type([False, False, True, True]) == ("partial", "last-k")


def test_rule_type_partial_contiguous_interior_run():
    # Interior run of length >= 2 -> "contiguous".
    assert vep.rule_type([False, True, True, False]) == ("partial", "contiguous")
    assert vep.rule_type([False, True, True, True, False]) == ("partial", "contiguous")


def test_rule_type_partial_other():
    # Single interior match and scattered sets both fold into "other".
    assert vep.rule_type([False, True, False]) == ("partial", "other")
    assert vep.rule_type([True, False, True]) == ("partial", "other")
    assert vep.rule_type([True, False, True, False]) == ("partial", "other")


def test_rule_type_coerces_truthy_values():
    # stage-3 may emit 0/1 ints; treat them as bools.
    assert vep.rule_type([1, 1, 1]) == ("alternative", None)
    assert vep.rule_type([1, 0, 0]) == ("partial", "first-k")


# ---------------------------------------------------------------------------
# relative_length_change
# ---------------------------------------------------------------------------


def test_rlc_normal():
    assert vep.relative_length_change(
        {"line_count_delta": -1, "original_line_count": 3}
    ) == pytest.approx(-1 / 3)


def test_rlc_zero_original_guard():
    assert vep.relative_length_change(
        {"line_count_delta": 2, "original_line_count": 0}
    ) is None


def test_rlc_missing_fields():
    assert vep.relative_length_change({}) is None
    assert vep.relative_length_change({"line_count_delta": 1}) is None
    assert vep.relative_length_change({"original_line_count": 5}) is None


def test_rlc_zero_delta_is_zero():
    assert vep.relative_length_change(
        {"line_count_delta": 0, "original_line_count": 4}
    ) == 0.0


# ---------------------------------------------------------------------------
# render_grid
# ---------------------------------------------------------------------------


def test_render_grid_cell_classes():
    out = vep.render_grid([[0, 1], [9, 5]])
    assert 'class="c0"' in out
    assert 'class="c1"' in out
    assert 'class="c9"' in out
    assert "<table" in out


def test_render_grid_embeds_scaled_cell_var():
    # small grid -> capped cell size; the per-table CSS var carries the size.
    small = vep.render_grid([[1, 2], [3, 4]])
    assert f"--cell:{vep.CELL_MAX_PX}px" in small
    # a large grid scales the cell down so the grid stays inside the box.
    big = vep.render_grid([[0] * 30 for _ in range(30)])
    assert f"--cell:{vep.cell_size_px(30, 30)}px" in big
    assert vep.cell_size_px(30, 30) < vep.CELL_MAX_PX


# ---------------------------------------------------------------------------
# cell_size_px (fixed-box scaling, square cells)
# ---------------------------------------------------------------------------


def test_cell_size_square_and_bounded():
    # largest dimension must fit within the box.
    for rows, cols in [(1, 1), (3, 3), (11, 22), (16, 19), (30, 30)]:
        cell = vep.cell_size_px(rows, cols)
        assert vep.CELL_MIN_PX <= cell <= vep.CELL_MAX_PX
        assert cell * max(rows, cols) <= vep.GRID_BOX_PX


def test_cell_size_small_grid_capped():
    assert vep.cell_size_px(1, 1) == vep.CELL_MAX_PX
    assert vep.cell_size_px(2, 3) == vep.CELL_MAX_PX


def test_cell_size_large_grid_shrinks_monotonically():
    assert vep.cell_size_px(10, 10) >= vep.cell_size_px(20, 20) >= vep.cell_size_px(30, 30)


def test_render_grid_out_of_range_fallback():
    out = vep.render_grid([[10, -1]])
    assert 'class="c-other"' in out
    assert 'class="c10"' not in out


def test_render_grid_empty():
    assert "empty" in vep.render_grid([])
    assert "empty" in vep.render_grid(None)


def test_render_grid_ragged_does_not_crash():
    # ragged grids render best-effort, cell by cell.
    out = vep.render_grid([[1, 2], [3]])
    assert "<table" in out
    assert 'class="c3"' in out


# ---------------------------------------------------------------------------
# classification_category
# ---------------------------------------------------------------------------


def test_classification_category_prefers_category_key():
    assert vep.classification_category({"category": "scope_restriction"}) == "scope_restriction"
    assert vep.classification_category({"error_category": "operator_confusion"}) == "operator_confusion"
    assert vep.classification_category(None) is None
    assert vep.classification_category({}) is None
