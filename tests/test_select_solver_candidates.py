"""Unit tests for ``scripts.select_solver_candidates`` and the
``load_synthesized_solvers`` loader the export feeds into."""

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import select_solver_candidates as ssc  # noqa: E402

from src.utils import collapse_solver_bundles, load_synthesized_solvers  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _record(
    *,
    task_id: str,
    rewritten_program: str,
    line_count: int,
    sample_index: int,
    strict_solve: bool = True,
):
    """Build a minimal solver-eval record (mirrors stage-C output keys)."""
    return {
        "task_id": task_id,
        "sample_index": sample_index,
        "rewritten_program": rewritten_program,
        "line_count": line_count,
        "strict_solve": strict_solve,
    }


def _prog(n: int) -> str:
    """A distinct full-source solver body of `n` assignment lines."""
    lines = "\n".join(f"    x{i} = identity(I)" for i in range(1, n + 1))
    return f"def solver(I):\n{lines}\n    O = x{n}\n    return O"


# ---------------------------------------------------------------------------
# select_solvers — shortest mode
# ---------------------------------------------------------------------------


def test_shortest_picks_min_line_count():
    records = [
        _record(task_id="t1", rewritten_program=_prog(5), line_count=5, sample_index=0),
        _record(task_id="t1", rewritten_program=_prog(2), line_count=2, sample_index=1),
        _record(task_id="t1", rewritten_program=_prog(3), line_count=3, sample_index=2),
    ]
    bundles, _ = ssc.select_solvers(records, mode="shortest")
    assert list(bundles) == ["t1"]
    assert bundles["t1"] == [_prog(2)]


def test_shortest_tie_broken_by_sample_index():
    # Two candidates tied at line_count=2; the lower sample_index wins.
    records = [
        _record(task_id="t1", rewritten_program=_prog(2) + "  # b", line_count=2, sample_index=3),
        _record(task_id="t1", rewritten_program=_prog(2) + "  # a", line_count=2, sample_index=1),
    ]
    bundles, _ = ssc.select_solvers(records, mode="shortest")
    assert bundles["t1"] == [_prog(2) + "  # a"]


# ---------------------------------------------------------------------------
# select_solvers — all-unique mode
# ---------------------------------------------------------------------------


def test_all_unique_dedupes_and_orders_shortest_first():
    p2, p4 = _prog(2), _prog(4)
    records = [
        _record(task_id="t1", rewritten_program=p4, line_count=4, sample_index=0),
        _record(task_id="t1", rewritten_program=p2, line_count=2, sample_index=1),
        # exact-duplicate program text (different sample) collapses away.
        _record(task_id="t1", rewritten_program=p2, line_count=2, sample_index=2),
    ]
    bundles, counters = ssc.select_solvers(records, mode="all-unique")
    assert bundles["t1"] == [p2, p4]  # shortest first, deduped
    assert counters["bodies_kept"] == 2


# ---------------------------------------------------------------------------
# Validity gate
# ---------------------------------------------------------------------------


def test_non_strict_solvers_excluded():
    records = [
        # train-only-correct (strict_solve False) must be dropped.
        _record(task_id="t1", rewritten_program=_prog(2), line_count=2, sample_index=0, strict_solve=False),
        _record(task_id="t1", rewritten_program=_prog(3), line_count=3, sample_index=1, strict_solve=True),
    ]
    bundles, counters = ssc.select_solvers(records, mode="all-unique")
    assert bundles["t1"] == [_prog(3)]
    assert counters["strict_solving"] == 1


def test_task_with_no_strict_solver_absent_from_output():
    records = [
        _record(task_id="t1", rewritten_program=_prog(2), line_count=2, sample_index=0, strict_solve=False),
    ]
    bundles, counters = ssc.select_solvers(records, mode="shortest")
    assert bundles == {}
    assert counters["tasks_with_solver"] == 0


def test_missing_program_text_skipped_defensively():
    records = [
        {"task_id": "t1", "sample_index": 0, "rewritten_program": None,
         "line_count": None, "strict_solve": True},
    ]
    bundles, counters = ssc.select_solvers(records, mode="shortest")
    assert bundles == {}
    assert counters["missing_program"] == 1


def test_multiple_tasks_grouped_independently():
    records = [
        _record(task_id="t1", rewritten_program=_prog(2), line_count=2, sample_index=0),
        _record(task_id="t2", rewritten_program=_prog(4), line_count=4, sample_index=0),
    ]
    bundles, _ = ssc.select_solvers(records, mode="shortest")
    assert set(bundles) == {"t1", "t2"}


# ---------------------------------------------------------------------------
# load_synthesized_solvers
# ---------------------------------------------------------------------------


def test_loader_roundtrips_valid_map(tmp_path):
    path = tmp_path / "synthesized_solvers.json"
    payload = {"t1": [_prog(2), _prog(3)], "t2": [_prog(1)]}
    path.write_text(json.dumps(payload))
    assert load_synthesized_solvers(path) == payload


def test_loader_allows_empty_bundle(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"t1": []}))
    assert load_synthesized_solvers(path) == {"t1": []}


def test_loader_rejects_non_dict(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps(["not", "a", "dict"]))
    with pytest.raises(ValueError):
        load_synthesized_solvers(path)


def test_loader_rejects_non_list_value(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"t1": "def solver(I): ..."}))
    with pytest.raises(ValueError):
        load_synthesized_solvers(path)


def test_loader_rejects_non_string_element(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"t1": [None]}))
    with pytest.raises(ValueError):
        load_synthesized_solvers(path)


# ---------------------------------------------------------------------------
# first-collapse contract — the shared selection rule both stage 2 and stage 3
# call (src.utils.collapse_solver_bundles).
# ---------------------------------------------------------------------------


def test_first_collapse_selects_element_zero_and_drops_empty():
    bundles = {"t1": [_prog(2), _prog(3)], "t2": [], "t3": [_prog(1)]}
    collapsed = collapse_solver_bundles(bundles)
    assert collapsed == {"t1": _prog(2), "t3": _prog(1)}


def test_collapse_rejects_unimplemented_selection():
    with pytest.raises(ValueError):
        collapse_solver_bundles({"t1": [_prog(1)]}, selection="all")
