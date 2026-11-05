"""Unit tests for ``scripts.filter_candidates`` (stage 4 dedup)."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import filter_candidates as fc  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _record(
    *,
    error_id: str,
    rewritten_program: str,
    modifications,
    program_output_grid=None,
    target_error_grid=None,
):
    """Build a minimal eval record that passes the executable/threshold guards."""
    target = target_error_grid or [[1, 2], [3, 4]]
    output = program_output_grid or [row[:] for row in target]
    return {
        "error_id": error_id,
        "task_id": "task-A",
        "sample_index": 0,
        "execution_success": True,
        "has_nested_violation": False,
        "target_error_grid": target,
        "program_output_grid": output,
        "modifications": modifications,
        "rewritten_program": rewritten_program,
    }


# ---------------------------------------------------------------------------
# _candidate_key
# ---------------------------------------------------------------------------


def test_candidate_key_is_order_independent():
    a = {
        "modifications": [
            {"variable_name": "x1", "original": "foo", "replacement": "bar"},
            {"variable_name": "x2", "original": "baz", "replacement": "qux"},
        ]
    }
    b = {
        "modifications": [
            {"variable_name": "x2", "original": "baz", "replacement": "qux"},
            {"variable_name": "x1", "original": "foo", "replacement": "bar"},
        ]
    }
    assert fc._candidate_key(a) == fc._candidate_key(b)


def test_candidate_key_distinguishes_different_mods():
    a = {"modifications": [{"variable_name": "x1", "original": "foo", "replacement": "bar"}]}
    b = {"modifications": [{"variable_name": "x1", "original": "foo", "replacement": "BAZ"}]}
    assert fc._candidate_key(a) != fc._candidate_key(b)


def test_candidate_key_freeform_falls_back_to_program_text():
    rec = {"modifications": [], "rewritten_program": "def solver(I): return I\n"}
    other = {"modifications": [], "rewritten_program": "def solver(I): return I\n"}
    different = {"modifications": [], "rewritten_program": "def solver(I): return O\n"}
    assert fc._candidate_key(rec) == fc._candidate_key(other)
    assert fc._candidate_key(rec) != fc._candidate_key(different)


# ---------------------------------------------------------------------------
# select_best_per_group
# ---------------------------------------------------------------------------


def test_pass1_only_when_each_record_has_unique_program():
    """With distinct rewritten_program texts, pass 2 is a no-op: every record
    survives with n_edit_paths == 1 and n_programs == |group|."""
    records = [
        _record(
            error_id="e1",
            rewritten_program=f"prog-{i}",
            modifications=[{"variable_name": f"x{i}", "original": "a", "replacement": "b"}],
        )
        for i in range(3)
    ]
    kept, _ = fc.select_best_per_group(
        records, tau=0.0, k_cells=None, min_grid_size=None, keep_nested=False
    )
    assert len(kept) == 3
    assert {r["rewritten_program"] for r in kept} == {"prog-0", "prog-1", "prog-2"}
    assert all(r["n_edit_paths"] == 1 for r in kept)
    assert all(r["n_programs"] == 3 for r in kept)


def test_pass2_collapses_same_program_text_keeping_simplest_edit():
    """Two records share a rewritten_program text but differ in mod count;
    the one with fewer mods wins and n_edit_paths == 2."""
    fewer_mods = _record(
        error_id="e1",
        rewritten_program="def solver(I): return I\n",
        modifications=[{"variable_name": "x1", "original": "foo", "replacement": "bar"}],
    )
    more_mods = _record(
        error_id="e1",
        rewritten_program="def solver(I): return I\n",
        modifications=[
            {"variable_name": "x1", "original": "foo", "replacement": "bar"},
            {"variable_name": "x2", "original": "baz", "replacement": "qux"},
        ],
    )
    kept, _ = fc.select_best_per_group(
        [more_mods, fewer_mods],
        tau=0.0,
        k_cells=None,
        min_grid_size=None,
        keep_nested=False,
    )
    assert len(kept) == 1
    winner = kept[0]
    assert len(winner["modifications"]) == 1
    assert winner["n_edit_paths"] == 2
    assert winner["n_programs"] == 1


def test_n_programs_is_per_error_group():
    """Group A has 3 distinct programs; group B has 1. Each kept record carries
    n_programs for *its own* group, not a global count."""
    group_a = [
        _record(
            error_id="eA",
            rewritten_program=f"progA-{i}",
            modifications=[{"variable_name": f"x{i}", "original": "a", "replacement": "b"}],
        )
        for i in range(3)
    ]
    group_b = [
        _record(
            error_id="eB",
            rewritten_program="progB-only",
            modifications=[{"variable_name": "y", "original": "a", "replacement": "b"}],
        )
    ]
    kept, _ = fc.select_best_per_group(
        group_a + group_b,
        tau=0.0,
        k_cells=None,
        min_grid_size=None,
        keep_nested=False,
    )
    by_error = {}
    for r in kept:
        by_error.setdefault(r["error_id"], []).append(r)

    assert len(by_error["eA"]) == 3
    assert all(r["n_programs"] == 3 for r in by_error["eA"])
    assert len(by_error["eB"]) == 1
    assert by_error["eB"][0]["n_programs"] == 1
