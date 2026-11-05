"""Unit tests for eval-split reference resolution in ``scripts/classify_errors.py``.

The classifier renders the task's ground-truth solver as
``{{ref_program}}``.  It resolves that reference via ``load_solvers()``, which
only covers the hand-written ARC-AGI-1 *training* solvers.  Eval-split tasks
have no such solver, so the classifier falls back to the synthesized reference
stage 3 stamps onto each record as ``original_program``.  These pin ``resolve_reference_solver``: GT-hits-first for
training, the ``original_program`` fallback for eval, GT precedence when both
exist, and ``None`` when neither does.
"""

import importlib.util
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

# The script lives under scripts/ (not an importable package), so load it by path.
_spec = importlib.util.spec_from_file_location(
    "classify_errors", _root / "scripts" / "classify_errors.py"
)
ce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ce)


def test_training_record_prefers_hand_written_solver():
    """A task present in the solver map resolves to that (hand-written) source."""
    rec = {"task_id": "abc123", "original_program": "def solve_abc123(I):\n    return I"}
    solvers = {"abc123": "def solve_abc123(I):\n    x = f(I)\n    return x"}
    assert ce.resolve_reference_solver(rec, solvers) == solvers["abc123"]


def test_eval_record_falls_back_to_original_program():
    """An eval-split task (absent from the solver map) uses the record's reference."""
    rec = {"task_id": "8fbca751", "original_program": "def solve_8fbca751(I):\n    return I"}
    solvers = {}  # eval-split: no hand-written solver
    assert ce.resolve_reference_solver(rec, solvers) == rec["original_program"]


def test_hand_written_solver_wins_when_both_present():
    """GT precedence: the solver map takes priority over the record's copy."""
    rec = {"task_id": "abc123", "original_program": "SYNTHESIZED"}
    solvers = {"abc123": "HAND_WRITTEN"}
    assert ce.resolve_reference_solver(rec, solvers) == "HAND_WRITTEN"


def test_returns_none_when_no_reference_available():
    """Neither a hand-written solver nor a carried reference → None (record skipped)."""
    assert ce.resolve_reference_solver({"task_id": "zzz"}, {}) is None
    # Empty original_program is falsy → also None.
    assert ce.resolve_reference_solver({"task_id": "zzz", "original_program": ""}, {}) is None
