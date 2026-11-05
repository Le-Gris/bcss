"""Contract tests for the solver evaluator.

Covers the body-extraction contract (the ``record["program"]["program"]``
nesting vs a bare string), the parse/exec/score path against a task's own
train/test grids, and the failure branches.  Conventions mirror
``tests/test_dsl_parser.py`` and ``tests/test_control_records.py``: explicit
``sys.path`` insertion, ``tmp_path``-materialised ARC task JSONs, plain
``assert``.
"""

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import evaluate_synthesized_solver_programs as ev  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def ctx():
    """A fresh (rewriter, dsl_namespace) pair, mirroring the script's main()."""
    rewriter = ev.ProgramRewriter(
        dsl_symbols_path=str(_root / "hodel_dsl" / "dsl.py"),
        dsl_constants_path=str(_root / "hodel_dsl" / "primitives.py"),
    )
    return rewriter, ev.build_dsl_namespace()


def _write_task(task_dir: Path, task_id: str, train: list, test: list) -> None:
    """Materialise an ARC-format task JSON at ``task_dir/<task_id>.json``."""
    (task_dir / f"{task_id}.json").write_text(
        json.dumps({"train": train, "test": test})
    )


def _vmirror_task(task_dir: Path, task_id: str = "t1", *, with_test_output: bool = True):
    """A task whose rule is 'reverse each row' (``vmirror``)."""
    test_pair = {"input": [[7, 8, 9]]}
    if with_test_output:
        test_pair["output"] = [[9, 8, 7]]
    _write_task(
        task_dir,
        task_id,
        train=[
            {"input": [[1, 2, 3]], "output": [[3, 2, 1]]},
            {"input": [[4, 5, 6]], "output": [[6, 5, 4]]},
        ],
        test=[test_pair],
    )
    return task_id


# ---------------------------------------------------------------------------
# evaluate_record — happy path and extraction shapes
# ---------------------------------------------------------------------------


def test_happy_path_nested_program(ctx, tmp_path):
    """A correct solver scores a perfect solve.

    Body arrives in the nested shape ``record["program"]["program"]``.
    """
    rewriter, ns = ctx
    tid = _vmirror_task(tmp_path)
    record = {"task_id": tid, "program": {"program": "O = vmirror(I)"}}

    r = ev.evaluate_record(record, tmp_path, rewriter, ns)

    assert r["is_syntactically_valid"]
    assert r["exec_success"]
    assert r["train_total"] == 2
    assert r["train_pass_count"] == r["train_total"]
    assert r["all_training_correct"] is True
    assert r["test_match"] is True
    assert r["strict_solve"] is True
    assert r["mismatch_proportion"] == 0.0


def test_bare_string_program(ctx, tmp_path):
    """A bare-string ``program`` evaluates identically to the nested dict shape."""
    rewriter, ns = ctx
    tid = _vmirror_task(tmp_path)
    record = {"task_id": tid, "program": "O = vmirror(I)"}

    r = ev.evaluate_record(record, tmp_path, rewriter, ns)

    assert r["exec_success"]
    assert r["strict_solve"] is True


def test_wrong_but_valid_solver(ctx, tmp_path):
    """A valid solver that doesn't match: executes, but fails train and test."""
    rewriter, ns = ctx
    tid = _vmirror_task(tmp_path)
    record = {"task_id": tid, "program": {"program": "O = identity(I)"}}

    r = ev.evaluate_record(record, tmp_path, rewriter, ns)

    assert r["exec_success"]
    assert r["all_training_correct"] is False
    assert r["train_pass_count"] == 0
    assert r["test_match"] is False
    assert r["strict_solve"] is False


# ---------------------------------------------------------------------------
# evaluate_record — failure branches
# ---------------------------------------------------------------------------


def test_parse_failure(ctx, tmp_path):
    """A syntactically broken body is flagged, not executed."""
    rewriter, ns = ctx
    tid = _vmirror_task(tmp_path)
    record = {"task_id": tid, "program": {"program": "O = "}}

    r = ev.evaluate_record(record, tmp_path, rewriter, ns)

    assert r["is_syntactically_valid"] is False
    assert r["parse_error"] is not None
    assert r["exec_success"] is False


def test_runtime_failure(ctx, tmp_path):
    """A program that parses but raises at execution is caught."""
    rewriter, ns = ctx
    tid = _vmirror_task(tmp_path)
    # divide(ONE, ZERO) -> 1 // 0 -> ZeroDivisionError at exec time.
    record = {"task_id": tid, "program": {"program": "O = divide(ONE, ZERO)"}}

    r = ev.evaluate_record(record, tmp_path, rewriter, ns)

    assert r["is_syntactically_valid"] is True
    assert r["exec_success"] is False
    assert r["runtime_error"] is not None


def test_no_test_output_is_tristate(ctx, tmp_path):
    """When the task has no test output, ``test_match`` is None but train scores."""
    rewriter, ns = ctx
    tid = _vmirror_task(tmp_path, with_test_output=False)
    record = {"task_id": tid, "program": {"program": "O = vmirror(I)"}}

    r = ev.evaluate_record(record, tmp_path, rewriter, ns)

    assert r["exec_success"]
    assert r["test_match"] is None
    assert r["all_training_correct"] is True
    assert r["train_pass_count"] == r["train_total"] == 2
    assert r["strict_solve"] is False  # no test confirmation


def test_invalid_json_sentinel(ctx, tmp_path):
    """The ``{"error": "Invalid JSON"}`` sentinel is an extraction failure, no crash."""
    rewriter, ns = ctx
    tid = _vmirror_task(tmp_path)
    record = {"task_id": tid, "program": {"error": "Invalid JSON", "raw": "not json"}}

    r = ev.evaluate_record(record, tmp_path, rewriter, ns)

    assert r["is_syntactically_valid"] is False
    assert r["parse_error"] is not None
    assert r["exec_success"] is False


# ---------------------------------------------------------------------------
# wrap_solver_body unit
# ---------------------------------------------------------------------------


def test_wrap_solver_body_multiline():
    """A multi-line body is indented under ``def solver(I):`` with a trailing return."""
    wrapped = ev.wrap_solver_body("x1 = vmirror(I)\nO = hconcat(I, x1)")
    assert wrapped == (
        "def solver(I):\n"
        "    x1 = vmirror(I)\n"
        "    O = hconcat(I, x1)\n"
        "    return O"
    )


def test_wrap_solver_body_passthrough_for_full_function():
    """A body that already carries its own ``def`` line is returned unchanged."""
    full = "def solver(I):\n    O = vmirror(I)\n    return O"
    assert ev.wrap_solver_body(full) == full
