"""Contract test for the stage-3 error evaluator's ``--solvers-file`` override.

By default, stage 3 (``evaluate_synthesized_error_programs.py``) loads the
hand-written ``hodel_dsl.solvers``, which only covers ARC-AGI-1 *training*
tasks.  The override lets it score runs whose reference programs came from
solver synthesis (e.g. the evaluation split).  This test pins the
contract: a solver body present *only* in a synthesized map resolves and
round-trips through ``evaluate_record``, and the "No solver found" guard still
fires when the task is absent.

Conventions mirror ``tests/test_evaluate_synthesized_solver_programs.py``:
explicit ``sys.path`` insertion, ``tmp_path``-materialised task JSON, plain
``assert``.
"""

import json
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "scripts"))

import evaluate_synthesized_error_programs as ev  # noqa: E402

from src.utils import collapse_solver_bundles  # noqa: E402

# A made-up task id that is *not* in hodel_dsl.solvers, standing in for an
# evaluation-split task that only has a synthesized solver.
_EVAL_TASK = "zzeval01"
# Minimal valid DSL solver: copy the input through unchanged.
_SOLVER = "def solver(I):\n    x1 = identity(I)\n    O = x1\n    return O"


@pytest.fixture
def ctx():
    """A fresh (rewriter, dsl_namespace) pair, mirroring the script's main()."""
    rewriter = ev.ProgramRewriter(
        dsl_symbols_path=str(_root / "hodel_dsl" / "dsl.py"),
        dsl_constants_path=str(_root / "hodel_dsl" / "primitives.py"),
    )
    return rewriter, ev.build_dsl_namespace()


def _write_task(task_dir: Path, task_id: str) -> None:
    """Materialise a trivial identity ARC task at ``task_dir/<task_id>.json``."""
    (task_dir / f"{task_id}.json").write_text(
        json.dumps(
            {
                "train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}],
                "test": [{"input": [[5, 6], [7, 8]], "output": [[5, 6], [7, 8]]}],
            }
        )
    )


def _record() -> dict:
    """A synthesis result record: one no-op modify on the identity solver."""
    return {
        "task_id": _EVAL_TASK,
        "error_id": "e-1",
        "response_id": None,
        "modifications": [
            {"action": "modify", "variable_name": "x1", "replacement": "identity(I)"}
        ],
    }


def test_solvers_file_body_resolves_and_executes(ctx, tmp_path):
    """A body only in the synthesized map resolves and runs through evaluate_record."""
    rewriter, dsl_namespace = ctx
    _write_task(tmp_path, _EVAL_TASK)

    # Same path the evaluator's --solvers-file branch produces: a {task: [body]}
    # bundle collapsed by the shared 'first' selection.
    solvers = collapse_solver_bundles({_EVAL_TASK: [_SOLVER]})
    assert _EVAL_TASK in solvers

    result = ev.evaluate_record(
        record=_record(),
        solvers=solvers,
        error_lookup={},  # no participant entry needed for this contract
        task_dir=tmp_path,
        rewriter=rewriter,
        dsl_namespace=dsl_namespace,
    )

    assert result["parse_error"] != f"No solver found for task {_EVAL_TASK}"
    assert result["is_syntactically_valid"] is True
    assert result["execution_success"] is True


def test_missing_solver_still_guarded(ctx, tmp_path):
    """Without the override, the same eval-split task hits the No-solver guard."""
    rewriter, dsl_namespace = ctx
    _write_task(tmp_path, _EVAL_TASK)

    result = ev.evaluate_record(
        record=_record(),
        solvers={},  # stands in for hodel_dsl.solvers lacking the eval task
        error_lookup={},
        task_dir=tmp_path,
        rewriter=rewriter,
        dsl_namespace=dsl_namespace,
    )

    assert result["parse_error"] == f"No solver found for task {_EVAL_TASK}"
    assert result["execution_success"] is False
