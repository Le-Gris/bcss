"""Unit tests for ``src.solver_pool``."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solver_pool import load_solver_pool


def test_loads_hodel_solvers():
    """The bundled hodel_dsl.solvers module must yield a non-empty pool of
    string bodies, one per ARC-AGI-1 training task that has a hand-written
    solver."""
    pool = load_solver_pool("hodel_dsl.solvers")
    assert isinstance(pool, dict)
    assert len(pool) > 0
    for task_id, body in pool.items():
        assert isinstance(task_id, str)
        assert task_id  # non-empty
        assert isinstance(body, str)
        assert body.strip()  # body is non-empty
        # No def line, no trailing return; should end with an assignment to O.
        assert not body.lstrip().startswith("def "), (
            f"Body for {task_id} still contains a def line; "
            "get_solver_body did not strip correctly."
        )


def test_missing_module_raises():
    with pytest.raises(ModuleNotFoundError):
        load_solver_pool("nonexistent.module.path")
