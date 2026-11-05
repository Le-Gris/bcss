"""Load the example solvers shown in solver-synthesis prompts.

``load_solver_pool`` reads a Python module of ``solve_<task_id>(I)`` functions
in the format of ``hodel_dsl/solvers.py`` (e.g. ``hodel_dsl.solvers``) and
returns ``{task_id: program body}``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Dict

from src.utils import get_solver_body, load_solvers


def load_solver_pool(source: str) -> Dict[str, str]:
    """Load a demo-solver pool keyed by task_id.

    *source* is an importable Python module path (e.g. ``hodel_dsl.solvers``)
    whose top-level functions are named ``solve_<task_id>(I)``.  The returned
    mapping holds the **function body** for each solver (no ``def`` line, no
    trailing ``return`` — ready to drop into a prompt as in-context example
    code).

    Raises
    ------
    ModuleNotFoundError
        The module path does not resolve to an importable Python file.
    ValueError
        The module resolves but contains no ``solve_*`` functions.
    """
    spec = importlib.util.find_spec(source)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(
            f"Solver source {source!r} is not importable. "
            "Pass a Python module path such as 'hodel_dsl.solvers'."
        )

    full_solvers = load_solvers(Path(spec.origin))
    if not full_solvers:
        raise ValueError(
            f"Solver source {source!r} resolved to {spec.origin} but contains "
            "no 'solve_<task_id>' functions."
        )

    return {task_id: get_solver_body(src) for task_id, src in full_solvers.items()}
