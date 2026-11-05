"""The tolerance for "same grid" and the checks for trivial errors.

``passes_threshold`` decides whether two grids match within the tolerance (at
most k wrong cells, or a maximum mismatch proportion). The same function is
used to drop trivial errors before synthesis (``scripts/prefilter_errors.py``)
and to accept synthesized programs afterwards (``scripts/filter_candidates.py``),
so both steps apply the same rule. The trivial-error checks are
``is_identity``, ``is_training_copy``, ``is_blank_grid`` and
``matches_ground_truth``.
"""

from __future__ import annotations

import copy
from typing import Any, Iterable, List, Optional, Tuple

from hodel_dsl.dsl_parser import ProgramRewriter
from src.utils import grid_array, grid_to_tuple


# ---------------------------------------------------------------------------
# Threshold predicate (the single source of truth)
# ---------------------------------------------------------------------------


def passes_threshold(
    candidate: Any,
    target: Any,
    *,
    tau: Optional[float] = None,
    k_cells: Optional[int] = None,
    min_grid_size: Optional[int] = None,
) -> bool:
    """Return True if *candidate* matches *target* within the configured tolerance.

    Exactly one of ``tau`` and ``k_cells`` should be provided.

    Semantics:

    * ``tau`` (mismatch-proportion mode): pass iff
      ``mismatch_proportion <= tau``.  ``mismatch_proportion`` is the fraction
      of cells that differ; shape mismatches and invalid grids count as 1.0
      (never pass).
    * ``k_cells`` (absolute-cell mode): pass iff
      ``mismatched_cells <= k_cells``.  Shape mismatches never pass.  When
      ``min_grid_size`` is provided, grids smaller than ``min_grid_size``
      cells must match exactly — this guards against accepting "3 cells off
      out of 9" as ground-truth on tiny grids.
    """
    if (tau is None) == (k_cells is None):
        raise ValueError("Exactly one of `tau` or `k_cells` must be provided.")

    a1 = grid_array(candidate)
    a2 = grid_array(target)
    if a1 is None or a2 is None:
        return False
    if a1.shape != a2.shape:
        return False
    if a1.size == 0:
        return True

    diff_count = int((a1 != a2).sum())

    if tau is not None:
        proportion = diff_count / a1.size
        return proportion <= tau

    # k_cells mode
    if min_grid_size is not None and a1.size < min_grid_size:
        return diff_count == 0
    return diff_count <= k_cells  # type: ignore[operator]


# ---------------------------------------------------------------------------
# Trivial-error detectors
# ---------------------------------------------------------------------------


def is_identity(
    error_grid: Any,
    test_input: Any,
    *,
    tau: Optional[float] = None,
    k_cells: Optional[int] = None,
    min_grid_size: Optional[int] = None,
) -> bool:
    """True if *error_grid* ≈ *test_input* under the configured threshold."""
    return passes_threshold(
        error_grid, test_input,
        tau=tau, k_cells=k_cells, min_grid_size=min_grid_size,
    )


def is_training_copy(
    error_grid: Any,
    train_examples: Iterable[dict],
) -> Optional[Tuple[str, int]]:
    """Exact match against any training example's input or output.

    Returns ``("input", i)`` or ``("output", i)`` when the *i*-th training
    example matches, otherwise ``None``.
    """
    target = grid_to_tuple(error_grid)
    for i, example in enumerate(train_examples):
        if "input" in example and grid_to_tuple(example["input"]) == target:
            return ("input", i)
        if "output" in example and grid_to_tuple(example["output"]) == target:
            return ("output", i)
    return None


def is_blank_grid(
    error_grid: Any,
    allowed_shapes: Optional[List[Tuple[int, int]]] = None,
) -> bool:
    """True if *error_grid* is all zeros.

    When ``allowed_shapes`` is ``None``, blank grids of any shape qualify.
    When provided, only blank grids whose ``(rows, cols)`` is in
    ``allowed_shapes`` qualify.
    """
    if not error_grid or not error_grid[0]:
        return False
    if not all(cell == 0 for row in error_grid for cell in row):
        return False
    if allowed_shapes is None:
        return True
    rows = len(error_grid)
    cols = len(error_grid[0])
    return (rows, cols) in allowed_shapes


def matches_ground_truth(
    error_grid: Any,
    test_output: Any,
    *,
    tau: Optional[float] = None,
    k_cells: Optional[int] = None,
    min_grid_size: Optional[int] = None,
) -> bool:
    """True if *error_grid* ≈ the dataset-provided *test_output* under threshold.

    The task JSON is treated as authoritative — we do not re-derive the
    canonical output from a DSL solver here.  The prefilter pairs this with
    the participant's submitted grid to drop errors that already match the
    labeled correct answer.
    """
    return passes_threshold(
        error_grid, test_output,
        tau=tau, k_cells=k_cells, min_grid_size=min_grid_size,
    )


# ---------------------------------------------------------------------------
# Description-text detectors
# ---------------------------------------------------------------------------


def passes_word_count(text: Optional[str], min_words: int) -> bool:
    """True if *text* has at least *min_words* whitespace-separated tokens.

    Empty, whitespace-only, or ``None`` text returns False for any
    ``min_words >= 1``.  A non-positive *min_words* disables the check and
    always returns True.
    """
    if min_words <= 0:
        return True
    return len((text or "").split()) >= min_words


# ---------------------------------------------------------------------------
# DSL namespace builder (shared by prefilter and evaluate)
# ---------------------------------------------------------------------------


def build_dsl_namespace() -> dict:
    """Return a dict containing all hodel_dsl functions and primitive constants."""
    ns: dict = {}
    exec("from hodel_dsl.dsl import *", ns)
    exec("from hodel_dsl.primitives import *", ns)
    ns.pop("__builtins__", None)
    return ns


# ---------------------------------------------------------------------------
# Program execution on training inputs (shared by the error-classification
# producer, which needs each error program's per-example output grid)
# ---------------------------------------------------------------------------


def _grid_to_list(grid: Any) -> List[List[int]]:
    """Convert a grid tuple-of-tuples back to a list of lists for JSON."""
    return [list(row) for row in grid]


def _make_json_serializable(obj: Any) -> Any:
    """Recursively convert DSL output types (frozenset, etc.) to JSON-safe types."""
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_make_json_serializable(x) for x in sorted(list(obj), key=str)]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


def compute_training_outputs(
    rewritten_program: str,
    train_inputs: List[List[List[int]]],
    dsl_namespace: dict,
    rewriter: ProgramRewriter,
) -> List[List[List[int]] | None]:
    """Execute a rewritten program on each training input and return output grids.

    Returns a list parallel to *train_inputs*.  Each entry is a grid (list of
    lists) on success or ``None`` on failure.  If the program cannot be parsed
    or compiled at all, every entry will be ``None``.
    """
    n = len(train_inputs)
    if not rewritten_program or n == 0:
        return [None] * n

    # Parse -> clean -> rewrite to canonical form, then compile once.
    try:
        rewriter.parse(rewritten_program)
        canonical = rewriter.clean().rewrite()
        exec_ns = copy.copy(dsl_namespace)
        exec(canonical, exec_ns)
        solver_fn = exec_ns.get("solver")
        if solver_fn is None:
            return [None] * n
    except Exception:
        return [None] * n

    # Run on each training input independently.
    outputs: List[List[List[int]] | None] = []
    for inp in train_inputs:
        try:
            result = solver_fn(grid_to_tuple(inp))
            if isinstance(result, (tuple, list)) and len(result) > 0:
                outputs.append(_make_json_serializable(_grid_to_list(result)))
            else:
                outputs.append(None)
        except Exception:
            outputs.append(None)
    return outputs
