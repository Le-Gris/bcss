"""Unit tests for ``src.filters``."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hodel_dsl.dsl_parser import ProgramRewriter

from src.filters import (
    build_dsl_namespace,
    compute_training_outputs,
    is_blank_grid,
    is_identity,
    is_training_copy,
    matches_ground_truth,
    passes_threshold,
    passes_word_count,
)


# ---------------------------------------------------------------------------
# passes_threshold
# ---------------------------------------------------------------------------


def test_passes_threshold_requires_exactly_one_threshold():
    g = [[0, 0], [0, 0]]
    with pytest.raises(ValueError):
        passes_threshold(g, g)
    with pytest.raises(ValueError):
        passes_threshold(g, g, tau=0.1, k_cells=1)


def test_passes_threshold_exact_match():
    g = [[1, 2], [3, 4]]
    assert passes_threshold(g, g, tau=0.0) is True
    assert passes_threshold(g, g, k_cells=0) is True


def test_passes_threshold_proportion():
    candidate = [[1, 2], [3, 5]]  # one of four cells differs
    target = [[1, 2], [3, 4]]
    assert passes_threshold(candidate, target, tau=0.25) is True
    assert passes_threshold(candidate, target, tau=0.24) is False


def test_passes_threshold_k_cells():
    candidate = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # one cell differs
    target = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert passes_threshold(candidate, target, k_cells=1) is True
    assert passes_threshold(candidate, target, k_cells=0) is False


def test_passes_threshold_min_grid_size_guards_small_grids():
    # 2x2 grid (4 cells) with 1 mismatched cell; min_grid_size=9 forces exact.
    candidate = [[1, 2], [3, 4]]
    target = [[1, 2], [3, 5]]
    assert passes_threshold(candidate, target, k_cells=1, min_grid_size=9) is False
    # Same grids without the guard pass at k_cells=1.
    assert passes_threshold(candidate, target, k_cells=1) is True
    # Large enough to skip the guard.
    big_candidate = [[0] * 5 for _ in range(5)]
    big_target = [row[:] for row in big_candidate]
    big_target[0][0] = 1
    assert passes_threshold(big_candidate, big_target, k_cells=1, min_grid_size=9) is True


def test_passes_threshold_rejects_shape_mismatch():
    a = [[0, 0], [0, 0]]
    b = [[0, 0, 0]]
    assert passes_threshold(a, b, tau=0.5) is False
    assert passes_threshold(a, b, k_cells=10) is False


# ---------------------------------------------------------------------------
# is_identity
# ---------------------------------------------------------------------------


def test_is_identity_exact_with_zero_tau():
    test_input = [[1, 0], [0, 1]]
    error = [[1, 0], [0, 1]]
    assert is_identity(error, test_input, tau=0.0) is True


def test_is_identity_tolerates_small_noise():
    test_input = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    error_with_noise = [[1, 0, 0], [0, 1, 0], [0, 0, 0]]  # 1/9 differs
    assert is_identity(error_with_noise, test_input, tau=0.2) is True
    assert is_identity(error_with_noise, test_input, tau=0.05) is False


# ---------------------------------------------------------------------------
# is_training_copy
# ---------------------------------------------------------------------------


def test_is_training_copy_matches_input():
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        {"input": [[5, 6], [7, 8]], "output": [[8, 7], [6, 5]]},
    ]
    assert is_training_copy([[5, 6], [7, 8]], train) == ("input", 1)


def test_is_training_copy_matches_output():
    train = [
        {"input": [[1, 2], [3, 4]], "output": [[9, 9], [9, 9]]},
    ]
    assert is_training_copy([[9, 9], [9, 9]], train) == ("output", 0)


def test_is_training_copy_no_match():
    train = [
        {"input": [[1, 2]], "output": [[3, 4]]},
    ]
    assert is_training_copy([[5, 6]], train) is None


# ---------------------------------------------------------------------------
# is_blank_grid
# ---------------------------------------------------------------------------


def test_is_blank_grid_all_zeros_any_shape():
    assert is_blank_grid([[0, 0], [0, 0]]) is True
    assert is_blank_grid([[0] * 5 for _ in range(5)]) is True


def test_is_blank_grid_rejects_non_zero():
    assert is_blank_grid([[0, 0], [0, 1]]) is False


def test_is_blank_grid_with_allowed_shapes():
    blank_3x3 = [[0] * 3 for _ in range(3)]
    blank_5x5 = [[0] * 5 for _ in range(5)]
    assert is_blank_grid(blank_3x3, allowed_shapes=[(3, 3)]) is True
    assert is_blank_grid(blank_5x5, allowed_shapes=[(3, 3)]) is False
    assert is_blank_grid(blank_5x5, allowed_shapes=[(3, 3), (5, 5)]) is True


def test_is_blank_grid_empty_input():
    assert is_blank_grid([]) is False
    assert is_blank_grid([[]]) is False


# ---------------------------------------------------------------------------
# matches_ground_truth
# ---------------------------------------------------------------------------


def test_matches_ground_truth_exact():
    test_output = ((3, 2, 1), (6, 5, 4))
    assert matches_ground_truth(test_output, test_output, tau=0.0) is True
    # A different grid does not match
    assert matches_ground_truth(((1, 2, 3), (4, 5, 6)), test_output, tau=0.0) is False


def test_matches_ground_truth_threshold_tolerance():
    test_output = [[1, 2, 3], [4, 5, 6]]
    one_off = [[1, 2, 3], [4, 5, 0]]  # 1/6 cells differ ≈ 0.167
    assert matches_ground_truth(one_off, test_output, tau=0.2) is True
    assert matches_ground_truth(one_off, test_output, tau=0.1) is False


def test_matches_ground_truth_shape_mismatch():
    assert matches_ground_truth([[0]], [[0, 0], [0, 0]], tau=1.0) is False


# ---------------------------------------------------------------------------
# passes_word_count
# ---------------------------------------------------------------------------


def test_passes_word_count_empty_and_none():
    assert passes_word_count("", 1) is False
    assert passes_word_count("   ", 1) is False
    assert passes_word_count(None, 1) is False


def test_passes_word_count_single_word():
    assert passes_word_count("rotated", 1) is True
    assert passes_word_count("rotated", 2) is False


def test_passes_word_count_multi_word_boundary():
    text = "rotated 90 degrees clockwise"
    assert passes_word_count(text, 4) is True
    assert passes_word_count(text, 5) is False


def test_passes_word_count_collapses_whitespace():
    # Tabs, newlines, runs of spaces all count as one separator.
    assert passes_word_count("a\tb\n c", 3) is True
    assert passes_word_count("a   b", 2) is True


def test_passes_word_count_non_positive_threshold_always_passes():
    assert passes_word_count("", 0) is True
    assert passes_word_count(None, 0) is True
    assert passes_word_count("any text", -1) is True


# ---------------------------------------------------------------------------
# compute_training_outputs
# ---------------------------------------------------------------------------


TRAIN_INPUTS = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]


def test_compute_training_outputs_identity_returns_inputs():
    ns = build_dsl_namespace()
    program = "def solver(I):\n    O = I\n    return O"
    out = compute_training_outputs(program, TRAIN_INPUTS, ns, ProgramRewriter())
    assert out == TRAIN_INPUTS


def test_compute_training_outputs_executes_transformation():
    ns = build_dsl_namespace()
    program = "def solver(I):\n    O = rot90(I)\n    return O"
    out = compute_training_outputs(program, TRAIN_INPUTS, ns, ProgramRewriter())
    assert out == [[[3, 1], [4, 2]], [[7, 5], [8, 6]]]


def test_compute_training_outputs_unparseable_program_yields_none_per_input():
    ns = build_dsl_namespace()
    out = compute_training_outputs(
        "def solver(I):\n    O = (((", TRAIN_INPUTS, ns, ProgramRewriter()
    )
    assert out == [None, None]


def test_compute_training_outputs_edge_cases():
    ns = build_dsl_namespace()
    program = "def solver(I):\n    O = I\n    return O"
    # Empty program -> one None per input; no inputs -> empty list.
    assert compute_training_outputs("", TRAIN_INPUTS, ns, ProgramRewriter()) == [
        None,
        None,
    ]
    assert compute_training_outputs(program, [], ns, ProgramRewriter()) == []
