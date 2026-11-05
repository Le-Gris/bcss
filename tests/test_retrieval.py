"""Unit tests for ``src.retrieval``.

Covers the contract that ``--retrieval embedding`` relies on:

* ``load_embeddings`` round-trips a small npz preserving ids and shape;
* ``retrieve_top_k`` orders by descending cosine similarity;
* ``keep`` and ``exclude`` filters are applied *post-sort* but pre-truncation;
* the ``1e-10`` epsilon prevents NaN for zero-norm queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import load_embeddings, retrieve_top_k


def _write_npz(tmp_path: Path, name: str, embeddings: np.ndarray, task_ids: list[str]):
    path = tmp_path / name
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float32),
        task_ids=np.array(task_ids),
    )
    return path


# ---------------------------------------------------------------------------
# load_embeddings
# ---------------------------------------------------------------------------


def test_load_embeddings_roundtrip(tmp_path: Path):
    embs = np.arange(12, dtype=np.float32).reshape(3, 4)
    ids = ["a", "b", "c"]
    path = _write_npz(tmp_path, "round.npz", embs, ids)

    loaded_embs, loaded_ids = load_embeddings(path)
    np.testing.assert_array_equal(loaded_embs, embs)
    assert loaded_ids == ids
    assert isinstance(loaded_ids, list)
    assert all(isinstance(t, str) for t in loaded_ids)


def test_load_embeddings_rejects_missing_keys(tmp_path: Path):
    path = tmp_path / "bad.npz"
    np.savez_compressed(path, foo=np.zeros(3))
    with pytest.raises(ValueError, match="expected keys"):
        load_embeddings(path)


def test_load_embeddings_rejects_row_count_mismatch(tmp_path: Path):
    path = tmp_path / "mismatch.npz"
    np.savez_compressed(
        path,
        embeddings=np.zeros((3, 4), dtype=np.float32),
        task_ids=np.array(["a", "b"]),
    )
    with pytest.raises(ValueError, match="rows but"):
        load_embeddings(path)


# ---------------------------------------------------------------------------
# retrieve_top_k — ordering
# ---------------------------------------------------------------------------


def test_retrieve_top_k_ordering_matches_hand_computed_cosine():
    # 4 docs in 2-D + 1 query pointed along the x-axis.  Hand-computed:
    #   doc a = (1, 0)  -> sim 1.0
    #   doc b = (1, 1)  -> sim 1/sqrt(2) ≈ 0.7071
    #   doc c = (0, 1)  -> sim 0
    #   doc d = (-1, 0) -> sim -1.0
    doc_embs = np.array(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    doc_ids = ["a", "b", "c", "d"]
    query = np.array([1.0, 0.0], dtype=np.float32)

    results = retrieve_top_k(query, doc_embs, doc_ids, k=3)
    ids = [tid for tid, _ in results]
    scores = [s for _, s in results]
    assert ids == ["a", "b", "c"]
    assert scores[0] == pytest.approx(1.0, abs=1e-6)
    assert scores[1] == pytest.approx(1 / np.sqrt(2), abs=1e-6)
    assert scores[2] == pytest.approx(0.0, abs=1e-6)


def test_retrieve_top_k_zero_query_returns_results_not_nan():
    """The 1e-10 epsilon must keep cosine well-defined for zero queries."""
    doc_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    doc_ids = ["a", "b"]
    query = np.zeros(2, dtype=np.float32)

    results = retrieve_top_k(query, doc_embs, doc_ids, k=2)
    assert len(results) == 2
    for _, score in results:
        assert not np.isnan(score)


# ---------------------------------------------------------------------------
# retrieve_top_k — filters
# ---------------------------------------------------------------------------


def test_retrieve_top_k_keep_filters_pool():
    doc_embs = np.array(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32
    )
    doc_ids = ["a", "b", "c", "d"]
    query = np.array([1.0, 0.0], dtype=np.float32)

    # Top-1 of {a, c} is a (sim 1.0).  Without keep we'd get a then b.
    results = retrieve_top_k(query, doc_embs, doc_ids, k=2, keep={"a", "c"})
    ids = [tid for tid, _ in results]
    assert ids == ["a", "c"]


def test_retrieve_top_k_exclude_drops_self_match():
    doc_embs = np.array(
        [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32
    )
    doc_ids = ["target", "neighbour", "far"]
    query = np.array([1.0, 0.0], dtype=np.float32)

    # Without exclude, 'target' would be top-1 (sim 1.0).  With exclude, the
    # ranking shifts to ['neighbour', 'far'].
    results = retrieve_top_k(
        query, doc_embs, doc_ids, k=2, exclude={"target"}
    )
    ids = [tid for tid, _ in results]
    assert ids == ["neighbour", "far"]


def test_retrieve_top_k_raises_when_eligible_below_k():
    doc_embs = np.eye(3, dtype=np.float32)
    doc_ids = ["a", "b", "c"]
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # keep={'a'} leaves only 1 eligible doc; asking for k=2 must raise so we
    # never silently corrupt downstream prompts.
    with pytest.raises(ValueError, match="eligible documents"):
        retrieve_top_k(query, doc_embs, doc_ids, k=2, keep={"a"})


def test_retrieve_top_k_rejects_nonpositive_k():
    doc_embs = np.eye(2, dtype=np.float32)
    with pytest.raises(ValueError, match="k must be positive"):
        retrieve_top_k(
            np.ones(2, dtype=np.float32), doc_embs, ["a", "b"], k=0
        )


def test_retrieve_top_k_rejects_shape_mismatch():
    doc_embs = np.eye(3, dtype=np.float32)
    with pytest.raises(ValueError, match="doc_embs has"):
        retrieve_top_k(
            np.ones(3, dtype=np.float32), doc_embs, ["a", "b"], k=1
        )
