"""Find the most similar tasks by cosine similarity of their embeddings.

``load_embeddings`` reads an ``.npz`` written by
``scripts/generate_task_embeddings.py``; ``retrieve_top_k`` returns the k
example tasks closest to a query task. Pure NumPy, no API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np


def load_embeddings(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Load a ``.npz`` with keys ``embeddings`` (N, D) and ``task_ids`` (N,).

    Returns ``(embeddings, task_ids)`` with ``task_ids`` as ``list[str]``.
    The ``datasets`` key, if present, is ignored — only the matrix and
    the per-row task ID are needed for retrieval.
    """
    data = np.load(path)
    if "embeddings" not in data.files or "task_ids" not in data.files:
        raise ValueError(
            f"{path}: expected keys 'embeddings' and 'task_ids', "
            f"found {data.files!r}"
        )
    embeddings = data["embeddings"]
    task_ids = [str(t) for t in data["task_ids"].tolist()]
    if embeddings.shape[0] != len(task_ids):
        raise ValueError(
            f"{path}: embeddings has {embeddings.shape[0]} rows but "
            f"task_ids has {len(task_ids)} entries."
        )
    return embeddings, task_ids


def retrieve_top_k(
    query_emb: np.ndarray,
    doc_embs: np.ndarray,
    doc_task_ids: Sequence[str],
    k: int,
    keep: Optional[Set[str]] = None,
    exclude: Optional[Set[str]] = None,
) -> List[Tuple[str, float]]:
    """Return the top-*k* ``(task_id, cosine_similarity)`` pairs.

    Filtering is applied **after** the full similarity sort: ``keep`` (if
    supplied) restricts the pool to that set; ``exclude`` always drops the
    listed IDs (e.g. the target task itself, to avoid self-matches when
    docs and queries share IDs).

    Cosine similarity uses a ``1e-10`` epsilon on both norms, which protects
    against zero-vector queries without changing rankings for normal inputs.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    if doc_embs.shape[0] != len(doc_task_ids):
        raise ValueError(
            f"doc_embs has {doc_embs.shape[0]} rows but doc_task_ids has "
            f"{len(doc_task_ids)} entries."
        )

    query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    similarities = doc_norms @ query_norm  # (N,)

    # Iterate in descending-similarity order, applying filters lazily.
    order = np.argsort(similarities)[::-1]
    results: List[Tuple[str, float]] = []
    for idx in order:
        tid = doc_task_ids[idx]
        if exclude is not None and tid in exclude:
            continue
        if keep is not None and tid not in keep:
            continue
        results.append((tid, float(similarities[idx])))
        if len(results) >= k:
            break

    if len(results) < k:
        raise ValueError(
            f"Only {len(results)} eligible documents after filtering "
            f"(keep={None if keep is None else len(keep)}, "
            f"exclude={None if exclude is None else len(exclude)}), "
            f"but k={k}."
        )
    return results
