"""Build the prompts for solver synthesis.

Fills ``prompts/synthesize_solver_program.md`` for each task with the DSL, the
task's training pairs (the test pair only with ``include_test``) and, depending
on the retrieval mode, no example tasks (``zero-shot``), random ones
(``random``) or the most similar ones (``embedding``, see ``src.retrieval``).
The section introducing the examples, and the optional task description, are
added only when present. Used by ``scripts/synthesize_solver_programs.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

import numpy as np

from src.retrieval import retrieve_top_k
from src.utils import format_grid, format_task_grids_for_prompt, load_grid

ExampleGridsMode = Literal["full", "minimal", "none"]
RetrievalMode = Literal["zero-shot", "random", "embedding"]

IN_CONTEXT_HEADER_TEMPLATE = (
    "## Similar Solved Tasks\n\n"
    "Below are {num_examples} similar tasks that have been solved, along "
    "with their solutions. Study these examples to understand common "
    "patterns and how the DSL is used to express transformation rules."
)

TASK_DESCRIPTION_HEADER = (
    "## Task Description\n\n"
    "The following description(s) of the transformation were written by "
    "human participants who solved this task correctly:"
)

# Kept for backwards compatibility with existing tests / call sites that
# compare against a constant header.  Produced by interpolating
# ``num_examples`` into :data:`IN_CONTEXT_HEADER_TEMPLATE`; ``{num_examples}``
# remains a literal placeholder in this constant.
IN_CONTEXT_HEADER = IN_CONTEXT_HEADER_TEMPLATE

# Retrieval scores are rounded to this many decimal places before being
# written to prompt records, so the records are stable against
# floating-point noise and can be compared byte for byte across runs.
RETRIEVAL_SCORE_PRECISION = 6


@dataclass(frozen=True)
class EmbeddingRetrievalState:
    """Pre-resolved embedding state for ``--retrieval embedding``.

    The caller loads the document (demo pool) and query (target) embedding
    files once, intersects ``doc_ids`` with the available solver pool to
    derive ``keep``, and hands the bundle to
    :func:`build_solver_prompt_records`.  Per-task retrieval is then a
    single dot-product + sort against the cached matrices.
    """

    doc_embs: np.ndarray            # (N_docs, D)
    doc_ids: List[str]
    query_embs: np.ndarray          # (N_queries, D)
    query_id_to_idx: Dict[str, int]
    keep: Set[str]                  # doc_ids ∩ solver_pool


def format_in_context_example(
    task_id: str,
    solver_body: str,
    grid_data: Optional[Dict[str, Any]],
    example_grids: ExampleGridsMode,
) -> str:
    """Render one in-context demo as a markdown block.

    ``example_grids`` controls *what grids* are shown alongside the solver
    program body:

    * ``"full"``     — all train pairs + test I/O for the demo task.
    * ``"minimal"``  — first train pair only (one Input + one Output).
    * ``"none"``     — solver body only, no grids.

    *grid_data* is the dict returned by :func:`src.utils.load_grid`; it is
    consulted only for ``full`` and ``minimal``.  ``None`` is allowed when
    ``example_grids == "none"``.
    """
    parts: List[str] = [f"### Task {task_id}"]

    if example_grids == "full":
        if grid_data is None:
            raise ValueError(
                f"example_grids='full' requires grid_data for task {task_id!r}."
            )
        parts.append(format_task_grids_for_prompt(grid_data).rstrip())
    elif example_grids == "minimal":
        if grid_data is None:
            raise ValueError(
                f"example_grids='minimal' requires grid_data for task {task_id!r}."
            )
        train = grid_data.get("train") or []
        if not train:
            raise ValueError(
                f"example_grids='minimal' needs at least one train pair for task {task_id!r}."
            )
        first = train[0]
        parts.append("Example Input:")
        parts.append(format_grid(first["input"]))
        parts.append("Example Output:")
        parts.append(format_grid(first["output"]))
        parts.append("")  # blank line between grid block and solver program
    elif example_grids == "none":
        pass
    else:
        raise ValueError(f"Unknown example_grids mode: {example_grids!r}")

    parts.append("**Solver Program:**")
    parts.append(f"```python\n{solver_body}\n```")
    return "\n".join(parts)


def _pick_demos(
    retrieval: RetrievalMode,
    target_task_id: str,
    pool_task_ids: Sequence[str],
    num_examples: int,
    rng: Optional[np.random.Generator],
    embedding_state: Optional[EmbeddingRetrievalState] = None,
) -> Tuple[List[str], Optional[List[float]]]:
    """Choose demo task IDs for *target_task_id* under *retrieval* mode.

    Returns ``(picks, scores)``.  ``scores`` is ``None`` for zero-shot and
    random (no similarity signal); embedding retrieval populates it with
    cosine similarities rounded to :data:`RETRIEVAL_SCORE_PRECISION`.
    """
    if retrieval == "zero-shot":
        return [], None
    if retrieval == "random":
        if rng is None:
            raise ValueError("random retrieval requires a seeded np.random.Generator.")
        eligible = [tid for tid in pool_task_ids if tid != target_task_id]
        if len(eligible) < num_examples:
            raise ValueError(
                f"Solver pool has {len(eligible)} eligible tasks (excluding the target "
                f"{target_task_id!r}) but --num-examples={num_examples}."
            )
        picks = rng.choice(eligible, size=num_examples, replace=False).tolist()
        return picks, None
    if retrieval == "embedding":
        if embedding_state is None:
            raise ValueError(
                "embedding retrieval requires an EmbeddingRetrievalState."
            )
        if target_task_id not in embedding_state.query_id_to_idx:
            raise KeyError(
                f"Target task {target_task_id!r} has no query embedding "
                f"(known query IDs: {len(embedding_state.query_id_to_idx)})."
            )
        qi = embedding_state.query_id_to_idx[target_task_id]
        results = retrieve_top_k(
            query_emb=embedding_state.query_embs[qi],
            doc_embs=embedding_state.doc_embs,
            doc_task_ids=embedding_state.doc_ids,
            k=num_examples,
            keep=embedding_state.keep,
            exclude={target_task_id},
        )
        picks = [tid for tid, _ in results]
        scores = [round(s, RETRIEVAL_SCORE_PRECISION) for _, s in results]
        return picks, scores
    raise ValueError(f"Unknown retrieval mode: {retrieval!r}")


def build_solver_prompt_records(
    task_ids: Sequence[str],
    template: str,
    dsl_code: str,
    task_dir: Path,
    demo_task_dir: Path,
    solver_pool: Dict[str, str],
    retrieval: RetrievalMode,
    num_examples: int,
    example_grids: ExampleGridsMode,
    include_test: bool,
    seed: Optional[int],
    verbose: bool = False,
    embedding_state: Optional[EmbeddingRetrievalState] = None,
    descriptions: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """Build one prompt record per task_id.

    Returned records contain ``task_id``, ``entry_index``, ``prompt``,
    ``retrieved_tasks``, ``retrieval_scores``, and ``has_description``.
    Tasks whose grids fail to load are skipped with a warning.

    *descriptions* optionally maps task_id to NL description text (see
    :func:`src.utils.load_descriptions`); tasks present in the mapping get a
    ``## Task Description`` block injected at the ``{{task_description}}``
    placeholder, all others render identically to a run without descriptions.

    For ``retrieval="embedding"`` the caller must supply an
    :class:`EmbeddingRetrievalState` carrying pre-loaded document and
    query matrices plus the ``keep`` set of pool-eligible document IDs.
    """
    rng: Optional[np.random.Generator] = None
    if retrieval == "random":
        if seed is None:
            raise ValueError("--retrieval random requires --seed for reproducibility.")
        rng = np.random.default_rng(seed)
    if retrieval == "embedding" and embedding_state is None:
        raise ValueError(
            "--retrieval embedding requires an EmbeddingRetrievalState "
            "(pass embedding_state=...)."
        )

    pool_ids: List[str] = sorted(solver_pool) if solver_pool else []

    records: List[Dict[str, Any]] = []
    skipped = 0
    for task_id in task_ids:
        target_grid = load_grid(task_id, [task_dir])
        if target_grid is None:
            skipped += 1
            continue

        target_grid_for_prompt = (
            target_grid
            if include_test
            else {k: v for k, v in target_grid.items() if k != "test"}
        )
        target_block = format_task_grids_for_prompt(target_grid_for_prompt).rstrip()

        picks, scores = _pick_demos(
            retrieval=retrieval,
            target_task_id=task_id,
            pool_task_ids=pool_ids,
            num_examples=num_examples,
            rng=rng,
            embedding_state=embedding_state,
        )

        if picks:
            formatted: List[str] = []
            for demo_id in picks:
                demo_grid = (
                    load_grid(demo_id, [demo_task_dir]) if example_grids != "none" else None
                )
                formatted.append(
                    format_in_context_example(
                        task_id=demo_id,
                        solver_body=solver_pool[demo_id],
                        grid_data=demo_grid,
                        example_grids=example_grids,
                    )
                )
            in_context_block = (
                IN_CONTEXT_HEADER_TEMPLATE.format(num_examples=len(picks))
                + "\n\n"
                + "\n\n".join(formatted)
            )
        else:
            in_context_block = ""

        desc_text = (descriptions or {}).get(task_id)
        description_block = (
            TASK_DESCRIPTION_HEADER + "\n\n" + desc_text if desc_text else ""
        )

        prompt = (
            template
            .replace("{{dsl_code}}", dsl_code)
            .replace("{{in_context_examples}}", in_context_block)
            .replace("{{target_examples}}", target_block)
            .replace("{{task_description}}", description_block)
        )
        prompt = re.sub(r"\n{3,}", "\n\n", prompt)

        records.append(
            {
                "task_id": task_id,
                "entry_index": len(records),
                "prompt": prompt,
                "retrieved_tasks": list(picks),
                "retrieval_scores": list(scores) if scores is not None else None,
                "has_description": bool(desc_text),
            }
        )

        if verbose and len(records) == 1:
            print("\n" + "=" * 60)
            print("EXAMPLE PROMPT (first record):")
            print("=" * 60)
            print(f"Task ID: {task_id}")
            print("-" * 60)
            print(prompt)
            print("=" * 60 + "\n")

    if skipped:
        print(f"Built {len(records)} prompt records, skipped {skipped} (missing grid).")
    else:
        print(f"Built {len(records)} prompt records.")
    return records


def build_solver_stem(
    num_examples: int,
    example_grids: ExampleGridsMode,
    n_tasks: int,
    num_samples: int,
    include_test: bool,
    with_descriptions: bool = False,
) -> str:
    """Build the output-file stem for a solver-synthesis run.

    Format: ``solver_k{K}_{example_grids}_{N}tasks[_withtest][_nl][_n{S}]``.
    Deviations from defaults are stamped; defaults (test hidden; no NL
    descriptions; ``--num-samples 1``) leave no marker.
    """
    stem = f"solver_k{num_examples}_{example_grids}_{n_tasks}tasks"
    if include_test:
        stem += "_withtest"
    if with_descriptions:
        stem += "_nl"
    if num_samples > 1:
        stem += f"_n{num_samples}"
    return stem
