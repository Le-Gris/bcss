"""Unit tests for ``src.solver_prompts`` (solver-synthesis prompt construction).

These tests use a small in-memory ARC fixture so they are independent of any
on-disk corpus.  They lock in:

* random retrieval is seed-deterministic and picks different demos per
  target task;
* zero-shot omits the header and emits no triple-newline runs;
* the three ``--example-grids`` modes (full / minimal / none) render the
  expected number of grids per demo;
* the target test pair is hidden by default and exposed with
  ``--include-test``;
* the stem builder follows the "stamp deviations from defaults" convention;
* the optional NL description block is injected only for tasks in the
  ``descriptions`` mapping, and its absence leaves prompts byte-identical
  to a template without the placeholder.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.solver_prompts import (
    IN_CONTEXT_HEADER,
    RETRIEVAL_SCORE_PRECISION,
    TASK_DESCRIPTION_HEADER,
    EmbeddingRetrievalState,
    build_solver_prompt_records,
    build_solver_stem,
    format_in_context_example,
)

TEMPLATE = (
    "Header\n\n"
    "{{dsl_code}}\n\n"
    "{{in_context_examples}}\n\n"
    "## Target Task\n\n"
    "{{target_examples}}\n"
)
DSL_CODE = "<dsl placeholder>"


def _grid(rows):
    return [list(r) for r in rows]


def _make_task_dir(tmp_path: Path, name: str, tasks: dict) -> Path:
    """Materialise a directory of ARC-format task JSONs in *tmp_path*."""
    d = tmp_path / name
    d.mkdir()
    for tid, payload in tasks.items():
        (d / f"{tid}.json").write_text(json.dumps(payload))
    return d


@pytest.fixture
def fixture(tmp_path: Path):
    """Two target tasks, four demo tasks, with simple distinct grids."""
    targets = {
        "T1": {
            "train": [{"input": _grid([[1, 2]]), "output": _grid([[2, 1]])}],
            "test": [{"input": _grid([[3, 3]]), "output": _grid([[3, 3]])}],
        },
        "T2": {
            "train": [
                {"input": _grid([[4]]), "output": _grid([[4, 4]])},
                {"input": _grid([[5]]), "output": _grid([[5, 5]])},
            ],
            "test": [{"input": _grid([[6]]), "output": _grid([[6, 6]])}],
        },
    }
    demos = {
        "D1": {"train": [{"input": _grid([[1]]), "output": _grid([[2]])}], "test": []},
        "D2": {"train": [{"input": _grid([[3]]), "output": _grid([[4]])}], "test": []},
        "D3": {"train": [{"input": _grid([[5]]), "output": _grid([[6]])}], "test": []},
        "D4": {"train": [{"input": _grid([[7]]), "output": _grid([[8]])}], "test": []},
    }
    return {
        "target_dir": _make_task_dir(tmp_path, "targets", targets),
        "demo_dir": _make_task_dir(tmp_path, "demos", demos),
        "pool": {"D1": "O = a(I)", "D2": "O = b(I)", "D3": "O = c(I)", "D4": "O = d(I)"},
    }


# ---------------------------------------------------------------------------
# Random retrieval
# ---------------------------------------------------------------------------


def test_random_retrieval_is_seed_deterministic(fixture):
    common = dict(
        task_ids=["T1", "T2"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool=fixture["pool"],
        retrieval="random",
        num_examples=2,
        example_grids="minimal",
        include_test=False,
    )
    rec_a = build_solver_prompt_records(seed=42, **common)
    rec_b = build_solver_prompt_records(seed=42, **common)
    assert [r["retrieved_tasks"] for r in rec_a] == [r["retrieved_tasks"] for r in rec_b]
    assert [r["prompt"] for r in rec_a] == [r["prompt"] for r in rec_b]


def test_random_retrieval_picks_different_demos_per_task(fixture):
    """RNG must be seeded once before the per-task loop; otherwise every task
    gets the same demos.  Use K=2 from a pool of 4 to make collisions
    statistically unlikely."""
    records = build_solver_prompt_records(
        task_ids=["T1", "T2"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool=fixture["pool"],
        retrieval="random",
        num_examples=2,
        example_grids="minimal",
        include_test=False,
        seed=42,
    )
    assert records[0]["retrieved_tasks"] != records[1]["retrieved_tasks"]


def test_random_requires_seed(fixture):
    with pytest.raises(ValueError, match="requires --seed"):
        build_solver_prompt_records(
            task_ids=["T1"],
            template=TEMPLATE,
            dsl_code=DSL_CODE,
            task_dir=fixture["target_dir"],
            demo_task_dir=fixture["demo_dir"],
            solver_pool=fixture["pool"],
            retrieval="random",
            num_examples=2,
            example_grids="minimal",
            include_test=False,
            seed=None,
        )


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------


def test_zero_shot_omits_header_and_collapses_newlines(fixture):
    records = build_solver_prompt_records(
        task_ids=["T1"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool={},
        retrieval="zero-shot",
        num_examples=0,
        example_grids="none",
        include_test=False,
        seed=None,
    )
    assert len(records) == 1
    r = records[0]
    assert r["retrieved_tasks"] == []
    assert r["retrieval_scores"] is None
    assert IN_CONTEXT_HEADER not in r["prompt"]
    assert "## Similar Solved Tasks" not in r["prompt"]
    assert "\n\n\n" not in r["prompt"]


# ---------------------------------------------------------------------------
# example_grids modes
# ---------------------------------------------------------------------------


def test_example_grids_minimal_renders_one_pair():
    grid_data = {
        "train": [
            {"input": [[1]], "output": [[2]]},
            {"input": [[3]], "output": [[4]]},
        ],
        "test": [{"input": [[5]], "output": [[6]]}],
    }
    block = format_in_context_example("D1", "O = a(I)", grid_data, "minimal")
    assert block.count("Example Input:") == 1
    assert block.count("Example Output:") == 1
    # Only the first train pair should appear.
    assert "1" in block and "2" in block
    assert "3" not in block and "4" not in block
    # Test pair must not appear in minimal mode.
    assert "5" not in block and "6" not in block


def test_example_grids_full_renders_all_pairs():
    grid_data = {
        "train": [
            {"input": [[1]], "output": [[2]]},
            {"input": [[3]], "output": [[4]]},
        ],
        "test": [{"input": [[5]], "output": [[6]]}],
    }
    block = format_in_context_example("D1", "O = a(I)", grid_data, "full")
    assert "Example 1 Input:" in block
    assert "Example 2 Input:" in block
    assert "Test Example Input:" in block


def test_example_grids_none_omits_grids():
    grid_data = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[5]], "output": [[6]]}],
    }
    block = format_in_context_example("D1", "O = a(I)", grid_data, "none")
    # No grid headers of any kind.
    assert "Example Input:" not in block
    assert "Example 1 Input:" not in block
    assert "Test Example Input:" not in block
    # The solver body still appears.
    assert "O = a(I)" in block


# ---------------------------------------------------------------------------
# Test-pair visibility on the *target* task
# ---------------------------------------------------------------------------


def test_target_test_pair_hidden_by_default(fixture):
    records = build_solver_prompt_records(
        task_ids=["T1"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool={},
        retrieval="zero-shot",
        num_examples=0,
        example_grids="none",
        include_test=False,
        seed=None,
    )
    assert "Test Example Input:" not in records[0]["prompt"]
    assert "Test Example Output:" not in records[0]["prompt"]


def test_target_test_pair_shown_with_include_test(fixture):
    records = build_solver_prompt_records(
        task_ids=["T1"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool={},
        retrieval="zero-shot",
        num_examples=0,
        example_grids="none",
        include_test=True,
        seed=None,
    )
    assert "Test Example Input:" in records[0]["prompt"]
    assert "Test Example Output:" in records[0]["prompt"]


# ---------------------------------------------------------------------------
# NL task descriptions
# ---------------------------------------------------------------------------

# Mirrors the real template's shape around the placeholder: the description
# slot sits between the target grids and a following section, so an empty
# block must collapse back to the placeholder-free layout.
TEMPLATE_PRE_NL = (
    "Header\n\n"
    "{{dsl_code}}\n\n"
    "{{in_context_examples}}\n\n"
    "## Target Task\n\n"
    "{{target_examples}}\n\n"
    "## Instructions\n"
)
TEMPLATE_NL = TEMPLATE_PRE_NL.replace(
    "## Instructions", "{{task_description}}\n\n## Instructions"
)


def _zero_shot_records(fixture, template, **kwargs):
    return build_solver_prompt_records(
        task_ids=["T1"],
        template=template,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool={},
        retrieval="zero-shot",
        num_examples=0,
        example_grids="none",
        include_test=False,
        seed=None,
        **kwargs,
    )


def test_description_present_injects_block(fixture):
    records = _zero_shot_records(
        fixture,
        TEMPLATE_NL,
        descriptions={"T1": "flip the grid horizontally"},
    )
    r = records[0]
    assert TASK_DESCRIPTION_HEADER in r["prompt"]
    assert "flip the grid horizontally" in r["prompt"]
    assert "\n\n\n" not in r["prompt"]
    assert r["has_description"] is True


def test_description_absent_omits_block(fixture):
    records = _zero_shot_records(
        fixture,
        TEMPLATE_NL,
        descriptions={"T2": "irrelevant — describes a different task"},
    )
    r = records[0]
    assert "## Task Description" not in r["prompt"]
    assert "\n\n\n" not in r["prompt"]
    assert r["has_description"] is False


def test_no_descriptions_matches_pre_placeholder_template(fixture):
    """Adding the placeholder to a template must not change prompts when no
    descriptions are supplied — the back-compat guarantee for old runs."""
    old = _zero_shot_records(fixture, TEMPLATE_PRE_NL)
    new = _zero_shot_records(fixture, TEMPLATE_NL, descriptions=None)
    assert new[0]["prompt"] == old[0]["prompt"]
    assert new[0]["has_description"] is False


def test_real_template_renders_without_leftover_placeholders(fixture):
    template = (
        Path(__file__).resolve().parents[1]
        / "prompts"
        / "synthesize_solver_program.md"
    ).read_text()
    assert "{{task_description}}" in template
    records = _zero_shot_records(
        fixture, template, descriptions={"T1": "flip the grid horizontally"}
    )
    assert "{{" not in records[0]["prompt"]
    assert TASK_DESCRIPTION_HEADER in records[0]["prompt"]


# ---------------------------------------------------------------------------
# Stem builder
# ---------------------------------------------------------------------------


def test_stem_builder_reference_settings():
    """The plan's reference run yields ``solver_k3_minimal_400tasks_n4``."""
    stem = build_solver_stem(
        num_examples=3,
        example_grids="minimal",
        n_tasks=400,
        num_samples=4,
        include_test=False,
    )
    assert stem == "solver_k3_minimal_400tasks_n4"


def test_stem_builder_stamps_withtest():
    stem = build_solver_stem(
        num_examples=3,
        example_grids="minimal",
        n_tasks=400,
        num_samples=4,
        include_test=True,
    )
    assert stem == "solver_k3_minimal_400tasks_withtest_n4"


def test_stem_builder_stamps_nl():
    stem = build_solver_stem(
        num_examples=3,
        example_grids="minimal",
        n_tasks=400,
        num_samples=1,
        include_test=False,
        with_descriptions=True,
    )
    assert stem == "solver_k3_minimal_400tasks_nl"


def test_stem_builder_nl_marker_ordering():
    """``_nl`` sits between ``_withtest`` and ``_n{S}``."""
    stem = build_solver_stem(
        num_examples=3,
        example_grids="minimal",
        n_tasks=400,
        num_samples=4,
        include_test=True,
        with_descriptions=True,
    )
    assert stem == "solver_k3_minimal_400tasks_withtest_nl_n4"


def test_stem_builder_omits_n1_default():
    """num_samples=1 is the default; no _n token."""
    stem = build_solver_stem(
        num_examples=3,
        example_grids="minimal",
        n_tasks=400,
        num_samples=1,
        include_test=False,
    )
    assert stem == "solver_k3_minimal_400tasks"


# ---------------------------------------------------------------------------
# Embedding retrieval
# ---------------------------------------------------------------------------


def _embedding_state_for(fixture, target_ids=("T1", "T2")):
    """Build an EmbeddingRetrievalState whose nearest neighbours are deterministic.

    Embedding axes are constructed so the cosine similarity to each query
    has a strict ordering:
      T1's closest demos are D1 > D2 > D3 > D4.
      T2's closest demos are D4 > D3 > D2 > D1.
    """
    doc_ids = ["D1", "D2", "D3", "D4"]
    # Place demos at angles 0°, 30°, 60°, 90° from the x-axis.
    angles = [0.0, np.pi / 6, np.pi / 3, np.pi / 2]
    doc_embs = np.array(
        [[np.cos(a), np.sin(a)] for a in angles], dtype=np.float32
    )

    # T1 query along x-axis -> nearest D1; T2 query along y-axis -> nearest D4.
    query_embs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query_id_to_idx = {tid: i for i, tid in enumerate(target_ids)}

    return EmbeddingRetrievalState(
        doc_embs=doc_embs,
        doc_ids=doc_ids,
        query_embs=query_embs,
        query_id_to_idx=query_id_to_idx,
        keep=set(fixture["pool"]) & set(doc_ids),
    )


def test_embedding_retrieval_requires_state(fixture):
    with pytest.raises(ValueError, match="EmbeddingRetrievalState"):
        build_solver_prompt_records(
            task_ids=["T1"],
            template=TEMPLATE,
            dsl_code=DSL_CODE,
            task_dir=fixture["target_dir"],
            demo_task_dir=fixture["demo_dir"],
            solver_pool=fixture["pool"],
            retrieval="embedding",
            num_examples=2,
            example_grids="minimal",
            include_test=False,
            seed=None,
        )


def test_embedding_retrieval_picks_top_k(fixture):
    state = _embedding_state_for(fixture)
    records = build_solver_prompt_records(
        task_ids=["T1", "T2"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool=fixture["pool"],
        retrieval="embedding",
        num_examples=2,
        example_grids="minimal",
        include_test=False,
        seed=None,
        embedding_state=state,
    )
    # T1's query is along x-axis -> D1, D2 are nearest.
    assert records[0]["retrieved_tasks"] == ["D1", "D2"]
    # T2's query is along y-axis -> D4, D3 are nearest.
    assert records[1]["retrieved_tasks"] == ["D4", "D3"]


def test_embedding_retrieval_scores_are_rounded(fixture):
    state = _embedding_state_for(fixture)
    records = build_solver_prompt_records(
        task_ids=["T1"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool=fixture["pool"],
        retrieval="embedding",
        num_examples=2,
        example_grids="minimal",
        include_test=False,
        seed=None,
        embedding_state=state,
    )
    for score in records[0]["retrieval_scores"]:
        # Every stored score must round-trip through round() with no change.
        assert score == round(score, RETRIEVAL_SCORE_PRECISION)


def test_embedding_retrieval_excludes_target(fixture):
    """If the target id happens to be in the doc pool, it must be skipped."""
    # Build a state where T1 is also a 'doc' aligned with its own query.
    doc_ids = ["T1", "D1", "D2"]
    doc_embs = np.array(
        [[1.0, 0.0], [np.cos(np.pi / 6), np.sin(np.pi / 6)], [0.0, 1.0]],
        dtype=np.float32,
    )
    query_embs = np.array([[1.0, 0.0]], dtype=np.float32)
    state = EmbeddingRetrievalState(
        doc_embs=doc_embs,
        doc_ids=doc_ids,
        query_embs=query_embs,
        query_id_to_idx={"T1": 0},
        keep={"T1", "D1", "D2"},
    )
    pool_with_self = {**fixture["pool"], "T1": "O = z(I)"}
    records = build_solver_prompt_records(
        task_ids=["T1"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool=pool_with_self,
        retrieval="embedding",
        num_examples=2,
        example_grids="none",
        include_test=False,
        seed=None,
        embedding_state=state,
    )
    assert "T1" not in records[0]["retrieved_tasks"]
    assert records[0]["retrieved_tasks"] == ["D1", "D2"]


def test_embedding_retrieval_respects_keep(fixture):
    """Docs outside the keep set must never be picked, even when ranked high."""
    state = _embedding_state_for(fixture)
    # Override keep to exclude D1 — the would-be top match for T1.
    restricted = EmbeddingRetrievalState(
        doc_embs=state.doc_embs,
        doc_ids=state.doc_ids,
        query_embs=state.query_embs,
        query_id_to_idx=state.query_id_to_idx,
        keep={"D2", "D3", "D4"},
    )
    records = build_solver_prompt_records(
        task_ids=["T1"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool=fixture["pool"],
        retrieval="embedding",
        num_examples=2,
        example_grids="none",
        include_test=False,
        seed=None,
        embedding_state=restricted,
    )
    assert "D1" not in records[0]["retrieved_tasks"]
    assert records[0]["retrieved_tasks"] == ["D2", "D3"]




# ---------------------------------------------------------------------------
# Target-task exclusion in random retrieval
# ---------------------------------------------------------------------------


def test_random_retrieval_excludes_self_from_demos(fixture):
    """If the target task happens to be in the demo pool, it must NOT be
    picked as its own demo — that would leak the solution."""
    pool_with_target = {**fixture["pool"], "T1": "O = z(I)"}
    records = build_solver_prompt_records(
        task_ids=["T1"],
        template=TEMPLATE,
        dsl_code=DSL_CODE,
        task_dir=fixture["target_dir"],
        demo_task_dir=fixture["demo_dir"],
        solver_pool=pool_with_target,
        retrieval="random",
        num_examples=2,
        example_grids="none",
        include_test=False,
        seed=42,
    )
    assert "T1" not in records[0]["retrieved_tasks"]
