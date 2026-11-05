"""Ask Gemini to write a DSL program that solves each ARC task.

Used for tasks without a human-written reference program (the evaluation split).
Each prompt holds the DSL, the task's training pairs and, optionally, solved
example tasks with their human-written programs: none (``zero-shot``), chosen
at random (``random``), or the most similar ones by embedding
(``embedding``, see ``generate_task_embeddings.py``). The test pair is hidden
unless ``--include-test`` is given.

Reads ARC task JSONs and, for retrieval, the example programs and embeddings.
Writes the model responses as JSONL under ``--output-dir``. ``--mode`` selects
token counting, the Batch API (``prepare`` / ``submit`` / ``check``) or direct
asynchronous calls.

Usage:
    uv run python scripts/synthesize_solver_programs.py \\
        --task-dir external/arc-agi-1/evaluation --demo-task-dir external/arc-agi-1/training \\
        --solver-source hodel_dsl.solvers --retrieval embedding \\
        --doc-embeddings <training>.npz --query-embeddings <evaluation>.npz \\
        --num-examples 5 --num-samples 5 \\
        --model gemini-3.5-flash --mode async
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np

# Ensure project root is on sys.path.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from src.pipeline import (  # noqa: E402
    PipelineConfig,
    add_common_args,
    apply_limit,
    expand_for_sampling,
    load_dsl_resources,
    run_async_mode,
    run_check_mode,
    run_count_mode,
    run_prepare_submit_mode,
    write_run_config,
)
from src.retrieval import load_embeddings  # noqa: E402
from src.solver_pool import load_solver_pool  # noqa: E402
from src.utils import load_descriptions  # noqa: E402
from src.solver_prompts import (  # noqa: E402
    EmbeddingRetrievalState,
    build_solver_prompt_records,
    build_solver_stem,
)

DEFAULT_TEMPLATE = "prompts/synthesize_solver_program.md"
DEFAULT_OUTPUT_DIR = "results/solver_synthesis"
DEFAULT_NUM_EXAMPLES = 3

SOLVER_METADATA_KEYS = [
    "task_id",
    "retrieved_tasks",
    "retrieval_scores",
    "has_description",
]

SOLVER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "program": {
            "type": "string",
            "description": "Solver program body in unnested canonical form.",
        }
    },
    "required": ["program"],
}


def estimate_solver_output_tokens(solver_pool: Dict[str, str]) -> int:
    """Estimate output tokens per response from the demo solver corpus.

    The response is a JSON object with a single ``program`` string.  Token
    count is estimated from the average character length of the bodies in
    *solver_pool* (rule of thumb: 1 token per 4 characters of code), plus
    fixed JSON framing overhead.
    """
    body_lengths = [len(body) for body in solver_pool.values() if body.strip()]
    if not body_lengths:
        avg_chars = 200
    else:
        avg_chars = int(np.mean(body_lengths))
    json_overhead = 20
    return json_overhead + (avg_chars // 4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate and optionally execute solver-synthesis prompts."
    )
    add_common_args(
        parser,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        input_file_required=False,
    )

    parser.add_argument(
        "--retrieval",
        type=str,
        default=None,
        choices=["zero-shot", "random", "embedding"],
        help=(
            "In-context retrieval mode. 'zero-shot' uses no demos; 'random' "
            "samples K demos uniformly from --solver-source; 'embedding' "
            "selects K demos by cosine top-K over precomputed task "
            "embeddings (requires --doc-embeddings + --query-embeddings). "
            "Required for count/prepare/submit/async; ignored in --mode check."
        ),
    )
    parser.add_argument(
        "--doc-embeddings",
        type=str,
        default=None,
        help=(
            "Path to .npz of RETRIEVAL_DOCUMENT embeddings (the demo "
            "pool side). Required for --retrieval embedding; rejected "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--query-embeddings",
        type=str,
        default=None,
        help=(
            "Path to .npz of RETRIEVAL_QUERY embeddings (the target "
            "side). Required for --retrieval embedding; rejected "
            "otherwise."
        ),
    )
    parser.add_argument(
        "--solver-source",
        type=str,
        default=None,
        help=(
            "Importable Python module path containing the demo solver pool "
            "(e.g. 'hodel_dsl.solvers'). Required for --retrieval random / "
            "embedding; ignored for zero-shot."
        ),
    )
    parser.add_argument(
        "--demo-task-dir",
        type=str,
        default=None,
        help=(
            "Directory holding ARC-format JSONs for the in-context demo tasks "
            "(typically a training split distinct from --task-dir). Defaults "
            "to --task-dir when omitted."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=DEFAULT_NUM_EXAMPLES,
        help=f"Number of in-context demos per target task (default: {DEFAULT_NUM_EXAMPLES}).",
    )
    parser.add_argument(
        "--example-grids",
        type=str,
        choices=["full", "minimal", "none"],
        default="minimal",
        help=(
            "What to show alongside each demo solver: 'full' (all train + "
            "test grids), 'minimal' (first train pair only; default), or "
            "'none' (solver body only)."
        ),
    )
    parser.add_argument(
        "--include-test",
        action="store_true",
        help=(
            "Include the target task's test pair(s) in the prompt. OFF by "
            "default — showing the test output trivially leaks the rule. "
            "Use only for ablation runs."
        ),
    )
    parser.add_argument(
        "--description-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL with {task_id, text} records. Multiple records "
            "per task are concatenated. Tasks without descriptions get no "
            "description block."
        ),
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum word count per description (shorter ones are dropped).",
    )

    args = parser.parse_args()
    config = PipelineConfig.from_args(args)

    output_dir = Path(args.output_dir)
    if args.run_tag:
        output_dir = output_dir / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = output_dir / "prompts"
    requests_dir = output_dir / "batch_requests"
    results_dir = output_dir / "results"

    if args.mode == "check":
        run_check_mode(
            config=config,
            requests_dir=requests_dir,
            results_dir=results_dir,
            response_key="program",
        )
        return

    if not args.model:
        parser.error("--model is required (except in --mode check).")
    if not args.task_dir:
        parser.error("--task-dir is required (except in --mode check).")
    if not args.retrieval:
        parser.error("--retrieval is required (except in --mode check).")

    if args.retrieval != "zero-shot" and not args.solver_source:
        parser.error(f"--solver-source is required for --retrieval {args.retrieval}.")
    if args.retrieval == "random" and args.seed is None:
        parser.error("--retrieval random requires --seed for reproducibility.")
    if args.retrieval == "embedding":
        if not args.doc_embeddings or not args.query_embeddings:
            parser.error(
                "--retrieval embedding requires both --doc-embeddings and "
                "--query-embeddings."
            )
    else:
        if args.doc_embeddings or args.query_embeddings:
            parser.error(
                "--doc-embeddings / --query-embeddings are only valid with "
                "--retrieval embedding."
            )

    task_dir = Path(args.task_dir)
    demo_task_dir = Path(args.demo_task_dir) if args.demo_task_dir else task_dir
    template_path = (
        Path(args.template_file) if args.template_file else Path(DEFAULT_TEMPLATE)
    )

    if not task_dir.exists():
        print(f"Error: Task directory {task_dir} not found.")
        return
    if not demo_task_dir.exists():
        print(f"Error: Demo task directory {demo_task_dir} not found.")
        return
    if not template_path.exists():
        print(f"Error: Template file {template_path} not found.")
        return

    # DSL resources are shared with error synthesis; only --solver-source
    # supplies the pool.
    dsl = load_dsl_resources(api_only=args.api_only)
    dsl_code = dsl["dsl_code"]

    if args.retrieval == "zero-shot":
        solver_pool: Dict[str, str] = {}
    else:
        solver_pool = load_solver_pool(args.solver_source)
        print(f"Loaded {len(solver_pool)} demo solvers from {args.solver_source}.")

    template = template_path.read_text()
    if args.description_file and "{{task_description}}" not in template:
        parser.error(
            f"--description-file given but template {template_path} has no "
            "{{task_description}} placeholder; descriptions would be "
            "silently dropped."
        )
    print(f"Template: {template_path}")
    print(
        f"Retrieval: {args.retrieval} (K={args.num_examples}, grids={args.example_grids})"
    )
    print(f"Include test pair: {args.include_test}")

    # Task discovery — sorted for determinism.
    all_task_ids = sorted(p.stem for p in task_dir.glob("*.json"))
    if args.tasks:
        target_ids = {tid.replace(".json", "") for tid in args.tasks}
        before = len(all_task_ids)
        all_task_ids = [tid for tid in all_task_ids if tid in target_ids]
        missing = target_ids - set(all_task_ids)
        if missing:
            print(f"Warning: tasks not present in --task-dir: {sorted(missing)}")
        print(f"Task-id filter: {before} -> {len(all_task_ids)} tasks.")

    all_task_ids = apply_limit(all_task_ids, args.limit, seed=args.seed, label="tasks")
    if not all_task_ids:
        print("No tasks selected. Exiting.")
        return

    descriptions: Dict[str, str] | None = None
    if args.description_file is not None:
        descriptions = load_descriptions(args.description_file, args.min_words)
        print(
            f"Loaded descriptions for {len(descriptions)} tasks "
            f"(>= {args.min_words} words) from {args.description_file}."
        )
        with_desc = set(all_task_ids) & set(descriptions)
        print(
            f"Coverage: {len(with_desc)}/{len(all_task_ids)} selected tasks "
            f"have descriptions."
        )

    embedding_state = None
    if args.retrieval == "embedding":
        doc_embs, doc_ids = load_embeddings(Path(args.doc_embeddings))
        query_embs, query_ids = load_embeddings(Path(args.query_embeddings))
        keep = set(doc_ids) & set(solver_pool)
        if not keep:
            parser.error(
                f"No overlap between --doc-embeddings task IDs "
                f"({len(doc_ids)}) and the loaded solver pool "
                f"({len(solver_pool)}). Check that the embeddings file "
                f"corresponds to the demo pool you intend to use."
            )
        embedding_state = EmbeddingRetrievalState(
            doc_embs=doc_embs,
            doc_ids=list(doc_ids),
            query_embs=query_embs,
            query_id_to_idx={tid: i for i, tid in enumerate(query_ids)},
            keep=keep,
        )
        print(
            f"Embeddings loaded: {len(doc_ids)} docs, {len(query_ids)} queries, "
            f"dim={doc_embs.shape[1]}. Pool-eligible docs: {len(keep)}."
        )

    prompt_records = build_solver_prompt_records(
        task_ids=all_task_ids,
        template=template,
        dsl_code=dsl_code,
        task_dir=task_dir,
        demo_task_dir=demo_task_dir,
        solver_pool=solver_pool,
        retrieval=args.retrieval,
        num_examples=args.num_examples,
        example_grids=args.example_grids,
        include_test=args.include_test,
        seed=args.seed,
        verbose=args.verbose,
        embedding_state=embedding_state,
        descriptions=descriptions,
    )

    if not prompt_records:
        print("No prompts generated. Exiting.")
        return

    stem = build_solver_stem(
        num_examples=args.num_examples,
        example_grids=args.example_grids,
        n_tasks=len(prompt_records),
        num_samples=args.num_samples,
        include_test=args.include_test,
        with_descriptions=args.description_file is not None,
    )

    prompts = [r["prompt"] for r in prompt_records]
    expanded = expand_for_sampling(
        prompt_records,
        args.num_samples,
        metadata_keys=SOLVER_METADATA_KEYS,
    )
    total_requests = len(expanded)
    print(
        f"Total requests (prompts x samples): {total_requests} "
        f"({len(prompt_records)} prompts x {args.num_samples} samples)"
    )

    write_run_config(
        output_dir,
        config={
            "script": "synthesize_solver_programs.py",
            "run_tag": args.run_tag,
            "model": args.model,
            "retrieval": args.retrieval,
            "solver_source": args.solver_source,
            "num_examples": args.num_examples,
            "example_grids": args.example_grids,
            "include_test": args.include_test,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "thinking_level": args.thinking_level,
            "api_only": args.api_only,
            "seed": args.seed,
            "task_dir": args.task_dir,
            "demo_task_dir": str(demo_task_dir),
            "template_file": str(template_path),
            "doc_embeddings": (
                str(args.doc_embeddings) if args.doc_embeddings else None
            ),
            "query_embeddings": (
                str(args.query_embeddings) if args.query_embeddings else None
            ),
            "description_file": (
                str(args.description_file) if args.description_file else None
            ),
            "min_words": args.min_words,
        },
        response_schema=SOLVER_RESPONSE_SCHEMA,
    )

    if args.mode == "count":
        est_output = estimate_solver_output_tokens(solver_pool) if solver_pool else 200
        run_count_mode(
            prompts=prompts,
            records=prompt_records,
            config=config,
            stem=stem,
            prompts_dir=prompts_dir,
            total_requests=total_requests,
            est_output_tokens=est_output,
        )

    elif args.mode in ("prepare", "submit"):
        run_prepare_submit_mode(
            expanded=expanded,
            config=config,
            response_schema=SOLVER_RESPONSE_SCHEMA,
            stem=stem,
            requests_dir=requests_dir,
            batch_config={
                "retrieval": args.retrieval,
                "num_examples": args.num_examples,
                "example_grids": args.example_grids,
                "include_test": args.include_test,
                "with_descriptions": args.description_file is not None,
            },
            submit=(args.mode == "submit"),
            metadata_keys=SOLVER_METADATA_KEYS,
        )

    elif args.mode == "async":
        run_async_mode(
            expanded=expanded,
            config=config,
            response_schema=SOLVER_RESPONSE_SCHEMA,
            stem=stem,
            results_dir=results_dir,
            response_key="program",
            metadata_keys=SOLVER_METADATA_KEYS,
        )


if __name__ == "__main__":
    main()
