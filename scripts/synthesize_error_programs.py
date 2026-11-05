"""Ask Gemini to edit a reference program so it reproduces an error (stage 2).

Each prompt holds the DSL, the reference program, the task's grids and one
participant's incorrect output. The model may replace up to ``--num-steps``
lines of the reference program. Reference programs are the human-written
solvers in ``hodel_dsl/solvers.py``, or those in ``--solvers-file`` (the
evaluation split). With ``--control``, the target is the correct output
instead of an error.

Reads the filtered errors and the ARC task JSONs. Writes the model responses as
JSONL under ``--output-dir``. ``--mode`` selects token counting, the Batch API
(``prepare`` / ``submit`` / ``check``) or direct asynchronous calls.

Usage:
    uv run python scripts/synthesize_error_programs.py \\
        --input-file OSF/00_inputs/human/harc_errors_training.filtered.jsonl \\
        --task-dir external/arc-agi-1/training \\
        --num-steps 2 --num-samples 5 --model gemini-3.5-flash --mode async
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Ensure project root and src are on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from src.pipeline import (  # noqa: E402
    PipelineConfig,
    add_common_args,
    apply_limit,
    apply_prompt_filter,
    build_control_error_records,
    build_output_stem,
    build_prompt_records,
    expand_for_sampling,
    load_common_resources,
    load_dsl_resources,
    load_filter_pairs,
    run_async_mode,
    run_check_mode,
    run_count_mode,
    run_prepare_submit_mode,
    write_run_config,
)
from src.program_edits import get_response_schema  # noqa: E402
from src.utils import (  # noqa: E402
    collapse_solver_bundles,
    get_solver_body,
    load_synthesized_solvers,
    load_unique_errors,
)

DEFAULT_TEMPLATE = "prompts/synthesize_error_program_modify_only.md"
DEFAULT_OUTPUT_DIR = "results/error_synthesis"
DEFAULT_NUM_STEPS = 2


# ---------------------------------------------------------------------------
# Script-specific helpers
# ---------------------------------------------------------------------------


def estimate_output_tokens_per_response(solvers: Dict[str, str], num_steps: int) -> int:
    """Estimate output tokens per response from the solver corpus.

    The response is a JSON object with up to *num_steps* modifications; each
    modification carries a variable name and a DSL expression.  Token count
    is estimated from the average character length of assignment lines in the
    solver corpus (rule of thumb: 1 token per 4 characters of code), plus
    fixed JSON framing overhead.
    """
    line_lengths: List[int] = []
    for source in solvers.values():
        body = get_solver_body(source)
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if re.match(r"^[a-zA-Z_]\w*\s*=\s*.+$", stripped):
                line_lengths.append(len(stripped))

    avg_chars = int(np.mean(line_lengths)) if line_lengths else 40
    tokens_per_modification = (avg_chars // 4) + 10
    json_overhead = 15
    return json_overhead + num_steps * tokens_per_modification


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate and optionally execute error-program-synthesis prompts."
    )
    add_common_args(
        parser,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        input_file_required=False,
    )

    parser.add_argument(
        "--num-steps",
        type=int,
        default=DEFAULT_NUM_STEPS,
        help=f"Maximum program-modification steps allowed (default: {DEFAULT_NUM_STEPS}).",
    )
    parser.add_argument(
        "--solvers-file",
        type=str,
        default=None,
        help=(
            "Path to a {task_id: [solver_source,...]} JSON map (output of "
            "scripts/select_solver_candidates.py). When given, replaces the "
            "hand-written hodel_dsl.solvers as the reference-program source. "
            "Works in both default and --control modes."
        ),
    )
    parser.add_argument(
        "--solver-selection",
        choices=["first", "all"],
        default="first",
        help=(
            "With --solvers-file and multiple solvers per task: 'first' uses the "
            "first body in each bundle (the shortest, given an all-unique export). "
            "'all' (per-task fan-out) is not yet implemented."
        ),
    )

    args = parser.parse_args()
    config = PipelineConfig.from_args(args)

    output_dir = Path(args.output_dir)
    if args.run_tag:
        output_dir = output_dir / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_dir = output_dir / "prompts"
    requests_dir = output_dir / "requests"
    results_dir = output_dir / "results_raw"

    # CHECK mode does not need any resources.
    if args.mode == "check":
        run_check_mode(
            config=config,
            requests_dir=requests_dir,
            results_dir=results_dir,
            response_key="modifications",
        )
        return

    if not args.model:
        parser.error("--model is required (except in --mode check).")
    if not args.task_dir:
        parser.error("--task-dir is required (except in --mode check).")
    if not args.control and not args.input_file:
        parser.error("one of --input-file or --control is required")
    if args.solver_selection == "all":
        parser.error("--solver-selection all is not yet implemented; use first")

    input_path = Path(args.input_file) if args.input_file else None
    task_dir = Path(args.task_dir)

    template_path = Path(args.template_file or DEFAULT_TEMPLATE)

    if input_path is not None and not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return
    if not task_dir.exists():
        print(f"Error: Task directory {task_dir} not found.")
        return
    if not template_path.exists():
        print(f"Error: Template file {template_path} not found.")
        return

    # Optionally override the reference-program source with a synthesized solver
    # map (select_solver_candidates.py output). Collapse each bundle to its first body under the default
    # 'first' selection. Loaded *before* the control/non-control branch so that
    # control mode derives its task set from the synthesized keys, not the
    # hand-written hodel_dsl.solvers.
    synthesized_solvers: Dict[str, str] | None = None
    if args.solvers_file:
        solvers_file = Path(args.solvers_file)
        if not solvers_file.exists():
            print(f"Error: Solvers file {solvers_file} not found.")
            return
        bundles = load_synthesized_solvers(solvers_file)
        synthesized_solvers = collapse_solver_bundles(bundles, args.solver_selection)
        print(
            f"Synthesized solvers: {len(synthesized_solvers)} tasks "
            f"(from {sum(len(v) for v in bundles.values())} bodies in {solvers_file.name})."
        )

    if args.control:
        dsl = load_dsl_resources(api_only=args.api_only)
        solvers = synthesized_solvers if synthesized_solvers is not None else dsl["solvers"]
        dsl_code = dsl["dsl_code"]

        solved = set(solvers.keys())
        available = {p.stem for p in task_dir.glob("*.json")}
        base = solved & available
        if input_path is not None:
            filter_ids = {rec["task_id"] for rec in load_unique_errors(input_path)}
            task_ids = sorted(base & filter_ids)
            print(
                f"Control mode: {len(task_ids)} tasks "
                f"(solvers∩task-dir={len(base)} ∩ input-file={len(filter_ids)})."
            )
        else:
            task_ids = sorted(base)
            print(
                f"Control mode: {len(task_ids)} tasks "
                f"(solvers∩task-dir; no input-file filter)."
            )

        errors = build_control_error_records(task_ids, task_dir)
    else:
        resources = load_common_resources(input_path, api_only=args.api_only)
        errors = resources["errors"]
        solvers = (
            synthesized_solvers
            if synthesized_solvers is not None
            else resources["solvers"]
        )
        dsl_code = resources["dsl_code"]
    n_errors_in_input = len(errors)

    print(f"Template: {template_path}")

    with open(template_path, "r") as f:
        template = f.read()

    # Restrict to specific task IDs if requested.
    if args.tasks:
        target_ids = {tid.replace(".json", "") for tid in args.tasks}
        before = len(errors)
        errors = [e for e in errors if e["task_id"] in target_ids]
        missing = target_ids - {e["task_id"] for e in errors}
        if missing:
            print(f"Warning: tasks not present in input: {sorted(missing)}")
        print(f"Task-id filter: {before} -> {len(errors)} errors.")

    # Build per-prompt records (expanding by descriptions when NL is on).
    prompt_records = build_prompt_records(
        errors=errors,
        template=template,
        solvers=solvers,
        dsl_code=dsl_code,
        task_dir=task_dir,
        num_steps=args.num_steps,
        include_nl=args.include_nl,
        verbose=args.verbose,
    )

    # Optional: filter prompt records based on a previous run's results.
    if args.filter_from:
        allowed = load_filter_pairs(
            args.filter_from, args.filter_where, args.filter_mode
        )
        prompt_records = apply_prompt_filter(prompt_records, allowed)

    prompt_records = apply_limit(
        prompt_records, args.limit, seed=args.seed, label="prompts"
    )

    if not prompt_records:
        print("No prompts generated. Exiting.")
        return

    # Build the output-file stem.
    nl_tag = "with_nl" if args.include_nl else "no_nl"
    dsl_tag = "api" if args.api_only else "full"
    edit_tag = "modify_only"
    temp_tag = f"t{args.temperature:g}"
    think_tag = f"think-{args.thinking_level}" if args.thinking_level else "no-think"

    stem_input_path = (
        input_path if input_path is not None else Path("control_correct_output.jsonl")
    )
    stem = build_output_stem(
        input_path=stem_input_path,
        n_errors_in_run=len({(r["task_id"], r["response_id"]) for r in prompt_records}),
        n_errors_in_input=n_errors_in_input,
        tags={
            "steps": f"steps{args.num_steps}",
            "nl": nl_tag,
            "dsl": dsl_tag,
            "edit": edit_tag,
            "temp": temp_tag,
            "think": think_tag,
        },
        num_samples=args.num_samples,
    )

    prompts = [r["prompt"] for r in prompt_records]
    expanded = expand_for_sampling(prompt_records, args.num_samples)
    total_requests = len(expanded)
    print(
        f"Total requests (prompts x samples): {total_requests} "
        f"({len(prompt_records)} prompts x {args.num_samples} samples)"
    )

    response_schema = get_response_schema(max_modifications=args.num_steps)

    write_run_config(
        output_dir,
        config={
            "run_tag": args.run_tag,
            "model": args.model,
            "num_steps": args.num_steps,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "thinking_level": args.thinking_level,
            "include_nl": args.include_nl,
            "api_only": args.api_only,
            "modify_only": True,
            "control": args.control,
            "input_file": args.input_file,
            "task_dir": args.task_dir,
            "template_file": str(template_path),
            "solvers_file": args.solvers_file,
            "solver_selection": args.solver_selection,
        },
        response_schema=response_schema,
    )

    if args.mode == "count":
        est_output = estimate_output_tokens_per_response(solvers, args.num_steps)
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
            response_schema=response_schema,
            stem=stem,
            requests_dir=requests_dir,
            batch_config={"num_steps": args.num_steps},
            submit=(args.mode == "submit"),
        )

    elif args.mode == "async":
        run_async_mode(
            expanded=expanded,
            config=config,
            response_schema=response_schema,
            stem=stem,
            results_dir=results_dir,
            response_key="modifications",
        )


if __name__ == "__main__":
    main()
