"""Ask Gemini to assign each accepted error program to an error category.

The categories are the seven of the paper's taxonomy plus ``unclassifiable``
(see ``src/category_definitions.py``). Each prompt shows the reference program,
the task's grids, the error program, and the reference and error outputs side
by side on each training input. Each program is classified ``--num-samples``
times to capture the classifier's uncertainty.

Reads accepted programs (the output of ``filter_candidates.py``) and the ARC
task JSONs. Writes one row per sample, keyed by ``error_id`` and
``program_id`` (the format of OSF's ``04_classifications/``). ``--mode``
selects token counting, the Batch API (``prepare`` / ``submit`` / ``check``)
or direct asynchronous calls.

Usage:
    uv run python scripts/classify_errors.py \\
        --input-file OSF/03_candidates/human/training.jsonl \\
        --task-dir external/arc-agi-1/training \\
        --num-samples 5 --model gemini-3.5-flash --mode async
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Ensure project root and src are on path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from hodel_dsl.dsl_parser import ProgramRewriter  # noqa: E402
from src.category_definitions import CATEGORY_NAMES  # noqa: E402
from src.filters import build_dsl_namespace, compute_training_outputs  # noqa: E402
from src.pipeline import (  # noqa: E402
    PipelineConfig,
    add_common_args,
    apply_limit,
    build_output_stem,
    construct_base_prompt,
    expand_for_sampling,
    run_async_mode,
    run_check_mode,
    run_count_mode,
    run_prepare_submit_mode,
    write_run_config,
)
from src.utils import (  # noqa: E402
    format_dsl_api_for_prompt,
    format_dsl_source_for_prompt,
    format_grid,
    get_solver_body,
    grid_hash,
    grid_to_tuple,
    load_dsl_api,
    load_dsl_preamble,
    load_dsl_source,
    load_grid,
    load_jsonl_records,
    load_solvers,
)

DEFAULT_TEMPLATE = "prompts/classify_errors.md"
DEFAULT_OUTPUT_DIR = "results/error_classification"

# Which per-record keys are threaded onto each output record (in addition to the
# always-present ``sample_index``). ``error_id`` serves the viz badge, ``grid_hash``
# serves the category-distribution analysis; ``program_id`` disambiguates multiple
# candidate programs for one error.
METADATA_KEYS = ["error_id", "grid_hash", "task_id", "program_id"]


# ---------------------------------------------------------------------------
# Response schema for Gemini structured output
# ---------------------------------------------------------------------------


def get_characterization_response_schema() -> Dict[str, Any]:
    """Return a Gemini-compatible JSON schema for structured error classification.

    Flat union schema: 4 always-required fields + 32 category-specific nullable
    fields.  All 36 fields appear in ``required`` (Gemini convention: nullable
    fields must be listed as required to guarantee they appear in the output).
    """
    return {
        "type": "OBJECT",
        "properties": {
            # -- Always filled ------------------------------------------
            "correct_rule_description": {
                "type": "STRING",
                "description": (
                    "What the ground-truth program does, as a rule a human could articulate, in perceptual/cognitive terms."
                ),
            },
            "error_rule_description": {
                "type": "STRING",
                "description": (
                    "What the error program does instead, as a rule in perceptual/cognitive terms."
                ),
            },
            "operation_domain": {
                "type": "STRING",
                "description": (
                    "A short sentence naming the cognitive/perceptual faculty at stake."
                ),
            },
            "error_category": {
                "type": "STRING",
                "enum": [
                    "attribute_substitution",
                    "scope_restriction",
                    "operator_confusion",
                    "transformation_substitution",
                    "compositional_sequencing",
                    "partial_program",
                    "boundary_precision",
                    "unclassifiable",
                ],
                "description": "One of the 8 pre-specified error categories.",
            },
            # -- Attribute Substitution ---------------------------------
            "target_parameter": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The quantity or parameter in the transformation that "
                    "the correct and substituted attributes determine "
                    "(e.g., 'width of the drawn patterns')."
                ),
            },
            "correct_attribute": {
                "type": "STRING",
                "nullable": True,
                "description": "The attribute the GT program computes.",
            },
            "substituted_attribute": {
                "type": "STRING",
                "nullable": True,
                "description": "The attribute the error program uses instead.",
            },
            "substitution_type": {
                "type": "STRING",
                "nullable": True,
                "enum": [
                    "computed_property",
                    "hardcoded_constant",
                    "surface_correlation",
                ],
                "description": "Type of attribute substitution ('hardcoded_constant' for literals like 3 or 'red', 'surface_correlation' for exploiting a spurious correlation in training examples, 'computed_property' for calculating a different logical property).",
            },
            "hardcoded_value": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The literal value substituted (only when "
                    "substitution_type = hardcoded_constant)."
                ),
            },
            # -- Scope Restriction --------------------------------------
            "restricted_operation": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The transformation that is correctly applied but "
                    "to a limited scope, in perceptual terms "
                    "(e.g., 'color fill', 'shape copying')."
                ),
            },
            "correct_scope": {
                "type": "STRING",
                "nullable": True,
                "description": "What the GT program applies the operation to.",
            },
            "actual_scope": {
                "type": "STRING",
                "nullable": True,
                "description": "What the error program applies the operation to.",
            },
            "fraction_covered": {
                "type": "NUMBER",
                "nullable": True,
                "description": ("Approximate fraction of targets covered (0.0-1.0)."),
            },
            # -- Operator Confusion -------------------------------------
            "operator_target": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "What aspect of the transformation the operator "
                    "determines, in perceptual terms (e.g., "
                    "'orientation of the copied shape')."
                ),
            },
            "operation_class": {
                "type": "STRING",
                "nullable": True,
                "description": ("The class of operation (e.g., rotation, mirroring)."),
            },
            "correct_operator": {
                "type": "STRING",
                "nullable": True,
                "description": "The GT operator.",
            },
            "confused_operator": {
                "type": "STRING",
                "nullable": True,
                "description": "The operator the error program uses.",
            },
            "confusion_axis": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "What dimension the confusion is along "
                    "(e.g., CW vs CCW, horizontal vs vertical)."
                ),
            },
            # -- Transformation Substitution ----------------------------
            "correct_operation_domain": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The family of transformation the GT program uses "
                    "(e.g., 'rotation-based tiling', 'concentric expansion')."
                ),
            },
            "applied_operation_domain": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The family of transformation the error program uses "
                    "instead (e.g., 'mirror-based tiling', 'spatial shift')."
                ),
            },
            "shared_structure": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "What aspects of the task structure the participant "
                    "correctly identified (e.g., 'correct target objects "
                    "and 2x2 grid layout')."
                ),
            },
            "domain_distance": {
                "type": "STRING",
                "nullable": True,
                "enum": ["adjacent", "distant"],
                "description": (
                    "Whether the confused operation families are from "
                    "neighboring domains (adjacent) or very different "
                    "domains (distant)."
                ),
            },
            # -- Compositional Sequencing -------------------------------
            "sequencing_sub_type": {
                "type": "STRING",
                "nullable": True,
                "enum": ["sequential_ordering", "parallel_binding"],
                "description": "Sub-type of compositional sequencing error.",
            },
            "operations_involved": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "A natural-language description of the operations "
                    "that are mis-sequenced (e.g., 'recoloring the "
                    "border and filling the interior')."
                ),
            },
            "sequencing_description": {
                "type": "STRING",
                "nullable": True,
                "description": "How the sequencing differs from GT.",
            },
            # -- Partial Program ----------------------------------------
            "steps_present": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "A natural-language description of the GT steps "
                    "that are preserved (e.g., 'identifies the colored "
                    "objects and extracts their bounding boxes')."
                ),
            },
            "steps_missing": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "A natural-language description of the GT steps "
                    "that are absent (e.g., 'does not recolor the "
                    "extracted shapes according to their size rank')."
                ),
            },
            "missing_phase_function": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "What the missing steps accomplish in perceptual "
                    "terms (e.g., 'recoloring the output objects', "
                    "'extending the pattern to fill the grid')."
                ),
            },
            "missing_phase_position": {
                "type": "STRING",
                "nullable": True,
                "enum": ["early", "middle", "late"],
                "description": ("Where in the pipeline the missing steps fall."),
            },
            # -- Boundary Precision -------------------------------------
            "target_quantity": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The perceptual quantity that is off by a small "
                    "constant, in intuitive terms (e.g., 'height of "
                    "the filled region', 'number of repetitions')."
                ),
            },
            "correct_value": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The GT boundary/count value, described with its "
                    "semantic type (e.g., 'a height of 3 rows', "
                    "'repeating 5 times')."
                ),
            },
            "participant_value": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "The participant's boundary/count value, in the "
                    "same format (e.g., 'a height of 2 rows', "
                    "'repeating 4 times')."
                ),
            },
            "offset": {
                "type": "INTEGER",
                "nullable": True,
                "description": ("The signed difference (participant - correct)."),
            },
            "training_discriminates": {
                "type": "BOOLEAN",
                "nullable": True,
                "description": (
                    "Whether training examples distinguish the two values."
                ),
            },
            # -- Unclassifiable -----------------------------------------
            "apparent_strategy": {
                "type": "STRING",
                "nullable": True,
                "description": (
                    "A one- to two-sentence description of the rule "
                    "the participant appears to have followed, in "
                    "perceptual/cognitive vocabulary."
                ),
            },
            "structural_relationship_to_gt": {
                "type": "STRING",
                "nullable": True,
                "enum": ["none", "minimal_scaffolding"],
                "description": (
                    "How much of the GT structure survives in the error program."
                ),
            },
        },
        "required": [
            "correct_rule_description",
            "error_rule_description",
            "operation_domain",
            "error_category",
            # Attribute Substitution
            "target_parameter",
            "correct_attribute",
            "substituted_attribute",
            "substitution_type",
            "hardcoded_value",
            # Scope Restriction
            "restricted_operation",
            "correct_scope",
            "actual_scope",
            "fraction_covered",
            # Operator Confusion
            "operator_target",
            "operation_class",
            "correct_operator",
            "confused_operator",
            "confusion_axis",
            # Transformation Substitution
            "correct_operation_domain",
            "applied_operation_domain",
            "shared_structure",
            "domain_distance",
            # Compositional Sequencing
            "sequencing_sub_type",
            "operations_involved",
            "sequencing_description",
            # Partial Program
            "steps_present",
            "steps_missing",
            "missing_phase_function",
            "missing_phase_position",
            # Boundary Precision
            "target_quantity",
            "correct_value",
            "participant_value",
            "offset",
            "training_discriminates",
            # Unclassifiable
            "apparent_strategy",
            "structural_relationship_to_gt",
        ],
    }


# ---------------------------------------------------------------------------
# Training comparison formatting
# ---------------------------------------------------------------------------


def format_training_comparison(
    train_examples: List[Dict[str, Any]],
    error_outputs: List[Any | None],
) -> str:
    """Format side-by-side expected vs error program outputs for each training example.

    Parameters
    ----------
    train_examples : list of dict
        Each dict has ``"input"`` and ``"output"`` keys (list-of-lists grids).
    error_outputs : list
        Parallel to *train_examples*. Each entry is a grid (list-of-lists) on
        success or ``None`` on failure.
    """
    lines: List[str] = []
    for i, (example, error_out) in enumerate(zip(train_examples, error_outputs)):
        lines.append(f"Training Example {i + 1}:")
        lines.append("Expected Output:")
        lines.append(format_grid(example["output"]))
        lines.append("Error Program Output:")
        if error_out is not None:
            lines.append(format_grid(error_out))
            expected_tuple = grid_to_tuple(example["output"])
            error_tuple = grid_to_tuple(error_out)
            match = "Yes" if expected_tuple == error_tuple else "No"
        else:
            lines.append("[EXECUTION FAILED]")
            match = "No"
        lines.append(f"Match: {match}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Minimal DSL extraction
# ---------------------------------------------------------------------------


def resolve_reference_solver(
    rec: Dict[str, Any], solvers: Dict[str, str]
) -> str | None:
    """Reference (ground-truth) program for a stage-4 record.

    Prefers the hand-written GT solver keyed by ``task_id`` (from
    ``hodel_dsl.solvers``); for eval-split tasks with no such solver, falls back
    to the synthesized reference carried on the record as ``original_program``
    (stamped in stage 3, evaluate_synthesized_error_programs.py). Returns
    ``None`` if neither is available.
    """
    return solvers.get(rec.get("task_id")) or rec.get("original_program") or None


def extract_function_names(program_source: str) -> set[str]:
    """Extract all function call names from a program body using AST parsing."""
    names: set[str] = set()
    try:
        tree = ast.parse(program_source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def build_minimal_dsl(
    func_names: set[str],
    dsl_api: Dict[str, Dict[str, str]],
    preamble: str,
    api_only: bool = False,
    dsl_sources: Dict[str, str] | None = None,
) -> str:
    """Build a compact DSL string containing only the specified functions.

    Prepends the preamble (type aliases + primitives) and appends either
    full source implementations (default) or API-only signatures + docstrings
    for the requested functions.
    """
    valid_names = sorted(name for name in func_names if name in dsl_api)
    if api_only or dsl_sources is None:
        func_str = format_dsl_api_for_prompt(dsl_api, valid_names)
    else:
        func_str = format_dsl_source_for_prompt(dsl_sources, valid_names)
    return f"{preamble}\n\n{func_str}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Classify cognitive errors from synthesized error programs "
            "(program-only, structured 7-category taxonomy)."
        )
    )
    add_common_args(
        parser,
        default_output_dir=DEFAULT_OUTPUT_DIR,
        input_file_required=False,  # enforced post-parse; --mode check needs no input
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

    # ------------------------------------------------------------------ #
    # CHECK MODE: early exit — no data loading needed                     #
    # ------------------------------------------------------------------ #
    if args.mode == "check":
        run_check_mode(
            config=config,
            requests_dir=requests_dir,
            results_dir=results_dir,
            response_key="characterization_result",
        )
        return

    if not args.model:
        parser.error("--model is required (except in --mode check).")
    if not args.task_dir:
        parser.error("--task-dir is required (except in --mode check).")
    if not args.input_file:
        parser.error("--input-file is required (except in --mode check).")

    input_path = Path(args.input_file)
    task_dir = Path(args.task_dir)
    template_path = (
        Path(args.template_file) if args.template_file is not None
        else Path(DEFAULT_TEMPLATE)
    )

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found.")
        return
    if not task_dir.exists():
        print(f"Error: Task directory {task_dir} not found.")
        return
    if not template_path.exists():
        print(f"Error: Template file {template_path} not found.")
        return

    # Sanity: the taxonomy SSOT must be a subset of the schema enum. The enum
    # additionally carries ``unclassifiable`` (a model escape hatch that is not
    # a taxonomy category), so a strict subset — not equality — is expected.
    schema_enum = set(
        get_characterization_response_schema()["properties"]["error_category"]["enum"]
    )
    assert set(CATEGORY_NAMES) <= schema_enum, (
        "CATEGORY_NAMES drifted from the response-schema enum: "
        f"{set(CATEGORY_NAMES) - schema_enum}"
    )

    # Load stage-4 filtered candidates (one record per kept (error_id, program)).
    print(f"Loading filtered candidates from {input_path}...")
    all_records = load_jsonl_records([input_path])
    print(f"  Total candidate records: {len(all_records)}")
    n_errors_in_input = len(all_records)

    if not all_records:
        print("No records to process. Exiting.")
        return

    print("Loading resources...")
    solvers = load_solvers()
    dsl_api = load_dsl_api()
    dsl_sources = None if args.api_only else load_dsl_source()
    preamble = load_dsl_preamble()
    print(f"DSL mode: {'API-only (signatures + docstrings)' if args.api_only else 'full source'}")
    dsl_namespace = build_dsl_namespace()
    rewriter = ProgramRewriter()

    print(f"Template: {template_path}")
    with open(template_path, "r") as f:
        template = f.read()

    # Restrict to specific task IDs if requested.
    if args.tasks:
        task_ids_filter = {tid.replace(".json", "") for tid in args.tasks}
        all_records = [r for r in all_records if r.get("task_id") in task_ids_filter]
        print(f"Filtered to {len(all_records)} records for specified tasks.")

    all_records = apply_limit(
        all_records, args.limit, seed=args.seed, label="candidate records"
    )

    # Generate prompts.
    results: List[Dict[str, Any]] = []
    skipped = 0
    skipped_analytical = 0

    for rec in tqdm(all_records, desc="Generating prompts"):
        task_id = rec.get("task_id")
        error_id = rec.get("error_id", "")
        rewritten_program = rec.get("rewritten_program")
        target_error_grid = rec.get("target_error_grid")
        program_output_grid = rec.get("program_output_grid")

        # Skip analytical baselines (GT / identity) — no error program. These are
        # already dropped upstream in stage 1, so this is a defensive no-op.
        if rec.get("candidate_type") == "analytical":
            skipped_analytical += 1
            continue

        if not rewritten_program or not target_error_grid:
            skipped += 1
            continue

        # Stage-4 records do not store grid_hash; derive it from the participant
        # error grid. This is the same grid_hash that error_id is minted from
        # (error_id == record_id(task_id, target_error_grid)), so the two keys
        # stay consistent. Use src.utils.grid_hash — never a local hash.
        grid_hash_val = grid_hash(target_error_grid)

        # Content-stable per-program id (disambiguates multiple candidate
        # programs for one error_id).
        program_id = grid_hash(rewritten_program)

        # Load grid data for training examples.
        grid_data = load_grid(task_id, [task_dir])
        if not grid_data:
            skipped += 1
            continue

        train_examples = grid_data.get("train", [])
        train_inputs = [ex["input"] for ex in train_examples]

        # Use the error program's output (the cognitive hypothesis) rather than
        # the participant's raw grid — they may differ by 1-2 cells when the
        # program is an approximate match under noise tolerance.
        prompt_output_grid = (
            program_output_grid if program_output_grid else target_error_grid
        )

        # Execute the error program on training inputs → side-by-side comparison.
        error_outputs = compute_training_outputs(
            rewritten_program, train_inputs, dsl_namespace, rewriter
        )
        training_comparison_str = format_training_comparison(
            train_examples, error_outputs
        )

        # Error program body (strip def/return).
        error_program_body = get_solver_body(rewritten_program)

        # Reference-solver body → function names for the per-record minimal
        # DSL. Prefer the hand-written GT solver; for eval-split tasks with
        # no such solver, fall back to the synthesized reference carried on
        # the record (original_program). Zero effect on training runs, where
        # solvers.get(task_id) hits first.
        solver_source = resolve_reference_solver(rec, solvers)
        if not solver_source:
            skipped += 1
            continue
        gt_body = get_solver_body(solver_source)

        gt_func_names = extract_function_names(gt_body)
        error_func_names = extract_function_names(error_program_body)
        all_func_names = gt_func_names | error_func_names

        per_record_dsl_code = build_minimal_dsl(
            all_func_names,
            dsl_api,
            preamble,
            api_only=args.api_only,
            dsl_sources=dsl_sources,
        )

        prompt = construct_base_prompt(
            task_id=task_id,
            output_grid=prompt_output_grid,
            description="",  # No NL in program-only classification.
            template=template,
            # One-entry map of the resolved reference: construct_base_prompt
            # re-looks-up solvers.get(task_id) internally to fill
            # {{ref_program}}, so this makes eval-split records resolve to
            # the same synthesized solver as the minimal-DSL extraction above
            # (rather than skipping). Identical to `solvers` for training.
            solvers={task_id: solver_source},
            dsl_code=per_record_dsl_code,
            task_dir=task_dir,
            include_nl=False,
            extra_substitutions={
                "error_program": error_program_body,
                "training_comparison": training_comparison_str,
            },
        )

        if prompt is None:
            skipped += 1
            continue

        record = {
            "task_id": task_id,
            "error_id": error_id,
            "description_id": error_id,  # pipeline uniqueness key
            "grid_hash": grid_hash_val,
            "program_id": program_id,
            "include_nl": False,
            "prompt": prompt,
        }
        results.append(record)

        if args.verbose and len(results) == 1:
            print("\n" + "=" * 60)
            print("EXAMPLE PROMPT (first entry):")
            print("=" * 60)
            print(f"Task ID: {task_id}")
            print(f"Error ID: {error_id}")
            print(f"Grid Hash: {grid_hash_val}")
            print(f"Program ID: {program_id}")
            print(f"DSL functions: {sorted(all_func_names)}")
            print("-" * 60)
            print(prompt)
            print("=" * 60 + "\n")

    print(
        f"Generated {len(results)} prompts, skipped {skipped} "
        f"(missing solver/grid/program), skipped {skipped_analytical} "
        "(analytical baselines)."
    )

    if not results:
        print("No prompts generated. Exiting.")
        return

    # Build the output-file stem.
    temp_tag = f"t{args.temperature:g}"
    think_tag = f"think-{args.thinking_level}" if args.thinking_level else "no-think"
    dsl_tag = "api" if args.api_only else "full"

    stem = build_output_stem(
        input_path=input_path,
        n_errors_in_run=len(results),
        n_errors_in_input=n_errors_in_input,
        tags={
            "script": "classify_err",
            "dsl": dsl_tag,
            "temp": temp_tag,
            "think": think_tag,
        },
        num_samples=args.num_samples,
    )

    prompts = [r["prompt"] for r in results]
    expanded = expand_for_sampling(results, args.num_samples, metadata_keys=METADATA_KEYS)
    total_requests = len(expanded)
    print(
        f"Total requests (prompts x samples): {total_requests} "
        f"({len(results)} prompts x {args.num_samples} samples)"
    )

    response_schema = get_characterization_response_schema()

    write_run_config(
        output_dir,
        config={
            "run_tag": args.run_tag,
            "model": args.model,
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "thinking_level": args.thinking_level,
            "api_only": args.api_only,
            "input_file": args.input_file,
            "task_dir": args.task_dir,
            "template_file": str(template_path),
        },
        response_schema=response_schema,
    )

    # ------------------------------------------------------------------ #
    # Execute selected mode                                               #
    # ------------------------------------------------------------------ #
    if args.mode == "count":
        run_count_mode(
            prompts=prompts,
            records=results,
            config=config,
            stem=stem,
            prompts_dir=prompts_dir,
            total_requests=total_requests,
            est_output_tokens=400,  # ~400 tokens for the structured schema.
        )

    elif args.mode in ("prepare", "submit"):
        run_prepare_submit_mode(
            expanded=expanded,
            config=config,
            response_schema=response_schema,
            stem=stem,
            requests_dir=requests_dir,
            submit=(args.mode == "submit"),
            metadata_keys=METADATA_KEYS,
        )

    elif args.mode == "async":
        run_async_mode(
            expanded=expanded,
            config=config,
            response_schema=response_schema,
            stem=stem,
            results_dir=results_dir,
            response_key="characterization_result",
            metadata_keys=METADATA_KEYS,
        )


if __name__ == "__main__":
    main()
