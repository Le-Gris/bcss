"""Check which synthesized solver programs actually solve their task.

Each program is parsed, rewritten to one function call per line, and run on the
task's training inputs and test input. A program is a valid solver
(``strict_solve``) only if it reproduces every training output and the test
output. Since the test pair is hidden during synthesis, the test output is a
held-out check.

Reads the output of ``synthesize_solver_programs.py`` and the ARC task JSONs.
Writes one row per program to ``eval_<input>.jsonl`` and prints a summary.

Usage:
    uv run python scripts/evaluate_synthesized_solver_programs.py \\
        --input-file <solver_synthesis_output>.jsonl \\
        --task-dir external/arc-agi-1/evaluation
"""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

# Ensure project root is on sys.path.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from hodel_dsl.dsl_parser import ProgramRewriter  # noqa: E402

from src.filters import build_dsl_namespace  # noqa: E402
from src.utils import (  # noqa: E402
    DEFAULT_PROGRAM_TIMEOUT_S,
    calculate_mismatch,
    grid_to_tuple,
    load_grid,
    time_limit,
)

DEFAULT_OUTPUT_DIR = "results/solver_synthesis_evaluation"


# ---------------------------------------------------------------------------
# Program wrapping / line counting
# ---------------------------------------------------------------------------


def wrap_solver_body(body: str) -> str:
    """Wrap an unnested program body in a ``def solver(I): ... return O``.

    The synthesis model returns only the *body* (assignment lines ending in an
    ``O = ...`` assignment), so we re-attach the function signature and a
    trailing ``return O`` before handing it to :class:`ProgramRewriter`.  If
    the body already carries its own ``def`` line (a model emitting a full
    function), it is passed through unchanged.
    """
    if body.lstrip().startswith("def "):
        return body
    indented = "\n".join("    " + line for line in body.splitlines())
    return f"def solver(I):\n{indented}\n    return O"


def count_body_lines(program_source: str) -> int:
    """Count assignment lines in a solver function (excludes ``def`` / ``return``)."""
    count = 0
    for line in program_source.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("def ") or stripped.startswith("return "):
            continue
        count += 1
    return count


def extract_program_body(record: Dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull the synthesized program body out of a results record.

    The synthesis pipeline writes the full parsed JSON response under the
    ``program`` key (``run_async_mode`` with ``response_key="program"``), so
    the body lives at ``record["program"]["program"]``.  We handle three shapes
    defensively:

    * dict with a ``program`` field → the body (async / batch output);
    * bare string → used directly (back-compat / hand-made fixtures);
    * dict carrying an ``{"error": "Invalid JSON", ...}`` sentinel (emitted by
      ``_parse_async_response`` when the model returned unparseable text) → an
      extraction failure.

    Returns ``(body, error)`` where exactly one is non-``None``.
    """
    raw = record.get("program")
    if isinstance(raw, str):
        body = raw
    elif isinstance(raw, dict):
        if "error" in raw and "program" not in raw:
            return None, f"Unparseable response: {raw.get('error')}"
        body = raw.get("program", "")
    elif raw is None:
        return None, "No 'program' field in record"
    else:
        return None, f"Unexpected 'program' type: {type(raw).__name__}"

    if not body or not body.strip():
        return None, "Empty program body"
    return body, None


# ---------------------------------------------------------------------------
# Core evaluation for a single record
# ---------------------------------------------------------------------------


def evaluate_record(
    record: Dict[str, Any],
    task_dir: Path,
    rewriter: ProgramRewriter,
    dsl_namespace: dict,
    timeout_s: float = DEFAULT_PROGRAM_TIMEOUT_S,
) -> Dict[str, Any]:
    """Evaluate a single solver-synthesis result record into a flat metrics dict."""
    task_id = record["task_id"]

    result: Dict[str, Any] = {
        "task_id": task_id,
        "entry_index": record.get("entry_index"),
        "sample_index": record.get("sample_index", 0),
        "retrieved_tasks": record.get("retrieved_tasks"),
        "retrieval_scores": record.get("retrieval_scores"),
        "is_syntactically_valid": False,
        "parse_error": None,
        "exec_success": False,
        "runtime_error": None,
        "rewritten_program": None,
        "line_count": None,
        "training_matches": None,
        "train_pass_count": None,
        "train_total": None,
        "all_training_correct": False,
        "any_training_correct": False,
        "test_match": None,
        "mismatch_proportion": None,
        "strict_solve": False,
        "program_output_grid": None,
        "cost": record.get("cost"),
        "usage": record.get("usage"),
    }

    # --- Extract program body ---
    body, extract_error = extract_program_body(record)
    if extract_error is not None:
        result["parse_error"] = extract_error
        return result

    # --- Wrap and parse ---
    source = wrap_solver_body(body)
    try:
        rewriter.parse(source)
        result["is_syntactically_valid"] = True
    except Exception as e:
        result["parse_error"] = str(e)
        return result

    # --- Rewrite to canonical form ---
    try:
        rewritten = rewriter.clean().rewrite()
        result["rewritten_program"] = rewritten
        result["line_count"] = count_body_lines(rewritten)
    except Exception as e:
        result["is_syntactically_valid"] = False
        result["parse_error"] = f"Rewrite failed: {e}"
        return result

    # --- Load grid data ---
    grid_data = load_grid(task_id, [task_dir])
    if not grid_data or "test" not in grid_data or not grid_data["test"]:
        result["runtime_error"] = f"No test grid data for task {task_id}"
        return result

    test_input = grid_data["test"][0]["input"]
    test_input_tuple = grid_to_tuple(test_input)
    test_output = grid_data["test"][0].get("output")
    gt_output_tuple = grid_to_tuple(test_output) if test_output is not None else None

    # --- Execute on the test input ---
    solver_fn = None
    try:
        exec_ns = copy.copy(dsl_namespace)
        exec(rewritten, exec_ns)
        solver_fn = exec_ns.get("solver")
        if solver_fn is None:
            result["runtime_error"] = "Rewritten program did not define 'solver'"
            return result
        with time_limit(timeout_s):
            program_output = solver_fn(test_input_tuple)
        result["exec_success"] = True
    except Exception as e:
        result["runtime_error"] = f"{type(e).__name__}: {e}"
        return result

    # Validate output is a 2D grid (tuple of tuples).
    if not isinstance(program_output, tuple) or not program_output:
        result["runtime_error"] = (
            f"Program output is not a grid: {type(program_output).__name__}"
        )
        return result
    if not all(isinstance(row, tuple) for row in program_output):
        result["runtime_error"] = "Program output rows are not tuples"
        return result

    result["program_output_grid"] = [list(row) for row in program_output]

    # --- Compare to ground-truth test output (tri-state) ---
    if gt_output_tuple is not None:
        result["test_match"] = program_output == gt_output_tuple
        result["mismatch_proportion"] = calculate_mismatch(program_output, gt_output_tuple)
    else:
        result["test_match"] = None
        result["mismatch_proportion"] = None

    # --- Evaluate on training examples ---
    train_examples = grid_data.get("train", [])
    if train_examples and solver_fn is not None:
        training_matches: List[bool] = []
        for pair in train_examples:
            train_input = pair.get("input")
            train_expected = pair.get("output")
            if train_input is None or train_expected is None:
                training_matches.append(False)
                continue
            try:
                with time_limit(timeout_s):
                    train_out = solver_fn(grid_to_tuple(train_input))
                training_matches.append(train_out == grid_to_tuple(train_expected))
            except Exception:
                training_matches.append(False)

        result["training_matches"] = training_matches
        result["train_total"] = len(training_matches)
        result["train_pass_count"] = sum(training_matches)
        result["all_training_correct"] = all(training_matches)
        result["any_training_correct"] = any(training_matches)

    # strict_solve: correct on ALL training pairs AND the test output. This is
    # the honest signal — by default the model never saw the test output, so
    # this measures genuine generalisation.
    result["strict_solve"] = bool(
        result["all_training_correct"] and result["test_match"]
    )

    return result


# ---------------------------------------------------------------------------
# Summary statistics (simple)
# ---------------------------------------------------------------------------


def print_summary(eval_results: List[Dict[str, Any]]) -> None:
    """Print per-sample summary stats, a per-task any-solved line, and failures."""
    total = len(eval_results)
    if total == 0:
        print("No results to summarize.")
        return

    valid = [r for r in eval_results if r["is_syntactically_valid"]]
    executed = [r for r in eval_results if r["exec_success"]]
    with_training = [r for r in eval_results if r["training_matches"] is not None]
    all_train = [r for r in with_training if r["all_training_correct"]]
    strict = [r for r in eval_results if r["strict_solve"]]
    has_test = [r for r in executed if r["test_match"] is not None]
    test_matched = [r for r in has_test if r["test_match"]]

    print("\n" + "=" * 65)
    print("  SOLVER EVALUATION SUMMARY")
    print("=" * 65)
    print(
        "\n  NOTE: test_match is a clean held-out signal only when the test pair"
        "\n  was hidden during synthesis (the default). If --include-test was"
        "\n  used, prefer strict_solve / all_training_correct instead.\n"
    )

    print(f"--- Per-sample (n={total}) ---")
    print(f"  Syntactically valid:       {len(valid)} ({len(valid)/total*100:.1f}%)")
    print(f"  Execution success:         {len(executed)} ({len(executed)/total*100:.1f}%)")
    if with_training:
        print(
            f"  All training correct:      {len(all_train)} "
            f"({len(all_train)/total*100:.1f}%)"
        )
    print(f"  Strict solve (train+test): {len(strict)} ({len(strict)/total*100:.1f}%)")
    if has_test:
        print(
            f"  Test match (of executed):  {len(test_matched)} "
            f"({len(test_matched)/len(has_test)*100:.1f}%)"
        )

    line_counts = [r["line_count"] for r in valid if r["line_count"] is not None]
    if line_counts:
        arr = np.array(line_counts)
        print(
            f"  Synthesized line count:    mean={np.mean(arr):.1f}  "
            f"median={np.median(arr):.1f}  min={np.min(arr)}  max={np.max(arr)}"
        )

    # Per-task: how many tasks had at least one strict-solving sample.
    by_task: Dict[str, List[Dict]] = defaultdict(list)
    for r in eval_results:
        by_task[r["task_id"]].append(r)
    num_tasks = len(by_task)
    task_solved = sum(1 for recs in by_task.values() if any(r["strict_solve"] for r in recs))
    print(
        f"\n--- Per-task (n={num_tasks}) ---\n"
        f"  Any sample strict-solved:  {task_solved}/{num_tasks} "
        f"({task_solved/num_tasks*100:.1f}%)"
    )

    # Failure breakdown.
    invalid = [r for r in eval_results if not r["is_syntactically_valid"]]
    runtime_failed = [
        r for r in eval_results if r["is_syntactically_valid"] and not r["exec_success"]
    ]
    if invalid or runtime_failed:
        print("\n--- Failure breakdown ---")
        if invalid:
            parse_errors: Dict[str, int] = defaultdict(int)
            for r in invalid:
                parse_errors[(r.get("parse_error") or "unknown")[:80]] += 1
            print(f"  Parse failures ({len(invalid)}):")
            for err, count in sorted(parse_errors.items(), key=lambda x: -x[1])[:10]:
                print(f"    [{count:>4}] {err}")
        if runtime_failed:
            runtime_errors: Dict[str, int] = defaultdict(int)
            for r in runtime_failed:
                runtime_errors[(r.get("runtime_error") or "unknown")[:80]] += 1
            print(f"  Runtime failures ({len(runtime_failed)}):")
            for err, count in sorted(runtime_errors.items(), key=lambda x: -x[1])[:10]:
                print(f"    [{count:>4}] {err}")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate synthesized solver programs against task train/test grids."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to solver-synthesis results JSONL file(s).",
    )
    parser.add_argument(
        "--task-dir",
        type=str,
        required=True,
        help="Directory containing ARC-format task JSONs (one per task_id).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag; nests output under <output-dir>/<run-tag>/.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one status line per record (useful for debugging).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PROGRAM_TIMEOUT_S,
        help=(
            "Per-program execution time budget in seconds (default: "
            f"{DEFAULT_PROGRAM_TIMEOUT_S:g}). A program exceeding it on the test "
            "input is recorded as a runtime error; on a train pair it counts as "
            "a "
            "mismatch. Set <= 0 to disable (NOT recommended: a pathological "
            "synthesized solver can hang the whole run)."
        ),
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input_file]
    task_dir = Path(args.task_dir)
    output_dir = Path(args.output_dir)
    if args.run_tag:
        output_dir = output_dir / args.run_tag

    for ip in input_paths:
        if not ip.exists():
            print(f"Error: Input file {ip} not found.")
            return
    if not task_dir.exists():
        print(f"Error: Task directory {task_dir} not found.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading resources...")
    dsl_path = str(_project_root / "hodel_dsl" / "dsl.py")
    primitives_path = str(_project_root / "hodel_dsl" / "primitives.py")
    rewriter = ProgramRewriter(
        dsl_symbols_path=dsl_path,
        dsl_constants_path=primitives_path,
    )
    print("  ProgramRewriter initialized.")

    dsl_namespace = build_dsl_namespace()
    print(f"  DSL namespace built ({len(dsl_namespace)} symbols).")

    all_eval_results: List[Dict[str, Any]] = []
    for input_path in input_paths:
        records: List[Dict[str, Any]] = []
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        print(f"\nLoaded {len(records)} records from {input_path.name}.")

        eval_results: List[Dict[str, Any]] = []
        for record in tqdm(records, desc=f"Evaluating {input_path.name}"):
            result = evaluate_record(
                record, task_dir, rewriter, dsl_namespace, timeout_s=args.timeout
            )
            eval_results.append(result)

            if args.verbose:
                if result["strict_solve"]:
                    status = "SOLVED"
                elif result["all_training_correct"]:
                    status = "TRAIN_OK"
                elif result["exec_success"]:
                    status = "EXEC_OK"
                elif result["is_syntactically_valid"]:
                    status = "VALID"
                else:
                    status = "INVALID"
                tp = result["train_pass_count"]
                tt = result["train_total"]
                train_str = f"  train={tp}/{tt}" if tt is not None else ""
                print(
                    f"  [{status:>8}] task={result['task_id']}  "
                    f"sample={result['sample_index']}{train_str}"
                )

        output_file = output_dir / f"eval_{input_path.stem}.jsonl"
        with open(output_file, "w") as f:
            for result in eval_results:
                f.write(json.dumps(result) + "\n")
        print(f"Detailed results saved to {output_file}")
        all_eval_results.extend(eval_results)

    print_summary(all_eval_results)


if __name__ == "__main__":
    main()
