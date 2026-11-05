"""Shared machinery for the scripts that prompt Gemini.

Provides the common command-line options, loading of tasks, reference programs
and the DSL, prompt construction, and the five run modes: ``count`` (token and
cost estimate), ``prepare`` / ``submit`` / ``check`` (Batch API) and ``async``
(direct calls with rate limiting). Used by ``synthesize_error_programs.py``,
``synthesize_solver_programs.py`` and ``classify_errors.py``.
"""

import argparse
import asyncio
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from google.genai import types
from tqdm import tqdm

from src.gemini_client import (
    AsyncRateLimiter,
    calculate_cost_from_tokens,
    count_tokens,
    process_batch_async,
    setup_gemini_client,
)
from src.utils import (
    format_full_dsl_for_prompt,
    format_grid,
    format_task_grids_for_prompt,
    get_solver_body,
    load_dsl_api,
    load_dsl_preamble,
    load_dsl_source,
    load_grid,
    load_solvers,
    load_unique_errors,
    record_id,
    unique_path,
)

DEFAULT_OUTPUT_DIR = "results/error_synthesis"


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Standard configuration shared across all pipeline scripts."""

    model: str | None = None  # required for run modes; check mode tolerates None
    temperature: float = 1.0
    thinking_level: Optional[str] = None
    num_samples: int = 1
    rpm: int = 60
    tpm: int = 1_000_000
    checkpoint_every: int = 50
    workers: int = 4
    no_token_count: bool = False
    include_nl: bool = False
    api_only: bool = False
    verbose: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        """Build from parsed argparse namespace, ignoring unknown fields."""
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in vars(args).items() if k in known})


# ---------------------------------------------------------------------------
# Argparse helpers
# ---------------------------------------------------------------------------


def add_common_args(
    parser: argparse.ArgumentParser,
    default_output_dir: str = DEFAULT_OUTPUT_DIR,
    input_file_required: bool = True,
) -> None:
    """Add CLI arguments shared by all pipeline scripts.

    Set ``input_file_required=False`` when the caller has an alternative
    source of per-error records (e.g. stage-2 ``--control`` mode synthesizes
    them on the fly from ``--task-dir``).
    """
    grp_io = parser.add_argument_group("Input / output")
    grp_io.add_argument(
        "--input-file",
        type=str,
        required=input_file_required,
        default=None,
        help="Path to the unique-errors JSONL (one record per unique error).",
    )
    grp_io.add_argument(
        "--control",
        action="store_true",
        help=(
            "Control mode: bypass --input-file and synthesize one record per "
            "task in --task-dir, using the task's correct test output as the "
            "participant output. If --input-file is also given, its task_ids "
            "are used purely as a filter (intersected with solvers ∩ task-dir)."
        ),
    )
    grp_io.add_argument(
        "--task-dir",
        type=str,
        default=None,
        help=(
            "Directory containing ARC-format task JSONs (one <task_id>.json "
            "per task with 'train' and 'test' lists of {input, output} grids). "
            "Required for count/prepare/submit/async; ignored in --mode check."
        ),
    )
    grp_io.add_argument(
        "--template-file",
        type=str,
        default=None,
        help="Path to the prompt template.",
    )
    grp_io.add_argument(
        "--output-dir",
        type=str,
        default=default_output_dir,
        help=f"Directory for output files (default: {default_output_dir}).",
    )
    grp_io.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag to group outputs under a subfolder.",
    )

    grp_sel = parser.add_argument_group("Task / error selection")
    grp_sel.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of prompts to generate.",
    )
    grp_sel.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed for reproducible sampling when --limit is used. "
            "Without --seed, --limit takes the first N entries; "
            "with --seed, --limit takes a random sample of N entries."
        ),
    )
    grp_sel.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=None,
        help="Specific task IDs to generate prompts for (space-separated).",
    )
    grp_sel.add_argument(
        "--filter-from",
        type=str,
        default=None,
        help="Path to a previous eval results JSONL to filter entries from.",
    )
    grp_sel.add_argument(
        "--filter-where",
        action="append",
        default=None,
        help=(
            "Filter condition as FIELD=VALUE (repeatable). "
            "Values are auto-cast to bool/int/float/string. "
            "Multiple conditions are ANDed together."
        ),
    )
    grp_sel.add_argument(
        "--filter-mode",
        type=str,
        choices=["any", "all", "none"],
        default="any",
        help=(
            "Aggregation across samples per (error_id, response_id) group: "
            "'any' (default) includes entry if any sample matches all conditions; "
            "'all' if every sample matches; 'none' if no sample matches."
        ),
    )

    grp_prompt = parser.add_argument_group("Prompt configuration")
    grp_prompt.add_argument(
        "--include-nl",
        action="store_true",
        help="Include each NL description in the prompt (one prompt per description).",
    )
    grp_prompt.add_argument(
        "--api-only",
        action="store_true",
        help="Include only DSL function signatures + docstrings instead of full source.",
    )

    grp_exec = parser.add_argument_group("Execution")
    grp_exec.add_argument(
        "--mode",
        type=str,
        choices=["count", "prepare", "async", "submit", "check"],
        default="count",
        help=(
            "Mode: count (token stats), prepare (Batch API JSONL), "
            "submit (upload+submit batch), async (live rate-limited), "
            "check (check batch job status)."
        ),
    )
    grp_exec.add_argument(
        "--model",
        "--model-name",
        type=str,
        default=None,
        help=(
            "Gemini model to use. Required for count/prepare/submit/async; "
            "ignored in --mode check (which reads the model name from the "
            "submitted batch's metadata). Pass an explicit model name such as "
            "'gemini-3.5-flash'."
        ),
    )
    grp_exec.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent samples to request per prompt (default: 1).",
    )
    grp_exec.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for generation (default: 1.0).",
    )
    grp_exec.add_argument(
        "--thinking-level",
        type=str,
        choices=["low", "high"],
        default=None,
        help="Thinking level (Gemini models that support it).",
    )

    grp_async = parser.add_argument_group("Async mode")
    grp_async.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="In async mode: write results to disk every N requests (default: 50).",
    )
    grp_async.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="RPM limit for async mode (default: 60).",
    )
    grp_async.add_argument(
        "--tpm",
        type=int,
        default=1_000_000,
        help="TPM limit for async mode (default: 1,000,000).",
    )

    grp_tok = parser.add_argument_group("Token counting")
    grp_tok.add_argument(
        "--no-token-count",
        action="store_true",
        help="Skip token counting in count mode.",
    )
    grp_tok.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers for parallel token counting (default: 4).",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the first generated prompt to stdout.",
    )


# ---------------------------------------------------------------------------
# Limit / sampling helper
# ---------------------------------------------------------------------------


def apply_limit(
    items: List[Any],
    limit: Optional[int],
    seed: Optional[int] = None,
    label: str = "entries",
) -> List[Any]:
    """Trim *items* to at most *limit*. With *seed*, take a reproducible random sample."""
    if limit is None or limit >= len(items):
        return items
    if seed is not None:
        rng = random.Random(seed)
        items = rng.sample(items, limit)
        print(f"Randomly sampled {len(items)} {label} (seed={seed}).")
    else:
        items = items[:limit]
        print(f"Limited to first {len(items)} {label}.")
    return items


# ---------------------------------------------------------------------------
# Previous-results filtering
# ---------------------------------------------------------------------------


def _cast_filter_value(value_str: str) -> Any:
    """Auto-cast a filter value string to bool, int, float, or str."""
    low = value_str.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(value_str)
    except ValueError:
        pass
    try:
        return float(value_str)
    except ValueError:
        pass
    return value_str


def parse_filter_conditions(conditions: Optional[List[str]]) -> List[tuple]:
    """Parse ``FIELD=VALUE`` strings into ``(field, value)`` pairs."""
    if not conditions:
        return []
    parsed: List[tuple] = []
    for cond in conditions:
        if "=" not in cond:
            raise ValueError(f"Invalid filter condition '{cond}': expected FIELD=VALUE")
        field, value_str = cond.split("=", 1)
        field = field.strip()
        value_str = value_str.strip()
        if not field:
            raise ValueError(f"Empty field name in filter condition '{cond}'")
        parsed.append((field, _cast_filter_value(value_str)))
    return parsed


def _record_matches(record: Dict[str, Any], conditions: List[tuple]) -> bool:
    for f, v in conditions:
        if record.get(f) != v:
            return False
    return True


def load_filter_pairs(
    filter_from: str,
    filter_where: Optional[List[str]],
    filter_mode: str = "any",
) -> set:
    """Load a previous results JSONL and return matching ``(error_id, response_id)`` pairs."""
    from collections import defaultdict

    filter_path = Path(filter_from)
    if not filter_path.exists():
        raise FileNotFoundError(f"Filter file not found: {filter_from}")

    conditions = parse_filter_conditions(filter_where)

    groups: Dict[tuple, List[bool]] = defaultdict(list)
    total_records = 0
    with open(filter_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total_records += 1
            key = (record["error_id"], record["response_id"])
            groups[key].append(_record_matches(record, conditions))

    result: set = set()
    for key, matches in groups.items():
        if filter_mode == "any" and any(matches):
            result.add(key)
        elif filter_mode == "all" and all(matches):
            result.add(key)
        elif filter_mode == "none" and not any(matches):
            result.add(key)

    cond_str = " AND ".join(f"{f}={v}" for f, v in conditions) if conditions else "(no conditions)"
    print(
        f"Filter: loaded {total_records} records from {filter_path.name}, "
        f"{len(groups)} unique entries, "
        f"{len(result)} pass [{filter_mode}: {cond_str}]"
    )
    return result


def apply_prompt_filter(
    prompt_records: List[Dict[str, Any]],
    allowed_pairs: set,
) -> List[Dict[str, Any]]:
    """Keep only prompt records whose ``(error_id, response_id)`` is in *allowed_pairs*."""
    before = len(prompt_records)
    filtered = [
        r for r in prompt_records if (r["error_id"], r["response_id"]) in allowed_pairs
    ]
    print(f"Previous-results filter: {before} -> {len(filtered)} prompt records")
    return filtered


# ---------------------------------------------------------------------------
# Run config writer
# ---------------------------------------------------------------------------


def write_run_config(
    output_dir: Path,
    config: Dict[str, Any],
    response_schema: Optional[Dict[str, Any]] = None,
) -> None:
    """Write a small run_config.json to capture key run metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    payload = {**config, "timestamp": int(time.time())}
    if response_schema is not None:
        payload["response_schema"] = response_schema
    with open(config_path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Output stem builder
# ---------------------------------------------------------------------------


def build_output_stem(
    input_path: Path,
    n_errors_in_run: int,
    n_errors_in_input: int,
    tags: Dict[str, str],
    num_samples: int = 1,
) -> str:
    """Build a descriptive filename stem from *tags* + input-file stem + scope.

    The input-file stem is used as-is, with a trailing ``.filtered`` segment
    stripped so the output stem reads naturally when the input was prefiltered
    (e.g. ``errors_raw.filtered.jsonl`` → input_tag ``errors_raw``).
    """
    input_stem = input_path.stem
    if input_stem.endswith(".filtered"):
        input_stem = input_stem[: -len(".filtered")]
    input_tag = input_stem or "errors"

    if n_errors_in_run < n_errors_in_input:
        scope_tag = "partial"
    else:
        scope_tag = "all"

    parts = list(tags.values()) + [input_tag, scope_tag]
    stem = "_".join(parts)
    if num_samples > 1:
        stem += f"_n{num_samples}"
    return stem


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------


def load_dsl_resources(api_only: bool = False) -> Dict[str, Any]:
    """Load the solvers and the formatted DSL prompt string.

    Returns ``{"solvers", "dsl_code"}``.  Shared between the error-driven and
    control-driven stage-2 paths.
    """
    solvers = load_solvers()
    dsl_api = load_dsl_api()
    dsl_sources = load_dsl_source()
    preamble = load_dsl_preamble()

    dsl_code = format_full_dsl_for_prompt(
        dsl_api=dsl_api,
        dsl_sources=dsl_sources,
        preamble=preamble,
        api_only=api_only,
    )
    dsl_mode = "API-only" if api_only else "full source"
    print(f"DSL mode: {dsl_mode} ({len(dsl_code):,} chars)")

    return {"solvers": solvers, "dsl_code": dsl_code}


def load_common_resources(
    input_path: Path,
    api_only: bool = False,
) -> Dict[str, Any]:
    """Load the unique-errors JSONL, the solvers, and the formatted DSL string.

    Returns a dict with keys: ``errors`` (list of records), ``solvers``, ``dsl_code``.
    """
    print("Loading resources...")
    errors = list(load_unique_errors(input_path))
    dsl = load_dsl_resources(api_only=api_only)
    print(f"Loaded {len(errors)} unique errors.")

    return {
        "errors": errors,
        "solvers": dsl["solvers"],
        "dsl_code": dsl["dsl_code"],
    }


def build_control_error_records(
    task_ids: List[str],
    task_dir: Path,
) -> List[Dict[str, Any]]:
    """Synthesize per-error records for the control condition.

    For each task in *task_ids*, reads ``<task_dir>/<task_id>.json`` and emits
    a record whose ``output_grid`` is the task's correct test output. Tasks
    whose JSON is missing are skipped (with a warning).  Mirrors the
    unique-errors schema consumed by :func:`build_prompt_records`.
    """
    records: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for task_id in task_ids:
        task_json = task_dir / f"{task_id}.json"
        if not task_json.exists():
            skipped.append(task_id)
            continue
        with open(task_json, "r") as f:
            data = json.load(f)
        test_examples = data.get("test", [])
        if not test_examples:
            skipped.append(task_id)
            continue
        records.append(
            {
                "task_id": task_id,
                "output_grid": test_examples[0]["output"],
                "dataset": "control_correct_output",
                "source": "control",
                "attempt_count": 1,
                "descriptions": [],
            }
        )
    if skipped:
        print(f"Warning: skipped {len(skipped)} control tasks (missing JSON or no test): {skipped[:5]}{'...' if len(skipped) > 5 else ''}")
    return records


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def construct_base_prompt(
    task_id: str,
    output_grid: List[List[int]],
    description: Optional[str],
    template: str,
    solvers: Dict[str, str],
    dsl_code: str,
    task_dir: Path,
    include_nl: bool = False,
    extra_substitutions: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Construct a filled prompt for a single (error, optional-description) pair.

    Performs the standard substitutions (``{{dsl_code}}``, ``{{ref_program}}``,
    ``{{ref_examples}}``, ``{{participant_description}}``,
    ``{{participant_output}}``).  Additional placeholders may be filled via
    *extra_substitutions* (e.g. ``{"num_modified_lines": "3"}``).

    Returns ``None`` if the task has no solver or no grid data.
    """
    solver_source = solvers.get(task_id)
    if not solver_source:
        return None

    program_body = get_solver_body(solver_source)

    grid_data = load_grid(task_id, [task_dir])
    if not grid_data:
        return None

    examples_str = format_task_grids_for_prompt(grid_data)
    participant_output_str = format_grid(output_grid)

    if include_nl and description:
        participant_description = f"**Natural Language Description:** {description.strip()}"
    else:
        participant_description = ""

    prompt = template
    prompt = prompt.replace("{{dsl_code}}", dsl_code)
    prompt = prompt.replace("{{ref_program}}", program_body)
    prompt = prompt.replace("{{ref_examples}}", examples_str)
    prompt = prompt.replace("{{participant_description}}", participant_description)
    prompt = prompt.replace("{{participant_output}}", participant_output_str)

    if extra_substitutions:
        for key, value in extra_substitutions.items():
            prompt = prompt.replace(f"{{{{{key}}}}}", value)

    # Collapse runs of 3+ blank lines into 2 (cleans up empty substitutions)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt


def build_prompt_records(
    errors: List[Dict[str, Any]],
    template: str,
    solvers: Dict[str, str],
    dsl_code: str,
    task_dir: Path,
    num_steps: int,
    include_nl: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Expand a list of unique-error records into per-prompt records.

    NL policy:
    - ``include_nl`` → one prompt per (error, description), or one prompt with
      no NL if ``descriptions`` is empty.
    - otherwise → one prompt per error, NL omitted.

    Each output record carries ``error_id``, ``task_id``, ``response_id``,
    ``output_grid``, ``prompt``, ``description`` (None or text), and minimal
    carry-through metadata (``dataset``, ``source``, ``attempt_count``).
    ``response_id`` comes from the description in NL modes and is ``None`` in
    the no-NL mode (use ``error_id`` to identify the error itself).  If the
    input record lacks ``error_id`` (older prefilter outputs), it is derived
    on the fly via :func:`record_id`.
    """
    records: List[Dict[str, Any]] = []
    skipped = 0

    for error in errors:
        task_id = error["task_id"]
        output_grid = error["output_grid"]
        descriptions = error.get("descriptions") or []
        error_id = error.get("error_id") or record_id(task_id, output_grid)
        common = {
            "error_id": error_id,
            "task_id": task_id,
            "output_grid": output_grid,
            "dataset": error.get("dataset"),
            "source": error.get("source"),
            "attempt_count": error.get("attempt_count"),
        }

        if include_nl:
            iter_descriptions = descriptions if descriptions else [None]
            use_nl = True
        else:
            iter_descriptions = [None]
            use_nl = False

        for desc in iter_descriptions:
            desc_text = desc.get("text", "") if desc else ""
            prompt = construct_base_prompt(
                task_id=task_id,
                output_grid=output_grid,
                description=desc_text,
                template=template,
                solvers=solvers,
                dsl_code=dsl_code,
                task_dir=task_dir,
                include_nl=use_nl,
                extra_substitutions={"num_modified_lines": str(num_steps)},
            )
            if prompt is None:
                skipped += 1
                continue

            response_id = desc["response_id"] if desc else None
            record = {
                **common,
                "response_id": response_id,
                "description": desc_text if desc else None,
                "prompt": prompt,
            }
            records.append(record)

            if verbose and len(records) == 1:
                print("\n" + "=" * 60)
                print("EXAMPLE PROMPT (first record):")
                print("=" * 60)
                print(f"Task ID: {task_id}")
                print(f"Response ID: {response_id}")
                print("-" * 60)
                print(prompt)
                print("=" * 60 + "\n")

    print(f"Built {len(records)} prompt records, skipped {skipped} (missing solver/grid).")
    return records


# ---------------------------------------------------------------------------
# Sampling expansion
# ---------------------------------------------------------------------------


DEFAULT_METADATA_KEYS = ("error_id", "task_id", "response_id")


def expand_for_sampling(
    records: List[Dict[str, Any]],
    num_samples: int,
    metadata_keys: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Expand prompt records into per-sample items.

    *metadata_keys* controls which keys are copied from each record onto each
    expanded item (in addition to ``prompt``, ``entry_index``, ``sample_index``
    which are always present).  When ``None``, falls back to the default
    ``("error_id", "task_id", "response_id")`` — the error-pipeline schema.
    Pipelines with a different metadata schema (e.g. solver synthesis with
    ``retrieved_tasks`` / ``retrieval_scores``) pass an explicit list.
    """
    keys = list(metadata_keys) if metadata_keys is not None else list(DEFAULT_METADATA_KEYS)
    expanded: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        for s in range(num_samples):
            item: Dict[str, Any] = {
                "prompt": record["prompt"],
                "entry_index": i,
                "sample_index": s,
            }
            for k in keys:
                item[k] = record[k]
            expanded.append(item)
    return expanded


# ---------------------------------------------------------------------------
# Response parsing helpers
# ---------------------------------------------------------------------------


def _parse_response_text(texts: List[str]) -> Dict[str, Any] | None:
    """Decode the JSON answer from a response's non-thought text parts.

    Gemini can split one JSON answer across several text parts (e.g.
    ``'{"modifications":'`` then ``' [...]}'``), so the parts are joined before
    decoding.  Undecodable text is kept as ``{"error", "raw"}`` rather than
    dropped, so it can be inspected and re-parsed later.
    """
    text = "".join(texts)
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception:
        return {"error": "Invalid JSON", "raw": text}


def _parse_async_response(
    result: Any,
    model: str,
    batch_mode: bool = False,
) -> Dict[str, Any]:
    """Parse a single async Gemini API response object."""
    response_data = None
    thoughts = None
    cost = 0.0
    usage = None

    if result:
        texts: List[str] = []
        thought_texts: List[str] = []
        if hasattr(result, "candidates") and result.candidates:
            candidate = result.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                for part in candidate.content.parts or []:
                    text = getattr(part, "text", None)
                    if not text:
                        continue
                    if getattr(part, "thought", False):
                        thought_texts.append(text)
                    else:
                        texts.append(text)
        thoughts = "\n".join(thought_texts) or None
        response_data = _parse_response_text(texts)

        if response_data is None:
            try:
                fallback_text = result.text
            except Exception:
                fallback_text = None
            if fallback_text:
                response_data = _parse_response_text([fallback_text])

    if result and hasattr(result, "usage_metadata") and result.usage_metadata:
        m = result.usage_metadata
        prompt_tok = getattr(m, "prompt_token_count", 0) or 0
        candidates_tok = getattr(m, "candidates_token_count", 0) or 0
        thinking_tok = getattr(m, "thinking_token_count", 0) or 0
        usage = {
            "prompt_tokens": prompt_tok,
            "candidates_tokens": candidates_tok,
            "total_tokens": getattr(m, "total_token_count", 0) or 0,
        }
        if thinking_tok > 0:
            usage["thinking_tokens"] = thinking_tok

        cost = calculate_cost_from_tokens(
            prompt_tokens=prompt_tok,
            completion_tokens=candidates_tok + thinking_tok,
            model=model,
            batch_mode=batch_mode,
        )

    return {"response_data": response_data, "thoughts": thoughts, "usage": usage, "cost": cost}


def _parse_batch_response(res_entry: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Parse a single batch result entry (raw JSON from Gemini batch API)."""
    response_data = None
    thoughts = None
    usage = None
    cost = 0.0

    if "response" in res_entry and res_entry["response"]:
        resp = res_entry["response"]

        if "usageMetadata" in resp:
            um = resp["usageMetadata"]
            prompt_tokens = um.get("promptTokenCount", 0)
            candidates_tokens = um.get("candidatesTokenCount", 0)
            thinking_tokens = um.get("thoughtsTokenCount", 0)

            usage = {
                "prompt_tokens": prompt_tokens,
                "candidates_tokens": candidates_tokens,
                "total_tokens": um.get("totalTokenCount", 0),
            }
            if thinking_tokens > 0:
                usage["thinking_tokens"] = thinking_tokens

            cost = calculate_cost_from_tokens(
                prompt_tokens=prompt_tokens,
                completion_tokens=candidates_tokens + thinking_tokens,
                model=model,
                batch_mode=True,
            )

        texts: List[str] = []
        thought_texts: List[str] = []
        if "candidates" in resp and resp["candidates"]:
            candidate = resp["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                for part in candidate["content"]["parts"]:
                    text = part.get("text")
                    if not text:
                        continue
                    if part.get("thought"):
                        thought_texts.append(text)
                    else:
                        texts.append(text)
        thoughts = "\n".join(thought_texts) or None
        response_data = _parse_response_text(texts)

        if response_data is None and resp.get("text"):
            response_data = _parse_response_text([resp["text"]])

    return {"response_data": response_data, "thoughts": thoughts, "usage": usage, "cost": cost}


def _print_usage_summary(
    total_prompt_tokens: int,
    total_candidates_tokens: int,
    total_thinking_tokens: int,
    total_cost: float,
    batch_mode: bool = False,
) -> None:
    """Print a usage & cost summary block."""
    print("\n--- Usage & Cost Summary ---")
    print(f"  Total prompt tokens:     {total_prompt_tokens:,}")
    print(f"  Total candidates tokens: {total_candidates_tokens:,}")
    if total_thinking_tokens > 0:
        print(f"  Total thinking tokens:   {total_thinking_tokens:,}")
    all_tokens = total_prompt_tokens + total_candidates_tokens + total_thinking_tokens
    print(f"  Total tokens:            {all_tokens:,}")
    cost_note = " (batch 50% discount applied)" if batch_mode else ""
    print(f"  Estimated cost (USD):    ${total_cost:.4f}{cost_note}")
    print("-----------------------------")


# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------


def run_count_mode(
    prompts: List[str],
    records: List[Dict[str, Any]],
    config: PipelineConfig,
    stem: str,
    prompts_dir: Path,
    total_requests: int,
    est_output_tokens: Optional[int] = None,
) -> None:
    """Count tokens, estimate cost, and save prompts JSONL."""
    if config.no_token_count:
        print("Token counting skipped (--no-token-count).")
    else:
        client = setup_gemini_client()
        print(f"Counting tokens for {len(prompts)} unique prompts using {config.workers} workers...")

        def count_tokens_task(p):
            try:
                return count_tokens(client, p, config.model)
            except Exception as e:
                print(f"Error counting tokens: {e}")
                return 0

        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            token_counts = list(
                tqdm(executor.map(count_tokens_task, prompts), total=len(prompts), desc="Counting tokens")
            )

        if token_counts:
            counts_arr = np.array(token_counts)
            print(f"\n--- Token count stats ({len(token_counts)} unique prompts, model={config.model}) ---")
            print(f"  Total tokens:  {np.sum(counts_arr):,}")
            print(f"  Min tokens:    {np.min(counts_arr)}")
            print(f"  Max tokens:    {np.max(counts_arr)}")
            print(f"  Mean tokens:   {np.mean(counts_arr):.1f}")
            print(f"  Median tokens: {np.median(counts_arr):.1f}")
            if config.num_samples > 1:
                print(
                    f"  Total tokens (x{config.num_samples} samples): "
                    f"{np.sum(counts_arr) * config.num_samples:,}"
                )
            print("----------------------------\n")

            if est_output_tokens is not None:
                total_prompt_tokens = int(np.sum(counts_arr)) * config.num_samples
                total_completion_tokens = est_output_tokens * total_requests
                est_cost = calculate_cost_from_tokens(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    model=config.model,
                )
                est_cost_batch = calculate_cost_from_tokens(
                    prompt_tokens=total_prompt_tokens,
                    completion_tokens=total_completion_tokens,
                    model=config.model,
                    batch_mode=True,
                )
                print(f"--- Estimated cost (model={config.model}) ---")
                print(f"  Est. output tokens/response: ~{est_output_tokens}")
                print(f"  Total prompt tokens:         {total_prompt_tokens:,}")
                print(f"  Total completion tokens:     {total_completion_tokens:,}")
                print(f"  Estimated cost (async):      ${est_cost:.4f}")
                print(f"  Estimated cost (batch, 50%): ${est_cost_batch:.4f}")
                if config.thinking_level:
                    print("  Note: thinking tokens not included in estimate; actual cost will be higher.")
                print("----------------------------\n")

    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = unique_path(prompts_dir / f"prompts_{stem}.jsonl")
    with open(prompts_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Prompts saved to {prompts_file}")


def run_prepare_submit_mode(
    expanded: List[Dict[str, Any]],
    config: PipelineConfig,
    response_schema: Dict[str, Any],
    stem: str,
    requests_dir: Path,
    batch_config: Optional[Dict[str, Any]] = None,
    submit: bool = False,
    metadata_keys: Optional[List[str]] = None,
) -> None:
    """Prepare Batch API JSONL and optionally submit the job.

    *metadata_keys* mirrors :func:`run_async_mode`: chooses which keys from
    each expanded item are recorded in the metadata mirror file alongside
    ``key`` / ``entry_index`` / ``sample_index``.  Saved in
    ``meta_output['metadata_keys']`` so :func:`run_check_mode` can replay
    the same schema when assembling output records.
    """
    keys = list(metadata_keys) if metadata_keys is not None else list(DEFAULT_METADATA_KEYS)
    total_requests = len(expanded)
    print(f"Preparing Batch API JSONL in {requests_dir}...")
    requests_dir.mkdir(parents=True, exist_ok=True)
    batch_filename = unique_path(requests_dir / f"batch_request_{stem}.jsonl")
    metadata_filename = unique_path(requests_dir / f"batch_metadata_{stem}.json")

    metadata_requests = []
    with open(batch_filename, "w") as f:
        for item in expanded:
            key = f"entry_{item['entry_index']}_sample_{item['sample_index']}"

            gen_config: Dict[str, Any] = {
                "temperature": config.temperature,
                "responseMimeType": "application/json",
                "responseJsonSchema": response_schema,
            }
            if config.thinking_level:
                gen_config["thinkingConfig"] = {
                    "includeThoughts": True,
                    "thinkingLevel": config.thinking_level.upper(),
                }

            request = {
                "key": key,
                "request": {
                    "contents": [{"parts": [{"text": item["prompt"]}]}],
                    "generationConfig": gen_config,
                },
            }
            f.write(json.dumps(request) + "\n")

            meta_record: Dict[str, Any] = {
                "key": key,
                "entry_index": item["entry_index"],
                "sample_index": item["sample_index"],
            }
            for k in keys:
                meta_record[k] = item[k]
            metadata_requests.append(meta_record)

    meta_config: Dict[str, Any] = {
        "model": config.model,
        "temperature": config.temperature,
        "thinking_level": config.thinking_level,
        "num_samples": config.num_samples,
        "include_nl": config.include_nl,
        "api_only": config.api_only,
    }
    if batch_config:
        meta_config.update(batch_config)

    meta_output: Dict[str, Any] = {
        "config": meta_config,
        "metadata_keys": keys,
        "requests": metadata_requests,
    }
    with open(metadata_filename, "w") as f:
        json.dump(meta_output, f, indent=2)

    print(f"Saved {total_requests} requests to {batch_filename}")
    print(f"Saved metadata to {metadata_filename}")

    if submit:
        client = setup_gemini_client()
        print(f"Uploading {batch_filename} to Gemini...")
        try:
            uploaded_file = client.files.upload(
                file=str(batch_filename),
                config=types.UploadFileConfig(
                    display_name=f"batch_{stem}",
                    mime_type="jsonl",
                ),
            )
            print(f"Uploaded file: {uploaded_file.name}")

            model_id = config.model
            if not model_id.startswith("models/"):
                model_id = f"models/{model_id}"

            print(f"Creating batch job for model {model_id}...")
            batch_job = client.batches.create(
                model=model_id,
                src=uploaded_file.name,
                config={"display_name": f"pipeline_{stem}_{int(time.time())}"},
            )
            print(f"Created batch job: {batch_job.name}")
            state = batch_job.state.name if hasattr(batch_job.state, "name") else batch_job.state
            print(f"Status: {state}")

            meta_output["job_info"] = {
                "job_name": batch_job.name,
                "file_name": uploaded_file.name,
                "model": config.model,
                "timestamp": int(time.time()),
            }
            with open(metadata_filename, "w") as f:
                json.dump(meta_output, f, indent=2)
            print(f"Updated metadata with job info: {metadata_filename}")

        except Exception as e:
            print(f"Error submitting batch job: {e}")
            import traceback

            traceback.print_exc()


def run_check_mode(
    config: PipelineConfig,
    requests_dir: Path,
    results_dir: Path,
    response_key: str = "response_data",
) -> None:
    """Check batch job status and download results.

    The metadata-key schema is read off the saved
    ``batch_metadata_*.json[metadata_keys]`` slot written by
    :func:`run_prepare_submit_mode`.  Metadata files lacking that key
    fall back to the error-pipeline default (``DEFAULT_METADATA_KEYS``).
    """
    client = setup_gemini_client()
    metadata_files = sorted(requests_dir.glob("batch_metadata_*.json"))
    if not metadata_files:
        print(f"Error: No batch_metadata_*.json found in {requests_dir}.")
        return

    metadata_filename = metadata_files[-1]
    print(f"Using metadata file: {metadata_filename}")

    with open(metadata_filename, "r") as f:
        meta_data = json.load(f)

    if "job_info" not in meta_data:
        print(f"Error: No job info found in {metadata_filename}. Has this job been submitted?")
        return

    job_info = meta_data["job_info"]
    job_name = job_info["job_name"]
    print(f"Checking status for job: {job_name}...")

    try:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name if hasattr(batch_job.state, "name") else batch_job.state
        print(f"Current State: {state}")

        if state == "JOB_STATE_SUCCEEDED":
            print("Job succeeded! Downloading results...")
            result_file_name = batch_job.dest.file_name
            content_bytes = client.files.download(file=result_file_name)
            content_str = content_bytes.decode("utf-8")

            results_by_key = {}
            for line in content_str.splitlines():
                if not line.strip():
                    continue
                res = json.loads(line)
                results_by_key[res["key"]] = res

            requests_meta = meta_data["requests"]
            keys = meta_data.get("metadata_keys") or list(DEFAULT_METADATA_KEYS)
            model_name = job_info.get("model", config.model)
            if model_name is None:
                print(
                    "Error: batch metadata predates job_info[model] and --model "
                    "was not supplied; re-run with --model <name> to enable "
                    "cost calculation and result download."
                )
                return

            results_filename = results_dir / (
                metadata_filename.name.replace("batch_metadata_", "batch_results_").replace(
                    ".json", ".jsonl"
                )
            )

            total_prompt_tokens = 0
            total_candidates_tokens = 0
            total_thinking_tokens = 0
            total_cost = 0.0

            results_dir.mkdir(parents=True, exist_ok=True)
            with open(results_filename, "w") as f:
                for req in requests_meta:
                    key = req["key"]
                    if key not in results_by_key:
                        print(f"Warning: Missing result for key {key}")
                        continue

                    parsed = _parse_batch_response(results_by_key[key], model_name)

                    if parsed["usage"]:
                        total_prompt_tokens += parsed["usage"].get("prompt_tokens", 0)
                        total_candidates_tokens += parsed["usage"].get("candidates_tokens", 0)
                        total_thinking_tokens += parsed["usage"].get("thinking_tokens", 0)
                    total_cost += parsed["cost"]

                    output: Dict[str, Any] = {k: req[k] for k in keys}
                    output["entry_index"] = req["entry_index"]
                    output["sample_index"] = req["sample_index"]
                    output[response_key] = parsed["response_data"]
                    output["thoughts"] = parsed["thoughts"]
                    output["usage"] = parsed["usage"]
                    output["cost"] = parsed["cost"]
                    output["job_id"] = job_name
                    f.write(json.dumps(output) + "\n")

            print(f"Successfully processed {len(results_by_key)} results.")
            print(f"Saved results to {results_filename}")

            _print_usage_summary(
                total_prompt_tokens,
                total_candidates_tokens,
                total_thinking_tokens,
                total_cost,
                batch_mode=True,
            )

        elif state == "JOB_STATE_FAILED":
            print(f"Job failed. Error: {batch_job.error}")
        else:
            print("Job is still processing. Please check again later.")

    except Exception as e:
        print(f"Error checking batch job: {e}")
        import traceback

        traceback.print_exc()


def run_async_mode(
    expanded: List[Dict[str, Any]],
    config: PipelineConfig,
    response_schema: Dict[str, Any],
    stem: str,
    results_dir: Path,
    response_key: str = "response_data",
    metadata_keys: Optional[List[str]] = None,
) -> None:
    """Run prompts through the async Gemini API with rate limiting.

    *metadata_keys* mirrors the param on :func:`expand_for_sampling`: which
    keys are read off each expanded item and threaded onto the output JSONL
    record.  ``None`` uses the error-pipeline ``error_id`` / ``task_id`` /
    ``response_id`` keys; explicit lists let other callers (e.g. solver
    synthesis) thread arbitrary
    metadata (e.g. ``retrieved_tasks``).
    """
    keys = list(metadata_keys) if metadata_keys is not None else list(DEFAULT_METADATA_KEYS)
    total_requests = len(expanded)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_filename = unique_path(results_dir / f"async_results_{stem}.jsonl")
    expanded_prompts = [item["prompt"] for item in expanded]

    print(
        f"Running async processing for {total_requests} requests, "
        f"{config.rpm} RPM, {config.tpm:,} TPM..."
    )

    rate_limiter = AsyncRateLimiter(config.rpm, config.tpm)
    checkpoint_every = max(1, config.checkpoint_every)

    total_prompt_tokens = 0
    total_candidates_tokens = 0
    total_thinking_tokens = 0
    total_cost = 0.0

    with open(results_filename, "w") as f:
        for chunk_start in range(0, total_requests, checkpoint_every):
            chunk_end = min(chunk_start + checkpoint_every, total_requests)
            chunk_prompts = expanded_prompts[chunk_start:chunk_end]
            chunk_items = expanded[chunk_start:chunk_end]

            async def run_chunk():
                return await process_batch_async(
                    prompts=chunk_prompts,
                    model=config.model,
                    rate_limiter=rate_limiter,
                    temperature=config.temperature,
                    thinking_level=config.thinking_level,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                )

            chunk_results = asyncio.run(run_chunk())

            for result, item, prompt_text in zip(chunk_results, chunk_items, chunk_prompts):
                parsed = _parse_async_response(result, config.model)

                if parsed["usage"]:
                    total_prompt_tokens += parsed["usage"].get("prompt_tokens", 0)
                    total_candidates_tokens += parsed["usage"].get("candidates_tokens", 0)
                    total_thinking_tokens += parsed["usage"].get("thinking_tokens", 0)
                total_cost += parsed["cost"]

                output: Dict[str, Any] = {k: item[k] for k in keys}
                output["entry_index"] = item["entry_index"]
                output["sample_index"] = item["sample_index"]
                output[response_key] = parsed["response_data"]
                output["thoughts"] = parsed["thoughts"]
                output["usage"] = parsed["usage"]
                output["cost"] = parsed["cost"]
                f.write(json.dumps(output) + "\n")

            f.flush()
            print(
                f"Checkpoint: wrote {len(chunk_results)} results "
                f"(requests {chunk_start}-{chunk_end - 1})"
            )

    print(f"Saved {total_requests} results to {results_filename}")
    _print_usage_summary(
        total_prompt_tokens, total_candidates_tokens, total_thinking_tokens, total_cost
    )
