#!/usr/bin/env python3
"""Embed ARC tasks with a Gemini embedding model, for retrieving similar tasks.

Each task's training grids (and, optionally, its test input and a text
description) are rendered and embedded in one call. Run it twice: once over the
tasks that serve as examples (``--task-type RETRIEVAL_DOCUMENT``) and once over
the tasks to solve (``--task-type RETRIEVAL_QUERY``). The two files are the
``--doc-embeddings`` / ``--query-embeddings`` inputs of
``synthesize_solver_programs.py --retrieval embedding``.

Reads a directory of ARC task JSONs. Writes one ``.npz`` of embeddings to
``--output-dir``; ``--mode count`` only reports what would be embedded.

Usage:
    uv run python scripts/generate_task_embeddings.py \\
        --task-dir external/arc-agi-1/training --task-type RETRIEVAL_DOCUMENT \\
        --model gemini-embedding-2-preview --mode embed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tenacity
from matplotlib.colors import ListedColormap
from tqdm import tqdm

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from google.genai import types  # noqa: E402

from src.gemini_client import _RETRY_POLICY, setup_gemini_client  # noqa: E402
from src.utils import load_descriptions, load_grid  # noqa: E402

# ---------------------------------------------------------------------------
# ARC palette (10 colours).
# ---------------------------------------------------------------------------

ARC_COLORS = [
    "#000000",  # 0: black
    "#0074D9",  # 1: blue
    "#FF4136",  # 2: red
    "#2ECC40",  # 3: green
    "#FFDC00",  # 4: yellow
    "#AAAAAA",  # 5: grey
    "#F012BE",  # 6: magenta
    "#FF851B",  # 7: orange
    "#7FDBFF",  # 8: light blue
    "#870C25",  # 9: dark red
]
ARC_CMAP = ListedColormap(ARC_COLORS)

VALID_TASK_TYPES = {
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "CLUSTERING",
}


# ---------------------------------------------------------------------------
# Grid rendering
# ---------------------------------------------------------------------------


def render_grid(grid: List[List[int]], ax: plt.Axes) -> None:
    arr = np.array(grid, dtype=int)
    ax.imshow(arr, cmap=ARC_CMAP, vmin=0, vmax=9, interpolation="nearest")
    rows, cols = arr.shape
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color="#333333", linewidth=0.5)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color="#333333", linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


def render_grid_pair(
    input_grid: List[List[int]], output_grid: List[List[int]]
) -> bytes:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.set_title("Input", fontsize=9)
    render_grid(input_grid, ax1)
    ax2.set_title("Output", fontsize=9)
    render_grid(output_grid, ax2)
    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def render_test_input(input_grid: List[List[int]]) -> bytes:
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title("Test Input", fontsize=9)
    render_grid(input_grid, ax)
    fig.tight_layout(pad=0.5)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Output naming
# ---------------------------------------------------------------------------


def derive_output_name(model: str, dimensions: int, task_type: str) -> str:
    """Build the canonical output-file stem from model + dim + task_type.

    For example, ``gemini-embedding-2-preview`` + 3072 + ``RETRIEVAL_DOCUMENT`` becomes
    ``gemini_embedding_2_3072d_retrieval_document``.  Trailing release-stage
    tags (``-preview``/``-experimental``/``-exp``/``-latest``) are stripped
    so reruns against the GA model land on the same filename.
    """
    slug = model.lower()
    slug = re.sub(r"-(preview|experimental|exp|latest)$", "", slug)
    slug = slug.replace("-", "_")
    return f"{slug}_{dimensions}d_{task_type.lower()}"


# ---------------------------------------------------------------------------
# Sync rate limiter
# ---------------------------------------------------------------------------


class SyncRateLimiter:
    """Timestamp-based RPM limiter for the synchronous embed_content API."""

    def __init__(self, rpm: int) -> None:
        self.rpm = rpm
        self.timestamps: deque[float] = deque()

    def wait_if_needed(self) -> None:
        now = time.monotonic()
        while self.timestamps and now - self.timestamps[0] > 60:
            self.timestamps.popleft()

        if len(self.timestamps) >= self.rpm:
            sleep_time = 60 - (now - self.timestamps[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
            now = time.monotonic()
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()

        self.timestamps.append(time.monotonic())


# ---------------------------------------------------------------------------
# Embedding generation
# ---------------------------------------------------------------------------


@tenacity.retry(**_RETRY_POLICY)
def embed_task(
    client: Any,
    model: str,
    task_id: str,
    grid_data: Dict[str, Any],
    descriptions: str,
    dimensions: int,
    task_type: str,
    max_train_pairs: int,
    include_test_input: bool,
) -> List[float]:
    parts: List[types.Part] = []

    if descriptions:
        parts.append(types.Part.from_text(text=descriptions))

    for example in grid_data.get("train", [])[:max_train_pairs]:
        png_bytes = render_grid_pair(example["input"], example["output"])
        parts.append(types.Part.from_bytes(data=png_bytes, mime_type="image/png"))

    if include_test_input:
        test_examples = grid_data.get("test", [])
        if test_examples:
            png_bytes = render_test_input(test_examples[0]["input"])
            parts.append(
                types.Part.from_bytes(data=png_bytes, mime_type="image/png")
            )

    if not parts:
        raise ValueError(
            f"Task {task_id!r}: no content to embed (no description, no grids)."
        )

    content = types.Content(parts=parts)
    config = types.EmbedContentConfig(
        output_dimensionality=dimensions,
        task_type=task_type,
    )
    response = client.models.embed_content(
        model=model,
        contents=content,
        config=config,
    )
    return response.embeddings[0].values


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_checkpoint(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[rec["task_id"]] = rec
    return records


def save_final_npz(
    checkpoint_path: Path,
    output_dir: Path,
    output_name: str,
) -> Optional[Path]:
    records = load_checkpoint(checkpoint_path)
    if not records:
        print("No embeddings found in checkpoint.")
        return None

    sorted_ids = sorted(records.keys())
    task_ids: List[str] = []
    datasets: List[str] = []
    embeddings: List[List[float]] = []
    for tid in sorted_ids:
        rec = records[tid]
        task_ids.append(tid)
        datasets.append(rec.get("dataset", ""))
        embeddings.append(rec["embedding"])

    emb_array = np.array(embeddings, dtype=np.float32)
    npz_path = output_dir / f"{output_name}.npz"
    np.savez_compressed(
        npz_path,
        task_ids=np.array(task_ids),
        embeddings=emb_array,
        datasets=np.array(datasets),
    )
    print(f"Saved npz: {npz_path} — shape {emb_array.shape}")
    return npz_path


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------


def discover_tasks(task_dir: Path) -> List[Tuple[str, str]]:
    """Return ``[(task_id, dataset_label), ...]`` in sorted-by-id order.

    ``dataset_label`` is purely descriptive — it's the parent directory's
    basename, carried through into the npz so downstream tools can tell
    splits apart without re-globbing.
    """
    if not task_dir.exists():
        raise FileNotFoundError(f"--task-dir not found: {task_dir}")
    label = task_dir.name
    return [(p.stem, label) for p in sorted(task_dir.glob("*.json"))]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description=(
            "Generate multimodal task embeddings via Gemini. Run once per "
            "(task-dir, task-type) pair."
        )
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help="Directory of ARC-format task JSONs to embed.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        required=True,
        choices=sorted(VALID_TASK_TYPES),
        help=(
            "Gemini embedding task type. Use RETRIEVAL_DOCUMENT for the "
            "demo pool side, RETRIEVAL_QUERY for the target side."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Gemini embedding model name (e.g. 'gemini-embedding-2-preview').",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        required=True,
        help="Output embedding dimensionality (e.g. 768, 1536, 3072).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["count", "embed"],
        default="count",
        help="'count' prints coverage stats only; 'embed' calls the API.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/task_embeddings"),
        help="Directory for output .npz and checkpoint files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help=(
            "Override the derived output filename stem. By default the stem "
            "is f'{model_slug}_{dims}d_{task_type_lower}'."
        ),
    )
    parser.add_argument(
        "--description-file",
        type=Path,
        default=None,
        help=(
            "Optional JSONL with {task_id, text} records. Multiple records "
            "per task are concatenated. Tasks without descriptions are "
            "embedded image-only."
        ),
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=5,
        help="Minimum word count per description (shorter ones are dropped).",
    )
    parser.add_argument(
        "--max-train-pairs",
        type=int,
        default=5,
        help="Maximum number of train pairs to render per task.",
    )
    parser.add_argument(
        "--no-test-input",
        dest="include_test_input",
        action="store_false",
        help="Skip rendering the test-input grid (default: include it).",
    )
    parser.set_defaults(include_test_input=True)
    parser.add_argument(
        "--rpm",
        type=int,
        default=100,
        help="Requests per minute (Gemini embedding RPM limit).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Embed only the first N tasks (post-task-filter).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Specific task IDs to embed (whitespace-separated).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing checkpoint instead of re-embedding.",
    )

    args = parser.parse_args()

    tasks = discover_tasks(args.task_dir)
    print(f"Discovered {len(tasks)} tasks in {args.task_dir}.")

    if args.tasks:
        wanted = set(args.tasks)
        tasks = [(tid, ds) for tid, ds in tasks if tid in wanted]
        print(f"Filtered to {len(tasks)} specified tasks.")
    if args.limit is not None:
        tasks = tasks[: args.limit]
        print(f"Limited to first {len(tasks)} tasks.")

    if not tasks:
        print("No tasks selected. Exiting.")
        return

    desc_map: Dict[str, str] = {}
    if args.description_file is not None:
        desc_map = load_descriptions(args.description_file, args.min_words)
        print(
            f"Loaded descriptions for {len(desc_map)} tasks "
            f"(>= {args.min_words} words) from {args.description_file}."
        )

    task_ids_set = {tid for tid, _ in tasks}
    with_desc = task_ids_set & set(desc_map)
    print(f"Coverage: {len(with_desc)}/{len(task_ids_set)} tasks have descriptions.")

    missing_grids = sum(
        1 for tid, _ in tasks if load_grid(tid, [args.task_dir]) is None
    )
    if missing_grids:
        print(f"  WARNING: {missing_grids} tasks have unloadable grid data.")

    if args.mode == "count":
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or derive_output_name(
        args.model, args.dimensions, args.task_type
    )
    checkpoint_path = (
        args.output_dir / f"embeddings_checkpoint_{args.task_type.lower()}.jsonl"
    )

    existing: Dict[str, dict] = {}
    if args.resume and checkpoint_path.exists():
        existing = load_checkpoint(checkpoint_path)
        print(f"Resuming: {len(existing)} tasks already embedded.")

    to_embed = [(tid, ds) for tid, ds in tasks if tid not in existing]
    print(f"Tasks to embed: {len(to_embed)}")

    if not to_embed:
        print("All tasks already embedded. Writing final npz...")
        save_final_npz(checkpoint_path, args.output_dir, output_name)
        return

    client = setup_gemini_client()
    rate_limiter = SyncRateLimiter(rpm=args.rpm)

    failed: List[Tuple[str, str]] = []
    with open(checkpoint_path, "a") as ckpt_f:
        for task_id, dataset in tqdm(to_embed, desc="Embedding tasks"):
            grid_data = load_grid(task_id, [args.task_dir])
            if grid_data is None:
                failed.append((task_id, "no grid data"))
                continue

            descriptions = desc_map.get(task_id, "")
            rate_limiter.wait_if_needed()

            try:
                embedding = embed_task(
                    client=client,
                    model=args.model,
                    task_id=task_id,
                    grid_data=grid_data,
                    descriptions=descriptions,
                    dimensions=args.dimensions,
                    task_type=args.task_type,
                    max_train_pairs=args.max_train_pairs,
                    include_test_input=args.include_test_input,
                )
                record = {
                    "task_id": task_id,
                    "dataset": dataset,
                    "embedding": embedding,
                }
                ckpt_f.write(json.dumps(record) + "\n")
                ckpt_f.flush()
            except Exception as e:
                print(f"  FAIL {task_id}: {e}")
                failed.append((task_id, str(e)))

    if failed:
        print(f"\n{len(failed)} tasks failed:")
        for tid, reason in failed:
            print(f"  {tid}: {reason}")

    save_final_npz(checkpoint_path, args.output_dir, output_name)


if __name__ == "__main__":
    main()
