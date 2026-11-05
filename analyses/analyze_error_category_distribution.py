"""Fit a hierarchical Bayesian categorical model to the error classifications
and plot the distribution of error categories.

Reads the classifier output from ``scripts/classify_errors.py`` (several
labels per error program). If a candidates file is given, it also weights each
error by how many participants made it. Writes the population, marginal and
attempt-weighted distribution plots, a variance-decomposition plot, and one
CSV summary (posterior median + 94% HDI) for each distribution.

The default model has three levels (error -> program -> sample); use
``--two-level`` to drop the program level.

Usage:
    uv run python analyses/analyze_error_category_distribution.py \
        --results-dir OSF/04_classifications/human/training.jsonl \
        --candidates-file OSF/03_candidates/human/training.jsonl \
        --output-dir results/analyses/error_categories/humans_training
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    import arviz as az
    import pymc as pm
except ImportError:
    az = pm = None  # type: ignore[assignment]

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.bayes_defaults import add_sampling_args
from src.categorical_model import (
    build_index_maps,
    compute_attempt_weighted_sample_marginal,
    compute_population_average_marginal,
    compute_population_probs,
    fit_categorical_model,
    hdi_bounds,
    plot_variance_decomposition,
    print_diagnostics,
    print_latent_vpcs,
)
from src.category_definitions import CATEGORY_NAMES

# ── Display labels ──────────────────────────────────────────────────────────

SHORT_NAMES = {
    "attribute_substitution": "Attr. Missel.",
    "scope_restriction": "Scope Restr.",
    "operator_confusion": "Op. Confusion",
    "transformation_substitution": "Transf. Missel.",
    "compositional_sequencing": "Comp. Seq.",
    "partial_program": "Partial Prog.",
    "boundary_precision": "Boundary Prec.",
    "unclassifiable": "Unclassifiable",
}

VALID_CATEGORIES = set(CATEGORY_NAMES) | {"unclassifiable"}


# ── Data loading ────────────────────────────────────────────────────────────


def load_data(results_dir: Path) -> pl.DataFrame:
    """Load ``classify_errors.py`` JSONL output into a flat polars DataFrame.

    The producer emits one record per (error program, sample) with flat
    top-level keys ``error_id``, ``grid_hash``, ``program_id``, ``task_id``,
    ``sample_index`` and a nested ``characterization_result`` carrying
    ``error_category``.  The returned frame keeps ``error_id`` so that
    :func:`build_index_maps` keys errors on the real id (it prefers a populated
    ``error_id`` column, else falls back to ``task_id:grid_hash``).

    ``results_dir`` may be a directory (every ``*.jsonl`` in it is read) or a
    single JSONL file.
    """
    if results_dir.is_file():
        jsonl_paths = [results_dir]
    else:
        jsonl_paths = sorted(results_dir.glob("*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"No JSONL files found in {results_dir}")

    rows: list[dict] = []
    skipped = 0
    for path in jsonl_paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                char_result = r.get("characterization_result")
                if not char_result or "error_category" not in char_result:
                    skipped += 1
                    continue
                cat = char_result["error_category"]
                if cat not in VALID_CATEGORIES:
                    print(f"  Warning: unknown category '{cat}', skipping")
                    skipped += 1
                    continue
                rows.append(
                    {
                        "error_id": r["error_id"],
                        "task_id": r["task_id"],
                        "grid_hash": r["grid_hash"],
                        "program_id": r["program_id"],
                        "sample_index": r.get("sample_index", 0),
                        "error_category": cat,
                    }
                )

    if skipped:
        print(f"  Skipped {skipped} records (missing or invalid characterization)")

    return pl.DataFrame(rows)


def build_attempt_counts(
    candidates_df: pl.DataFrame,
    error_ids: list[str],
) -> np.ndarray:
    """Build an ``attempt_count`` array aligned with *error_ids*.

    Reads the ``attempt_count`` already present on stage-4 filtered-candidates
    records (max over each error's rows), keyed on ``error_id``.  Missing ids ->
    1.  Mirrors ``analyze_rule_type_distribution.build_attempt_counts``.
    """
    if "attempt_count" not in candidates_df.columns:
        return np.ones(len(error_ids), dtype=float)

    per_error = candidates_df.group_by("error_id").agg(
        pl.max("attempt_count").alias("attempt_count")
    )
    lookup = dict(
        zip(
            per_error["error_id"].to_list(),
            per_error["attempt_count"].to_list(),
        )
    )
    return np.array([lookup.get(eid, 1) for eid in error_ids], dtype=float)


# ── Posterior reporting ────────────────────────────────────────────────────


def _summarize_posterior(
    samples: np.ndarray,
    label: str,
    hdi_prob: float = 0.94,
) -> tuple[float, float, float]:
    """Print and return (median, hdi_low, hdi_high) for a 1D posterior."""
    med = float(np.median(samples))
    lo, hi = az.hdi(samples, hdi_prob=hdi_prob)
    print(f"  {label}: {med:.1%}  [{int(hdi_prob * 100)}% HDI: {lo:.1%}–{hi:.1%}]")
    return med, float(lo), float(hi)


def _report_attempt_weighted(
    aw_probs: np.ndarray,
    categories: list[str],
    label: str,
    complement_of: str | None = "unclassifiable",
    combined_label: str = "Any classifiable category",
    print_order: list[str] | None = None,
) -> None:
    """Print per-category attempt-weighted posterior summaries with HDIs.

    Uses :data:`SHORT_NAMES` for friendlier console labels.  When
    *complement_of* names a category present in *categories*, also prints
    ``1 - P(complement_of)`` with the given *combined_label*, computed from
    joint posterior draws so the HDI is correct for the derived quantity.
    """
    print(f"\n{label} — attempt-weighted posterior (94% HDI):")
    if print_order is None:
        ordered = list(categories)
    else:
        ordered = [c for c in print_order if c in categories]
        ordered += [c for c in categories if c not in ordered]

    for cat in ordered:
        k = categories.index(cat)
        _summarize_posterior(aw_probs[:, k], SHORT_NAMES.get(cat, cat))

    if complement_of is not None and complement_of in categories:
        k = categories.index(complement_of)
        _summarize_posterior(1.0 - aw_probs[:, k], combined_label)


# ── Plots ───────────────────────────────────────────────────────────────────


def _ordered_categories(categories: list[str], medians: np.ndarray | None = None) -> list[str]:
    """Return a stable display order with unclassifiable last.

    Categories listed in :data:`CATEGORY_NAMES` keep that canonical order. Any
    unexpected-but-modelled categories are appended, and ``unclassifiable`` is
    pinned last for readability.
    """
    ordered = [c for c in CATEGORY_NAMES if c in categories]
    extras = [
        c
        for c in categories
        if c not in ordered and c != "unclassifiable"
    ]
    if medians is not None:
        extras = sorted(extras, key=lambda c: -medians[categories.index(c)])
    ordered += extras
    if "unclassifiable" in categories:
        ordered.append("unclassifiable")
    return ordered


def _save_distribution_summary(
    probs: np.ndarray,
    categories: list[str],
    output_path: Path,
    mode: str,
    n_obs: int,
    n_errors: int,
    n_programs: int,
    total_attempts: int | None = None,
) -> None:
    """Save posterior medians and 94% HDIs in tidy CSV form."""
    medians = np.median(probs, axis=0)
    lo, hi = hdi_bounds(probs)
    ordered = _ordered_categories(categories, medians)

    rows = []
    for cat in ordered:
        k = categories.index(cat)
        rows.append(
            {
                "analysis": "error_category",
                "mode": mode,
                "category": cat,
                "display_category": SHORT_NAMES.get(cat, cat),
                "median": float(medians[k]),
                "hdi_low": float(lo[k]),
                "hdi_high": float(hi[k]),
                "hdi_prob": 0.94,
                "n_obs": n_obs,
                "n_errors": n_errors,
                "n_programs": n_programs,
                "total_attempts": total_attempts,
            }
        )

    pl.DataFrame(rows).write_csv(output_path)
    print(f"Saved {output_path}")


# ── Plots ───────────────────────────────────────────────────────────────────


def _plot_count_distribution(
    probs: np.ndarray,
    categories: list[str],
    scale: int,
    xlabel: str,
    title: str,
    output_path: Path,
    transparent: bool = False,
) -> plt.Figure:
    """Horizontal bar chart of a category distribution with 94% HDI whiskers.

    *probs* has shape ``(n_draws, K)``; counts are ``prob * scale``.  Categories
    are sorted classifiable-descending with ``unclassifiable`` pinned last.
    """
    medians = np.median(probs, axis=0)
    lo, hi = hdi_bounds(probs)

    classifiable = [i for i, c in enumerate(categories) if c != "unclassifiable"]
    unclass = [i for i, c in enumerate(categories) if c == "unclassifiable"]
    order = sorted(classifiable, key=lambda i: -medians[i]) + unclass

    cats_ord = [categories[i] for i in order]
    med_ord = medians[order]
    lo_ord = lo[order]
    hi_ord = hi[order]
    labels = [SHORT_NAMES.get(c, c) for c in cats_ord]

    count_med = med_ord * scale
    count_lo = lo_ord * scale
    count_hi = hi_ord * scale
    colors = ["#4C72B0" if c != "unclassifiable" else "#AAAAAA" for c in cats_ord]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    y_pos = np.arange(len(labels))[::-1]
    xerr = [count_med - count_lo, count_hi - count_med]

    ax.barh(
        y_pos,
        count_med,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        xerr=xerr,
        error_kw=dict(lw=1.2, capsize=3, color="0.3"),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for yi, med, l, h in zip(y_pos, count_med, count_lo, count_hi):
        pct = med / scale * 100
        ax.text(
            h + scale * 0.01,
            yi,
            f"{med:.0f}  ({pct:.1f}%)  [{l:.0f}–{h:.0f}]",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, max(count_hi) * 1.35)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=transparent)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", transparent=transparent)
    print(f"Saved figure to {output_path} (+ .pdf)")
    return fig


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="classify_errors.py per-sample JSONL output: one file, or a "
        "directory whose *.jsonl files are all read.",
    )
    parser.add_argument(
        "--candidates-file",
        type=Path,
        default=None,
        help=(
            "Stage-4 filter_candidates.py JSONL; when given, adds the "
            "attempt-weighted distribution plot + posterior report "
            "(weights read from the per-error attempt_count)."
        ),
    )
    parser.add_argument(
        "--two-level",
        action="store_true",
        help=(
            "Fit a 2-level model (error -> obs) omitting program-level random "
            "effects, rather than the default 3-level model (error -> program -> sample)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root / "results" / "analyses" / "error_categories" / "humans_training",
        help="Directory for saved figures.",
    )
    add_sampling_args(parser)
    parser.add_argument(
        "--save-idata",
        type=str,
        default=None,
        help="Path to save InferenceData as NetCDF.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save figures with transparent background.",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots interactively."
    )
    args = parser.parse_args()

    if pm is None:
        print("Error: pymc/arviz/pytensor required. Install with: uv add pymc arviz")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Loading characterization results...")
    df = load_data(args.results_dir)
    print(f"  {len(df)} observations loaded")

    # 2. Index maps (keys on error_id when present)
    three_level = not args.two_level
    idx = build_index_maps(df, three_level=three_level)
    prog_str = f", {idx['n_programs']} programs" if three_level else ""
    print(
        f"  {idx['n_errors']} errors{prog_str}, "
        f"{idx['K']} categories"
    )
    print(f"  Reference category: {idx['categories'][0]}")

    print("\nCategory frequencies:")
    for i, cat in enumerate(idx["categories"]):
        count = int((idx["y"] == i).sum())
        print(f"  {cat}: {count} ({count / idx['n_obs'] * 100:.1f}%)")

    # 3. Fit
    level_str = "3-level" if three_level else "2-level"
    print(
        f"\nFitting {level_str} hierarchical model "
        f"(draws={args.draws}, tune={args.tune}, chains={args.chains})..."
    )
    idata = fit_categorical_model(
        idx,
        three_level=three_level,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
        keep_u_raw=True,
    )

    # 4. Diagnostics
    print_diagnostics(idata, idx, three_level=three_level)
    print_latent_vpcs(
        idata,
        idx["categories"],
        three_level=three_level,
        short_names=SHORT_NAMES,
    )

    # 5. Save idata
    if args.save_idata:
        idata_path = Path(args.save_idata)
        idata_path.parent.mkdir(parents=True, exist_ok=True)
        idata.to_netcdf(str(idata_path))
        print(f"Saved InferenceData to {idata_path}")

    figures: list[plt.Figure] = []

    # 6. Population distribution (conditional at u=v=0, kept for reference)
    probs = compute_population_probs(idata, idx["K"])
    fig_pop = _plot_count_distribution(
        probs,
        idx["categories"],
        idx["n_obs"],
        xlabel="Expected number of errors",
        title="Population-Level Error Category Distribution\n(Hierarchical Model, 94% HDI)",
        output_path=args.output_dir / "error_category_distribution_hierarchical.png",
        transparent=args.transparent,
    )
    figures.append(fig_pop)
    _save_distribution_summary(
        probs,
        idx["categories"],
        args.output_dir / "error_category_population_summary.csv",
        mode="population",
        n_obs=idx["n_obs"],
        n_errors=idx["n_errors"],
        n_programs=idx["n_programs"],
    )

    # 7. Marginal distribution (integrates over random effects)
    print("\nComputing marginal distribution (integrating over random effects)...")
    marginal_probs = compute_population_average_marginal(
        idata, idx["K"], three_level=three_level, random_seed=args.seed
    )
    fig_marginal = _plot_count_distribution(
        marginal_probs,
        idx["categories"],
        idx["n_errors"],
        xlabel="Expected number of errors",
        title=(
            "Marginal Error Category Distribution\n"
            "(Hierarchical Model, Random Effects Integrated, 94% HDI)"
        ),
        output_path=args.output_dir / "error_category_distribution_marginal.png",
        transparent=args.transparent,
    )
    figures.append(fig_marginal)
    _save_distribution_summary(
        marginal_probs,
        idx["categories"],
        args.output_dir / "error_category_marginal_summary.csv",
        mode="marginal",
        n_obs=idx["n_obs"],
        n_errors=idx["n_errors"],
        n_programs=idx["n_programs"],
    )

    # 8. Attempt-weighted distribution (requires --candidates-file)
    if args.candidates_file:
        # Read only the two columns we need via projection pushdown.  A full
        # ``pl.read_ndjson`` type-infers *every* column from a sample, which
        # can crash on heterogeneous files (a mixed grid/text column inferred
        # as i64 that later hits a string).  ``attempt_count`` may be absent
        # -> build_attempt_counts falls back to all-ones.
        lf = pl.scan_ndjson(args.candidates_file)
        wanted = [c for c in ("error_id", "attempt_count") if c in lf.collect_schema().names()]
        cand_df = lf.select(wanted).collect()
        attempt_counts = build_attempt_counts(cand_df, idx["error_ids"])
        n_matched = int((attempt_counts > 1).sum())
        total_attempts = int(attempt_counts.sum())
        print(
            f"\nMatched attempt counts for {idx['n_errors']} errors "
            f"({n_matched} with >1 attempt); {total_attempts} total attempts."
        )
        print("Computing attempt-weighted distribution...")
        aw_probs = compute_attempt_weighted_sample_marginal(idata, idx, attempt_counts)
        fig_aw = _plot_count_distribution(
            aw_probs,
            idx["categories"],
            total_attempts,
            xlabel="Expected number of attempts",
            title="Attempt-Weighted Error Category Distribution\n(Hierarchical Model, 94% HDI)",
            output_path=args.output_dir
            / "error_category_distribution_attempt_weighted.png",
            transparent=args.transparent,
        )
        figures.append(fig_aw)
        _save_distribution_summary(
            aw_probs,
            idx["categories"],
            args.output_dir / "error_category_attempt_weighted_summary.csv",
            mode="attempt_weighted",
            n_obs=idx["n_obs"],
            n_errors=idx["n_errors"],
            n_programs=idx["n_programs"],
            total_attempts=total_attempts,
        )
        _report_attempt_weighted(
            aw_probs,
            idx["categories"],
            "Error characterization",
            complement_of="unclassifiable",
            combined_label="Any classifiable category",
        )
    else:
        print("\nNo --candidates-file given; skipping attempt-weighted plot.")

    # 9. Variance decomposition (3-level only)
    if three_level:
        var_path = args.output_dir / "error_category_variance_decomposition.png"
        fig_var = plot_variance_decomposition(
            idata,
            idx["categories"],
            var_path,
            short_names=SHORT_NAMES,
            transparent=args.transparent,
        )
        fig_var.savefig(var_path.with_suffix(".pdf"), bbox_inches="tight",
                        transparent=args.transparent)
        figures.append(fig_var)

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == "__main__":
    main()
