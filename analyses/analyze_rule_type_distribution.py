"""Fit a hierarchical Bayesian categorical model to the rule type of each error
program, and to the order effects within partial matches.

Each program's rule type (neither / partial match / alternative rule) comes
from which training examples it reproduces (:func:`src.rule_types.rule_type`).
The order-effect analysis models, among partial matches, which examples are
reproduced (first-k / last-k / contiguous / other). Both use a two-level model
(error -> program).

Reads a candidates file. Writes population, marginal and attempt-weighted
plots, each with a CSV summary (posterior median + 94% HDI); order-effect
outputs go in ``<output-dir>/order_effects/``.

Usage:
    uv run python analyses/analyze_rule_type_distribution.py --analysis both \\
        --dsl-input OSF/03_candidates/human/training.jsonl \\
        --output-dir results/analyses/rule_types/humans_training
"""

from __future__ import annotations

import argparse
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
    print_diagnostics,
    print_latent_vpcs,
)
from src.rule_types import (
    ORDER_TIER_ORDER,
    RULE_TYPE_DISPLAY,
    RULE_TYPE_ORDER,
    rule_type,
)

# ── Style ─────────────────────────────────────────────────────────────────

FONTSIZE = 11
COLOR = "#4D4D4D"
# Written to the ``source`` column of every summary CSV and used in plot titles.
LABEL = "DSL"


def _apply_pub_style(ax: plt.Axes, fontsize: int = FONTSIZE) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize - 1)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)


# ── Defaults ──────────────────────────────────────────────────────────────

DSL_DEFAULT = _project_root / "OSF" / "03_candidates" / "human" / "training.jsonl"


# ── Data loading ──────────────────────────────────────────────────────────


def load_classified(path: Path) -> pl.DataFrame:
    """Load edits JSONL and classify each program's rule type.

    Adds two string columns derived from ``training_matches`` via
    :func:`src.rule_types.rule_type`:

    * ``program_rule_type`` — display label (Neither / Partial match /
      Alt. rule).  Null for unclassifiable rows (empty ``training_matches``),
      which are dropped.
    * ``program_order_tier`` — first-k / last-k / contiguous / other, set
      only on partial-match rows (null otherwise).

    Rows are restricted to ``candidate_type == "model"`` when that column is
    present and populated; stage-4 files where it is absent/null are already
    model-only and pass through unchanged.
    """
    # Read only the columns this analysis consumes via projection pushdown.
    # A full ``pl.read_ndjson`` type-infers *every* column from a sample, which
    # can crash on heterogeneous files (a mixed grid/text column inferred as
    # i64 that later hits a string).  All selected columns are
    # scalar/list-of-bool, so none can trip that inference.  Every column
    # except ``training_matches`` is optional and selected only when present.
    lf = pl.scan_ndjson(path)
    available = lf.collect_schema().names()
    wanted = [
        c
        for c in (
            "error_id",
            "program_id",
            "sample_index",
            "training_matches",
            "candidate_type",
            "attempt_count",
        )
        if c in available
    ]
    df = lf.select(wanted).collect()

    if "candidate_type" in df.columns and df["candidate_type"].null_count() < df.height:
        df = df.filter(pl.col("candidate_type") == "model")

    labels = [rule_type(tm) for tm in df["training_matches"].to_list()]
    rule_labels = [RULE_TYPE_DISPLAY.get(lbl) for lbl, _ in labels]
    order_tags = [tag for _, tag in labels]

    df = df.with_columns(
        pl.Series("program_rule_type", rule_labels, dtype=pl.Utf8),
        pl.Series("program_order_tier", order_tags, dtype=pl.Utf8),
    )
    # Drop unclassifiable rows (no training_matches → no rule type).
    df = df.filter(pl.col("program_rule_type").is_not_null())
    return df


def restrict_to_partial(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only partial-match rows (those carrying an order tier).

    The order-effect question is undefined for alternative-rule and
    neither-match programs, so they are dropped.
    """
    n_before = df.select(pl.col("error_id").n_unique()).item()
    df = df.filter(pl.col("program_order_tier").is_not_null())
    n_after = df.select(pl.col("error_id").n_unique()).item()
    print(
        f"  Order-tier filter: kept {n_after}/{n_before} errors "
        f"({len(df)} partial-match program rows)"
    )
    return df


def build_attempt_counts(
    df: pl.DataFrame,
    error_ids: list[str],
) -> np.ndarray:
    """Build an attempt_count array aligned with *error_ids*.

    Reads the ``attempt_count`` already present on stage-4 records (max over
    each error's rows).  Returns ones if the column is absent.
    """
    if "attempt_count" not in df.columns:
        return np.ones(len(error_ids), dtype=float)

    per_error = df.group_by("error_id").agg(
        pl.max("attempt_count").alias("attempt_count")
    )
    lookup = dict(
        zip(
            per_error["error_id"].to_list(),
            per_error["attempt_count"].to_list(),
        )
    )
    return np.array([lookup.get(eid, 1) for eid in error_ids], dtype=float)


# ── Plotting ──────────────────────────────────────────────────────────────


def plot_distribution(
    probs: np.ndarray,
    categories: list[str],
    n_errors: int,
    title: str = "Rule Type Distribution",
    color: str = "#555555",
    transparent: bool = False,
    category_order: list[str] = RULE_TYPE_ORDER,
    count_label: str = "errors",
) -> plt.Figure:
    """Vertical bar chart of category proportions with 94% HDI.

    *category_order* dictates the left-to-right plot order; categories
    not in the list are dropped, and the model's alphabetical category
    order is rearranged accordingly.
    """
    medians = np.median(probs, axis=0)
    lo, hi = hdi_bounds(probs)

    order = [categories.index(c) for c in category_order if c in categories]
    cats_ord = [categories[i] for i in order]
    med_ord = medians[order]
    lo_ord = lo[order]
    hi_ord = hi[order]

    x = np.arange(len(cats_ord))
    yerr = np.array([med_ord - lo_ord, hi_ord - med_ord])

    fig, ax = plt.subplots(figsize=(7, 4))
    if not transparent:
        fig.patch.set_facecolor("white")

    ax.bar(
        x,
        med_ord,
        width=0.55,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        yerr=yerr,
        capsize=4,
        error_kw={"linewidth": 1.2, "color": "#333333"},
    )

    for xi, med, l, h in zip(x, med_ord, lo_ord, hi_ord):
        ax.text(
            xi,
            h + 0.015,
            f"{med:.1%}\n[{l:.1%}–{h:.1%}]",
            ha="center",
            va="bottom",
            fontsize=FONTSIZE - 1,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(cats_ord, fontsize=FONTSIZE)
    ax.set_ylim(0, max(hi_ord) * 1.35)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title(
        f"{title}  (n={n_errors} {count_label})",
        fontsize=FONTSIZE + 1,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    _apply_pub_style(ax)
    ax.spines["left"].set_visible(False)

    plt.tight_layout()
    return fig


# ── Model fitting pipeline ────────────────────────────────────────────────


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
    complement_of: str | None = "Neither",
    combined_label: str = "Captures ≥1 training example",
    print_order: list[str] | None = None,
) -> None:
    """Print per-category attempt-weighted posterior summaries with HDIs.

    *categories* is the model-internal order (alphabetical, used to index
    *aw_probs*). *print_order*, if given, controls the order in which
    rows are printed; categories not in *print_order* are appended at
    the end so nothing is silently dropped.

    When *complement_of* names a category present in *categories*, also
    prints ``1 − P(complement_of)`` as a derived posterior with the
    given *combined_label* (computed from joint posterior draws so the
    HDI is correct for the derived quantity).
    """
    print(f"\n{label} — attempt-weighted posterior (94% HDI):")
    if print_order is None:
        ordered_cats = list(categories)
    else:
        ordered_cats = [c for c in print_order if c in categories]
        ordered_cats += [c for c in categories if c not in ordered_cats]

    for cat in ordered_cats:
        k = categories.index(cat)
        _summarize_posterior(aw_probs[:, k], cat)

    if complement_of is not None and complement_of in categories:
        idx = categories.index(complement_of)
        combined = 1.0 - aw_probs[:, idx]
        _summarize_posterior(combined, combined_label)


def _save_distribution_summary(
    probs: np.ndarray,
    categories: list[str],
    output_path: Path,
    label: str,
    mode: str,
    analysis_name: str,
    n_errors: int,
    total_attempts: int,
    category_order: list[str],
) -> None:
    """Save posterior medians and 94% HDIs in tidy CSV form."""
    medians = np.median(probs, axis=0)
    lo, hi = hdi_bounds(probs)

    ordered_cats = [c for c in category_order if c in categories]
    ordered_cats += [c for c in categories if c not in ordered_cats]

    rows = []
    for cat in ordered_cats:
        k = categories.index(cat)
        rows.append(
            {
                "analysis": analysis_name,
                "source": label,
                "mode": mode,
                "category": cat,
                "median": float(medians[k]),
                "hdi_low": float(lo[k]),
                "hdi_high": float(hi[k]),
                "hdi_prob": 0.94,
                "n_errors": n_errors,
                "total_attempts": total_attempts,
            }
        )

    pl.DataFrame(rows).write_csv(output_path)
    print(f"Saved {output_path}")


def _save_figure(fig: plt.Figure, path: Path, transparent: bool = False) -> None:
    """Save figure as both PNG and PDF, then close."""
    save_kwargs = dict(dpi=300, bbox_inches="tight", transparent=transparent)
    if not transparent:
        save_kwargs["facecolor"] = "white"
    fig.savefig(path, **save_kwargs)
    fig.savefig(path.with_suffix(".pdf"), **save_kwargs)
    print(f"Saved {path}")
    plt.close(fig)


def fit_and_report(
    df: pl.DataFrame,
    output_dir: Path,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    transparent: bool,
    category_col: str = "program_rule_type",
    category_order: list[str] = RULE_TYPE_ORDER,
    analysis_name: str = "rule_type",
    analysis_title: str = "Rule Type",
    complement_of: str | None = "Neither",
    combined_label: str = "Captures ≥1 training example",
) -> tuple[np.ndarray, np.ndarray, list[str], int, int]:
    """Fit the 2-level model, print diagnostics, save plots and summaries.

    Parameters
    ----------
    category_col, category_order
        The column name to model and the desired plot order. Defaults
        target the rule-type analysis; for order effects pass
        ``"program_order_tier"`` and :data:`ORDER_TIER_ORDER`.
    analysis_name
        Used as the filename prefix (``{analysis_name}_*.png``).  Rule-type
        outputs go directly in *output_dir*; any other analysis goes in
        ``<output_dir>/<analysis_name>/``.
    analysis_title
        Human-readable analysis name woven into figure titles.
    complement_of, combined_label
        Forwarded to :func:`_report_attempt_weighted`. Set
        ``complement_of=None`` to suppress the derived combined number.

    Returns
    -------
    (population_probs, attempt_weighted_probs, categories, n_errors,
    total_attempts)
    """
    label = LABEL
    idx = build_index_maps(df, category_col=category_col, three_level=False)
    print(
        f"\n  {label}: {idx['n_errors']} errors, {idx['n_obs']} obs, "
        f"{idx['K']} categories"
    )
    print(f"  Reference category: {idx['categories'][0]}")

    print("\nCategory frequencies:")
    for i, cat in enumerate(idx["categories"]):
        count = int((idx["y"] == i).sum())
        print(f"  {cat}: {count} ({count / idx['n_obs'] * 100:.1f}%)")

    print(
        f"\nFitting 2-level model "
        f"(draws={draws}, tune={tune}, chains={chains})..."
    )
    idata = fit_categorical_model(
        idx,
        three_level=False,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=seed,
        keep_u_raw=True,
    )

    print_diagnostics(idata, idx, three_level=False)
    print_latent_vpcs(idata, idx["categories"], three_level=False)

    # Rule-type outputs go directly in output_dir; other analyses get a
    # dedicated subdirectory.
    sub_dir = output_dir if analysis_name == "rule_type" else output_dir / analysis_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    plot_kwargs = dict(
        color=COLOR,
        transparent=transparent,
        category_order=category_order,
    )

    # Population distribution
    pop_probs = compute_population_probs(idata, idx["K"])
    pop_path = sub_dir / f"{analysis_name}_population.png"
    fig_pop = plot_distribution(
        pop_probs,
        idx["categories"],
        idx["n_errors"],
        title=f"{label} — Population ({analysis_title})",
        **plot_kwargs,
    )
    _save_figure(fig_pop, pop_path, transparent)

    # Marginal distribution (uniform over errors)
    print("\nComputing population-average marginal distribution ...")
    marg_probs = compute_population_average_marginal(
        idata, idx["K"], three_level=False, random_seed=seed
    )
    marg_path = sub_dir / f"{analysis_name}_marginal.png"
    fig_marg = plot_distribution(
        marg_probs,
        idx["categories"],
        idx["n_errors"],
        title=f"{label} — Marginal ({analysis_title})",
        **plot_kwargs,
    )
    _save_figure(fig_marg, marg_path, transparent)

    # Attempt-weighted marginal
    attempt_counts = build_attempt_counts(df, idx["error_ids"])
    total_attempts = int(attempt_counts.sum())
    has_weights = "attempt_count" in df.columns
    if has_weights:
        print(
            f"\nComputing attempt-weighted sample-average marginal "
            f"({total_attempts} total attempts) ..."
        )
        aw_probs = compute_attempt_weighted_sample_marginal(idata, idx, attempt_counts)
        aw_path = sub_dir / f"{analysis_name}_attempt_weighted.png"
        fig_aw = plot_distribution(
            aw_probs,
            idx["categories"],
            total_attempts,
            title=f"{label} — Attempt-Weighted ({analysis_title})",
            count_label="attempts",
            **plot_kwargs,
        )
        _save_figure(fig_aw, aw_path, transparent)
    else:
        print("\n  No attempt_count column — skipping attempt-weighted plot")
        aw_probs = marg_probs

    _save_distribution_summary(
        pop_probs,
        idx["categories"],
        sub_dir / f"{analysis_name}_population_summary.csv",
        label=label,
        mode="population",
        analysis_name=analysis_name,
        n_errors=idx["n_errors"],
        total_attempts=total_attempts,
        category_order=category_order,
    )
    _save_distribution_summary(
        marg_probs,
        idx["categories"],
        sub_dir / f"{analysis_name}_marginal_summary.csv",
        label=label,
        mode="marginal",
        analysis_name=analysis_name,
        n_errors=idx["n_errors"],
        total_attempts=total_attempts,
        category_order=category_order,
    )
    _save_distribution_summary(
        aw_probs,
        idx["categories"],
        sub_dir / f"{analysis_name}_attempt_weighted_summary.csv",
        label=label,
        mode="attempt_weighted",
        analysis_name=analysis_name,
        n_errors=idx["n_errors"],
        total_attempts=total_attempts,
        category_order=category_order,
    )

    _report_attempt_weighted(
        aw_probs,
        idx["categories"],
        f"{label} {analysis_title}",
        complement_of=complement_of,
        combined_label=combined_label,
        print_order=category_order,
    )

    return pop_probs, aw_probs, idx["categories"], idx["n_errors"], total_attempts


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--analysis",
        choices=["rule_type", "order_effects", "both"],
        default="both",
        help=(
            "Which categorical distribution(s) to fit. "
            "'rule_type' = Neither/Partial/Alt; "
            "'order_effects' = first-k/last-k/contiguous/other "
            "(conditional on partial match)."
        ),
    )
    parser.add_argument(
        "--dsl-input",
        type=Path,
        default=DSL_DEFAULT,
        help="Path to DSL stage-4 filtered candidates JSONL "
        "(default: OSF/03_candidates/human/training.jsonl).",
    )
    add_sampling_args(parser)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_project_root / "results" / "analyses" / "rule_types" / "humans_training",
    )
    parser.add_argument("--transparent", action="store_true")
    args = parser.parse_args()

    if pm is None:
        print("Error: pymc/arviz/pytensor required. Install with: uv add pymc arviz")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine which analyses to run ───────────────────────────────
    analyses_to_run: list[str] = (
        ["rule_type", "order_effects"] if args.analysis == "both" else [args.analysis]
    )

    specs: dict[str, dict] = {
        "rule_type": dict(
            category_col="program_rule_type",
            category_order=RULE_TYPE_ORDER,
            analysis_name="rule_type",
            analysis_title="Rule Type",
            complement_of="Neither",
            combined_label="Captures ≥1 training example",
        ),
        "order_effects": dict(
            category_col="program_order_tier",
            category_order=ORDER_TIER_ORDER,
            analysis_name="order_effects",
            analysis_title="Order Effects",
            complement_of="other",
            combined_label="Structured (1 − other)",
        ),
    }

    for analysis in analyses_to_run:
        spec = specs[analysis]
        print("\n" + "#" * 60)
        print(f"# Analysis: {analysis}")
        print("#" * 60)

        # ── Load + classify data ──────────────────────────────────
        print(f"Loading DSL data from {args.dsl_input.name} ...")
        df = load_classified(args.dsl_input)
        if analysis == "order_effects":
            df = restrict_to_partial(df)
        n_e = df.select(pl.col("error_id").n_unique()).item()
        print(f"  {len(df)} programs, {n_e} unique errors")

        if len(df) == 0:
            print(f"\n  DSL data empty after {analysis} filter — skipping fit.")
            continue

        # ── Fit model ─────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"DSL (2-level model) — {analysis}")
        print("=" * 60)
        fit_and_report(
            df,
            output_dir=args.output_dir,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            seed=args.seed,
            transparent=args.transparent,
            **spec,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
