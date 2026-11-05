"""Compare the length of error programs with control programs using a Bayesian
mixed-effects regression.

Treatment programs were edited to reproduce a participant's error; control
programs were edited while the target was still the correct output. The model
predicts program length from condition, with random intercepts for task and
error. A negative treatment effect means error programs are shorter than plain
edits would make them. With ``--relative`` the outcome is the relative length
change, (edited - reference) / reference; ``--ast`` counts AST nodes instead of
lines.

Reads one or more candidate files per condition, keeping only tasks present in
both. Writes the condition-distribution, means, posterior and Q-Q plots, and
prints the posterior contrast and variance decomposition.

Usage:
    uv run python analyses/analyze_program_simplicity.py \\
        --treatment-candidates OSF/03_candidates/human/training.jsonl \\
        --control-candidates OSF/03_candidates/control/human_training.jsonl \\
        --relative --output-dir results/analyses/program_simplicity/humans_training
"""

import argparse
import sys
from pathlib import Path

import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.bayes_defaults import add_sampling_args, fit_bambi
from src.program_complexity import safe_node_count as _safe_node_count
from src.utils import load_jsonl_records

# ── Outcome metadata ─────────────────────────────────────────────────────────

_OUTCOME_META: dict[str, dict[str, str]] = {
    "rewritten_line_count": {
        "label": "rewritten line count",
        "axis_label": "Effect on rewritten line count (lines)",
        "y_axis_label": "Rewritten line count (lines)",
        "title": "Treatment vs. control: program length",
        "prefix": "rwlc",
    },
    "log_rewritten_line_count": {
        "label": "log rewritten line count",
        "axis_label": "Effect on log rewritten line count",
        "y_axis_label": "Log rewritten line count",
        "title": "Treatment vs. control: program length",
        "prefix": "lrwlc",
    },
    "rewritten_node_count": {
        "label": "rewritten node count",
        "axis_label": "Effect on rewritten node count",
        "y_axis_label": "Rewritten node count (nodes)",
        "title": "Treatment vs. control: AST complexity",
        "prefix": "rwnc",
    },
    "log_rewritten_node_count": {
        "label": "log rewritten node count",
        "axis_label": "Effect on log rewritten node count",
        "y_axis_label": "Log rewritten node count",
        "title": "Treatment vs. control: AST complexity",
        "prefix": "lrwnc",
    },
    "relative_length_change": {
        "label": "relative length change",
        "axis_label": "Effect on relative length change",
        "y_axis_label": "Relative length change (rewritten / original − 1)",
        "title": "Treatment vs. control: relative program length",
        "prefix": "rlc",
    },
}

# ── Style ────────────────────────────────────────────────────────────────────

BLUE = "#4878CF"
RED = "#C44E52"
GRAY = "#888888"
CONDITION_COLORS = {"control": GRAY, "treatment": BLUE}


def _apply_pub_style(ax: plt.Axes, fontsize: int = 11) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize - 1)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)


# ── Data ─────────────────────────────────────────────────────────────────────


def _load_condition(
    candidate_files: list[Path],
    condition: str,
    compute_ast: bool = False,
    dedup_subtrees: bool = True,
) -> pl.DataFrame:
    """Load stage-4 candidate JSONL files and tag rows with *condition*.

    Only a cheap null-guard is applied (stage 4 already enforces validity):
    rows missing ``error_id`` or with null / non-positive line counts are
    dropped with a printed count.
    """
    records = load_jsonl_records(candidate_files)

    # Grids are list-of-lists and would blow up Polars schema inference, so
    # only scalar columns are materialized into the DataFrame.
    rows: list[dict] = []
    dropped_no_error_id = 0
    for r in records:
        error_id = r.get("error_id")
        if not error_id:
            dropped_no_error_id += 1
            continue
        row = {
            "task_id": r.get("task_id"),
            "error_id": error_id,
            "line_count_delta": r.get("line_count_delta"),
            "original_line_count": r.get("original_line_count"),
            "rewritten_line_count": r.get("rewritten_line_count"),
        }
        if compute_ast:
            orig_source = r.get("original_canonical") or r.get("original_program")
            row["original_node_count"] = _safe_node_count(
                orig_source, dedup_subtrees=dedup_subtrees
            )
            row["rewritten_node_count"] = _safe_node_count(
                r.get("rewritten_program"), dedup_subtrees=dedup_subtrees
            )
        rows.append(row)

    if dropped_no_error_id:
        print(f"  [{condition}] Dropped {dropped_no_error_id} records with no error_id.")

    raw = pl.from_dicts(rows, infer_schema_length=10_000)
    pre_guard_n = raw.height
    df = raw.filter(
        pl.col("line_count_delta").is_not_null()
        & pl.col("original_line_count").is_not_null()
        & (pl.col("original_line_count") > 0)
        & pl.col("rewritten_line_count").is_not_null()
        & (pl.col("rewritten_line_count") > 0)
    )
    if df.height < pre_guard_n:
        print(
            f"  [{condition}] Null-guard dropped {pre_guard_n - df.height} rows "
            f"with missing/non-positive line counts."
        )

    if compute_ast:
        pre_ast = df.height
        df = df.filter(
            pl.col("original_node_count").is_not_null()
            & pl.col("rewritten_node_count").is_not_null()
            & (pl.col("original_node_count") > 0)
            & (pl.col("rewritten_node_count") > 0)
        )
        print(f"  [{condition}] AST parsing kept {df.height:,}/{pre_ast:,} rows.")

    df = df.with_columns(
        pl.col("rewritten_line_count").log().alias("log_rewritten_line_count"),
        (pl.col("line_count_delta") / pl.col("original_line_count")).alias(
            "relative_length_change"
        ),
        pl.lit(condition).alias("condition"),
    )

    if compute_ast:
        df = df.with_columns(
            pl.col("rewritten_node_count").log().alias("log_rewritten_node_count"),
        )

    return df


def load_data(
    treatment_files: list[Path],
    control_files: list[Path],
    compute_ast: bool = False,
    dedup_subtrees: bool = True,
) -> pl.DataFrame:
    treat = _load_condition(treatment_files, "treatment", compute_ast, dedup_subtrees)
    ctrl = _load_condition(control_files, "control", compute_ast, dedup_subtrees)

    print(f"Treatment rows: {treat.height:,}")
    print(f"Control   rows: {ctrl.height:,}")

    treat_tasks = set(treat["task_id"].unique().to_list())
    ctrl_tasks = set(ctrl["task_id"].unique().to_list())
    shared = treat_tasks & ctrl_tasks
    print(
        f"Tasks: treatment={len(treat_tasks)}, control={len(ctrl_tasks)}, "
        f"shared={len(shared)}"
    )
    if not shared:
        raise SystemExit(
            "No overlapping task_ids between treatment and control candidate files."
        )

    combined = pl.concat([treat, ctrl], how="vertical_relaxed").filter(
        pl.col("task_id").is_in(list(shared))
    )
    print(
        "After restricting to shared tasks: "
        f"treatment={combined.filter(pl.col('condition') == 'treatment').height:,}, "
        f"control={combined.filter(pl.col('condition') == 'control').height:,}"
    )

    # Diagnostics: show the per-unique-error record multiplicity for each
    # condition.  The (1 | error_id) random effect is motivated by values >1
    # on the treatment side (multiple alternative programs explaining the
    # same error).  Control side is expected to be mostly 1 per task.
    for cond in ("treatment", "control"):
        sub = combined.filter(pl.col("condition") == cond)
        if sub.height == 0:
            continue
        n_err = sub.select(pl.col("error_id").n_unique()).item()
        mean_mult = sub.height / n_err if n_err else 0
        max_mult = (
            sub.group_by("error_id")
            .agg(pl.len().alias("n"))
            .select(pl.col("n").max())
            .item()
        )
        print(
            f"  [{cond}] unique errors={n_err:,}, "
            f"records/error mean={mean_mult:.2f}, max={max_mult}"
        )

    return combined


# ── Model ────────────────────────────────────────────────────────────────────


def _build_priors(outcome: str) -> dict[str, bmb.Prior]:
    """Weakly informative priors for the two outcomes we report:

    - 'log_rewritten_node_count': anchored on a typical program of ~15 AST
      nodes (mu=log 15, sigma=log 4.5) — a multiplicative ±4.5x range at
      ±1 SD.  Same width applied to the slope, group SDs, and the residual
      auxiliary parameter so all width-shaped quantities share one scale.
    - 'relative_length_change': centered at zero with sigma=1.5 on the
      common and group effects.  Residual sigma is left to Bambi's
      data-scaled default since the outcome's natural scale (a ratio in
      roughly [-1, +inf)) is well-handled there.

    All other outcomes fall back to Bambi's defaults.
    """
    if outcome == "log_rewritten_node_count":
        log_node_count_mu = 2.5
        sigma = 1.5
        return {
            "Intercept": bmb.Prior("Normal", mu=log_node_count_mu, sigma=sigma),
            "condition": bmb.Prior("Normal", mu=0, sigma=sigma),
            "1|task_id": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
            ),
            "1|error_id": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
            ),
        }

    if outcome == "relative_length_change":
        sigma = 1.5
        return {
            "Intercept": bmb.Prior("Normal", mu=0, sigma=sigma),
            "condition": bmb.Prior("Normal", mu=0, sigma=sigma),
            "1|task_id": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
            ),
            "1|error_id": bmb.Prior(
                "Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)
            ),
        }

    return {}


def fit_model(
    df: pl.DataFrame,
    outcome: str,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> tuple[az.InferenceData, bmb.Model]:
    model_df = df.select(outcome, "condition", "task_id", "error_id").to_pandas()
    model_df["task_id"] = model_df["task_id"].astype(str)
    model_df["error_id"] = model_df["error_id"].astype(str)
    # Set control as the reference level so the coefficient name reads
    # "condition[treatment]" and represents the treatment-minus-control
    # contrast.
    model_df["condition"] = model_df["condition"].astype("category")
    model_df["condition"] = model_df["condition"].cat.reorder_categories(
        ["control", "treatment"], ordered=False
    )

    # Crossed random intercepts:
    #   (1 | task_id)  -- per-task baseline program length
    #   (1 | error_id) -- per-unique-error baseline (handles the fact
    #                     that stage 4 keeps several tied candidate
    #                     programs per error, which are NOT independent
    #                     observations of the length effect).
    #
    # On the control side error_id is constant per task and the nested
    # variance component will be estimated near zero — harmless.
    formula = f"{outcome} ~ condition + (1 | task_id) + (1 | error_id)"
    priors = _build_priors(outcome)
    model = bmb.Model(formula, data=model_df, priors=priors)
    print("\nModel structure:")
    print(model)

    idata = fit_bambi(
        model,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
    )

    coef_name = _treatment_coef_name(idata)
    summary = az.summary(
        idata,
        var_names=["Intercept", coef_name],
        hdi_prob=0.94,
    )
    print("\nPosterior summary:")
    print(summary)

    post = idata.posterior[coef_name].values.flatten()
    print(f"\nP(treatment - control > 0) = {(post > 0).mean():.3f}")
    print(f"P(treatment - control < 0) = {(post < 0).mean():.3f}")
    print(f"Mean effect:  {post.mean():.4f}")
    hdi_lo, hdi_hi = np.asarray(az.hdi(post, hdi_prob=0.94), dtype=float)
    print(f"94% HDI:     [{hdi_lo:.4f}, {hdi_hi:.4f}]")
    # The variance decomposition is reported by main(), which owns the
    # pooled-vs-treatment-only choice (--variance-subset).
    return idata, model


def fit_variance_model(
    df: pl.DataFrame,
    outcome: str,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
) -> az.InferenceData:
    """Treatment-only refit whose only purpose is the variance decomposition.

    In the pooled model the control arm carries exactly one ``error_id`` per
    task, so those clusters are per-task offsets rather than per-error ones
    and σ_error pools two structurally different populations.  Restricting to
    treatment rows makes every ``error_id`` a genuine human error, so the
    three components mean what their labels say.

    The condition contrast is NOT estimable here — read it off ``fit_model``.
    """
    treat = df.filter(pl.col("condition") == "treatment")
    if treat.height == 0:
        raise SystemExit("No treatment rows available for the variance refit.")

    model_df = treat.select(outcome, "task_id", "error_id").to_pandas()
    model_df["task_id"] = model_df["task_id"].astype(str)
    model_df["error_id"] = model_df["error_id"].astype(str)

    n_task = model_df["task_id"].nunique()
    n_err = model_df["error_id"].nunique()
    print(
        f"\nVariance refit (treatment only): {len(model_df):,} rows, "
        f"{n_task:,} tasks, {n_err:,} errors "
        f"({n_err / n_task:.2f} errors/task, {len(model_df) / n_err:.2f} rows/error)"
    )
    # Separating σ_task from σ_error needs >1 error per task; at ~1.0 the two
    # components are collinear and the refit buys nothing over the pooled fit.
    if n_err / n_task < 1.05:
        print(
            "  WARNING: ~1 error per task — σ_task and σ_error are not "
            "separately identified in this subset."
        )

    # Intercept-only: with a single condition present there is no contrast to
    # estimate, and the 'condition' prior would not match any model term.
    formula = f"{outcome} ~ 1 + (1 | task_id) + (1 | error_id)"
    priors = {k: v for k, v in _build_priors(outcome).items() if k != "condition"}
    model = bmb.Model(formula, data=model_df, priors=priors)

    return fit_bambi(
        model,
        draws=draws,
        tune=tune,
        chains=chains,
        random_seed=random_seed,
    )


# Display names for the three variance components of
#   outcome ~ condition + (1 | task_id) + (1 | error_id)
# keyed by the Bambi posterior variable name.
_VARIANCE_LABELS: dict[str, str] = {
    "1|task_id_sigma": "σ_task (Task baseline)",
    "1|error_id_sigma": "σ_error (Human error)",
}
_RESIDUAL_LABEL = "σ_ε (Candidate noise)"

# In the POOLED fit the control arm carries exactly one error_id per task, so
# those clusters are per-task offsets rather than per-error ones and σ_error
# pools two structurally different populations.  Say so rather than print the
# clean label over a contaminated estimate; fit_variance_model() below refits
# on treatment rows only, where the clean label is earned.
_POOLED_ERROR_LABEL = "σ_error (Error + ctrl offset)"


def _sigma_var_names(
    idata: az.InferenceData, outcome: str
) -> tuple[list[str], str | None]:
    """Locate the group-level SDs and the residual SD in the posterior.

    Bambi names group SDs '<term>_sigma' (e.g. '1|task_id_sigma') and the
    residual either 'sigma' or '<outcome>_sigma' depending on version, so
    resolve both rather than hard-coding.
    """
    names = list(idata.posterior.data_vars)
    group = [n for n in names if n.endswith("_sigma") and "|" in n]
    # Keep formula order (task, then error) regardless of posterior order.
    preferred = [n for n in _VARIANCE_LABELS if n in group]
    group = preferred + [n for n in group if n not in preferred]

    residual = None
    for cand in ("sigma", f"{outcome}_sigma"):
        if cand in names:
            residual = cand
            break
    if residual is None:
        leftovers = [n for n in names if n.endswith("sigma") and "|" not in n]
        residual = leftovers[0] if leftovers else None
    return group, residual


def print_variance_components(
    idata: az.InferenceData,
    outcome: str,
    hdi_prob: float = 0.94,
    error_label: str | None = None,
    note: str = "",
) -> None:
    """Print the variance decomposition (ICC) of the fitted model.

    Each component's share is computed *per draw* — prop_k = σ_k² / Σ σ² —
    so the reported percentage is the median of a posterior distribution
    over proportions, not a ratio of independently-summarised medians.

    ``error_label`` overrides the display name of the error-level component
    (used to flag the pooled fit's contaminated σ_error); ``note`` is appended
    to the table title to identify which fit produced it.
    """
    group, residual = _sigma_var_names(idata, outcome)
    if residual is None or not group:
        print("\n[variance components] could not resolve sigma terms; skipping.")
        return

    terms = group + [residual]
    labels = [
        error_label if (error_label and "error_id" in t) else _VARIANCE_LABELS.get(t, t)
        for t in group
    ] + [_RESIDUAL_LABEL]
    # reshape(-1) keeps (chain, draw) alignment identical across terms, so
    # the per-draw ratios below pair the right samples together.
    sds = {t: idata.posterior[t].values.reshape(-1) for t in terms}
    variances = np.stack([sds[t] ** 2 for t in terms])  # (n_terms, n_draws)
    props = variances / variances.sum(axis=0, keepdims=True)

    label_w = max(len(x) for x in labels + ["Variance Component"]) + 2
    stat_w = 28
    header = (
        f"{'Variance Component':<{label_w}}"
        f"{f'Median [{hdi_prob:.0%} HDI]':<{stat_w}}"
        "% Total Variance"
    )
    rule = "─" * (label_w + stat_w + 26)
    print(f"\nVariance decomposition ({outcome}){note}:")
    print(header)
    print(rule)
    for i, (term, label) in enumerate(zip(terms, labels)):
        lo, hi = np.asarray(az.hdi(sds[term], hdi_prob=hdi_prob), dtype=float)
        median = float(np.median(sds[term]))
        pct = 100.0 * float(np.median(props[i]))
        stat = f"{median:.2f} [{lo:.2f}, {hi:.2f}]"
        bar = "■" * max(1, int(round(pct / 5)))
        print(f"{label:<{label_w}}{stat:<{stat_w}}{pct:>3.0f}%  {bar}")
    print(rule)


def _treatment_coef_name(idata: az.InferenceData) -> str:
    """Bambi names the level contrast differently depending on version; find it."""
    candidates = [
        name for name in idata.posterior.data_vars if name.startswith("condition")
    ]
    if not candidates:
        raise KeyError(
            "Could not locate a 'condition' coefficient in the posterior. "
            f"Available vars: {list(idata.posterior.data_vars)}"
        )
    # Prefer a name that explicitly mentions 'treatment'.
    for c in candidates:
        if "treatment" in c:
            return c
    return candidates[0]


# ── Figure 1: posterior of the condition contrast ────────────────────────────


def _executed_trace_suffix(executed_trace: bool) -> str:
    """File suffix so DAG and executed-trace AST runs don't clobber each other."""
    return "_trace" if executed_trace else ""


def plot_posterior(
    idata: az.InferenceData,
    output_dir: Path,
    outcome: str,
    transparent: bool = False,
    executed_trace: bool = False,
) -> None:
    from scipy.stats import gaussian_kde

    coef_name = _treatment_coef_name(idata)
    post = idata.posterior[coef_name].values.flatten()
    mean_val = post.mean()
    hdi_lo, hdi_hi = np.asarray(az.hdi(post, hdi_prob=0.94), dtype=float)

    fig, ax = plt.subplots(figsize=(5, 4))

    kde = gaussian_kde(post, bw_method="scott")
    x = np.linspace(post.min() - 0.005, post.max() + 0.005, 500)
    y = kde(x)
    ax.plot(x, y, color=BLUE, lw=2)

    mask = (x >= hdi_lo) & (x <= hdi_hi)
    ax.fill_between(x, y, where=mask, color=BLUE, alpha=0.25)

    ax.hlines(
        -0.3, hdi_lo, hdi_hi, color=BLUE, lw=2.5, transform=ax.get_xaxis_transform()
    )
    ax.text(
        hdi_lo,
        -0.045,
        f"{hdi_lo:.3f}",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
        color=BLUE,
    )
    ax.text(
        hdi_hi,
        -0.045,
        f"{hdi_hi:.3f}",
        ha="center",
        va="top",
        fontsize=9,
        transform=ax.get_xaxis_transform(),
        color=BLUE,
    )

    ax.axvline(0, color=GRAY, lw=1, linestyle="--", alpha=0.7)
    ax.axvline(mean_val, color=BLUE, lw=1, linestyle=":", alpha=0.8)
    ax.text(
        mean_val,
        kde(mean_val)[0] * 1.05,
        f"  mean = {mean_val:.3f}",
        va="bottom",
        ha="left",
        fontsize=9.5,
        color=BLUE,
    )

    meta = _OUTCOME_META[outcome]
    ax.set_title("Posterior: treatment − control", fontsize=12, pad=10)
    ax.set_xlabel(meta["axis_label"], fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_ylim(bottom=0)

    _apply_pub_style(ax)
    fig.tight_layout()

    prefix = meta["prefix"] + _executed_trace_suffix(executed_trace)
    for ext in ("pdf", "png"):
        path = output_dir / f"{prefix}_condition_posterior.{ext}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            transparent=(transparent and ext == "png"),
        )
        print(f"Saved {path}")

    plt.close(fig)


# ── Figure 2: condition means with posterior 94% HDIs + contrast overlay ─────


def plot_condition_means(
    df: pl.DataFrame,
    idata: az.InferenceData,
    output_dir: Path,
    outcome: str,
    transparent: bool = False,
    executed_trace: bool = False,
) -> None:
    """Show the per-condition mean of *outcome* with posterior 94% HDIs, plus
    the model's posterior-mean condition contrast as an arrow annotation."""
    pdf = df.select(outcome, "condition").to_pandas()
    coef_name = _treatment_coef_name(idata)
    post = idata.posterior[coef_name].values.flatten()
    intercept = idata.posterior["Intercept"].values.flatten()

    posterior_means = {
        "control": intercept,
        "treatment": intercept + post,
    }

    summary_rows = []
    csv_rows = []
    for cond in ("control", "treatment"):
        n = int((pdf["condition"] == cond).sum())
        if n == 0:
            continue
        draws = posterior_means[cond]
        mean = float(np.mean(draws))
        lo, hi = np.asarray(az.hdi(draws, hdi_prob=0.94), dtype=float)
        summary_rows.append((cond, mean, lo, hi, n))
        csv_rows.append(
            {
                "outcome": outcome,
                "condition": cond,
                "mean": mean,
                "hdi_low": float(lo),
                "hdi_high": float(hi),
                "hdi_prob": 0.94,
                "n": n,
            }
        )

    fig, ax = plt.subplots(figsize=(5, 4))
    x_pos = {"control": 0, "treatment": 1}
    for cond, mean, lo, hi, n in summary_rows:
        color = CONDITION_COLORS[cond]
        ax.errorbar(
            x_pos[cond],
            mean,
            yerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=8,
            capsize=4,
            elinewidth=1.2,
            label=f"{cond} (n={n:,})",
            zorder=3,
        )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["control", "treatment"], fontsize=11)
    ax.set_xlim(-0.5, 1.5)
    if summary_rows:
        y_lows = [lo for _, _, lo, _, _ in summary_rows]
        y_highs = [hi for _, _, _, hi, _ in summary_rows]
        y_min = min(y_lows)
        y_max = max(y_highs)
        y_span = y_max - y_min
        pad = 0.12 * y_span if y_span > 0 else 0.1 * max(abs(y_min), 1.0)
        ax.set_ylim(y_min - pad, y_max + pad)

    # Annotate with the posterior mean contrast.
    post_mean = float(post.mean())
    hdi_lo, hdi_hi = np.asarray(az.hdi(post, hdi_prob=0.94), dtype=float)
    meta = _OUTCOME_META[outcome]
    ax.set_ylabel(meta["y_axis_label"], fontsize=11)
    ax.set_title(meta["title"], fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="lightgray", loc="upper right")

    # Place posterior summary just below the legend
    ax.text(
        0.97,
        0.68,
        f"Posterior (treat − control): {post_mean:.3f}\n"
        f"94% HDI: [{hdi_lo:.3f}, {hdi_hi:.3f}]",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color=RED,
    )

    _apply_pub_style(ax)

    # Secondary y-axis on the right showing raw (un-logged) counts, so
    # readers don't have to mentally exponentiate. Uses a log scale so
    # exp(y_primary) lines up with y_secondary at every vertical position.
    if outcome.startswith("log_"):
        raw_outcome = outcome.removeprefix("log_")
        raw_meta = _OUTCOME_META.get(raw_outcome)
        if raw_meta is not None:
            y_lo_log, y_hi_log = ax.get_ylim()
            y_lo_raw, y_hi_raw = float(np.exp(y_lo_log)), float(np.exp(y_hi_log))
            ax_raw = ax.twinx()
            ax_raw.set_yscale("log")
            ax_raw.set_ylim(y_lo_raw, y_hi_raw)
            tick_candidates = np.array(
                [
                    1,
                    2,
                    3,
                    5,
                    7,
                    10,
                    15,
                    20,
                    25,
                    30,
                    40,
                    50,
                    75,
                    100,
                    150,
                    200,
                    300,
                    500,
                    750,
                    1000,
                    1500,
                    2000,
                ]
            )
            visible = tick_candidates[
                (tick_candidates >= y_lo_raw) & (tick_candidates <= y_hi_raw)
            ]
            if visible.size:
                ax_raw.set_yticks(visible)
                ax_raw.set_yticklabels([str(int(t)) for t in visible])
            ax_raw.minorticks_off()
            ax_raw.set_ylabel(raw_meta["y_axis_label"], fontsize=11)
            ax_raw.tick_params(labelsize=10)
            ax_raw.spines["top"].set_visible(False)
            ax_raw.spines["left"].set_visible(False)
            ax_raw.spines["right"].set_visible(True)
            ax_raw.spines["right"].set_linewidth(0.8)

    fig.tight_layout()

    prefix = meta["prefix"] + _executed_trace_suffix(executed_trace)
    for ext in ("pdf", "png"):
        path = output_dir / f"{prefix}_condition_means.{ext}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            transparent=(transparent and ext == "png"),
        )
        print(f"Saved {path}")

    plt.close(fig)

    means_path = output_dir / f"{prefix}_condition_means_summary.csv"
    pl.DataFrame(csv_rows).write_csv(means_path)
    print(f"Saved {means_path}")

    n_by_condition = pdf["condition"].value_counts().to_dict()
    contrast_path = output_dir / f"{prefix}_condition_contrast_summary.csv"
    pl.DataFrame(
        [
            {
                "outcome": outcome,
                "contrast": "treatment_minus_control",
                "mean": post_mean,
                "hdi_low": float(hdi_lo),
                "hdi_high": float(hdi_hi),
                "hdi_prob": 0.94,
                "p_gt_zero": float((post > 0).mean()),
                "p_lt_zero": float((post < 0).mean()),
                "n_control": int(n_by_condition.get("control", 0)),
                "n_treatment": int(n_by_condition.get("treatment", 0)),
            }
        ]
    ).write_csv(contrast_path)
    print(f"Saved {contrast_path}")


# ── Figure 3: per-sample distribution split by condition ─────────────────────


def plot_distribution(
    df: pl.DataFrame,
    output_dir: Path,
    outcome: str,
    transparent: bool = False,
    executed_trace: bool = False,
) -> None:
    """Histogram of per-sample outcome values split by condition, to show
    distribution shape in addition to means."""
    pdf = df.select(outcome, "condition").to_pandas()

    fig, ax = plt.subplots(figsize=(5, 4))

    for cond in ("control", "treatment"):
        vals = pdf.loc[pdf["condition"] == cond, outcome].to_numpy()
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=40,
            alpha=0.55,
            color=CONDITION_COLORS[cond],
            label=f"{cond} (n={len(vals):,})",
            density=True,
        )

    ax.axvline(0, color=GRAY, lw=0.8, linestyle=":", zorder=0)

    meta = _OUTCOME_META[outcome]
    ax.set_xlabel(meta["y_axis_label"], fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Per-sample distribution", fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="lightgray")

    _apply_pub_style(ax)
    fig.tight_layout()

    prefix = meta["prefix"] + _executed_trace_suffix(executed_trace)
    for ext in ("pdf", "png"):
        path = output_dir / f"{prefix}_condition_distribution.{ext}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            transparent=(transparent and ext == "png"),
        )
        print(f"Saved {path}")

    plt.close(fig)


# ── Figure 4: Q-Q plot against Normal distribution ───────────────────────────


def plot_qq(
    df: pl.DataFrame,
    output_dir: Path,
    outcome: str,
    transparent: bool = False,
    executed_trace: bool = False,
) -> None:
    """Q-Q plot comparing outcome distributions to a theoretical Gaussian."""
    import scipy.stats as stats

    pdf = df.select(outcome, "condition").to_pandas()

    fig, ax = plt.subplots(figsize=(5, 4))

    for cond, color in [("control", GRAY), ("treatment", BLUE)]:
        vals = pdf.loc[pdf["condition"] == cond, outcome].to_numpy()
        if len(vals) == 0:
            continue

        # probplot returns (theoretical_quantiles, ordered_values), (slope, intercept, r)
        (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")

        ax.plot(
            osm,
            osr,
            "o",
            color=color,
            markersize=4,
            alpha=0.5,
            label=f"{cond} (n={len(vals):,})",
        )
        # Plot the best-fit reference line for the distribution
        ax.plot(
            osm, slope * osm + intercept, color=color, linestyle="--", lw=1.5, alpha=0.8
        )

    meta = _OUTCOME_META[outcome]
    ax.set_xlabel("Theoretical Quantiles (Normal)", fontsize=11)
    ax.set_ylabel(meta["y_axis_label"], fontsize=11)
    ax.set_title("Q-Q Plot vs Normal Distribution", fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="lightgray")

    _apply_pub_style(ax)
    fig.tight_layout()

    prefix = meta["prefix"] + _executed_trace_suffix(executed_trace)
    for ext in ("pdf", "png"):
        path = output_dir / f"{prefix}_condition_qq.{ext}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            transparent=(transparent and ext == "png"),
        )
        print(f"Saved {path}")

    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bayesian regression comparing program complexity (length / node count) "
            "between the error-fitting treatment and correct-output control "
            "conditions. Inputs are stage-4 filter_candidates.py output."
        )
    )
    parser.add_argument(
        "--treatment-candidates",
        nargs="+",
        required=True,
        help=(
            "Stage-4 filtered-candidate JSONL file(s) for the treatment "
            "(error-fitting) condition."
        ),
    )
    parser.add_argument(
        "--control-candidates",
        nargs="+",
        required=True,
        help=(
            "Stage-4 filtered-candidate JSONL file(s) for the control "
            "(correct-output) condition."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/analyses/program_simplicity/humans_training",
        help="Directory for plots and summaries "
        "(default: results/analyses/program_simplicity/humans_training).",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save PNGs with transparent background",
    )
    parser.add_argument(
        "--ast",
        action="store_true",
        help="Run regression using AST node counts instead of line counts.",
    )
    parser.add_argument(
        "--log-outcome",
        action="store_true",
        help="Use log-transformed outcomes for both models.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help=(
            "Use relative_length_change = (rewritten - original) / original as "
            "the outcome. Mutually exclusive with --ast and --log-outcome."
        ),
    )
    parser.add_argument(
        "--executed-trace",
        action="store_true",
        help=(
            "Count every variable reference as expanding a fresh subtree "
            "(executed-trace complexity). Default is to deduplicate "
            "structurally-identical subtrees via computation_id (DAG count). "
            "Only meaningful with --ast."
        ),
    )
    parser.add_argument(
        "--variance-subset",
        choices=("pooled", "treatment", "both"),
        default="both",
        help=(
            "Which fit the variance decomposition is reported from. "
            "'treatment' refits on treatment rows only, where every error_id "
            "is a genuine human error; 'pooled' reuses the contrast model, "
            "whose sigma_error mixes in one per-task offset per control task. "
            "Default 'both' prints the two tables for comparison (costs a "
            "second MCMC run). The contrast and all figures always come from "
            "the pooled fit."
        ),
    )
    add_sampling_args(parser)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(paths: list[str]) -> list[Path]:
        out = []
        for p in paths:
            path = Path(p)
            if not path.is_absolute():
                path = _project_root / path
            out.append(path)
        return out

    treatment_files = _resolve(args.treatment_candidates)
    control_files = _resolve(args.control_candidates)

    for label, files in (("treatment", treatment_files), ("control", control_files)):
        missing = [p for p in files if not p.exists()]
        if missing:
            print(f"Error: {label} candidate files not found:")
            for p in missing:
                print(f"  {p}")
            sys.exit(1)

    if args.relative and (args.ast or args.log_outcome):
        parser.error("--relative is mutually exclusive with --ast and --log-outcome.")

    if args.executed_trace and not args.ast:
        parser.error("--executed-trace only applies in AST mode; pass --ast as well.")

    dedup_subtrees = not args.executed_trace
    if args.ast:
        print(
            f"\nAST node-count mode: "
            f"{'executed-trace (every reference expanded)' if args.executed_trace else 'DAG (subtrees deduplicated by computation_id)'}"
        )

    df = load_data(
        treatment_files,
        control_files,
        compute_ast=args.ast,
        dedup_subtrees=dedup_subtrees,
    )

    if args.relative:
        outcome = "relative_length_change"
        print(f"\nRelative mode: outcome={outcome} ({df.height:,} rows)")
    elif args.ast:
        outcome = (
            "log_rewritten_node_count" if args.log_outcome else "rewritten_node_count"
        )
        print(f"\nAST mode: outcome={outcome} ({df.height:,} rows)")
    else:
        outcome = (
            "log_rewritten_line_count" if args.log_outcome else "rewritten_line_count"
        )
        print(f"\nLine mode: outcome={outcome} ({df.height:,} rows)")

    print(f"\n── Regression Model: {outcome} ──────────────────────────────────")
    idata, _ = fit_model(df, outcome, args.draws, args.tune, args.chains, args.seed)

    if args.variance_subset in ("pooled", "both"):
        print_variance_components(
            idata,
            outcome,
            error_label=_POOLED_ERROR_LABEL,
            note=" — pooled fit (see sigma_error caveat)",
        )
    if args.variance_subset in ("treatment", "both"):
        variance_idata = fit_variance_model(
            df, outcome, args.draws, args.tune, args.chains, args.seed
        )
        print_variance_components(
            variance_idata, outcome, note=" — treatment-only refit"
        )

    plot_posterior(
        idata,
        output_dir,
        outcome,
        args.transparent,
        executed_trace=args.executed_trace,
    )
    plot_condition_means(
        df,
        idata,
        output_dir,
        outcome,
        args.transparent,
        executed_trace=args.executed_trace,
    )
    plot_distribution(
        df,
        output_dir,
        outcome,
        args.transparent,
        executed_trace=args.executed_trace,
    )
    plot_qq(
        df,
        output_dir,
        outcome,
        args.transparent,
        executed_trace=args.executed_trace,
    )


if __name__ == "__main__":
    main()
