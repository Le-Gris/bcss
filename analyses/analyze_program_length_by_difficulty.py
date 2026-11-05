"""Predict error-program length from task difficulty and reference-program
length with a Bayesian regression, optionally adding an error-category term.

Difficulty is each task's human success rate; the model has a random intercept
per error. With ``--relative`` the outcome is the relative length change,
(edited - reference) / reference. Given ``--char-file`` (the classifications of
the same candidates), each program enters with its category proportions over
the classifier's samples; ``--alpha`` sets how much those proportions are
smoothed, and ``--hard-labels`` uses the most frequent category instead.

Reads a candidates file, a task success-rate CSV and optionally a
classification file. Writes posterior, partial-residual and category-forest
plots, and CSV summaries of the data, the coefficients and the category
effects.

Usage:
    uv run python analyses/analyze_program_length_by_difficulty.py \\
        --candidates-file OSF/03_candidates/human/training.jsonl \\
        --char-file OSF/04_classifications/human/training.jsonl \\
        --success-rate-file OSF/00_inputs/human/harc_task_difficulty.csv \\
        --relative --alpha 0.1 \\
        --output-dir results/analyses/program_length_by_difficulty/humans_training
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter
from scipy.stats import gaussian_kde

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:  # keep --help importable without the Bayesian stack
    az = pm = pt = None  # type: ignore[assignment]

from src.bayes_defaults import (
    INFERENCE_METHOD,
    TARGET_ACCEPT,
    add_sampling_args,
)
from src.category_definitions import CATEGORY_NAMES, DISPLAY_LABEL_OVERRIDES
from src.utils import grid_hash, load_jsonl_records

# Category order for the count-tally / mixture components: the seven cognitive
# categories plus the catch-all. Any missing/None label folds into UNCLASSIFIED.
UNCLASSIFIED = "unclassifiable"
CAT_ORDER: list[str] = list(CATEGORY_NAMES) + [UNCLASSIFIED]

# Default success-rate source: H-ARC human solve rates. Override --success-rate-file
# (and --success-rate-column) to use any solver's per-task success rate.
DEFAULT_SUCCESS_RATE_FILE = "OSF/00_inputs/human/harc_task_difficulty.csv"

# ── Style ────────────────────────────────────────────────────────────────────

BLUE = "#4878CF"
RED = "#C44E52"
GRAY = "#888888"

_OUTCOME_LABELS = {
    "relative_length_change": "Relative length change",
    "rewritten_line_count": "Rewritten line count",
}
_PREFIX_MAP = {
    "relative_length_change": "rlc",
    "rewritten_line_count": "rwlc",
}
# Predictor display labels (posterior x-axis + filenames).
_PREDICTOR_LABELS = {
    "b_success_rate": "success_rate",
    "b_complexity": "original_line_count",
}


def _apply_pub_style(ax: plt.Axes, fontsize: int = 11) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=fontsize - 1)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)


def _format_category_label(name: str) -> str:
    """Category slug → display label.

    Honours ``DISPLAY_LABEL_OVERRIDES`` (e.g. ``attribute_substitution`` →
    "Attribute Misselection") before falling back to title-casing the slug.
    """
    return DISPLAY_LABEL_OVERRIDES.get(name, name.replace("_", " ").title())


# ── Data ─────────────────────────────────────────────────────────────────────


def _resolve_grid_hash(record: dict) -> str | None:
    """Prefer a stored ``grid_hash``; else hash the error grid (never a local hash)."""
    h = record.get("grid_hash")
    if h:
        return h
    grid = record.get("target_error_grid")
    if grid is not None:
        return grid_hash(grid)
    return None


def load_eval_data(candidate_files: list[Path]) -> pl.DataFrame:
    """Load stage-4 ``filter_candidates.py`` output into a modelling frame.

    Every field is carried through stage 4 unchanged from stage 3.  ``program_id``
    is reconstructed as ``grid_hash(rewritten_program)`` — the exact key
    ``scripts/classify_errors.py`` uses — so the optional category join lines up.
    Rows are filtered to executed, non-nested programs with a valid GT length.
    """
    records = load_jsonl_records(candidate_files)
    rows: list[dict] = []
    for r in records:
        h = _resolve_grid_hash(r)
        rewritten = r.get("rewritten_program")
        if h is None or rewritten is None:
            continue
        rows.append(
            {
                "task_id": r.get("task_id"),
                "error_id": r.get("error_id"),
                "grid_hash": h,
                "program_id": grid_hash(rewritten),
                "execution_success": r.get("execution_success"),
                "has_nested_violation": r.get("has_nested_violation"),
                "line_count_delta": r.get("line_count_delta"),
                "original_line_count": r.get("original_line_count"),
                "rewritten_line_count": r.get("rewritten_line_count"),
            }
        )

    raw = pl.from_dicts(rows, infer_schema_length=10_000)
    df = raw.filter(
        pl.col("execution_success")
        & pl.col("line_count_delta").is_not_null()
        & pl.col("original_line_count").is_not_null()
        & (pl.col("original_line_count") > 0)
        & ~pl.col("has_nested_violation").fill_null(False)
    )
    df = df.with_columns(
        (pl.col("line_count_delta") / pl.col("original_line_count")).alias(
            "relative_length_change"
        ),
        (pl.col("task_id") + pl.lit("::") + pl.col("grid_hash")).alias("task_err"),
    )
    return df


def build_category_counts(char_files: list[Path]) -> pl.DataFrame:
    """Tally each program's per-sample classifications into a K-vector of counts.

    Returns one row per ``(error_id, program_id)`` with an integer column per
    category (``count_<category>``) in :data:`CAT_ORDER` and ``n_samples`` (the
    number of non-null samples).  Missing / ``None`` labels fold into
    ``unclassifiable`` so every sample is counted.  Pure function — unit-tested.
    """
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {c: 0 for c in CAT_ORDER}
    )
    records = load_jsonl_records(char_files)
    for r in records:
        eid = r.get("error_id")
        pid = r.get("program_id")
        if eid is None or pid is None:
            continue
        cat = (r.get("characterization_result") or {}).get("error_category")
        if cat not in CAT_ORDER:
            cat = UNCLASSIFIED
        counts[(eid, pid)][cat] += 1

    rows: list[dict] = []
    for (eid, pid), cvec in counts.items():
        row = {"error_id": eid, "program_id": pid}
        n = 0
        for c in CAT_ORDER:
            row[f"count_{c}"] = cvec[c]
            n += cvec[c]
        row["n_samples"] = n
        rows.append(row)

    schema = {"error_id": pl.Utf8, "program_id": pl.Utf8}
    schema |= {f"count_{c}": pl.Int64 for c in CAT_ORDER}
    schema["n_samples"] = pl.Int64
    return pl.DataFrame(rows, schema=schema)


def load_success_rates(
    path: Path,
    value_column: str = "success_rate",
    task_column: str | None = None,
) -> pl.DataFrame:
    """Load per-task success rates — the difficulty covariate — from a CSV.

    Generic over the *solver*: the default H-ARC file holds human solve rates,
    but any per-task success rate works (e.g. an LLM's pass rate on the same
    tasks).  The CSV needs a task-id column and a value column.

    * Task-id column: ``task_column`` if given, else auto-detected as ``task_id``
      or the H-ARC ``tasks`` column.  A trailing ``.json`` is stripped so ids
      match the pipeline's ``task_id`` either way.
    * Value column: ``value_column`` (default ``success_rate``), returned under
      the internal ``success_rate`` predictor name.
    """
    df = pl.read_csv(path)
    df = df.drop([c for c in df.columns if c in ("", "column_0")])

    if task_column is None:
        task_column = next((c for c in ("task_id", "tasks") if c in df.columns), None)
        if task_column is None:
            raise SystemExit(
                f"{path}: no task-id column found (expected 'task_id' or 'tasks'; "
                "pass --task-column for a differently-named column)."
            )
    if task_column not in df.columns:
        raise SystemExit(f"{path}: task column '{task_column}' not present.")
    if value_column not in df.columns:
        raise SystemExit(
            f"{path}: success-rate column '{value_column}' not present "
            "(pass --success-rate-column)."
        )

    df = df.with_columns(
        pl.col(task_column).cast(pl.Utf8).str.replace(r"\.json$", "").alias("task_id")
    )
    return df.select(["task_id", pl.col(value_column).alias("success_rate")])


def merge_data(
    eval_df: pl.DataFrame,
    diff_df: pl.DataFrame,
    counts_df: pl.DataFrame | None,
) -> pl.DataFrame:
    """Join difficulty (always) and category counts (when modelling categories)."""
    joined = eval_df.join(diff_df, on="task_id", how="inner")
    print(f"Eval rows valid: {eval_df.height:,}")
    print(f"Rows after joining difficulty: {joined.height:,}")
    if counts_df is not None:
        joined = joined.join(counts_df, on=["error_id", "program_id"], how="inner")
        print(f"Rows after joining category counts: {joined.height:,}")
        if joined.height == 0:
            raise SystemExit(
                "Category join produced 0 rows — the --char-file does not match "
                "this --candidates-file (program_id = grid_hash(rewritten_program))."
            )
    return joined


# ── Model ────────────────────────────────────────────────────────────────────


class FitResult:
    """Container bundling the fit outputs the plotting layer consumes.

    ``re`` and ``cat_effect`` are per-row posterior-mean adjustments used to form
    strict partial residuals: ``re`` is the per-error random intercept
    ``u_{err[i]}``; ``cat_effect`` is ``E[theta_i] . E[gamma]`` (0 in the reduced
    model).  ``cat_posteriors`` maps each category to flattened ``gamma`` draws.
    """

    def __init__(
        self,
        idata: "az.InferenceData",
        center_means: dict[str, float],
        re: np.ndarray,
        cat_effect: np.ndarray,
        cat_posteriors: dict[str, np.ndarray] | None,
    ) -> None:
        self.idata = idata
        self.center_means = center_means
        self.re = re
        self.cat_effect = cat_effect
        self.cat_posteriors = cat_posteriors


def _priors_for(outcome: str, y: np.ndarray) -> dict[str, float]:
    """Weakly-informative prior scales matched to the outcome's magnitude.

    ``relative_length_change`` is order-1, so fixed unit-ish scales mirror the
    control-vs-treatment regression; ``rewritten_line_count`` is data-scaled.
    """
    if outcome == "relative_length_change":
        return {
            "coef": 1.5,
            "intercept_mu": 0.0,
            "intercept_sigma": 1.5,
            "sigma_y": 1.5,
        }
    sd = float(y.std()) or 1.0
    return {
        "coef": sd,
        "intercept_mu": float(y.mean()),
        "intercept_sigma": 2 * sd,
        "sigma_y": 2 * sd,
    }


def fit_length_model(
    df: pl.DataFrame,
    *,
    outcome: str,
    complexity_col: str,
    has_category: bool,
    draws: int,
    tune: int,
    chains: int,
    random_seed: int,
    hard: bool = False,
    alpha: float = 0.1,
) -> FitResult:
    """Fit the reduced or category model via NUTS (numpyro backend).

    In category mode the per-program category weights are *fixed* data from the
    classification counts (no latent allocation), so the model always converges
    cleanly.  Two weightings:

    * ``hard=False`` (default) — soft weights: each ``theta_i`` is the
      Dirichlet-multinomial posterior mean of the counts (symmetric ``alpha=1``),
      an uncertainty-aware label that keeps the classifier's per-program
      disagreement instead of collapsing it to a point.  The category effect
      enters the mean linearly as the probability-weighted average
      ``theta_i . gamma`` (a plain linear-Gaussian term, not a latent mixture).
      Needs ``--num-samples >= 2`` to carry any disagreement signal.  ``alpha``
      sets the symmetric Dirichlet smoothing (``alpha -> 0`` = raw proportions,
      so ``gamma`` reflects the true per-category magnitude).
    * ``hard=True`` — point labels: ``theta_i`` is the one-hot of the plurality
      category, i.e. a plain categorical covariate.  Use this when the
      classification has a single sample per program (``--num-samples 1``), where
      the alpha=1 smoothing would otherwise wash a lone count toward uniform.
    """
    if pm is None:
        raise SystemExit("pymc / arviz not importable — install the Bayesian stack.")

    y = df[outcome].to_numpy().astype(float)
    sr = df["success_rate"].to_numpy().astype(float)
    comp = df[complexity_col].to_numpy().astype(float)
    task_err = df["task_err"].to_numpy()

    uniques, error_idx = np.unique(task_err, return_inverse=True)
    n_obs = y.shape[0]
    n_errors = uniques.shape[0]

    center_means = {
        "success_rate": float(sr.mean()),
        complexity_col: float(comp.mean()),
    }
    sr_c = sr - center_means["success_rate"]
    comp_c = comp - center_means[complexity_col]
    print(
        f"\nCentering: success_rate mean = {center_means['success_rate']:.4f}, "
        f"{complexity_col} mean = {center_means[complexity_col]:.4f}"
    )
    print(f"Rows: {n_obs:,} | unique errors (task_err): {n_errors:,}")

    pr = _priors_for(outcome, y)

    if has_category:
        counts = df.select([f"count_{c}" for c in CAT_ORDER]).to_numpy().astype(int)
        n_samples = counts.sum(axis=1)
        K = len(CAT_ORDER)
        if hard:
            # Point labels: one-hot the plurality category (plain categorical
            # covariate). The right choice at --num-samples 1, where the soft
            # weighting's alpha=1 smoothing would wash a lone count toward uniform.
            label_idx = counts.argmax(axis=1)
        else:
            # Soft labels: theta fixed to the Dirichlet-multinomial posterior mean
            # with symmetric concentration `alpha`, treated as data. alpha -> 0
            # gives raw empirical proportions (a unanimous program loads weight 1
            # on its category, so gamma is on the true per-category magnitude
            # scale); larger alpha smooths a lone count toward uniform (wanted at
            # --num-samples 1). With alpha=1 and K categories the dominant weight
            # caps near (n+1)/(n+K), which inflates gamma by ~1/that factor.
            theta_fixed = (counts + alpha) / (n_samples[:, None] + K * alpha)

    coords = {"obs": np.arange(n_obs), "error": uniques}
    if has_category:
        coords["category"] = CAT_ORDER
        coords["real_category"] = list(CATEGORY_NAMES)

    with pm.Model(coords=coords):
        b0 = pm.Normal("Intercept", mu=pr["intercept_mu"], sigma=pr["intercept_sigma"])
        b_sr = pm.Normal("b_success_rate", mu=0, sigma=pr["coef"])
        b_len = pm.Normal("b_complexity", mu=0, sigma=pr["coef"])

        sigma_u = pm.HalfNormal("sigma_u", sigma=pr["sigma_y"])
        u_raw = pm.Normal("u_raw", mu=0, sigma=1, dims="error")
        u = pm.Deterministic("u", u_raw * sigma_u, dims="error")

        base = b0 + b_sr * sr_c + b_len * comp_c + u[error_idx]
        sigma_y = pm.HalfNormal("sigma_y", sigma=pr["sigma_y"])

        if has_category:
            # Category effects.  The seven *cognitive* categories share a
            # sum-to-zero prior (ZeroSumNormal — enforced in the
            # parameterization, not by post-hoc mean subtraction) so they are
            # contrasted against their own grand mean, which is what the science
            # asks.  ``unclassifiable`` is NOT a peer category: it is a catch-all
            # for short degenerate programs (empirically ~5 vs ~13 lines), with
            # an effect several times larger than any real category.  Folding it
            # into the same shared-scale sum-to-zero prior stretches that scale
            # to fit one outlier, wrecking NUTS geometry (R-hat > 1.5) and
            # dragging the grand-mean anchor.  Give it its own free effect.
            gamma_real = pm.ZeroSumNormal(
                "gamma_real", sigma=pr["coef"], dims="real_category"
            )
            gamma_unclf = pm.Normal("gamma_unclassifiable", mu=0, sigma=pr["coef"])
            # Assemble the full effect vector in CAT_ORDER (real cats, then unclf).
            gamma = pm.Deterministic(
                "gamma",
                pt.concatenate([gamma_real, gamma_unclf[None]]),
                dims="category",
            )

            if hard:
                # Plain categorical: each program gets its one label's effect.
                pm.Normal(
                    "y_obs",
                    mu=base + gamma[label_idx],
                    sigma=sigma_y,
                    observed=y,
                    dims="obs",
                )
            else:
                # Soft covariate: the classifier-probability-weighted category
                # effect enters the mean linearly (theta_i . gamma). theta is
                # fixed data, so this is a plain linear-Gaussian term -- unimodal
                # geometry that samples cleanly on both outcomes, unlike a
                # pm.Mixture over the same weights (whose latent-allocation
                # geometry will not mix on the relative scale).
                pm.Normal(
                    "y_obs",
                    mu=base + pt.dot(theta_fixed, gamma),
                    sigma=sigma_y,
                    observed=y,
                    dims="obs",
                )
        else:
            pm.Normal("y_obs", mu=base, sigma=sigma_y, observed=y, dims="obs")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            target_accept=TARGET_ACCEPT,
            nuts_sampler=INFERENCE_METHOD,
            idata_kwargs={"log_likelihood": False},
        )

    # Per-row posterior-mean adjustments for strict partial residuals.
    u_mean = idata.posterior["u"].mean(("chain", "draw")).values  # aligned to `uniques`
    re = u_mean[error_idx]

    cat_posteriors: dict[str, np.ndarray] | None = None
    if has_category:
        gda = idata.posterior["gamma"]
        cat_posteriors = {c: gda.sel(category=c).values.flatten() for c in CAT_ORDER}
        gamma_mean = gda.mean(("chain", "draw")).values  # (K,)
        # Per-row category contribution for strict partial residuals: the label's
        # own effect (hard) or the soft-weighted average (soft). Weights are data
        # in both, so nothing to read back from the posterior. In soft mode this
        # is exactly the theta_i . gamma term the likelihood now uses.
        if hard:
            cat_effect = gamma_mean[label_idx]
        else:
            cat_effect = theta_fixed @ gamma_mean
    else:
        cat_effect = np.zeros(n_obs)

    if "u_raw" in idata.posterior:
        del idata.posterior["u_raw"]

    _print_diagnostics(idata, outcome, complexity_col, center_means, has_category)
    return FitResult(idata, center_means, re, cat_effect, cat_posteriors)


def _print_diagnostics(
    idata: "az.InferenceData",
    outcome: str,
    complexity_col: str,
    center_means: dict[str, float],
    has_category: bool,
) -> None:
    scalar_vars = ["Intercept", "b_success_rate", "b_complexity", "sigma_u", "sigma_y"]
    var_names = scalar_vars + (["gamma"] if has_category else [])
    print("\nPosterior summary:")
    print(az.summary(idata, var_names=var_names, hdi_prob=0.94))

    n_div = int(idata.sample_stats["diverging"].sum()) if "sample_stats" in idata else 0
    print(f"\nDivergences: {n_div}")

    for var in ("b_success_rate", "b_complexity"):
        post = idata.posterior[var].values.flatten()
        lo, hi = az.hdi(post, hdi_prob=0.94)
        pred = _PREDICTOR_LABELS[var]
        print(f"\n[{pred}]")
        print(f"  P(> 0) = {(post > 0).mean():.3f}   P(< 0) = {(post < 0).mean():.3f}")
        print(f"  Mean: {post.mean():.4f}   94% HDI: [{lo:.4f}, {hi:.4f}]")

    if has_category:
        for cat in CAT_ORDER:
            post = idata.posterior["gamma"].sel(category=cat).values.flatten()
            lo, hi = az.hdi(post, hdi_prob=0.94)
            print(f"\n[error_category: {cat}]")
            print(
                f"  P(> 0) = {(post > 0).mean():.3f}   P(< 0) = {(post < 0).mean():.3f}"
            )
            print(f"  Mean: {post.mean():.4f}   94% HDI: [{lo:.4f}, {hi:.4f}]")


# ── Summaries ────────────────────────────────────────────────────────────────

# Reference program lengths at which predictions are reported (and annotated
# on the partial-residual plot).
PREDICT_AT: tuple[float, ...] = (7.0, 14.0, 28.0)


def _mean_hdi(draws: np.ndarray) -> dict[str, float]:
    draws = np.asarray(draws).flatten()
    lo, hi = np.asarray(az.hdi(draws, hdi_prob=0.94), dtype=float)
    return {
        "mean": float(draws.mean()),
        "hdi_lower": float(lo),
        "hdi_upper": float(hi),
        "p_gt0": float((draws > 0).mean()),
    }


def save_length_summaries(
    fit: FitResult,
    df: pl.DataFrame,
    complexity_col: str,
    output_dir: Path,
    predict_at: tuple[float, ...] = PREDICT_AT,
) -> None:
    """Write the reported estimands as CSVs (posterior means and 94% HDIs).

    * ``length_data_summary.csv``: rows, unique errors and the centering means.
    * ``length_coefficients_summary.csv``: intercept, slopes, and the predicted
      outcome ``Intercept + b_complexity * (L - mean L)`` at each ``predict_at``
      reference length (at the mean success rate, category-neutral).
    * ``length_category_summary.csv`` (category model only): each category's
      effect ``gamma`` and its predicted outcome ``Intercept + gamma``.

    Derived quantities are computed per draw, so their HDIs are HDIs of the
    sum, not sums of HDIs.  Values are on the outcome's scale; for
    ``--relative`` a percentage shortening is ``-100 * value``.
    """
    post = fit.idata.posterior
    b0 = post["Intercept"].values.flatten()
    slope = post["b_complexity"].values.flatten()
    center_len = float(fit.center_means[complexity_col])

    pl.DataFrame(
        {
            "quantity": [
                "n_rows",
                "n_errors",
                "center_success_rate",
                f"center_{complexity_col}",
            ],
            "value": [
                float(df.height),
                float(df["task_err"].n_unique()),
                float(fit.center_means["success_rate"]),
                center_len,
            ],
        }
    ).write_csv(output_dir / "length_data_summary.csv")

    rows = [
        {"term": "Intercept", **_mean_hdi(b0)},
        {"term": "b_success_rate", **_mean_hdi(post["b_success_rate"].values)},
        {"term": "b_complexity", **_mean_hdi(slope)},
    ]
    for L in predict_at:
        rows.append(
            {"term": f"predicted_at_L{L:g}", **_mean_hdi(b0 + slope * (L - center_len))}
        )
    path = output_dir / "length_coefficients_summary.csv"
    pl.DataFrame(rows).write_csv(path)
    print(f"Saved {path}")

    if "gamma" in post:
        gamma = post["gamma"]
        cat_rows = []
        for c in CAT_ORDER:
            g = gamma.sel(category=c).values.flatten()
            eff, pred = _mean_hdi(g), _mean_hdi(b0 + g)
            cat_rows.append(
                {
                    "category": c,
                    "label": _format_category_label(c),
                    **{f"gamma_{k}": v for k, v in eff.items()},
                    **{f"predicted_{k}": v for k, v in pred.items() if k != "p_gt0"},
                }
            )
        path = output_dir / "length_category_summary.csv"
        pl.DataFrame(cat_rows).write_csv(path)
        print(f"Saved {path}")


# ── Plotting ─────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, output_dir: Path, stem: str, transparent: bool) -> None:
    for ext in ("pdf", "png"):
        path = output_dir / f"{stem}.{ext}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            transparent=(transparent and ext == "png"),
        )
        print(f"Saved {path}")
    plt.close(fig)


def _plot_single_posterior(
    post: np.ndarray,
    var_label: str,
    outcome: str,
    output_dir: Path,
    stem: str,
    transparent: bool = False,
    no_title: bool = False,
) -> None:
    """KDE posterior density with 94% HDI shading for a single scalar."""
    mean_val = post.mean()
    hdi_lo, hdi_hi = az.hdi(post, hdi_prob=0.94)

    fig, ax = plt.subplots(figsize=(5, 4))
    kde = gaussian_kde(post, bw_method="scott")
    x = np.linspace(post.min() - 0.005, post.max() + 0.005, 500)
    y = kde(x)
    ax.plot(x, y, color=BLUE, lw=2)
    ax.fill_between(x, y, where=(x >= hdi_lo) & (x <= hdi_hi), color=BLUE, alpha=0.25)

    ax.hlines(
        -0.03, hdi_lo, hdi_hi, color=BLUE, lw=2.5, transform=ax.get_xaxis_transform()
    )
    for xv in (hdi_lo, hdi_hi):
        ax.text(
            xv,
            -0.045,
            f"{xv:.3f}",
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

    if not no_title:
        ax.set_title(f"Posterior: {var_label}", fontsize=12, pad=10)
    y_axis_label = _OUTCOME_LABELS.get(outcome, outcome)
    ax.set_xlabel(f"Effect on {y_axis_label.lower()}", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_ylim(bottom=0)
    _apply_pub_style(ax)
    fig.tight_layout()
    _save(fig, output_dir, stem, transparent)


def plot_scalar_posterior(
    idata: "az.InferenceData",
    outcome: str,
    var_name: str,
    output_dir: Path,
    file_prefix: str,
    transparent: bool = False,
    no_title: bool = False,
) -> None:
    post = idata.posterior[var_name].values.flatten()
    pred = _PREDICTOR_LABELS.get(var_name, var_name)
    prefix = _PREFIX_MAP.get(outcome, outcome)
    _plot_single_posterior(
        post,
        var_label=pred,
        outcome=outcome,
        output_dir=output_dir,
        stem=f"{file_prefix}{prefix}_posterior_{pred}",
        transparent=transparent,
        no_title=no_title,
    )


def plot_category_forest(
    cat_posteriors: dict[str, np.ndarray],
    outcome: str,
    output_dir: Path,
    file_prefix: str,
    transparent: bool = False,
    no_title: bool = False,
) -> None:
    """Horizontal forest plot comparing all category-level (sum-to-zero) effects."""
    # Order by posterior mean so the plot reads as a monotone ladder: lowest
    # effect at the bottom (i=0), highest at the top.
    categories = sorted(cat_posteriors, key=lambda c: cat_posteriors[c].mean())
    n_cats = len(categories)
    fig, ax = plt.subplots(figsize=(7, 0.55 * n_cats + 1.5))

    for i, cat in enumerate(categories):
        post = cat_posteriors[cat]
        lo, hi = az.hdi(post, hdi_prob=0.94)
        q25, q75 = np.percentile(post, [25, 75])
        ax.hlines(i, lo, hi, color=BLUE, lw=1.5)
        ax.hlines(i, q25, q75, color=BLUE, lw=4, alpha=0.7)
        ax.plot(post.mean(), i, "o", color=BLUE, markersize=6, zorder=5)

    ax.axvline(0, color=GRAY, lw=1, linestyle="--", alpha=0.7)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels([_format_category_label(c) for c in categories])
    y_axis_label = _OUTCOME_LABELS.get(outcome, outcome)
    ax.set_xlabel(
        f"Effect on {y_axis_label.lower()} (deviation from grand mean)", fontsize=11
    )
    if not no_title:
        ax.set_title("Error category effects (sum-to-zero)", fontsize=12, pad=10)
    _apply_pub_style(ax)
    fig.tight_layout()
    prefix = _PREFIX_MAP.get(outcome, outcome)
    _save(fig, output_dir, f"{file_prefix}{prefix}_category_forest", transparent)


def _partial_residual_plot(
    df: pl.DataFrame,
    idata: "az.InferenceData",
    fit: "FitResult",
    output_dir: Path,
    outcome: str,
    complexity_col: str,
    file_prefix: str,
    *,
    predictor_col: str,
    slope_var: str,
    other_slope_var: str,
    other_col: str,
    x_label: str,
    annotate_at: tuple[float, ...] = (),
    transparent: bool = False,
    no_title: bool = False,
) -> None:
    """Strict partial residuals of ``outcome`` against ``predictor_col``.

    Subtracts the *other* continuous predictor's (centered) effect, the per-row
    category contribution ``E[theta]·E[gamma]``, and the per-error random
    intercept — all supplied as posterior means on ``fit``.
    """
    model_df = df.select(outcome, predictor_col, other_col).to_pandas()
    model_df["re"] = fit.re
    model_df["cat_effect"] = fit.cat_effect

    intercept = idata.posterior["Intercept"].values.flatten()
    slope = idata.posterior[slope_var].values.flatten()
    other_slope_mean = idata.posterior[other_slope_var].values.flatten().mean()

    mu_pred = fit.center_means[predictor_col]
    mu_other = fit.center_means[other_col]

    model_df["y_partial"] = model_df[outcome] - (
        (model_df[other_col] - mu_other) * other_slope_mean
        + model_df["cat_effect"]
        + model_df["re"]
    )

    x_min = float(model_df[predictor_col].min())
    x_max = float(model_df[predictor_col].max())
    x_grid = np.linspace(x_min, x_max, 200)
    x_centered = x_grid - mu_pred
    y_lines = intercept[:, None] + slope[:, None] * x_centered[None, :]
    # az.hdi wants (chain, draw, *event); fold flattened draws into one chain.
    hdi = az.hdi(y_lines[None, :, :], hdi_prob=0.94)
    band_lo, band_hi = hdi[:, 0], hdi[:, 1]

    mean_intercept = intercept.mean()
    mean_slope = slope.mean()

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(
        model_df[predictor_col],
        model_df["y_partial"],
        s=8,
        alpha=0.15,
        color=BLUE,
        edgecolors="none",
        rasterized=True,
        label="Partial residuals",
    )
    ax.fill_between(x_grid, band_lo, band_hi, color=RED, alpha=0.2, label="94% HDI")
    ax.plot(
        x_grid,
        mean_intercept + mean_slope * x_centered,
        color=RED,
        lw=2,
        linestyle="--",
        label=f"Posterior mean (β = {mean_slope:.4f})",
    )
    ax.axhline(0, color=GRAY, lw=0.8, linestyle=":", zorder=0)

    is_relative = "relative" in outcome
    for L in annotate_at:
        if L < x_min or L > x_max:
            continue
        pred = mean_intercept + mean_slope * (L - mu_pred)
        # Format the worked prediction in the outcome's own units: a percentage
        # for relative_length_change, a line count for the absolute outcome.
        pred_label = f"{pred * 100:+.1f}%" if is_relative else f"{pred:.1f} lines"
        ax.axvline(L, color=GRAY, lw=0.6, linestyle=":", alpha=0.55)
        ax.plot(
            L,
            pred,
            "o",
            color=RED,
            markersize=7,
            zorder=6,
            markeredgecolor="white",
            markeredgewidth=1.2,
        )
        ax.annotate(
            f"L={int(L)}\n{pred_label}",
            xy=(L, pred),
            xytext=(8, 12),
            textcoords="offset points",
            fontsize=9,
            color=RED,
            ha="left",
            va="bottom",
            fontweight="medium",
        )

    y_axis_label = _OUTCOME_LABELS.get(outcome, outcome)
    if is_relative:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(f"Partial residual ({y_axis_label.lower()})", fontsize=11)
    if not no_title:
        ax.set_title(f"Partial residuals vs {x_label.lower()}", fontsize=12, pad=10)
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="lightgray", loc="lower left")
    _apply_pub_style(ax)
    fig.tight_layout()
    prefix = _PREFIX_MAP.get(outcome, outcome)
    _save(
        fig,
        output_dir,
        f"{file_prefix}{prefix}_partial_residuals_{predictor_col}",
        transparent,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def _abs_paths(paths: list[str]) -> list[Path]:
    return [p if (p := Path(s)).is_absolute() else _project_root / s for s in paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian regression predicting synthesized program length "
        "from task difficulty and GT length (optional error-category term)."
    )
    parser.add_argument(
        "--candidates-file",
        nargs="+",
        required=True,
        help="Stage-4 filter_candidates.py output JSONL(s).",
    )
    parser.add_argument(
        "--char-file",
        nargs="+",
        default=None,
        help="Optional classify_errors.py output JSONL(s) for the SAME candidates; "
        "enables the error-category term.",
    )
    parser.add_argument(
        "--success-rate-file",
        default=DEFAULT_SUCCESS_RATE_FILE,
        help="CSV of per-task success rates used as the difficulty covariate. "
        "Defaults to the H-ARC human solve rates; pass any solver's per-task "
        "success rate (e.g. an LLM's pass rate) to model non-human outputs.",
    )
    parser.add_argument(
        "--success-rate-column",
        default="success_rate",
        help="Name of the success-rate value column in --success-rate-file.",
    )
    parser.add_argument(
        "--task-column",
        default=None,
        help="Name of the task-id column in --success-rate-file "
        "(default: auto-detect 'task_id' or the H-ARC 'tasks' column).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/analyses/program_length_by_difficulty/humans_training",
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Use relative_length_change (delta / GT length) as the outcome.",
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Save PNGs with a transparent background.",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles (useful for publication figures).",
    )
    parser.add_argument(
        "--hard-labels",
        action="store_true",
        help="Category mode: use point labels (one-hot the plurality category) "
        "instead of the default soft weights. Use this for single-sample "
        "classifications (--num-samples 1), where the soft weighting's alpha=1 "
        "smoothing would wash a lone count toward uniform.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Soft-label Dirichlet smoothing (symmetric concentration): theta = "
        "(counts + alpha) / (n + K*alpha). Default 0.1 (light): unanimous programs "
        "load ~fully on their category so gamma reflects the true per-category "
        "magnitude, with a touch of smoothing (no exact zeros). Use alpha=1 for "
        "Laplace (inflates gamma ~2-3x), alpha=0 for raw proportions. No effect "
        "with --hard-labels.",
    )
    add_sampling_args(parser)
    args = parser.parse_args()

    if args.hard_labels and args.char_file is None:
        parser.error(
            "--hard-labels only applies in category mode; pass --char-file too."
        )
    if args.alpha < 0:
        parser.error("--alpha must be >= 0 (0 = raw empirical proportions).")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = _project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_files = _abs_paths(args.candidates_file)
    success_rate_path = Path(args.success_rate_file)
    if not success_rate_path.is_absolute():
        success_rate_path = _project_root / success_rate_path

    existing = [p for p in candidate_files if p.exists()]
    if not existing:
        raise SystemExit(
            "No candidates files found:\n" + "\n".join(map(str, candidate_files))
        )

    has_category = args.char_file is not None
    counts_df = None
    if has_category:
        char_files = [p for p in _abs_paths(args.char_file) if p.exists()]
        if not char_files:
            raise SystemExit("No --char-file paths exist.")
        counts_df = build_category_counts(char_files)
        print(f"Category counts: {counts_df.height:,} classified programs.")

    eval_df = load_eval_data(existing)
    diff_df = load_success_rates(
        success_rate_path,
        value_column=args.success_rate_column,
        task_column=args.task_column,
    )
    df = merge_data(eval_df, diff_df, counts_df)

    complexity_col = "original_line_count"
    outcome = "relative_length_change" if args.relative else "rewritten_line_count"
    file_prefix = ""

    if has_category:
        cat_tag = "+ hard category" if args.hard_labels else "+ soft category"
    else:
        cat_tag = "reduced"
    print(f"\n── Regression Model: {outcome} ({cat_tag}) ──")
    fit = fit_length_model(
        df,
        outcome=outcome,
        complexity_col=complexity_col,
        has_category=has_category,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
        hard=args.hard_labels,
        alpha=args.alpha,
    )
    save_length_summaries(fit, df, complexity_col, output_dir)

    # Continuous-predictor posteriors.
    plot_scalar_posterior(
        fit.idata,
        outcome,
        "b_success_rate",
        output_dir,
        file_prefix,
        args.transparent,
        args.no_title,
    )
    plot_scalar_posterior(
        fit.idata,
        outcome,
        "b_complexity",
        output_dir,
        file_prefix,
        args.transparent,
        args.no_title,
    )

    # Category plots (category mode only).
    if has_category and fit.cat_posteriors is not None:
        prefix = _PREFIX_MAP.get(outcome, outcome)
        for cat, post in sorted(fit.cat_posteriors.items()):
            _plot_single_posterior(
                post,
                var_label=f"error_category: {cat}",
                outcome=outcome,
                output_dir=output_dir,
                stem=f"{file_prefix}{prefix}_posterior_category_{cat}",
                transparent=args.transparent,
                no_title=args.no_title,
            )
        plot_category_forest(
            fit.cat_posteriors,
            outcome,
            output_dir,
            file_prefix,
            args.transparent,
            args.no_title,
        )

    # Partial residuals. Annotate reference GT lengths only in absolute line-count mode.
    _partial_residual_plot(
        df,
        fit.idata,
        fit,
        output_dir,
        outcome,
        complexity_col,
        file_prefix,
        predictor_col="success_rate",
        slope_var="b_success_rate",
        other_slope_var="b_complexity",
        other_col=complexity_col,
        x_label="Success rate",
        transparent=args.transparent,
        no_title=args.no_title,
    )
    _partial_residual_plot(
        df,
        fit.idata,
        fit,
        output_dir,
        outcome,
        complexity_col,
        file_prefix,
        predictor_col=complexity_col,
        slope_var="b_complexity",
        other_slope_var="b_success_rate",
        other_col="success_rate",
        x_label="Reference program line count",
        annotate_at=PREDICT_AT,
        transparent=args.transparent,
        no_title=args.no_title,
    )


if __name__ == "__main__":
    main()
