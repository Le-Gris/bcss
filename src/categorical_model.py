"""Hierarchical Bayesian categorical model used by the rule-type and
error-category analyses.

The model has two levels (error -> program) or three (error -> program ->
classifier sample), a softmax link with the first category as reference, and is
sampled with NUTS (settings from ``src.bayes_defaults``). The module also
computes the reported distributions (population, marginal and
attempt-weighted), the variance decomposition across levels, HDIs and sampler
diagnostics, and plots the variance decomposition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

try:
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
except ImportError:
    az = pm = pt = None  # type: ignore[assignment]

from src.bayes_defaults import (
    DEFAULT_CHAINS,
    DEFAULT_DRAWS,
    DEFAULT_SEED,
    DEFAULT_TUNE,
    INFERENCE_METHOD,
    TARGET_ACCEPT,
)


def hdi_bounds(
    samples: np.ndarray, hdi_prob: float = 0.94
) -> tuple[np.ndarray, np.ndarray]:
    """Per-column highest-density interval bounds.

    *samples* has shape ``(n_draws, D)``; returns ``(lo, hi)`` arrays each of
    shape ``(D,)``.  Uses :func:`arviz.hdi` column-by-column so each entry is a
    genuine HDI (not a central percentile interval).
    """
    samples = np.asarray(samples)
    if samples.ndim == 1:
        samples = samples[:, None]
    d = samples.shape[1]
    lo = np.empty(d)
    hi = np.empty(d)
    for j in range(d):
        lo[j], hi[j] = az.hdi(samples[:, j], hdi_prob=hdi_prob)
    return lo, hi


def build_index_maps(
    df: Any,
    category_col: str = "error_category",
    three_level: bool = True,
) -> dict[str, Any]:
    """Build integer index arrays for the hierarchical model.

    Parameters
    ----------
    df : polars DataFrame.  Must carry an error identifier — either a ready-made
         ``error_id`` column (used as-is when present and fully populated) or
         ``task_id`` + ``grid_hash`` columns from which ``task_id:grid_hash`` is
         constructed.  Also needs *category_col*; ``program_id`` is required only
         when *three_level* is True.
    category_col : Column containing category labels.
    three_level : If True, builds program-level indices for an
                  error → program → observation hierarchy.
                  If False, each observation is a program (error → obs).

    Returns
    -------
    dict with ``y``, ``error_idx``, ``categories``, ``n_errors``,
    ``n_obs``, ``K``, ``error_ids``.
    If *three_level*: also ``program_idx``, ``n_programs``, ``program_uids``.
    """
    import polars as pl

    # Schema-aware error id: prefer an existing fully-populated ``error_id``
    # column (stage-4 and classification files), otherwise build it from
    # task_id + grid_hash for records that only carry those keys.
    if "error_id" in df.columns and df["error_id"].null_count() == 0:
        df2 = df
    else:
        df2 = df.with_columns(
            (pl.col("task_id") + ":" + pl.col("grid_hash")).alias("error_id"),
        )

    categories = sorted(df2[category_col].unique().to_list())
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    error_ids = sorted(df2["error_id"].unique().to_list())
    error_to_idx = {e: i for i, e in enumerate(error_ids)}

    y = np.array([cat_to_idx[c] for c in df2[category_col].to_list()])
    error_idx = np.array([error_to_idx[e] for e in df2["error_id"].to_list()])

    # Only meaningful when a real program identifier is present (it is absent /
    # all-null on DSL stage-4 files, where each row is itself one program).
    has_program_id = (
        "program_id" in df2.columns and df2["program_id"].null_count() < df2.height
    )
    if has_program_id:
        progs_per_error = df2.group_by("error_id").agg(pl.col("program_id").n_unique())[
            "program_id"
        ]
        print(
            f"  Programs per error: min={progs_per_error.min()}, "
            f"max={progs_per_error.max()}, mean={progs_per_error.mean():.2f}"
        )

    result: dict[str, Any] = {
        "y": y,
        "error_idx": error_idx,
        "categories": categories,
        "n_errors": len(error_ids),
        "n_obs": len(y),
        "K": len(categories),
        "error_ids": error_ids,
    }

    if three_level:
        df2 = df2.with_columns(
            (pl.col("error_id") + ":" + pl.col("program_id")).alias("program_uid"),
        )
        program_uids = sorted(df2["program_uid"].unique().to_list())
        prog_to_idx = {p: i for i, p in enumerate(program_uids)}
        program_idx = np.array([prog_to_idx[p] for p in df2["program_uid"].to_list()])

        if "sample_index" in df2.columns:
            max_sample = df2["sample_index"].max()
            print(f"  sample_index range: 0–{max_sample}")
            if max_sample == 0:
                print(
                    "  Warning: n_samples=1 — program-level and classifier "
                    "noise are confounded"
                )

        result.update(
            {
                "program_idx": program_idx,
                "n_programs": len(program_uids),
                "program_uids": program_uids,
            }
        )

    return result


def fit_categorical_model(
    idx: dict[str, Any],
    three_level: bool = True,
    draws: int = DEFAULT_DRAWS,
    tune: int = DEFAULT_TUNE,
    chains: int = DEFAULT_CHAINS,
    random_seed: int = DEFAULT_SEED,
    keep_u_raw: bool = False,
) -> "az.InferenceData":
    """Fit the hierarchical categorical model via NUTS (numpyro backend).

    Parameters
    ----------
    idx : Output of :func:`build_index_maps`.
    three_level : If True, includes program-level random effects
                  (``sigma_v``, ``v_raw``).
    keep_u_raw : If True, retains error-level random effects in the
                 posterior (needed for attempt-weighted marginals).
    """
    K = idx["K"]
    n_errors = idx["n_errors"]
    n_obs = idx["n_obs"]
    y = idx["y"]
    error_idx = idx["error_idx"]

    with pm.Model():
        # Population intercepts (K-1 free dims, category 0 = reference)
        beta0 = pm.Normal("beta0", mu=0, sigma=2.5, shape=K - 1)

        # Error-level random effects — non-centered (deterministic mu * variance )
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0, shape=K - 1)
        u_raw = pm.Normal("u_raw", mu=0, sigma=1, shape=(n_errors, K - 1))

        eta = beta0 + (u_raw * sigma_u)[error_idx]  # broadcasting

        if three_level:
            n_programs = idx["n_programs"]
            program_idx = idx["program_idx"]

            sigma_v = pm.HalfNormal("sigma_v", sigma=1.0, shape=K - 1)
            v_raw = pm.Normal("v_raw", mu=0, sigma=1, shape=(n_programs, K - 1))

            eta = eta + (v_raw * sigma_v)[program_idx]

        eta_full = pt.concatenate([pt.zeros((n_obs, 1)), eta], axis=1)
        p = pm.math.softmax(eta_full, axis=1)

        pm.Categorical("obs", p=p, observed=y)

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            target_accept=TARGET_ACCEPT,
            nuts_sampler=INFERENCE_METHOD,
        )

    # Drop large random-effect arrays to save memory
    drop_vars = ["v_raw"]
    if not keep_u_raw:
        drop_vars.append("u_raw")
    for var in drop_vars:
        if var in idata.posterior:
            del idata.posterior[var]

    return idata


def compute_population_probs(idata: "az.InferenceData", K: int) -> np.ndarray:
    """Softmax of [0, beta0] posterior draws → (n_samples, K) probabilities."""
    beta0 = idata.posterior["beta0"].values  # (chains, draws, K-1)
    flat = beta0.reshape(-1, beta0.shape[-1])  # (n_samples, K-1)

    logits = np.concatenate([np.zeros((flat.shape[0], 1)), flat], axis=1)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def compute_population_average_marginal(
    idata: "az.InferenceData",
    K: int,
    three_level: bool = True,
    n_mc: int = 2000,
    random_seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """Monte Carlo estimate of the marginal category distribution.

    This is computing the population-average marginal distribution.

    For each posterior draw, samples *n_mc* hypothetical errors
    ``z ~ N(0,1)`` and computes
    ``mean softmax([0, beta0 + z * sigma_total])``.

    When *three_level* is True:  ``sigma_total = sqrt(sigma_u² + sigma_v²)``
    When *three_level* is False: ``sigma_total = sigma_u``

    Returns array of shape ``(n_posterior_draws, K)``.
    """
    rng = np.random.default_rng(random_seed)

    beta0 = idata.posterior["beta0"].values
    sigma_u = idata.posterior["sigma_u"].values

    flat_b = beta0.reshape(-1, K - 1)
    flat_su = sigma_u.reshape(-1, K - 1)

    if three_level:
        sigma_v = idata.posterior["sigma_v"].values
        flat_sv = sigma_v.reshape(-1, K - 1)
        sigma_total = np.sqrt(flat_su**2 + flat_sv**2)
    else:
        sigma_total = flat_su

    S = flat_b.shape[0]
    z = rng.standard_normal((n_mc, K - 1))

    marginal_probs = np.empty((S, K))
    for s in range(S):
        eta = flat_b[s] + z * sigma_total[s]
        eta_full = np.concatenate([np.zeros((n_mc, 1)), eta], axis=1)
        eta_full -= eta_full.max(axis=1, keepdims=True)
        exp_eta = np.exp(eta_full)
        probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)
        marginal_probs[s] = probs.mean(axis=0)

    return marginal_probs


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical stability."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def compute_attempt_weighted_sample_marginal(
    idata: "az.InferenceData",
    idx: dict[str, Any],
    attempt_counts: np.ndarray,
) -> np.ndarray:
    """Attempt-weighted marginal from per-error posterior random effects.

    This is computing the sample-average marginal reweighted by attempt count.

    For each posterior draw *s*, computes each error's rule type probability
    from its realized random effect, then takes the attempt-weighted average:

        P_weighted(k) = sum_e  w_e * softmax([0, beta0_s + sigma_u_s * u_raw_{e,s}])_k

    where ``w_e = attempt_count_e / sum(attempt_counts)``.

    Requires ``u_raw`` in the posterior (use ``keep_u_raw=True`` when
    fitting).

    Parameters
    ----------
    idata : InferenceData with ``beta0``, ``sigma_u``, and ``u_raw``.
    idx : Output of :func:`build_index_maps`.
    attempt_counts : Array of shape ``(n_errors,)`` with per-error attempt
                     counts, ordered to match ``idx["error_ids"]``.

    Returns
    -------
    Array of shape ``(n_posterior_draws, K)``.
    """
    if "u_raw" not in idata.posterior:
        raise ValueError("u_raw not in posterior — refit with keep_u_raw=True")

    K = idx["K"]
    n_errors = idx["n_errors"]

    beta0 = idata.posterior["beta0"].values  # (chains, draws, K-1)
    sigma_u = idata.posterior["sigma_u"].values
    u_raw = idata.posterior["u_raw"].values  # (chains, draws, n_errors, K-1)

    flat_b = beta0.reshape(-1, K - 1)  # (S, K-1)
    flat_su = sigma_u.reshape(-1, K - 1)
    flat_u = u_raw.reshape(-1, n_errors, K - 1)  # (S, n_errors, K-1)

    S = flat_b.shape[0]
    weights = attempt_counts / attempt_counts.sum()  # (n_errors,)

    # For each draw: eta_e = beta0 + sigma_u * u_raw_e → softmax → weighted avg
    # Vectorised over errors: (S, n_errors, K-1)
    eta = flat_b[:, None, :] + flat_su[:, None, :] * flat_u
    # Prepend reference column of zeros → (S, n_errors, K)
    eta_full = np.concatenate([np.zeros((S, n_errors, 1)), eta], axis=2)
    p = _softmax(eta_full)  # (S, n_errors, K)

    # Attempt-weighted average over errors
    weighted = (weights[None, :, None] * p).sum(axis=1)  # (S, K)
    return weighted


def print_diagnostics(
    idata: "az.InferenceData",
    idx: dict[str, Any],
    three_level: bool = True,
) -> None:
    """Print MCMC diagnostic summary."""
    print("\nData dimensions:")
    print(f"  Errors:     {idx['n_errors']}")
    if three_level:
        print(f"  Programs:   {idx['n_programs']}")
    print(f"  Obs:        {idx['n_obs']}")
    print(f"  Categories: {idx['K']} ({', '.join(idx['categories'])})")
    print(f"  Reference:  {idx['categories'][0]}")

    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        divs = int(idata.sample_stats["diverging"].sum().item())
        print(f"\n  Divergences: {divs}")

    var_names = ["beta0", "sigma_u"]
    if three_level:
        var_names.append("sigma_v")

    summary = az.summary(idata, var_names=var_names)
    print(
        f"\n  R-hat range: [{summary['r_hat'].min():.4f}, {summary['r_hat'].max():.4f}]"
    )
    print(f"  ESS bulk min: {summary['ess_bulk'].min():.0f}")
    print(f"  ESS tail min: {summary['ess_tail'].min():.0f}")
    print()
    print(summary.to_string())


def print_cluster_sizes(idx: dict[str, Any]) -> None:
    """Observations per error, plus a model-free check for between-error signal.

    sigma_u is only informed by errors with >= 2 observations: with a single
    categorical draw per error, between- and within-error variation are not
    separable and the sigma_u posterior falls back on its prior.  Intended for
    2-level fits; for 3-level fits the relevant count is programs per error
    (already printed by :func:`build_index_maps`).
    """
    y = idx["y"]
    error_idx = idx["error_idx"]
    K = idx["K"]
    sizes = np.bincount(error_idx, minlength=idx["n_errors"])

    print("\nObservations per error:")
    print(
        f"  min={sizes.min()}, median={np.median(sizes):.0f}, "
        f"mean={sizes.mean():.2f}, max={sizes.max()}"
    )
    for n in range(1, 5):
        c = int((sizes == n).sum())
        print(f"  n={n}:  {c:>5d} errors ({c / len(sizes):.1%})")
    c = int((sizes >= 5).sum())
    print(f"  n>=5: {c:>5d} errors ({c / len(sizes):.1%})")

    multi = np.flatnonzero(sizes >= 2)
    if multi.size == 0:
        print("  No error has >= 2 observations — sigma_u is prior-only.")
        return

    # Within-error agreement vs. what independent draws from the pooled
    # category frequencies would give (sum_k p_k^n for an error of size n).
    p = np.bincount(y, minlength=K) / len(y)
    counts = np.zeros((idx["n_errors"], K), dtype=int)
    np.add.at(counts, (error_idx, y), 1)
    observed = (counts[multi].max(axis=1) == sizes[multi]).mean()
    expected = np.mean([(p**n).sum() for n in sizes[multi]])
    print(
        f"  Errors with >=2 obs: {multi.size}; all-same-category rate "
        f"{observed:.1%} vs. {expected:.1%} expected with no clustering"
    )


# ── Latent-response variance decomposition (Snijders & Bosker 2012) ───────────

# For categorical logit models, the observation error in the latent-response
# formulation follows a standard logistic distribution with fixed variance π²/3.
# Reference: Snijders & Bosker (2012), Chapter 17.3.3; Goldstein et al. (2002).
LATENT_LOGISTIC_VAR: float = (np.pi**2) / 3.0  # ≈ 3.289868
LATENT_LOGISTIC_SD: float = np.pi / np.sqrt(3.0)  # ≈ 1.813799


def compute_latent_vpcs(
    idata: "az.InferenceData",
    three_level: bool = False,
) -> dict[str, np.ndarray]:
    """Compute latent-response Variance Partition Coefficients (VPCs).

    Appeals to the latent-response formulation for categorical logit models
    (Snijders & Bosker 2012, Chapter 17.3.3; Goldstein et al. 2002; Hedeker 2003).
    Each of the K-1 non-reference categories has an underlying continuous
    propensity with standard logistic disturbance variance π² / 3 ≈ 3.29.

    Parameters
    ----------
    idata : MCMC posterior InferenceData containing ``sigma_u`` (and optionally
            ``sigma_v`` if *three_level* is True).
    three_level : If True and ``sigma_v`` is present in the posterior, partitions
                  variance across error, program, and latent residual levels.

    Returns
    -------
    dict with keys:
      - ``"error"``: array of shape ``(n_draws, K - 1)``
      - ``"program"``: array of shape ``(n_draws, K - 1)`` (only if three_level and sigma_v present)
      - ``"residual"``: array of shape ``(n_draws, K - 1)``
    """
    sigma_u = np.asarray(idata.posterior["sigma_u"].values)
    if sigma_u.ndim == 3:
        su = sigma_u.reshape(-1, sigma_u.shape[-1])
    elif sigma_u.ndim == 1:
        su = sigma_u[:, None]
    else:
        su = sigma_u

    var_u = su**2

    has_v = bool(
        three_level
        and hasattr(idata, "posterior")
        and "sigma_v" in idata.posterior
    )
    if has_v:
        sigma_v = np.asarray(idata.posterior["sigma_v"].values)
        if sigma_v.ndim == 3:
            sv = sigma_v.reshape(-1, sigma_v.shape[-1])
        elif sigma_v.ndim == 1:
            sv = sigma_v[:, None]
        else:
            sv = sigma_v
        var_v = sv**2
    else:
        var_v = np.zeros_like(var_u)

    total_var = var_u + var_v + LATENT_LOGISTIC_VAR

    res: dict[str, np.ndarray] = {
        "error": var_u / total_var,
    }
    if has_v:
        res["program"] = var_v / total_var
    res["residual"] = np.broadcast_to(
        LATENT_LOGISTIC_VAR / total_var,
        var_u.shape,
    ).copy()

    return res


def print_latent_vpcs(
    idata: "az.InferenceData",
    categories: list[str],
    three_level: bool = False,
    hdi_prob: float = 0.94,
    short_names: dict[str, str] | None = None,
) -> None:
    """Print the latent-response variance decomposition (VPC) for each category.

    Parameters
    ----------
    idata : MCMC posterior InferenceData.
    categories : Full list of K categories. The first category (index 0) is the
                 reference level (eta = 0).
    three_level : Whether the model includes program-level random effects.
    hdi_prob : Probability mass for highest density intervals (default 0.94).
    short_names : Optional mapping from category names to compact display labels.
    """
    if short_names is None:
        short_names = {}

    vpcs = compute_latent_vpcs(idata, three_level=three_level)
    non_ref = categories[1:]
    ref_label = short_names.get(categories[0], categories[0])

    sigma_u = np.asarray(idata.posterior["sigma_u"].values)
    su = sigma_u.reshape(-1, sigma_u.shape[-1]) if sigma_u.ndim == 3 else sigma_u
    if su.ndim == 1:
        su = su[:, None]

    has_v = "program" in vpcs
    sv: np.ndarray | None = None
    if has_v:
        sigma_v = np.asarray(idata.posterior["sigma_v"].values)
        sv = sigma_v.reshape(-1, sigma_v.shape[-1]) if sigma_v.ndim == 3 else sigma_v
        if sv.ndim == 1:
            sv = sv[:, None]

    label_w = 34
    scale_w = 22
    vpc_w = 26
    rule = "─" * (label_w + scale_w + vpc_w + 14)

    print(
        f"\nLatent-response variance decomposition (Snijders & Bosker 2012, ref: '{ref_label}'):"
    )

    for i, cat in enumerate(non_ref):
        cat_label = short_names.get(cat, cat)
        print(f"\n  Dimension: {cat_label} (vs. {ref_label})")
        header = (
            f"    {'Variance Component':<{label_w}}"
            f"{f'Scale σ [{hdi_prob:.0%} HDI]':<{scale_w}}"
            f"{f'VPC % [{hdi_prob:.0%} HDI]':<{vpc_w}}"
        )
        print(header)
        print("    " + rule)

        # 1. Error component
        u_lo, u_hi = az.hdi(su[:, i], hdi_prob=hdi_prob)
        u_med = float(np.median(su[:, i]))
        u_stat = f"{u_med:.2f} [{u_lo:.2f}, {u_hi:.2f}]"

        evpc = vpcs["error"][:, i]
        evpc_lo, evpc_hi = az.hdi(evpc, hdi_prob=hdi_prob)
        evpc_med = float(np.median(evpc))
        evpc_stat = f"{evpc_med * 100:>5.1f}% [{evpc_lo * 100:>4.1f}%, {evpc_hi * 100:>4.1f}%]"
        bar = "■" * max(0, round(evpc_med * 20))
        print(
            f"    {'σ_u (Between-error)':<{label_w}}"
            f"{u_stat:<{scale_w}}"
            f"{evpc_stat:<{vpc_w}}  {bar}"
        )

        # 2. Program component (if 3-level)
        if has_v and sv is not None:
            v_lo, v_hi = az.hdi(sv[:, i], hdi_prob=hdi_prob)
            v_med = float(np.median(sv[:, i]))
            v_stat = f"{v_med:.2f} [{v_lo:.2f}, {v_hi:.2f}]"

            pvpc = vpcs["program"][:, i]
            pvpc_lo, pvpc_hi = az.hdi(pvpc, hdi_prob=hdi_prob)
            pvpc_med = float(np.median(pvpc))
            pvpc_stat = f"{pvpc_med * 100:>5.1f}% [{pvpc_lo * 100:>4.1f}%, {pvpc_hi * 100:>4.1f}%]"
            pbar = "■" * max(0, round(pvpc_med * 20))
            print(
                f"    {'σ_v (Between-program)':<{label_w}}"
                f"{v_stat:<{scale_w}}"
                f"{pvpc_stat:<{vpc_w}}  {pbar}"
            )

        # 3. Residual component
        res_stat = f"{LATENT_LOGISTIC_SD:.2f} [fixed]"
        rvpc = vpcs["residual"][:, i]
        rvpc_lo, rvpc_hi = az.hdi(rvpc, hdi_prob=hdi_prob)
        rvpc_med = float(np.median(rvpc))
        rvpc_stat = f"{rvpc_med * 100:>5.1f}% [{rvpc_lo * 100:>4.1f}%, {rvpc_hi * 100:>4.1f}%]"
        rbar = "■" * max(0, round(rvpc_med * 20))
        print(
            f"    {'σ_ε (Logistic noise π/√3)':<{label_w}}"
            f"{res_stat:<{scale_w}}"
            f"{rvpc_stat:<{vpc_w}}  {rbar}"
        )
        print("    " + rule)


def plot_variance_decomposition(
    idata: "az.InferenceData",
    categories: list[str],
    output_path: Path,
    short_names: dict[str, str] | None = None,
    transparent: bool = False,
) -> plt.Figure:
    """Two-panel figure: (a) sigma_u vs sigma_v, (b) reliability ratio.

    Only meaningful for 3-level models where both ``sigma_u`` and
    ``sigma_v`` exist in the posterior.

    Parameters
    ----------
    short_names : Optional mapping from category names to display labels.
    """
    if short_names is None:
        short_names = {}

    sigma_u = idata.posterior["sigma_u"].values  # (chains, draws, K-1)
    sigma_v = idata.posterior["sigma_v"].values

    su = sigma_u.reshape(-1, sigma_u.shape[-1])
    sv = sigma_v.reshape(-1, sigma_v.shape[-1])

    su_med = np.median(su, axis=0)
    sv_med = np.median(sv, axis=0)
    su_lo, su_hi = hdi_bounds(su)
    sv_lo, sv_hi = hdi_bounds(sv)

    rel = su / (su + sv)
    rel_med = np.median(rel, axis=0)
    rel_lo, rel_hi = hdi_bounds(rel)

    non_ref = categories[1:]
    labels = [short_names.get(c, c) for c in non_ref]
    y_pos = np.arange(len(labels))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel (a): sigma_u vs sigma_v
    ax1.barh(
        y_pos + 0.15,
        su_med,
        height=0.3,
        color="#4C72B0",
        label="σ_u (error)",
    )
    ax1.barh(
        y_pos - 0.15,
        sv_med,
        height=0.3,
        color="#DD8452",
        label="σ_v (program)",
    )
    ax1.errorbar(
        su_med,
        y_pos + 0.15,
        xerr=[su_med - su_lo, su_hi - su_med],
        fmt="none",
        color="0.3",
        capsize=2,
        lw=1,
    )
    ax1.errorbar(
        sv_med,
        y_pos - 0.15,
        xerr=[sv_med - sv_lo, sv_hi - sv_med],
        fmt="none",
        color="0.3",
        capsize=2,
        lw=1,
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Random effect scale (σ)")
    ax1.set_title("(a) Variance Components")
    ax1.legend(loc="lower right")

    # Panel (b): reliability ratio
    ax2.barh(y_pos, rel_med, color="#55A868", edgecolor="white", linewidth=0.5)
    ax2.errorbar(
        rel_med,
        y_pos,
        xerr=[rel_med - rel_lo, rel_hi - rel_med],
        fmt="none",
        color="0.3",
        capsize=2,
        lw=1,
    )
    ax2.axvline(0.5, color="0.6", ls="--", lw=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("σ_u / (σ_u + σ_v)")
    ax2.set_title("(b) Characterization Reliability")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=transparent)
    print(f"Saved figure to {output_path}")

    print("\nReliability (σ_u / (σ_u + σ_v)) per category dimension:")
    for label, r, lo, hi in zip(labels, rel_med, rel_lo, rel_hi):
        print(f"  {label:<20s}: {r:.3f}  [{lo:.3f}–{hi:.3f}]")

    return fig
