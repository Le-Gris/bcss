"""Unit tests for ``src/categorical_model.py``.

Covers the data-schema coupling (``build_index_maps``) and the three
posterior estimands, all without running NUTS — the estimands operate on a
hand-built ``idata``-shaped stub so the tests stay fast and deterministic.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.categorical_model import (
    LATENT_LOGISTIC_VAR,
    build_index_maps,
    compute_attempt_weighted_sample_marginal,
    compute_latent_vpcs,
    compute_population_average_marginal,
    compute_population_probs,
    hdi_bounds,
    print_latent_vpcs,
)


# ── build_index_maps: schema awareness ────────────────────────────────────


def test_build_index_maps_uses_existing_error_id():
    # DSL stage-4 schema: ready-made error_id, no grid_hash/program_id.
    df = pl.DataFrame(
        {
            "task_id": ["t1", "t1", "t2"],
            "error_id": ["e_a", "e_a", "e_b"],
            "program_id": [None, None, None],
            "program_rule_type": ["Neither", "Alt. rule", "Neither"],
        }
    )
    idx = build_index_maps(df, category_col="program_rule_type", three_level=False)

    assert idx["error_ids"] == ["e_a", "e_b"]
    assert idx["n_errors"] == 2
    assert idx["n_obs"] == 3
    assert idx["categories"] == ["Alt. rule", "Neither"]  # alphabetical
    assert idx["K"] == 2
    # Reference (index 0) is the alphabetically-first category.
    assert idx["categories"][0] == "Alt. rule"
    # error_idx groups the two t1 rows under the same error.
    assert list(idx["error_idx"]) == [0, 0, 1]
    # y maps category labels to their sorted index.
    assert list(idx["y"]) == [1, 0, 1]


def test_build_index_maps_constructs_error_id_from_grid_hash():
    # Records keyed by task_id + grid_hash only (no error_id column).
    df = pl.DataFrame(
        {
            "task_id": ["t1", "t1", "t2"],
            "grid_hash": ["h1", "h1", "h2"],
            "program_id": ["p0", "p1", "p0"],
            "program_rule_type": ["Neither", "Neither", "Partial match"],
        }
    )
    idx = build_index_maps(df, category_col="program_rule_type", three_level=False)

    assert idx["error_ids"] == ["t1:h1", "t2:h2"]
    assert idx["n_errors"] == 2
    assert list(idx["error_idx"]) == [0, 0, 1]


def test_build_index_maps_three_level_builds_program_index():
    df = pl.DataFrame(
        {
            "task_id": ["t1", "t1", "t1"],
            "grid_hash": ["h1", "h1", "h1"],
            "program_id": ["p0", "p0", "p1"],
            "program_rule_type": ["Neither", "Neither", "Alt. rule"],
        }
    )
    idx = build_index_maps(df, category_col="program_rule_type", three_level=True)

    assert idx["n_errors"] == 1
    # Two distinct programs under the one error.
    assert idx["n_programs"] == 2
    assert idx["program_uids"] == ["t1:h1:p0", "t1:h1:p1"]
    assert list(idx["program_idx"]) == [0, 0, 1]


# ── hdi_bounds ─────────────────────────────────────────────────────────────


def test_hdi_bounds_shape_and_ordering():
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(2000, 3)) + np.array([0.0, 5.0, -3.0])
    lo, hi = hdi_bounds(samples, hdi_prob=0.9)

    assert lo.shape == (3,)
    assert hi.shape == (3,)
    assert np.all(lo < hi)
    # Bounds should bracket each column's median.
    med = np.median(samples, axis=0)
    assert np.all(lo < med) and np.all(med < hi)


def test_hdi_bounds_handles_1d():
    rng = np.random.default_rng(1)
    lo, hi = hdi_bounds(rng.normal(size=500))
    assert lo.shape == (1,) and hi.shape == (1,)


# ── Estimands on a stub posterior ─────────────────────────────────────────


class _Var:
    """Minimal stand-in for an xarray DataArray exposing ``.values``."""

    def __init__(self, arr: np.ndarray) -> None:
        self.values = arr


class _Idata:
    """Minimal stand-in for arviz InferenceData (only ``.posterior`` used)."""

    def __init__(self, posterior: dict[str, _Var]) -> None:
        self.posterior = posterior


def _make_idata(
    K: int,
    n_errors: int,
    chains: int = 2,
    draws: int = 50,
    seed: int = 0,
    three_level: bool = False,
):
    rng = np.random.default_rng(seed)
    beta0 = rng.normal(size=(chains, draws, K - 1))
    sigma_u = np.abs(rng.normal(size=(chains, draws, K - 1))) + 0.1
    u_raw = rng.normal(size=(chains, draws, n_errors, K - 1))
    posterior = {
        "beta0": _Var(beta0),
        "sigma_u": _Var(sigma_u),
        "u_raw": _Var(u_raw),
    }
    if three_level:
        sigma_v = np.abs(rng.normal(size=(chains, draws, K - 1))) + 0.1
        posterior["sigma_v"] = _Var(sigma_v)
    return _Idata(posterior)


def test_population_probs_are_valid_simplex():
    K = 3
    idata = _make_idata(K, n_errors=5)
    probs = compute_population_probs(idata, K)

    assert probs.shape == (2 * 50, K)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-10)
    assert np.all(probs >= 0)


def test_population_average_marginal_is_valid_simplex():
    K = 3
    idata = _make_idata(K, n_errors=5)
    marg = compute_population_average_marginal(
        idata, K, three_level=False, n_mc=200, random_seed=0
    )
    assert marg.shape == (2 * 50, K)
    np.testing.assert_allclose(marg.sum(axis=1), 1.0, atol=1e-10)


def test_population_average_marginal_three_level_is_valid_simplex():
    K = 3
    idata = _make_idata(K, n_errors=5, three_level=True)
    marg = compute_population_average_marginal(
        idata, K, three_level=True, n_mc=200, random_seed=0
    )
    assert marg.shape == (2 * 50, K)
    np.testing.assert_allclose(marg.sum(axis=1), 1.0, atol=1e-10)


def test_attempt_weighted_marginal_weights_and_simplex():
    K = 3
    n_errors = 4
    idata = _make_idata(K, n_errors=n_errors)
    idx = {"K": K, "n_errors": n_errors}
    counts = np.array([1.0, 3.0, 2.0, 4.0])

    aw = compute_attempt_weighted_sample_marginal(idata, idx, counts)
    assert aw.shape == (2 * 50, K)
    np.testing.assert_allclose(aw.sum(axis=1), 1.0, atol=1e-10)


def test_attempt_weighted_requires_u_raw():
    idata = _Idata({"beta0": _Var(np.zeros((2, 5, 2))), "sigma_u": _Var(np.ones((2, 5, 2)))})
    with pytest.raises(ValueError, match="u_raw"):
        compute_attempt_weighted_sample_marginal(
            idata, {"K": 3, "n_errors": 4}, np.ones(4)
        )


# ── Latent-response variance decomposition (VPCs) ──────────────────────────


def test_compute_latent_vpcs_two_level():
    K = 3
    idata = _make_idata(K, n_errors=5, chains=2, draws=50, three_level=False)
    vpcs = compute_latent_vpcs(idata, three_level=False)

    assert set(vpcs.keys()) == {"error", "residual"}
    assert vpcs["error"].shape == (100, K - 1)
    assert vpcs["residual"].shape == (100, K - 1)

    # Sum of variance components must be 1.0 for every draw and category
    total = vpcs["error"] + vpcs["residual"]
    np.testing.assert_allclose(total, 1.0, atol=1e-12)

    # Check bounds
    assert np.all(vpcs["error"] >= 0) and np.all(vpcs["error"] <= 1)
    assert np.all(vpcs["residual"] >= 0) and np.all(vpcs["residual"] <= 1)

    # Check analytical formula for the first entry
    su = idata.posterior["sigma_u"].values.reshape(-1, K - 1)
    expected_error_0 = (su[:, 0] ** 2) / (su[:, 0] ** 2 + LATENT_LOGISTIC_VAR)
    np.testing.assert_allclose(vpcs["error"][:, 0], expected_error_0)


def test_compute_latent_vpcs_three_level():
    K = 4
    idata = _make_idata(K, n_errors=5, chains=2, draws=50, three_level=True)
    vpcs = compute_latent_vpcs(idata, three_level=True)

    assert set(vpcs.keys()) == {"error", "program", "residual"}
    for key in ("error", "program", "residual"):
        assert vpcs[key].shape == (100, K - 1)
        assert np.all(vpcs[key] >= 0) and np.all(vpcs[key] <= 1)

    # Sum of variance components must be 1.0 for every draw and category
    total = vpcs["error"] + vpcs["program"] + vpcs["residual"]
    np.testing.assert_allclose(total, 1.0, atol=1e-12)

    # Check analytical formula
    su = idata.posterior["sigma_u"].values.reshape(-1, K - 1)
    sv = idata.posterior["sigma_v"].values.reshape(-1, K - 1)
    tot = su**2 + sv**2 + LATENT_LOGISTIC_VAR
    np.testing.assert_allclose(vpcs["error"], su**2 / tot)
    np.testing.assert_allclose(vpcs["program"], sv**2 / tot)
    np.testing.assert_allclose(vpcs["residual"], LATENT_LOGISTIC_VAR / tot)


def test_print_latent_vpcs_smoke(capsys):
    K = 3
    categories = ["Alt. rule", "Neither", "Partial match"]
    idata_2l = _make_idata(K, n_errors=5, chains=2, draws=30, three_level=False)
    print_latent_vpcs(idata_2l, categories, three_level=False)
    out_2l = capsys.readouterr().out
    assert "Latent-response variance decomposition" in out_2l
    assert "Neither" in out_2l
    assert "Partial match" in out_2l
    assert "σ_u (Between-error)" in out_2l
    assert "σ_ε (Logistic noise" in out_2l
    assert "σ_v (Between-program)" not in out_2l

    idata_3l = _make_idata(K, n_errors=5, chains=2, draws=30, three_level=True)
    print_latent_vpcs(idata_3l, categories, three_level=True, short_names={"Neither": "N"})
    out_3l = capsys.readouterr().out
    assert "σ_v (Between-program)" in out_3l
    assert "Dimension: N (vs. Alt. rule)" in out_3l
