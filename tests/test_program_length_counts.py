"""Unit tests for the pure helpers in
``analyses.analyze_program_length_by_difficulty``: the category count-tally
(``build_category_counts``) and the generic per-task success-rate loader
(``load_success_rates``).
"""

import json
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_program_length_by_difficulty import (
    CAT_ORDER,
    UNCLASSIFIED,
    build_category_counts,
    load_success_rates,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return path


def _rec(error_id: str, program_id: str, category) -> dict:
    return {
        "error_id": error_id,
        "program_id": program_id,
        "characterization_result": {"error_category": category},
    }


def test_tally_preserves_split_and_folds_unknown(tmp_path):
    # One program classified 5x: 2 attribute_substitution, 1 unclassifiable,
    # 1 None (folds to unclassifiable), 1 unknown label (folds to unclassifiable).
    recs = [
        _rec("e1", "p1", "attribute_substitution"),
        _rec("e1", "p1", "attribute_substitution"),
        _rec("e1", "p1", "unclassifiable"),
        _rec("e1", "p1", None),
        _rec("e1", "p1", "not_a_real_category"),
    ]
    df = build_category_counts([_write_jsonl(tmp_path / "c.jsonl", recs)])

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["error_id"] == "e1" and row["program_id"] == "p1"
    assert row["count_attribute_substitution"] == 2
    assert row[f"count_{UNCLASSIFIED}"] == 3  # 1 explicit + None + unknown
    assert row["n_samples"] == 5
    # Every category column is present and non-negative; they sum to n_samples.
    assert sum(row[f"count_{c}"] for c in CAT_ORDER) == row["n_samples"]


def test_separate_programs_are_distinct_rows(tmp_path):
    recs = [
        _rec("e1", "p1", "scope_restriction"),
        _rec("e1", "p2", "operator_confusion"),  # same error, different program
        _rec("e2", "p3", "partial_program"),
    ]
    df = build_category_counts([_write_jsonl(tmp_path / "c.jsonl", recs)])
    assert df.height == 3
    keys = set(zip(df["error_id"].to_list(), df["program_id"].to_list()))
    assert keys == {("e1", "p1"), ("e1", "p2"), ("e2", "p3")}


def test_records_missing_ids_are_skipped(tmp_path):
    recs = [
        _rec("e1", "p1", "boundary_precision"),
        {"program_id": "p9", "characterization_result": {"error_category": "x"}},  # no error_id
        {"error_id": "e9", "characterization_result": {"error_category": "x"}},  # no program_id
    ]
    df = build_category_counts([_write_jsonl(tmp_path / "c.jsonl", recs)])
    assert df.height == 1
    assert df.row(0, named=True)["count_boundary_precision"] == 1


# ---------------------------------------------------------------------------
# load_success_rates — generic over the solver (human IRT file or any per-task
# success rate, e.g. an LLM's pass rate).
# ---------------------------------------------------------------------------


def test_success_rates_generic_task_id_and_custom_column(tmp_path):
    path = tmp_path / "model_rates.csv"
    pl.DataFrame(
        {"task_id": ["a1", "b2"], "model_pass_rate": [0.1, 0.9]}
    ).write_csv(path)
    df = load_success_rates(path, value_column="model_pass_rate")
    assert df.columns == ["task_id", "success_rate"]
    assert dict(zip(df["task_id"], df["success_rate"])) == {"a1": 0.1, "b2": 0.9}


def test_success_rates_harc_tasks_column_strips_json(tmp_path):
    path = tmp_path / "harc.csv"
    pl.DataFrame({"tasks": ["a1.json", "b2.json"], "success_rate": [0.2, 0.7]}).write_csv(path)
    df = load_success_rates(path)  # auto-detects 'tasks', default value column
    assert set(df["task_id"]) == {"a1", "b2"}  # .json stripped


def test_success_rates_explicit_task_column(tmp_path):
    path = tmp_path / "custom.csv"
    pl.DataFrame({"puzzle": ["p1"], "rate": [0.5]}).write_csv(path)
    df = load_success_rates(path, value_column="rate", task_column="puzzle")
    assert df.row(0, named=True) == {"task_id": "p1", "success_rate": 0.5}


def test_success_rates_missing_value_column_errors(tmp_path):
    path = tmp_path / "x.csv"
    pl.DataFrame({"task_id": ["a"], "foo": [1.0]}).write_csv(path)
    with pytest.raises(SystemExit):
        load_success_rates(path)  # default 'success_rate' column absent


def test_success_rates_no_task_column_errors(tmp_path):
    path = tmp_path / "x.csv"
    pl.DataFrame({"weird": ["a"], "success_rate": [0.5]}).write_csv(path)
    with pytest.raises(SystemExit):
        load_success_rates(path)  # neither task_id nor tasks


# ---------------------------------------------------------------------------
# fit_length_model — soft (theta.gamma) smoke fit. Guards the linear-Gaussian
# geometry fix: the old pm.Mixture soft path did not mix on the relative
# outcome (r_hat ~2.85). A fit on well-behaved synthetic data must converge.
# ---------------------------------------------------------------------------


def test_fit_length_model_soft_smoke():
    pytest.importorskip("pymc")
    pytest.importorskip("numpyro")
    import arviz as az
    import numpy as np

    from analyses.analyze_program_length_by_difficulty import (
        FitResult,
        fit_length_model,
    )

    rng = np.random.default_rng(0)
    K = len(CAT_ORDER)
    n_errors, programs_per_error = 40, 2
    n = n_errors * programs_per_error

    true_gamma = np.linspace(-0.3, 0.3, K)  # small effects, relative scale
    err_idx = np.repeat(np.arange(n_errors), programs_per_error)
    u_true = rng.normal(0, 0.05, size=n_errors)
    true_cat = rng.integers(0, K, size=n)
    success_rate = rng.uniform(0, 1, size=n)
    original_line_count = rng.integers(4, 30, size=n).astype(float)

    y = (
        -0.15
        + 0.05 * (success_rate - success_rate.mean())
        - 0.004 * (original_line_count - original_line_count.mean())
        + u_true[err_idx]
        + true_gamma[true_cat]
        + rng.normal(0, 0.03, size=n)
    )

    # 5-sample count vectors peaked on the true category (so theta carries it).
    counts = np.zeros((n, K), dtype=int)
    for i, k in enumerate(true_cat):
        counts[i, k] = 4
        counts[i, (k + 1) % K] = 1

    data = {
        "relative_length_change": y,
        "success_rate": success_rate,
        "original_line_count": original_line_count,
        "task_err": [f"e{e}" for e in err_idx],
    }
    for j, c in enumerate(CAT_ORDER):
        data[f"count_{c}"] = counts[:, j]
    df = pl.DataFrame(data)

    fit = fit_length_model(
        df,
        outcome="relative_length_change",
        complexity_col="original_line_count",
        has_category=True,
        draws=500,
        tune=500,
        chains=2,
        random_seed=0,
        hard=False,
    )

    assert isinstance(fit, FitResult)
    post = fit.idata.posterior
    assert "Intercept" in post and "gamma" in post
    assert post["gamma"].sizes["category"] == K
    gamma_mean = post["gamma"].mean(("chain", "draw")).values
    assert np.all(np.isfinite(gamma_mean))

    # Geometry guard: the linear theta.gamma model mixes where the mixture did not.
    summ = az.summary(
        fit.idata,
        var_names=["Intercept", "b_success_rate", "b_complexity",
                   "sigma_u", "sigma_y", "gamma"],
    )
    assert summ["r_hat"].max() < 1.15


# ---------------------------------------------------------------------------
# save_length_summaries — reported estimands from a known posterior (no fit).
# ---------------------------------------------------------------------------


def test_save_length_summaries_matches_hand_computation(tmp_path):
    az = pytest.importorskip("arviz")
    import numpy as np

    from analyses.analyze_program_length_by_difficulty import (
        FitResult,
        save_length_summaries,
    )

    rng = np.random.default_rng(1)
    K = len(CAT_ORDER)
    b0 = rng.normal(-0.09, 0.01, size=(2, 200))
    slope = rng.normal(-0.005, 0.0005, size=(2, 200))
    sr = rng.normal(0.02, 0.02, size=(2, 200))
    gamma = rng.normal(0, 0.1, size=(2, 200, K))
    idata = az.from_dict(
        posterior={
            "Intercept": b0,
            "b_complexity": slope,
            "b_success_rate": sr,
            "gamma": gamma,
        },
        coords={"category": CAT_ORDER},
        dims={"gamma": ["category"]},
    )
    center = 14.1
    fit = FitResult(
        idata,
        {"success_rate": 0.6, "original_line_count": center},
        re=np.zeros(1),
        cat_effect=np.zeros(1),
        cat_posteriors=None,
    )
    df = pl.DataFrame({"task_err": ["a", "a", "b"]})

    save_length_summaries(fit, df, "original_line_count", tmp_path)

    data = dict(pl.read_csv(tmp_path / "length_data_summary.csv").iter_rows())
    assert data["n_rows"] == 3 and data["n_errors"] == 2
    assert data["center_original_line_count"] == pytest.approx(center)

    coef = {r["term"]: r for r in pl.read_csv(tmp_path / "length_coefficients_summary.csv").to_dicts()}
    assert coef["b_complexity"]["mean"] == pytest.approx(slope.mean())
    pred7 = (b0 + slope * (7 - center)).flatten()
    lo, hi = az.hdi(pred7, hdi_prob=0.94)
    assert coef["predicted_at_L7"]["mean"] == pytest.approx(pred7.mean())
    assert coef["predicted_at_L7"]["hdi_lower"] == pytest.approx(lo)
    assert coef["predicted_at_L7"]["hdi_upper"] == pytest.approx(hi)

    cats = {r["category"]: r for r in pl.read_csv(tmp_path / "length_category_summary.csv").to_dicts()}
    assert list(cats) == CAT_ORDER
    j = CAT_ORDER.index("partial_program")
    g = gamma[..., j].flatten()
    assert cats["partial_program"]["gamma_mean"] == pytest.approx(g.mean())
    assert cats["partial_program"]["gamma_p_gt0"] == pytest.approx((g > 0).mean())
    assert cats["partial_program"]["predicted_mean"] == pytest.approx((b0.flatten() + g).mean())
