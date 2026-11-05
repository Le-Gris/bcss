"""Checks the summary CSVs that ``plot_condition_means`` in
``analyses.analyze_program_simplicity`` writes: the per-condition posterior
means and the treatment − control contrast, each with a 94% HDI.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_condition_summaries_match_hand_computation(tmp_path):
    az = pytest.importorskip("arviz")
    pytest.importorskip("bambi")
    import matplotlib

    matplotlib.use("Agg")
    import numpy as np

    from analyses.analyze_program_simplicity import plot_condition_means

    rng = np.random.default_rng(0)
    intercept = rng.normal(-0.05, 0.01, size=(2, 300))
    contrast = rng.normal(-0.14, 0.012, size=(2, 300))
    idata = az.from_dict(
        posterior={"Intercept": intercept, "condition[treatment]": contrast}
    )
    outcome = "relative_length_change"
    df = pl.DataFrame(
        {
            outcome: rng.normal(-0.1, 0.1, size=7).tolist(),
            "condition": ["control"] * 3 + ["treatment"] * 4,
        }
    )

    plot_condition_means(df, idata, tmp_path, outcome)

    b0 = intercept.flatten()
    b1 = contrast.flatten()

    means = pl.read_csv(tmp_path / "rlc_condition_means_summary.csv")
    assert means.columns == [
        "outcome", "condition", "mean", "hdi_low", "hdi_high", "hdi_prob", "n",
    ]
    assert means["condition"].to_list() == ["control", "treatment"]
    assert means["n"].to_list() == [3, 4]
    for row, draws in zip(means.iter_rows(named=True), (b0, b0 + b1)):
        lo, hi = az.hdi(draws, hdi_prob=0.94)
        assert row["outcome"] == outcome
        assert row["mean"] == pytest.approx(draws.mean())
        assert (row["hdi_low"], row["hdi_high"]) == pytest.approx((lo, hi))
        assert row["hdi_prob"] == 0.94

    contrast_df = pl.read_csv(tmp_path / "rlc_condition_contrast_summary.csv")
    assert contrast_df.height == 1
    row = contrast_df.row(0, named=True)
    lo, hi = az.hdi(b1, hdi_prob=0.94)
    assert row["contrast"] == "treatment_minus_control"
    assert row["mean"] == pytest.approx(b1.mean())
    assert (row["hdi_low"], row["hdi_high"]) == pytest.approx((lo, hi))
    assert row["p_gt_zero"] == pytest.approx((b1 > 0).mean())
    assert row["p_lt_zero"] == pytest.approx((b1 < 0).mean())
    assert (row["n_control"], row["n_treatment"]) == (3, 4)

    assert not list(tmp_path.glob("*.npy"))
