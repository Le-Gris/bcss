"""Render the result insets of Figures 2C, 3C and 4C as PDFs.

Reads the summaries written by ``run_analyses.sh`` (no model is fit here) and
writes to ``figures/insets/``:

    fig2c_simplicity.pdf       relative length change, treatment vs control
    fig3c_rule_type.pdf        neither / partial match / alternative rule
    fig3c_order_effects.pdf    which training examples partial matches satisfy
    fig4c_error_category.pdf   error-category distribution (training)

Bars are attempt-weighted posterior medians with 94% HDIs.  Axis titles are
added in the composite figures, so the insets have none.

Usage:
    uv run python figures/render_insets.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from inset_style import (  # noqa: E402
    ORDER_TIER_ORDER,
    RULE_TYPE_ORDER,
    save_condition_dist_grid,
    save_estimand_figure,
    save_estimand_grouped,
)

SPLITS = (("humans_training", "Training"), ("humans_evaluation", "Evaluation"))


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _probs(path: Path, key: str = "category") -> dict[str, list[float]]:
    """{category: [median, hdi_low, hdi_high]} from a *_summary.csv."""
    return {r[key]: [float(r["median"]), float(r["hdi_low"]), float(r["hdi_high"])]
            for r in _rows(path)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analyses-dir", type=Path,
                   default=HERE.parent / "results" / "analyses")
    p.add_argument("--output-dir", type=Path, default=HERE / "insets")
    args = p.parse_args()
    a, out = args.analyses_dir, args.output_dir

    # Figure 2C
    sim = []
    for src, label in SPLITS:
        rows = _rows(a / "program_simplicity" / src / "per_task_relative_length_change.csv")
        sim.append((label, [float(r["treatment"]) for r in rows],
                    [float(r["control"]) for r in rows]))
    save_condition_dist_grid(sim, out / "fig2c_simplicity", transparent=True)

    # Figure 3C
    rt = {src: a / "rule_types" / src for src, _ in SPLITS}
    save_estimand_grouped(
        [(label, _probs(rt[src] / "rule_type_attempt_weighted_summary.csv"))
         for src, label in SPLITS],
        RULE_TYPE_ORDER, out / "fig3c_rule_type", transparent=True,
    )
    save_estimand_grouped(
        [(label, _probs(rt[src] / "order_effects" / "order_effects_attempt_weighted_summary.csv"))
         for src, label in SPLITS],
        ORDER_TIER_ORDER, out / "fig3c_order_effects", transparent=True,
        legend_loc="upper right",  # lower right collides with the "Other" label
    )

    # Figure 4C: classifiable categories by descending median, unclassifiable last
    ec = _probs(a / "error_categories" / "humans_training"
                / "error_category_attempt_weighted_summary.csv", key="display_category")
    unclass = [c for c in ec if c.lower().startswith("unclassifiable")]
    order = sorted((c for c in ec if c not in unclass), key=lambda c: -ec[c][0]) + unclass
    save_estimand_figure(ec, order, out / "fig4c_error_category",
                         figsize=(3.9, 4.6), transparent=True)


if __name__ == "__main__":
    main()
