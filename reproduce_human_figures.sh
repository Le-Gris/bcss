#!/usr/bin/env bash
# Render the human-data insets (Figures 2C, 3C, 4C) from the summaries
# written by run_human_analyses.sh.  No model is fit; this takes seconds.
#
#   figures/insets/fig2c_simplicity.{pdf,png}
#   figures/insets/fig3c_rule_type.{pdf,png}
#   figures/insets/fig3c_order_effects.{pdf,png}
#   figures/insets/fig4c_error_category.{pdf,png}
#
# The PNGs are transparent and gitignored; only the PDFs are committed.
#
# Usage:  bash reproduce_human_figures.sh
#   OUT overrides the analyses root (default: results/analyses; the summaries
#   are read from $OUT/human/), FIG the output directory (default: figures/insets).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv run python figures/render_insets.py \
  --analyses-dir "${OUT:-results/analyses}" --output-dir "${FIG:-figures/insets}"
