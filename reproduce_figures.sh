#!/usr/bin/env bash
# Render the main-results insets (Figures 2C, 3C, 4C) from the summaries
# written by run_analyses.sh.  No model is fit; this takes seconds.
#
#   figures/insets/fig2c_simplicity.pdf
#   figures/insets/fig3c_rule_type.pdf
#   figures/insets/fig3c_order_effects.pdf
#   figures/insets/fig4c_error_category.pdf
#
# Usage:  bash reproduce_figures.sh      (OUT/FIG override the input/output dirs)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

uv run python figures/render_insets.py \
  --analyses-dir "${OUT:-results/analyses}" --output-dir "${FIG:-figures/insets}"
