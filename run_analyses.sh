#!/usr/bin/env bash
# Fit the models behind the main results and write their summaries.
#
# Output layout (one directory per source, as in the machine-analyses code):
#
#   results/analyses/program_simplicity/humans_{training,evaluation}/
#   results/analyses/rule_types/humans_{training,evaluation}/
#   results/analyses/error_categories/humans_training/
#   results/analyses/program_length_by_difficulty/humans_training/
#
# Each directory holds the analysis plots and *_summary.csv files (posterior
# medians and 94% HDIs).  figures/render_insets.py reads only those CSVs.
#
# Models are fit with NUTS using src/bayes_defaults.py (5000 draws, 2000 tuning
# steps, 4 chains, seed 42).  The error-category model is the slowest step
# (about 2 hours on a laptop CPU).
#
# Usage:  bash run_analyses.sh [OSF_DIR] [sampler options...]
#   OSF_DIR defaults to ./OSF.  Sampler options are passed to every fit, e.g.
#   a quick smoke test:  OUT=/tmp/smoke bash run_analyses.sh OSF --draws 50 --tune 50 --chains 1
#   OUT overrides the output directory (default: results/analyses).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OSF="${1:-OSF}"
shift || true
SAMPLER=("$@")
OUT="${OUT:-results/analyses}"

run() { echo; echo ">>> $*"; uv run python "$@"; }

# Simplicity: per-task data for Figure 2C (no model), then the regression.
run analyses/compute_simplicity_per_task.py --osf-dir "$OSF" \
  --output-root "$OUT/program_simplicity"
for split in training evaluation; do
  run analyses/analyze_program_simplicity.py \
    --treatment-candidates "$OSF/03_candidates/human/$split.jsonl" \
    --control-candidates "$OSF/03_candidates/control/human_$split.jsonl" \
    --relative \
    --output-dir "$OUT/program_simplicity/humans_$split" \
    ${SAMPLER[@]+"${SAMPLER[@]}"}
done

# Rule types and order effects (Figure 3C).
for split in training evaluation; do
  run analyses/analyze_rule_type_distribution.py \
    --analysis both \
    --dsl-input "$OSF/03_candidates/human/$split.jsonl" \
    --output-dir "$OUT/rule_types/humans_$split" \
    ${SAMPLER[@]+"${SAMPLER[@]}"}
done

# Error categories, training split (Figure 4C).
run analyses/analyze_error_category_distribution.py \
  --results-dir "$OSF/04_classifications/human/training.jsonl" \
  --candidates-file "$OSF/03_candidates/human/training.jsonl" \
  --output-dir "$OUT/error_categories/humans_training" \
  ${SAMPLER[@]+"${SAMPLER[@]}"}

# Error-program length vs. reference length, task success rate and error
# category, training split (Results and Supplementary Figure S3).
run analyses/analyze_program_length_by_difficulty.py \
  --candidates-file "$OSF/03_candidates/human/training.jsonl" \
  --char-file "$OSF/04_classifications/human/training.jsonl" \
  --success-rate-file "$OSF/00_inputs/human/harc_task_difficulty.csv" \
  --relative --alpha 0.1 \
  --output-dir "$OUT/program_length_by_difficulty/humans_training" \
  ${SAMPLER[@]+"${SAMPLER[@]}"}

echo
echo "Done.  Summaries are in $OUT/.  Next: bash reproduce_figures.sh"
