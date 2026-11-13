#!/usr/bin/env bash
# Fit the models behind the human-data results and write their summaries.
#
# Output layout (one directory per source, as in the machine-analyses code):
#
#   results/analyses/human/program_simplicity/humans_{training,evaluation}/
#   results/analyses/human/rule_types/humans_{training,evaluation}/
#   results/analyses/human/error_categories/humans_training/
#   results/analyses/human/program_length_by_difficulty/humans_training/
#
# Each directory holds the analysis plots and *_summary.csv files (posterior
# medians and 94% HDIs).  figures/render_insets.py reads only those CSVs.
#
# Models are fit with NUTS using src/bayes_defaults.py (5000 draws, 2000 tuning
# steps, 4 chains, seed 42).  The error-category model is the slowest step
# (about 2 hours on a laptop CPU).
#
# Figures are saved as PNG and PDF with transparent backgrounds.
#
# Usage:  bash run_human_analyses.sh [OSF_DIR] [sampler options...]
#   OSF_DIR defaults to ./OSF.  Sampler options are passed to every fit, e.g.
#   a quick smoke test:  OUT=/tmp/smoke bash run_human_analyses.sh OSF --draws 50 --tune 50 --chains 1
#   OUT overrides the output root (default: results/analyses); the fits are
#   written under $OUT/human/.
#   ANALYSES picks which analyses to refit (default: all four), e.g. to skip
#   the slow error-category model:
#     ANALYSES="simplicity rule_types program_length" bash run_human_analyses.sh OSF
#   Valid names: simplicity rule_types error_categories program_length.
#   DRY_RUN=1 prints the commands without running them.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

OSF="${1:-OSF}"
shift || true
SAMPLER=("$@")
OUT="${OUT:-results/analyses}"
HUMAN="$OUT/human"
ALL_ANALYSES="simplicity rule_types error_categories program_length"
ANALYSES="${ANALYSES:-$ALL_ANALYSES}"
DRY_RUN="${DRY_RUN:-0}"

for name in $ANALYSES; do
  case " $ALL_ANALYSES " in
    *" $name "*) ;;
    *) echo "run_human_analyses.sh: unknown analysis '$name' (valid: $ALL_ANALYSES)" >&2
       exit 1 ;;
  esac
done

want() { case " $ANALYSES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

run() {
  echo; echo ">>> $*"
  if [ "$DRY_RUN" = 1 ]; then return 0; fi
  uv run python "$@"
}

# Simplicity: per-task data for Figure 2C (no model), then the regression.
if want simplicity; then
  run analyses/compute_simplicity_per_task.py --osf-dir "$OSF" \
    --output-root "$HUMAN/program_simplicity"
  for split in training evaluation; do
    run analyses/analyze_program_simplicity.py \
      --treatment-candidates "$OSF/03_candidates/human/$split.jsonl" \
      --control-candidates "$OSF/03_candidates/control/human_$split.jsonl" \
      --relative --transparent \
      --output-dir "$HUMAN/program_simplicity/humans_$split" \
      ${SAMPLER[@]+"${SAMPLER[@]}"}
  done
fi

# Rule types and order effects (Figure 3C).
if want rule_types; then
  for split in training evaluation; do
    run analyses/analyze_rule_type_distribution.py \
      --analysis both --transparent \
      --dsl-input "$OSF/03_candidates/human/$split.jsonl" \
      --output-dir "$HUMAN/rule_types/humans_$split" \
      ${SAMPLER[@]+"${SAMPLER[@]}"}
  done
fi

# Error categories, training split (Figure 4C).
if want error_categories; then
  run analyses/analyze_error_category_distribution.py \
    --results-dir "$OSF/04_classifications/human/training.jsonl" \
    --candidates-file "$OSF/03_candidates/human/training.jsonl" \
    --transparent \
    --output-dir "$HUMAN/error_categories/humans_training" \
    ${SAMPLER[@]+"${SAMPLER[@]}"}
fi

# Error-program length vs. reference length, task success rate and error
# category, training split (Results and Supplementary Figure S3).
if want program_length; then
  run analyses/analyze_program_length_by_difficulty.py \
    --candidates-file "$OSF/03_candidates/human/training.jsonl" \
    --char-file "$OSF/04_classifications/human/training.jsonl" \
    --success-rate-file "$OSF/00_inputs/human/harc_task_difficulty.csv" \
    --relative --alpha 0.1 --transparent \
    --output-dir "$HUMAN/program_length_by_difficulty/humans_training" \
    ${SAMPLER[@]+"${SAMPLER[@]}"}
fi

echo
if [ "$DRY_RUN" = 1 ]; then
  echo "Dry run: nothing was run."
else
  echo "Done ($ANALYSES).  Summaries are in $HUMAN/.  Next: bash reproduce_human_figures.sh"
fi
