# BCSS: Behavior-Conditioned Symbolic Synthesis

Code for *Synthesizing symbolic models of human and machine reasoning at scale*
(LeGris, Mondorf, Jin, Jagadish, Lake & Gureckis). Preprint: [PREPRINT LINK].

Behavior-Conditioned Symbolic Synthesis (BCSS) infers programs that explain
individual responses. Given a reference program implementing the target rule
of a task and one observed incorrect response, a large language model proposes
small sets of line edits to the reference program. Each edited program is then
executed on the test input and kept only if it reproduces the observed
response. The LLM supplies a proposal distribution over plausible deviations and
program execution is a hard verification step. Because a response is
often reachable by more than one program, BCSS returns a collection of programs
per response rather than a single model.

We apply BCSS to the Abstraction and Reasoning Corpus (ARC-AGI-1), using the
H-ARC dataset of human responses. Programs are written in the ARC
domain-specific language (DSL) of Michael Hodel, adapted in `hodel_dsl/`. This
repository contains the BCSS pipeline, the error classifier, and the
statistical analyses and figure code for the human results in the paper.

## Data

The data used in the paper (filtered errors, reference programs, every
synthesis sample, the accepted programs, error classifications and validation
data) are deposited on OSF. As stated in the paper's *Data availability*
section, they will be made publicly available with a permanent DOI upon
publication.

Download the bundle and unpack it at the root of this repository as `OSF/`:

```
OSF/
  00_inputs/              errors before and after filtering, task difficulty
  01_reference_programs/  synthesized reference programs (evaluation split)
  02_samples/             all synthesis samples
  03_candidates/          accepted programs (the main result) and controls
  04_classifications/     error categories
  05_validation/          pipeline validation data
```

`OSF/README.md` describes every file. The human behavioral data come from the
[**H-ARC** dataset](https://doi.org/10.17605/OSF.IO/BH8YQ).

## Installation

We use [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
git clone https://github.com/Le-Gris/bcss
cd bcss
uv sync --extra analysis
```

The `analysis` extra installs the statistical stack (PyMC, NumPyro, ArviZ,
Polars). The pipeline alone needs only `uv sync`.

Re-running the LLM stages also requires the ARC-AGI-1 tasks and a Gemini API
key. The scripts expect the tasks under `external/arc-agi-1/` (gitignored),
with one JSON file per task:

```
external/arc-agi-1/
├── training/     # 400 tasks, e.g. 007bbfb7.json
└── evaluation/   # 400 tasks
```

To set this up:

```bash
mkdir -p external/arc-agi-1
curl -L https://github.com/fchollet/ARC-AGI/archive/refs/heads/master.tar.gz \
  | tar -xz -C external/arc-agi-1 --strip-components=2 ARC-AGI-master/data
```

Pass the split directory to the scripts with `--task-dir`, e.g.
`--task-dir external/arc-agi-1/training`. Set the API key as an environment
variable in the shell you run the scripts from:

```bash
export GEMINI_API_KEY=<your-key>
```

You won't need either if you are reproducing the results from the OSF data.

**macOS 26 and later.** PyTensor's C compilation fails with the current clang
linker. Set `PYTENSOR_FLAGS=cxx=` for the analyses and tests. This is safe:
models are sampled with NumPyro (JAX), so PyTensor's C backend is not used for
sampling, and the results are identical.

## Reproducing the results

All commands read from `OSF/`; nothing calls an API.

**1. Fit the models** (the Bayesian analyses behind Figures 2C, 3C and 4C and
the program-length analysis):

```bash
bash run_analyses.sh OSF
```

This fits the simplicity and rule-type models on both splits, and the
error-category and program-length models on the `training` split, with NUTS
(4 chains, 5,000 draws, 2,000 tuning steps, seed 42). It writes plots and
`*_summary.csv` files (posterior point estimates and 94% HDIs) to
`results/analyses/`, one directory per analysis. The error-category model
dominates the run time (about 2 hours on a laptop CPU).

**2. Render the figures** (seconds):

```bash
bash reproduce_figures.sh
```

| Output (`figures/insets/`) | Paper |
|---|---|
| `fig2c_simplicity.pdf` | Figure 2C, simplicity bias |
| `fig3c_rule_type.pdf` | Figure 3C, rule types |
| `fig3c_order_effects.pdf` | Figure 3C, order effects |
| `fig4c_error_category.pdf` | Figure 4C, error categories |

The summaries in `results/analyses/` and the rendered insets are committed, so
step 2 can be run, and the numbers checked, without step 1.

**3. Compute the counts reported in the Results and Methods** (seconds):

```bash
uv run python scripts/paper_counts.py --osf-dir OSF
```

This prints the number of unique incorrect responses, reference-program
coverage, reproduction rates (unweighted and weighted by attempts), programs
per error, edits per program, the effect of the noise tolerance, the
exhaustive depth-1 baseline, and the filtering rates.

## Paper → code → data

| Paper (Methods) | Code | OSF |
|---|---|---|
| Data and filtering | `scripts/prefilter_errors.py` | `00_inputs/human/` |
| Synthesizing additional reference programs | `scripts/{generate_task_embeddings,synthesize_solver_programs,evaluate_synthesized_solver_programs,select_solver_candidates}.py` | `01_reference_programs/` |
| Symbolic model synthesis | `scripts/{synthesize_error_programs,evaluate_synthesized_error_programs,filter_candidates}.py` | `02_samples/human/`, `03_candidates/human/` |
| Control programs (simplicity bias) | `scripts/synthesize_error_programs.py --control`, then the same evaluation and filtering | `02_samples/control/`, `03_candidates/control/` |
| Error classification | `scripts/classify_errors.py`, `prompts/classify_errors.md` | `04_classifications/human/training.jsonl` |
| Program length, task difficulty and error category | `analyses/analyze_program_length_by_difficulty.py` | `03_candidates/human/training.jsonl`, `04_classifications/human/training.jsonl`, `00_inputs/human/harc_task_difficulty.csv` |
| Statistical analyses | `analyses/`, `src/categorical_model.py` | see *Reproducing the results* |
| Counts in Results and Methods | `scripts/paper_counts.py` | `00_inputs/`, `03_candidates/` |

The synthesis prompts shown in the Supplementary Information are in `prompts/`.

## Pipeline

```
prefilter_errors → synthesize_error_programs → evaluate_synthesized_error_programs → filter_candidates
```

Stage 1. **Filter** errors that are trivially explained (copies of the test input or
   a training output, blank grids, near-misses of the correct output).
Stage 2. **Synthesize** line edits to the reference program with Gemini, conditioned
   on each error.
Stage 3. **Evaluate** each edited program: apply the edits, remove code that no
   longer affects the output, execute it on the test input and compare the
   result with the error.
Stage 4. **Keep** the programs that reproduce the error within the noise tolerance.

Reference programs for the `training` split are the human-written solvers in
`hodel_dsl/solvers.py`. For the `evaluation` split, the solver scripts
synthesize and verify reference programs, which are passed to stage 2 with
`--solvers-file`. `scripts/classify_errors.py` then assigns each accepted
program to an error category. Every script documents its options with `--help`.

## Repository layout

```
src/           shared pipeline code: filtering, prompts, edits, Gemini client, statistical models
scripts/       pipeline, solver synthesis, classification, paper counts
analyses/      statistical analyses
figures/       figure rendering (insets/ holds the rendered figures)
prompts/       LLM prompt templates
hodel_dsl/     the ARC DSL and human-written training-split solvers
results/       committed analysis summaries
tests/         test suite
```

## Tests

```bash
uv run pytest tests/ -q
```

On macOS 26+, prefix with `PYTENSOR_FLAGS=cxx=` (see *Installation*).

## Not in this release

The following will be added in a later release:

- the analyses of machine (large reasoning model) errors and the human–machine
  comparisons;
- the code for the remaining supplementary analyses (filtering policy and
  coverage) and for the pipeline validation;
- the code for Table 1 and the supplementary tables;
- the generators for the remaining figure panels (task grids, edit diagrams,
  example programs);
- the exhaustive depth-1 enumerator (its output is on OSF and is used by
  `paper_counts.py`);
- detailed documentation of each pipeline stage.

## License and citation

The code is released under the [MIT License](LICENSE). The data on OSF are
released under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
`hodel_dsl/` is adapted from
[Michael Hodel's `arc-dsl`](https://github.com/michaelhodel/arc-dsl) (MIT; see
[`hodel_dsl/LICENSE`](hodel_dsl/LICENSE)). The ARC-AGI tasks are published
separately by François Chollet.

If you use this code, please cite:

```bibtex
[BIBTEX PLACEHOLDER]
```
