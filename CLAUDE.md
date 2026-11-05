# CLAUDE.md

Guidance for AI coding agents (such as Claude Code) working in this
repository. `README.md` is the main documentation: it covers the method, the data, installation and how
to reproduce the paper's results. This file adds what an agent needs to work on
the code safely.

## What this repository is

The code for *Synthesizing symbolic models of human and machine reasoning at
scale*. BCSS (Behavior-Conditioned Symbolic Synthesis) explains each incorrect
response to an ARC-AGI-1 task with a program. Gemini proposes line edits to a
reference program written in Michael Hodel's ARC DSL. The edited program is run
on the test input and kept only if it reproduces the response. The repository
contains the pipeline, the error classifier, and the statistical analyses and
figure code for the human results.

## Commands

Use `uv` for everything Python. Run commands from the repository root: the
pipeline scripts in `scripts/` resolve their default prompt templates
(`prompts/...`) and output directories (`results/...`) against the current
working directory.

```bash
uv sync --extra analysis              # install (the extra adds PyMC, NumPyro, ArviZ, Polars)
uv run pytest tests/ -q               # run the tests
uv run python scripts/<name>.py --help
```

On macOS 26 and later, prefix test and analysis commands with
`PYTENSOR_FLAGS=cxx=`. PyTensor's C compilation fails with the current clang
linker. The workaround is safe because sampling runs in NumPyro (JAX).

No linter, formatter or CI is configured. The test suite is the correctness
check.

## Before running anything expensive

Check with the user before you run any of these:

- **Scripts that call Gemini** (`synthesize_error_programs.py`,
  `synthesize_solver_programs.py`, `classify_errors.py`,
  `generate_task_embeddings.py`). These cost money. Every LLM script has the
  same `--mode` options:
  - `count`: estimates token counts through the API's free token counter. The
    estimate leaves out thinking tokens, so real costs are higher.
  - `prepare`: writes the Batch API request file locally and makes no API
    call.
  - `submit`: uploads that file and starts a paid batch job.
  - `check`: reports the job's status and downloads the results.
  - `async`: sends live, rate-limited requests, which are paid.

  The API key is read from the environment (`GEMINI_API_KEY`, with
  `GOOGLE_API_KEY` as a fallback). Nothing loads a `.env` file.
- **`bash run_analyses.sh OSF`** fits every Bayesian model with NUTS. It takes
  about two hours, and the error-category model takes most of that time. It
  overwrites the committed summaries in `results/analyses/`. For a quick
  end-to-end check, write somewhere else with tiny sampler settings:
  `OUT=/tmp/smoke bash run_analyses.sh OSF --draws 50 --tune 50 --chains 1`.

`bash reproduce_figures.sh` and `scripts/paper_counts.py` take seconds and make
no API calls.

## Data

The paper's data are on OSF and are expected unpacked at `OSF/`, which is
gitignored (see the README's *Data* section and `OSF/README.md`). The ARC tasks
go under `external/arc-agi-1/{training,evaluation}/`, which is also gitignored.
Treat both as read-only inputs and never write into them. New outputs go under
`results/` (gitignored, except `results/analyses/`).

The committed results are `results/analyses/**` (summary CSVs and analysis
PDFs) and `figures/insets/*.pdf` (the paper's Figure 2C, 3C and 4C insets).
They are produced by `run_analyses.sh` and `reproduce_figures.sh`. Change them
only by re-running those scripts, never by editing them by hand.

## Layout

| Path | Contents |
|---|---|
| `scripts/prefilter_errors.py` → `synthesize_error_programs.py` → `evaluate_synthesized_error_programs.py` → `filter_candidates.py` | The four pipeline stages: filter trivial errors, synthesize edits, execute and score, keep the matches. Each reads and writes JSONL. |
| `scripts/{generate_task_embeddings,synthesize_solver_programs,evaluate_synthesized_solver_programs,select_solver_candidates}.py` | Reference-program synthesis for tasks with no human-written solver (the evaluation split). The result goes to stage 2 through `--solvers-file`. |
| `scripts/classify_errors.py` | Assigns each accepted program to an error category. |
| `scripts/paper_counts.py`, `scripts/visualize_error_pages.py` | The counts reported in the paper, and a static HTML browser for accepted programs. |
| `src/` | Shared code. `pipeline.py` holds the CLI arguments, prompt construction and the five run modes; `filters.py` the match tolerance; `program_edits.py` the edit schema; `gemini_client.py` the API client; `categorical_model.py` the hierarchical Bayesian model; `category_definitions.py` and `rule_types.py` the two taxonomies. |
| `analyses/` | Statistical analyses (simplicity, rule types, error categories, program length). |
| `figures/` | `render_insets.py` and `inset_style.py` draw the insets from the summary CSVs. |
| `prompts/` | The LLM prompt templates, written in Markdown with `{{placeholder}}` substitutions. |
| `hodel_dsl/` | The DSL, its parser and rewriter, and the human-written training-split solvers (`solvers.py`). |
| `tests/` | The test suite and a small fixture (`tests/fixtures/errors_sample.jsonl`). |

## Rules that keep results reproducible

These rules matter more than style. Breaking one can silently change a number
reported in the paper.

- **The match tolerance is defined once.** `src/filters.passes_threshold`
  decides whether two grids count as the same. Stage 1 uses it to drop trivial
  errors and stage 4 uses it to accept programs. Changing it changes the whole
  pipeline. Add a test in `tests/test_filters.py` for any new behaviour.
- **Golden tests pin behaviour.** `tests/test_pipeline_golden.py` hashes what
  the synthesis and classification scripts would send to the API, and what the
  rule-type model is fit on. If a hash changes, your edit changed results.
  Find out why. Do not re-freeze the hash to make the test pass.
- **Identifiers in the data are frozen.** Category slugs, rule-type labels, the
  edit schema's class and field names, and output field names all appear in
  the OSF files. Pydantic also copies the edit schema's class names and
  docstrings into the API request. Don't rename them. To change how a category
  is displayed, use `DISPLAY_LABEL_OVERRIDES` in `src/category_definitions.py`.
- **Edits are modify-only.** The model can only replace the right-hand side of
  an existing line. The evaluator rejects any other action.
- **Use the shared helpers.** Hash grids with `src.utils.grid_hash`, and never
  write a local hash function: the hashes are join keys across files. Compute
  credible intervals with `az.hdi` (94%), not percentiles. Use the sampler
  settings in `src/bayes_defaults.py`.
- **No silent defaults for the main inputs.** `--input-file`, `--task-dir` and
  `--model` have no defaults in the LLM scripts and must be given explicitly.
  The exception is `--mode check`, which only needs `--output-dir` and
  `--run-tag`.

## Code conventions

- Python 3.12+. Write `X | None`, not `Optional[X]`.
- No pandas. Use NumPy in the pipeline code and Polars in `analyses/`.
- DSL programs have one variable assignment per line, ending in `O = ...` and
  `return O` (see `hodel_dsl/solvers.py`).
- Comments and docstrings describe the code as it is now, not how it changed.

## Not in this release

The machine (LRM) analyses, the supplementary and validation analyses, the
table builders, the remaining figure generators and the per-stage
documentation will come in a later release (see the README). Don't add
replacements for them here unless the user asks.
