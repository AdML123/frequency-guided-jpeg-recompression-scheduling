# Frequency-Guided JPEG Recompression Scheduling

This repository is the reproducibility artifact for the 2026 11th International Conference on Signal and Image Processing
submission "Frequency-Guided JPEG Recompression Scheduling for Adversarial
Defense." Use this repository to reproduce the paper data results: included
data figures, table snippets, smoke tests, and runtime microbenchmark. It is
not a LaTeX paper-build repository.

## Paper Claim

JPEG recompression is usually treated as a fixed preprocessing step. The exact
paper claim is that the order of JPEG quality factors (QFs) is a predictable
design variable for multi-generation recompression. On the included nonadaptive
CIFAR-10 evaluation summaries, the perturbation discrete cosine transform (DCT)
centroid rule `omega_delta > tau` predicts whether front-loaded (FL) or fixed
(Fix) recompression gives the lower attack success rate (ASR) in 5/8
model-attack conditions. The remaining 3/8 conditions differ by less than one
percentage point and are treated as practically equivalent. For standard
ResNet-18 attacks, changing the schedule order changes ASR by 12-19 percentage
points.

## Safe GitHub Authentication

No password authentication is used for GitHub operations. Use GitHub CLI with
`gh auth login --web`, a personal access token stored outside the repository, or
SSH. Do not write, paste, commit, or transmit account credentials in this
repository.

See `REPRODUCIBILITY.md` for exact commands, `ARTIFACTS.md` for the
figure/table source map, and `SECURITY.md` for the credential and reporting
policy.

## Repository Layout

- `src/jpeg_defense/`: package code for schedules, metrics, plotting, table
  generation, model wrappers, and attacks.
- `scripts/`: command-line entrypoints for tests, figures, tables, resource
  checks, full reruns, and secret scanning.
- `tests/`: pytest suite for package behavior and repository-delivery checks.
- `data/derived/`: derived CSV and JSON files used by the paper figures and
  tables.
- `results/`: default output directory for regenerated local artifacts.
- `third_party/diffjpeg/`: vendored DiffJPEG code and its license notice.
- `.github/workflows/ci.yml`: GitHub Actions smoke test workflow.

## Conda Installation

The verified local environment name is `paper20-cu128`. From the repository
root, install into that environment with:

```powershell
conda create -n paper20-cu128 python=3.10 -y
conda run -n paper20-cu128 python -m pip install -r requirements.txt
conda run -n paper20-cu128 python -m pip install -e .
```

If `paper20-cu128` already exists, skip the `conda create` command and run the
two install commands.

For CI-style checks with the smaller test dependency set:

```powershell
conda run -n paper20-cu128 python -m pip install -r requirements-ci.txt
conda run -n paper20-cu128 python -m pip install -e .
```

## Smoke Test

Run the repository tests and secret scan from the repository root:

```powershell
conda run -n paper20-cu128 python -m pytest tests -q
conda run -n paper20-cu128 python scripts/check_no_secrets.py
```

## Reproduce Figures

The included derived data regenerate the figure PDFs and PNGs used by the
paper:

```powershell
conda run -n paper20-cu128 python scripts/make_figures.py --metrics-csv data/derived/metrics.csv --asr-metrics-csv data/derived/figure_metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/figures
```

Expected outputs:

- `results/figures/figure2_asr_by_generation.pdf`
- `results/figures/figure2_asr_by_generation.png`
- `results/figures/figure4_centroid_vs_delta.pdf`
- `results/figures/figure4_centroid_vs_delta.png`
- `results/figures/figure3_attack_range_summary.pdf`
- `results/figures/figure3_attack_range_summary.png`

## Reproduce Tables

The table script regenerates every LaTeX table snippet delivered with the paper
from `data/derived`:

```powershell
conda run -n paper20-cu128 python scripts/make_tables.py --metrics-csv data/derived/metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/tables
```

Expected outputs:

- `results/tables/table_i_prediction_rules.tex`
- `results/tables/table_ii.tex`
- `results/tables/table_iii.tex`
- `results/tables/table_iv.tex`
- `results/tables/table_full_four_schedule_asr.tex`
- `results/tables/table_frequency_diagnostics.tex`
- `results/tables/table_mcnemar_audit.tex`
- `results/tables/table_tau_independence.tex`
- `results/tables/table_robust_training_gradient.tex`
- `results/tables/table_jpeg_aware_boundary.tex`

This repository is responsible for regenerating data-result artifacts from
included derived data. It does not require raw datasets or model checkpoints for
the figure and table reproduction commands above.

## Full Experiment Rerun Requirements

Full attack reruns are optional and require resources that are intentionally not
included in this repository:

- CIFAR-10 images from the official CIFAR-10 distribution.
- ImageNet validation images from the official ImageNet access process.
- RobustBench checkpoints from the official RobustBench package or model zoo.
- Any additional torchvision or timm checkpoints from their official project
  sources.
- A CUDA-capable PyTorch installation if GPU acceleration is needed.

After placing local resources in ignored data or checkpoint directories, check
availability and run the full pipeline:

```powershell
conda run -n paper20-cu128 python scripts/check_resources.py
bash scripts/run_full_pipeline.sh
```

Set only local environment variables required by your machine, such as
`DATA_DIR`, `RESULTS_DIR`, `ROBUSTBENCH_ROOT`, `AUTO_ATTACK_ROOT`, or
`DIFFJPEG_ROOT`. Keep these settings outside version control.

## Data Availability

The paper's derived data are included under `data\derived`:

- `metrics.csv`
- `figure_metrics.csv`
- `frequency_metrics.csv`
- `source_data.csv`
- `summary.json`
- `full_audit_summary.json`
- `frequency_full8_summary.json`

These files are sufficient for the smoke tests, figure regeneration, and table
regeneration commands in this README. Raw CIFAR-10 archives, ImageNet files,
model weights, and training checkpoints are not redistributed here; obtain them
from the official dataset, benchmark, or model-provider sources listed above.

## Third-Party Licenses

Original repository code is released under the MIT License in `LICENSE`.
Vendored DiffJPEG code in `third_party/diffjpeg` retains its bundled license
and attribution. Dataset and model resources used for full reruns remain subject
to their official upstream terms.

## Citation

Use `CITATION.cff` for software citation metadata. Author metadata remains
anonymous for double-blind review.

## Security/No Credentials

No credentials are needed to reproduce the included figures, tables, or tests.
Do not commit account secrets, local configuration, raw private data, or model
checkpoints. Run the scanner before publishing:

```powershell
conda run -n paper20-cu128 python scripts/check_no_secrets.py
```
