# Reproducibility

## Verified Environment

The verified local conda environment is `paper20-cu128`. Create it from the
repository root with:

```powershell
conda env create -f environment.yml
```

For an existing environment, install the CI dependencies and editable package:

```powershell
conda run -n paper20-cu128 python -m pip install -r requirements-ci.txt
conda run -n paper20-cu128 python -m pip install -e .
```

## Smoke Tests

Run the lightweight repository checks and secret scan:

```powershell
conda run -n paper20-cu128 python -m pytest tests -q
conda run -n paper20-cu128 python scripts/check_no_secrets.py
```

Expected result: pytest completes without failures and the scanner reports no
likely secrets.

## Scope

This repository reproduces the paper data results: derived-data figures, table
snippets, lightweight tests, and runtime measurements. It does not regenerate
or compile manuscript files.

## Figures

Regenerate the paper data-figure assets from included derived data:

```powershell
conda run -n paper20-cu128 python scripts/make_figures.py --metrics-csv data/derived/metrics.csv --asr-metrics-csv data/derived/figure_metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/figures
```

Expected outputs include:

- `results/figures/figure2_asr_by_generation.pdf`
- `results/figures/figure2_asr_by_generation.png`
- `results/figures/figure4_centroid_vs_delta.pdf`
- `results/figures/figure4_centroid_vs_delta.png`
- `results/figures/figure3_attack_range_summary.pdf`
- `results/figures/figure3_attack_range_summary.png`

## Tables

Regenerate the paper data-table snippets from included derived data:

```powershell
conda run -n paper20-cu128 python scripts/make_tables.py --metrics-csv data/derived/metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/tables
```

Expected outputs include:

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

## Runtime Microbenchmark

Measure JPEG schedule roundtrip runtime on a synthetic RGB image:

```powershell
conda run -n paper20-cu128 python scripts/benchmark_runtime.py --image-size 32 --repeats 5 --generations 5
```

Expected output is CSV with columns `schedule,generations,ms_per_image` and rows
for `front_loaded`, `fixed`, `geometric`, and `arithmetic`.

## Full Reruns

Full experiment reruns are optional. They require third-party datasets and model
checkpoints that are not included in this repository, including CIFAR-10,
ImageNet validation images, RobustBench checkpoints, and any torchvision or
timm checkpoints used for boundary probes.

Place local resources only in ignored data or checkpoint directories. Then run:

```powershell
conda run -n paper20-cu128 python scripts/check_resources.py
bash scripts/run_full_pipeline.sh
```

Keep machine-specific environment variables such as `DATA_DIR`, `RESULTS_DIR`,
`ROBUSTBENCH_ROOT`, `AUTO_ATTACK_ROOT`, and `DIFFJPEG_ROOT` outside version
control.
