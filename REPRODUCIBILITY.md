# Reproducibility

This package supports three checks, from lightest to heaviest.

## 1. Inspect Included Paper Results

The final data artifacts used by the paper are committed under `results/`:

- `results/figures/`: final PDF/PNG figures and the Mermaid source for Fig. 1.
- `results/tables/`: final LaTeX table snippets.
- `results/data/`: final CSV and JSON data snapshots.

These files allow direct comparison without running attacks or rebuilding the
paper.

## 2. Regenerate Figures and Tables From Included Data

Create the reference environment:

```powershell
conda env create -f environment.yml
conda run -n fgjpeg python -m pip install -e .
```

Run smoke checks:

```powershell
conda run -n fgjpeg python -m pytest tests -q
conda run -n fgjpeg python scripts/check_no_secrets.py
```

Regenerate figures:

```powershell
conda run -n fgjpeg python scripts/make_figures.py --metrics-csv data/derived/metrics.csv --asr-metrics-csv data/derived/figure_metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv.gz --out-dir results/figures
```

Regenerate tables:

```powershell
conda run -n fgjpeg python scripts/make_tables.py --metrics-csv data/derived/metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv.gz --out-dir results/tables
```

## 3. Rerun Full Experiments

Full reruns require external resources that are not redistributed here:

- CIFAR-10 images.
- ImageNet validation images for boundary probes.
- RobustBench, torchvision, and timm checkpoints used by the paper.
- A CUDA-capable PyTorch installation for practical runtime.

The required local layout and redistribution policy are recorded in
`resources_manifest.yml`. Prepare ignored resource directories:

```powershell
conda run -n fgjpeg python scripts/prepare_resources.py --data-root <local-data-root>
```

To register an authorized local file, pass `--resource NAME=PATH`, for example:

```powershell
conda run -n fgjpeg python scripts/prepare_resources.py --data-root <local-data-root> --resource cifar10=<path-to-cifar-10-python.tar.gz>
```

After placing local resources in ignored directories, check availability and
optionally print local file hashes:

```powershell
conda run -n fgjpeg python scripts/check_resources.py --data-root <local-data-root> --results-root <local-results-root>
conda run -n fgjpeg python scripts/verify_resources.py --data-root <local-data-root> --hash
```

Then run:

```powershell
bash scripts/run_full_pipeline.sh
```

The default `PIPELINE_PROFILE=paper` is intended to regenerate the paper data
artifacts from local resources. Use `PIPELINE_PROFILE=smoke` only to check that
the local environment, resources, and scripts are wired correctly.

Keep machine-specific variables such as `DATA_DIR`, `RESULTS_DIR`,
`ROBUSTBENCH_ROOT`, `AUTO_ATTACK_ROOT`, and `DIFFJPEG_ROOT` outside version
control.
