# Paper Assets

This folder documents the paper-facing artifacts mirrored by the repository.
The authoritative conference LaTeX source is delivered separately from this
repository candidate, while the reproducible figure and table inputs live under
`data/derived`.

Regenerate the mirrored assets from the repository root:

```powershell
conda run -n paper20-cu128 python scripts/make_figures.py --metrics-csv data/derived/metrics.csv --asr-metrics-csv data/derived/figure_metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/figures
conda run -n paper20-cu128 python scripts/make_tables.py --metrics-csv data/derived/metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/tables
```

The commands above use only included derived data and write local outputs under
`results`.
