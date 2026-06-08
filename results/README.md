# Generated Results

This directory is the default output location for regenerated paper figures
and tables. Generated artifacts are intentionally ignored by Git. Recreate them
from the repository root with:

```powershell
conda run -n paper20-cu128 python scripts/make_figures.py --metrics-csv data/derived/metrics.csv --asr-metrics-csv data/derived/figure_metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/figures
conda run -n paper20-cu128 python scripts/make_tables.py --metrics-csv data/derived/metrics.csv --frequency-csv data/derived/frequency_metrics.csv --source-data-csv data/derived/source_data.csv --out-dir results/tables
```
