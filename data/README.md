# Data Directory

`derived/` contains the redistributable CSV and JSON files used to produce the
paper figures and tables. These files are also mirrored under `results/data`
with the final submitted result set.

Included:

- `derived/metrics.csv`
- `derived/figure_metrics.csv`
- `derived/frequency_metrics.csv`
- `derived/source_data.csv.gz`
- `derived/summary.json`
- `derived/full_audit_summary.json`
- `derived/frequency_full8_summary.json`

Not included:

- Raw CIFAR-10 archives.
- ImageNet validation images.
- Model checkpoints or weights.
- Local cache directories.

Use `scripts/check_resources.py` to validate a local data root before running
the full experiment pipeline. `resources_manifest.yml` documents the expected
external resources, and `scripts/prepare_resources.py` can create or populate
the ignored local layout from already-authorized files. No runtime downloads are
attempted by the resource checks.
