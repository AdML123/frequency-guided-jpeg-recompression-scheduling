# Data Directory

This folder keeps redistributable derived data and documents required external
inputs for full reruns.

Included:

- `derived/metrics.csv`
- `derived/figure_metrics.csv`
- `derived/frequency_metrics.csv`
- `derived/source_data.csv`
- JSON summaries used by the paper audit

Not included:

- CIFAR-10 archives
- ImageNet validation data
- model checkpoints or weights
- local cache directories

Use `scripts/check_resources.py` to validate a local data root before running
the full experiment pipeline. No runtime downloads are attempted.
