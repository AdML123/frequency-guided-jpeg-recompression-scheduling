# Artifact Map

| Data-result item | Script | Inputs | Output |
|---|---|---|---|
| ASR trajectory data figure | `scripts/make_figures.py` | `data/derived/figure_metrics.csv` | `results/figures/figure2_asr_by_generation.pdf` |
| Compact schedule-count data figure | `scripts/make_figures.py` | `data/derived/metrics.csv` | `results/figures/figure3_attack_range_summary.pdf` |
| Centroid audit data figure | `scripts/make_figures.py` | `data/derived/frequency_metrics.csv`, `data/derived/source_data.csv` | `results/figures/figure4_centroid_vs_delta.pdf` |
| Prediction-rule table snippet | `scripts/make_tables.py` | `data/derived/frequency_metrics.csv`, `data/derived/source_data.csv` | `results/tables/table_i_prediction_rules.tex` |
| Schedule-level ASR table snippet | `scripts/make_tables.py` | `data/derived/metrics.csv` | `results/tables/table_ii.tex` |
| Generation-level ASR table snippet | `scripts/make_tables.py` | `data/derived/metrics.csv` | `results/tables/table_iii.tex` |
| Boundary-check table snippet | `scripts/make_tables.py` | `data/derived/metrics.csv`, `data/derived/source_data.csv` | `results/tables/table_iv.tex` |
| Full four-schedule ASR table snippet | `scripts/make_tables.py` | `data/derived/metrics.csv` | `results/tables/table_full_four_schedule_asr.tex` |
| McNemar audit table snippet | `scripts/make_tables.py` | `data/derived/source_data.csv` | `results/tables/table_mcnemar_audit.tex` |
| Tau independence table snippet | `scripts/make_tables.py` | `data/derived/frequency_metrics.csv`, `data/derived/source_data.csv` | `results/tables/table_tau_independence.tex` |
| Robust-training and JPEG-aware boundary table snippets | `scripts/make_tables.py` | `data/derived/metrics.csv`, `data/derived/source_data.csv` | `results/tables/table_robust_training_gradient.tex`, `results/tables/table_jpeg_aware_boundary.tex` |
| Runtime check | `scripts/benchmark_runtime.py` | synthetic RGB images | terminal CSV output |
