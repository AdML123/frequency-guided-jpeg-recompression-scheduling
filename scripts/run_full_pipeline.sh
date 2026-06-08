#!/bin/sh
set -eu

# Reproduce the paper data-result artifacts from local, user-authorized
# datasets and checkpoints. Use PIPELINE_PROFILE=smoke for a quick connectivity
# check; the default paper profile uses full CIFAR-10 settings for the audit
# and frequency diagnostics.

CODE_DIR="${CODE_DIR:-/code}"
DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-/results}"
PYTHON="${PYTHON:-python}"
PIPELINE_PROFILE="${PIPELINE_PROFILE:-paper}"

export CODE_DIR DATA_DIR RESULTS_DIR

METRICS_DIR="$RESULTS_DIR/metrics"
FIGURES_DIR="$RESULTS_DIR/figures"
TABLES_DIR="$RESULTS_DIR/tables"
LOGS_DIR="$RESULTS_DIR/logs"

mkdir -p "$METRICS_DIR" "$FIGURES_DIR" "$TABLES_DIR" "$LOGS_DIR"

cd "$CODE_DIR"

case "$PIPELINE_PROFILE" in
  paper)
    AUDIT_MAX_SAMPLES="${AUDIT_MAX_SAMPLES:-10000}"
    FIGURE_MAX_SAMPLES="${FIGURE_MAX_SAMPLES:-512}"
    FREQUENCY_MAX_SAMPLES="${FREQUENCY_MAX_SAMPLES:-10000}"
    DEVICE="${DEVICE:-cuda}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-64}"
    CIFAR_FIGURE_MODELS="${CIFAR_FIGURE_MODELS:-cifar_resnet18,robustbench_wong2020fast,robustbench_engstrom2019,robustbench_rice2020}"
    JPEG_AWARE_FIGURE_MODELS="${JPEG_AWARE_FIGURE_MODELS:-cifar_resnet18,robustbench_wong2020fast}"
    IMAGENET_FIGURE_MODELS="${IMAGENET_FIGURE_MODELS:-imagenet_vit_b_16,imagenet_swin_t,imagenet_deit_tiny}"
    ;;
  smoke)
    AUDIT_MAX_SAMPLES="${AUDIT_MAX_SAMPLES:-8}"
    FIGURE_MAX_SAMPLES="${FIGURE_MAX_SAMPLES:-8}"
    FREQUENCY_MAX_SAMPLES="${FREQUENCY_MAX_SAMPLES:-8}"
    DEVICE="${DEVICE:-cpu}"
    ATTACK_BATCH_SIZE="${ATTACK_BATCH_SIZE:-8}"
    CIFAR_FIGURE_MODELS="${CIFAR_FIGURE_MODELS:-cifar_resnet18}"
    JPEG_AWARE_FIGURE_MODELS="${JPEG_AWARE_FIGURE_MODELS:-cifar_resnet18}"
    IMAGENET_FIGURE_MODELS="${IMAGENET_FIGURE_MODELS:-}"
    ;;
  *)
    echo "Unsupported PIPELINE_PROFILE: $PIPELINE_PROFILE" >&2
    echo "Use PIPELINE_PROFILE=paper or PIPELINE_PROFILE=smoke." >&2
    exit 2
    ;;
esac

echo "Checking expected external resources under $DATA_DIR"
"$PYTHON" scripts/check_resources.py \
  --data-root "$DATA_DIR" \
  --results-root "$RESULTS_DIR" \
  > "$LOGS_DIR/check_resources.log" 2>&1

AUDIT_DIR="$METRICS_DIR/full_audit"
FIGURE_METRICS_DIR="$METRICS_DIR/figure_metrics"
FIGURE_FGSM_DIR="$FIGURE_METRICS_DIR/cifar_fgsm"
FIGURE_PGD_DIR="$FIGURE_METRICS_DIR/cifar_pgd"
FIGURE_JPEG_AWARE_DIR="$FIGURE_METRICS_DIR/cifar_jpeg_aware_pgd"
FIGURE_IMAGENET_DIR="$FIGURE_METRICS_DIR/imagenet_fgsm"
FREQUENCY_DIR="$METRICS_DIR/frequency_full8"

mkdir -p "$AUDIT_DIR" "$FIGURE_METRICS_DIR" "$FIGURE_FGSM_DIR" "$FIGURE_PGD_DIR" "$FIGURE_JPEG_AWARE_DIR" "$FIGURE_IMAGENET_DIR" "$FREQUENCY_DIR"

echo "Running FL-vs-Fix McNemar audit"
"$PYTHON" scripts/run_experiments.py \
  --mode mcnemar-audit \
  --data-root "$DATA_DIR" \
  --results-dir "$AUDIT_DIR" \
  --max-samples "$AUDIT_MAX_SAMPLES" \
  --attacks fgsm,pgd \
  --models cifar_resnet18,robustbench_engstrom2019,robustbench_rice2020,robustbench_wong2020fast \
  --ranges "R1:75:50,R2:85:55,R3:90:60" \
  --generations "5" \
  --device "$DEVICE" \
  --attack-batch-size "$ATTACK_BATCH_SIZE" \
  > "$LOGS_DIR/run_mcnemar_audit.log" 2>&1

echo "Running figure-metric CIFAR-10 FGSM schedule grid"
"$PYTHON" scripts/run_experiments.py \
  --mode multimodel \
  --data-root "$DATA_DIR" \
  --results-dir "$FIGURE_FGSM_DIR" \
  --max-samples "$FIGURE_MAX_SAMPLES" \
  --attack fgsm \
  --models "$CIFAR_FIGURE_MODELS" \
  --ranges "R1:75:50,R2:85:55,R3:90:60" \
  --generations "3,5" \
  --device "$DEVICE" \
  --attack-batch-size "$ATTACK_BATCH_SIZE" \
  > "$LOGS_DIR/run_figure_metrics_cifar_fgsm.log" 2>&1

echo "Running figure-metric CIFAR-10 PGD schedule grid"
"$PYTHON" scripts/run_experiments.py \
  --mode multimodel \
  --data-root "$DATA_DIR" \
  --results-dir "$FIGURE_PGD_DIR" \
  --max-samples "$FIGURE_MAX_SAMPLES" \
  --attack pgd \
  --models "$CIFAR_FIGURE_MODELS" \
  --ranges "R1:75:50,R2:85:55,R3:90:60" \
  --generations "3,5" \
  --device "$DEVICE" \
  --attack-batch-size "$ATTACK_BATCH_SIZE" \
  > "$LOGS_DIR/run_figure_metrics_cifar_pgd.log" 2>&1

echo "Running figure-metric JPEG-aware boundary grid"
"$PYTHON" scripts/run_experiments.py \
  --mode multimodel \
  --data-root "$DATA_DIR" \
  --results-dir "$FIGURE_JPEG_AWARE_DIR" \
  --max-samples "$FIGURE_MAX_SAMPLES" \
  --attack jpeg-aware-pgd \
  --models "$JPEG_AWARE_FIGURE_MODELS" \
  --ranges "R1:75:50" \
  --generations "3,5" \
  --device "$DEVICE" \
  --attack-batch-size "$ATTACK_BATCH_SIZE" \
  > "$LOGS_DIR/run_figure_metrics_jpeg_aware.log" 2>&1

if [ -n "$IMAGENET_FIGURE_MODELS" ]; then
  echo "Running figure-metric ImageNet FGSM boundary grid"
  "$PYTHON" scripts/run_experiments.py \
    --mode multimodel \
    --data-root "$DATA_DIR" \
    --results-dir "$FIGURE_IMAGENET_DIR" \
    --max-samples "$FIGURE_MAX_SAMPLES" \
    --attack fgsm \
    --models "$IMAGENET_FIGURE_MODELS" \
    --ranges "R1:75:50,R2:85:55,R3:90:60" \
    --generations "3,5" \
    --device "$DEVICE" \
    --attack-batch-size "$ATTACK_BATCH_SIZE" \
    > "$LOGS_DIR/run_figure_metrics_imagenet.log" 2>&1
fi

echo "Running frequency diagnostics"
"$PYTHON" scripts/run_experiments.py \
  --mode frequency \
  --data-root "$DATA_DIR" \
  --results-dir "$FREQUENCY_DIR" \
  --max-samples "$FREQUENCY_MAX_SAMPLES" \
  --device "$DEVICE" \
  --attack-batch-size "$ATTACK_BATCH_SIZE" \
  --model-attacks "cifar_resnet18:fgsm,cifar_resnet18:pgd,robustbench_wong2020fast:fgsm,robustbench_wong2020fast:pgd,robustbench_engstrom2019:fgsm,robustbench_engstrom2019:pgd,robustbench_rice2020:fgsm,robustbench_rice2020:pgd" \
  > "$LOGS_DIR/run_frequency_diagnostics.log" 2>&1

METRICS_CSV="$AUDIT_DIR/metrics.csv"
SOURCE_DATA_CSV="$AUDIT_DIR/source_data.csv"
FIGURE_METRICS_CSV="$FIGURE_METRICS_DIR/metrics.csv"
FREQUENCY_METRICS_CSV="$FREQUENCY_DIR/frequency_metrics.csv"

FIGURE_INPUTS="$FIGURE_FGSM_DIR/metrics.csv $FIGURE_PGD_DIR/metrics.csv $FIGURE_JPEG_AWARE_DIR/metrics.csv"
if [ -n "$IMAGENET_FIGURE_MODELS" ]; then
  FIGURE_INPUTS="$FIGURE_INPUTS $FIGURE_IMAGENET_DIR/metrics.csv"
fi

for required_csv in "$METRICS_CSV" "$SOURCE_DATA_CSV" $FIGURE_INPUTS "$FREQUENCY_METRICS_CSV"; do
  if [ ! -s "$required_csv" ]; then
    echo "Expected CSV was not generated: $required_csv" >&2
    exit 1
  fi
done

"$PYTHON" - "$FIGURE_METRICS_CSV" $FIGURE_INPUTS <<'PY'
import csv
import sys
from pathlib import Path

output = Path(sys.argv[1])
inputs = [Path(value) for value in sys.argv[2:]]
fieldnames = None
rows = []
for path in inputs:
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        elif reader.fieldnames != fieldnames:
            raise SystemExit(f"CSV schema mismatch in {path}")
        rows.extend(reader)
with output.open("w", newline="", encoding="utf-8") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY

echo "Generating figures"
"$PYTHON" scripts/make_figures.py \
  --metrics-csv "$METRICS_CSV" \
  --asr-metrics-csv "$FIGURE_METRICS_CSV" \
  --frequency-csv "$FREQUENCY_METRICS_CSV" \
  --source-data-csv "$SOURCE_DATA_CSV" \
  --out-dir "$FIGURES_DIR" \
  > "$LOGS_DIR/make_figures.log" 2>&1

echo "Generating tables"
"$PYTHON" scripts/make_tables.py \
  --metrics-csv "$METRICS_CSV" \
  --frequency-csv "$FREQUENCY_METRICS_CSV" \
  --source-data-csv "$SOURCE_DATA_CSV" \
  --out-dir "$TABLES_DIR" \
  > "$LOGS_DIR/make_tables.log" 2>&1

echo "Data-result reproduction complete. Outputs are under $RESULTS_DIR"
