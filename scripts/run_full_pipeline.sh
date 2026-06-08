#!/bin/sh
set -eu

# Container-style defaults are /code, /data, and /results. The same script can
# be run locally by setting CODE_DIR, DATA_DIR, and RESULTS_DIR.
CODE_DIR="${CODE_DIR:-/code}"
DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-/results}"
PYTHON="${PYTHON:-python}"

export CODE_DIR DATA_DIR RESULTS_DIR

METRICS_DIR="$RESULTS_DIR/metrics"
FIGURES_DIR="$RESULTS_DIR/figures"
TABLES_DIR="$RESULTS_DIR/tables"
LOGS_DIR="$RESULTS_DIR/logs"

mkdir -p "$METRICS_DIR" "$FIGURES_DIR" "$TABLES_DIR" "$LOGS_DIR"

cd "$CODE_DIR"

echo "Checking CIFAR-10 smoke resources under $DATA_DIR"
CIFAR_ARCHIVE="$DATA_DIR/cifar10/cifar-10-python.tar.gz"
CIFAR_CHECKPOINT="$DATA_DIR/checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt"
WONG_CHECKPOINT="$DATA_DIR/checkpoints/robustbench/cifar10/Linf/Wong2020Fast.pt"
ENGSTROM_CHECKPOINT="$DATA_DIR/checkpoints/robustbench/cifar10/Linf/Engstrom2019Robustness.pt"
RICE_CHECKPOINT="$DATA_DIR/checkpoints/robustbench/cifar10/Linf/Rice2020Overfitting.pt"
if [ ! -f "$CIFAR_ARCHIVE" ]; then
  echo "Missing CIFAR-10 archive: $CIFAR_ARCHIVE" >&2
  echo "No downloads are attempted by this smoke entrypoint." >&2
  exit 1
fi
if [ ! -f "$CIFAR_CHECKPOINT" ]; then
  echo "Missing CIFAR-10 checkpoint: $CIFAR_CHECKPOINT" >&2
  echo "No downloads are attempted by this smoke entrypoint." >&2
  exit 1
fi
for required_path in \
  "$WONG_CHECKPOINT" \
  "$ENGSTROM_CHECKPOINT" \
  "$RICE_CHECKPOINT"; do
  if [ ! -f "$required_path" ]; then
    echo "Missing model checkpoint required for 10-row frequency diagnostics: $required_path" >&2
    echo "No downloads are attempted by this smoke entrypoint." >&2
    exit 1
  fi
done

FGSM_METRICS_DIR="$METRICS_DIR/fgsm"
PGD_METRICS_DIR="$METRICS_DIR/pgd"
ADAPTIVE_METRICS_DIR="$METRICS_DIR/jpeg_aware_pgd"
FREQUENCY_METRICS_DIR="$METRICS_DIR/frequency"
COMBINED_METRICS_CSV="$METRICS_DIR/metrics.csv"
COMBINED_SOURCE_DATA_CSV="$METRICS_DIR/source_data.csv"

mkdir -p "$FGSM_METRICS_DIR" "$PGD_METRICS_DIR" "$ADAPTIVE_METRICS_DIR" "$FREQUENCY_METRICS_DIR"

echo "Running CIFAR-10 FGSM schedule experiment"
"$PYTHON" scripts/run_experiments.py \
  --mode smoke \
  --data-root "$DATA_DIR" \
  --results-dir "$FGSM_METRICS_DIR" \
  --ranges "R1:75:50,R2:85:55,R3:90:60" \
  --generations "1,2,3,5" \
  --attack fgsm \
  > "$LOGS_DIR/run_experiments_fgsm.log" 2>&1

echo "Running CIFAR-10 PGD schedule experiment"
"$PYTHON" scripts/run_experiments.py \
  --mode smoke \
  --data-root "$DATA_DIR" \
  --results-dir "$PGD_METRICS_DIR" \
  --ranges "R1:75:50,R2:85:55,R3:90:60" \
  --generations "3,5" \
  --attack pgd \
  > "$LOGS_DIR/run_experiments_pgd.log" 2>&1

echo "Running CIFAR-10 JPEG-aware PGD boundary experiment"
"$PYTHON" scripts/run_experiments.py \
  --mode smoke \
  --data-root "$DATA_DIR" \
  --results-dir "$ADAPTIVE_METRICS_DIR" \
  --ranges "R1:75:50,R2:85:55,R3:90:60" \
  --generations "3,5" \
  --attack jpeg-aware-pgd \
  > "$LOGS_DIR/run_experiments_jpeg_aware_pgd.log" 2>&1

echo "Running CIFAR-10 frequency-matching diagnostic"
"$PYTHON" scripts/run_experiments.py \
  --mode frequency \
  --data-root "$DATA_DIR" \
  --results-dir "$FREQUENCY_METRICS_DIR" \
  --max-samples "${FREQUENCY_MAX_SAMPLES:-64}" \
  --device "${FREQUENCY_DEVICE:-cpu}" \
  --attack-batch-size "${FREQUENCY_ATTACK_BATCH_SIZE:-8}" \
  --model-attacks "cifar_resnet18:fgsm,cifar_resnet18:pgd,cifar_resnet18:jpeg-aware-pgd,robustbench_wong2020fast:fgsm,robustbench_wong2020fast:pgd,robustbench_wong2020fast:jpeg-aware-pgd,robustbench_engstrom2019:fgsm,robustbench_engstrom2019:pgd,robustbench_rice2020:fgsm,robustbench_rice2020:pgd" \
  > "$LOGS_DIR/run_frequency_diagnostics.log" 2>&1

FGSM_METRICS_CSV="$FGSM_METRICS_DIR/metrics.csv"
PGD_METRICS_CSV="$PGD_METRICS_DIR/metrics.csv"
ADAPTIVE_METRICS_CSV="$ADAPTIVE_METRICS_DIR/metrics.csv"
FGSM_SOURCE_DATA_CSV="$FGSM_METRICS_DIR/source_data.csv"
PGD_SOURCE_DATA_CSV="$PGD_METRICS_DIR/source_data.csv"
ADAPTIVE_SOURCE_DATA_CSV="$ADAPTIVE_METRICS_DIR/source_data.csv"
FREQUENCY_METRICS_CSV="$FREQUENCY_METRICS_DIR/frequency_metrics.csv"
if [ ! -s "$FGSM_METRICS_CSV" ]; then
  echo "Expected FGSM metrics CSV was not generated: $FGSM_METRICS_CSV" >&2
  exit 1
fi
if [ ! -s "$PGD_METRICS_CSV" ]; then
  echo "Expected PGD metrics CSV was not generated: $PGD_METRICS_CSV" >&2
  exit 1
fi
if [ ! -s "$ADAPTIVE_METRICS_CSV" ]; then
  echo "Expected JPEG-aware PGD metrics CSV was not generated: $ADAPTIVE_METRICS_CSV" >&2
  exit 1
fi
for source_csv in "$FGSM_SOURCE_DATA_CSV" "$PGD_SOURCE_DATA_CSV" "$ADAPTIVE_SOURCE_DATA_CSV"; do
  if [ ! -s "$source_csv" ]; then
    echo "Expected source-data CSV was not generated: $source_csv" >&2
    exit 1
  fi
done
if [ ! -s "$FREQUENCY_METRICS_CSV" ]; then
  echo "Expected frequency metrics CSV was not generated: $FREQUENCY_METRICS_CSV" >&2
  exit 1
fi
"$PYTHON" - "$FREQUENCY_METRICS_CSV" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(newline="", encoding="utf-8") as csv_file:
    row_count = sum(1 for _row in csv.DictReader(csv_file))
if row_count != 10:
    raise SystemExit(f"Expected 10 frequency diagnostic rows, found {row_count}: {path}")
PY

"$PYTHON" - "$COMBINED_METRICS_CSV" "$FGSM_METRICS_CSV" "$PGD_METRICS_CSV" "$ADAPTIVE_METRICS_CSV" <<'PY'
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

"$PYTHON" - "$COMBINED_SOURCE_DATA_CSV" "$FGSM_SOURCE_DATA_CSV" "$PGD_SOURCE_DATA_CSV" "$ADAPTIVE_SOURCE_DATA_CSV" <<'PY'
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

METRICS_CSV="$COMBINED_METRICS_CSV"
if [ ! -s "$METRICS_CSV" ]; then
  echo "Expected metrics CSV was not generated: $METRICS_CSV" >&2
  exit 1
fi
SOURCE_DATA_CSV="$COMBINED_SOURCE_DATA_CSV"
if [ ! -s "$SOURCE_DATA_CSV" ]; then
  echo "Expected combined source-data CSV was not generated: $SOURCE_DATA_CSV" >&2
  exit 1
fi

echo "Generating figures from $METRICS_CSV"
"$PYTHON" scripts/make_figures.py \
  --metrics-csv "$METRICS_CSV" \
  --frequency-csv "$FREQUENCY_METRICS_CSV" \
  --source-data-csv "$SOURCE_DATA_CSV" \
  --out-dir "$FIGURES_DIR" \
  > "$LOGS_DIR/make_figures.log" 2>&1

echo "Generating tables from $METRICS_CSV"
"$PYTHON" scripts/make_tables.py \
  --metrics-csv "$METRICS_CSV" \
  --frequency-csv "$FREQUENCY_METRICS_CSV" \
  --source-data-csv "$SOURCE_DATA_CSV" \
  --out-dir "$TABLES_DIR" \
  > "$LOGS_DIR/make_tables.log" 2>&1

echo "Data-result reproduction complete. Outputs are under $RESULTS_DIR"
