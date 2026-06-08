"""Generate manuscript figures from a metrics CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jpeg_defense.plotting import (  # noqa: E402
    make_attack_range_summary,
    make_asr_by_generation,
    make_centroid_vs_delta_scatter,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--asr-metrics-csv", type=Path, default=None)
    parser.add_argument("--frequency-csv", type=Path, default=None)
    parser.add_argument("--source-data-csv", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    asr_metrics_csv = args.asr_metrics_csv or args.metrics_csv
    outputs = [
        *make_asr_by_generation(asr_metrics_csv, args.out_dir),
        *make_attack_range_summary(args.metrics_csv, args.out_dir),
    ]
    if args.frequency_csv and args.source_data_csv:
        outputs.extend(
            make_centroid_vs_delta_scatter(
                args.frequency_csv,
                args.source_data_csv,
                args.out_dir,
            )
        )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
