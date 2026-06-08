"""Render the manuscript LaTeX file from a draft, metrics, figures, and tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jpeg_defense.manuscript import render_latex_manuscript  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--frequency-csv", type=Path, default=None)
    parser.add_argument("--source-data-csv", type=Path, default=None)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--tables-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--template-cls", type=Path, default=None)
    args = parser.parse_args(argv)

    out_path = render_latex_manuscript(
        draft_path=args.draft,
        metrics_csv=args.metrics_csv,
        frequency_csv=args.frequency_csv,
        source_data_csv=args.source_data_csv,
        figures_dir=args.figures_dir,
        tables_dir=args.tables_dir,
        out_path=args.out,
        template_cls_path=args.template_cls,
    )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
