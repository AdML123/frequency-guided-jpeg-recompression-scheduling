"""Generate all paper LaTeX table snippets from derived CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jpeg_defense.manuscript import (  # noqa: E402
    make_full_four_schedule_asr_table,
    make_frequency_diagnostics_table,
    make_jpeg_aware_boundary_table,
    make_mcnemar_audit_table,
    make_prediction_rule_table,
    make_robust_training_gradient_table,
    make_table_ii,
    make_table_iii,
    make_tau_independence_table,
    make_threat_model_boundary_table,
)
from jpeg_defense.plotting import load_metrics_csv  # noqa: E402


def _record_output(outputs: list[tuple[str, Path]], label: str, path: Path) -> Path:
    outputs.append((label, path))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--frequency-csv", type=Path, default=None)
    parser.add_argument("--source-data-csv", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[str, Path]] = []
    metrics_df = load_metrics_csv(args.metrics_csv)
    prediction_table = _record_output(
        outputs,
        "prediction rules",
        args.out_dir / "table_i_prediction_rules.tex",
    )
    prediction_table.write_text(make_prediction_rule_table(), encoding="utf-8")
    if args.frequency_csv is not None:
        frequency_df = load_metrics_csv(args.frequency_csv)
        frequency_table = _record_output(
            outputs,
            "frequency diagnostics",
            args.out_dir / "table_frequency_diagnostics.tex",
        )
        frequency_table.write_text(
            make_frequency_diagnostics_table(frequency_df),
            encoding="utf-8",
        )
    table_ii = _record_output(outputs, "schedule-level ASR", args.out_dir / "table_ii.tex")
    table_iii = _record_output(
        outputs,
        "generation-level ASR",
        args.out_dir / "table_iii.tex",
    )
    table_iv = _record_output(
        outputs,
        "threat-model boundary",
        args.out_dir / "table_iv.tex",
    )
    table_full_asr = _record_output(
        outputs,
        "four-schedule ASR audit",
        args.out_dir / "table_full_four_schedule_asr.tex",
    )
    table_tau = args.out_dir / "table_tau_independence.tex"
    table_mcnemar = args.out_dir / "table_mcnemar_audit.tex"
    table_robust_grad = args.out_dir / "table_robust_training_gradient.tex"
    table_jpeg_aware = args.out_dir / "table_jpeg_aware_boundary.tex"
    table_ii.write_text(make_table_ii(metrics_df), encoding="utf-8")
    table_iii.write_text(make_table_iii(metrics_df), encoding="utf-8")
    table_iv.write_text(make_threat_model_boundary_table(metrics_df), encoding="utf-8")
    table_full_asr.write_text(
        make_full_four_schedule_asr_table(metrics_df),
        encoding="utf-8",
    )
    if args.source_data_csv is not None:
        source_data_df = load_metrics_csv(args.source_data_csv)
        table_mcnemar.write_text(
            make_mcnemar_audit_table(
                source_data_df,
                frequency_df if args.frequency_csv is not None else None,
            ),
            encoding="utf-8",
        )
        outputs.append(("paired McNemar audit", table_mcnemar))
        if args.frequency_csv is not None:
            frequency_df = load_metrics_csv(args.frequency_csv)
            table_tau.write_text(
                make_tau_independence_table(frequency_df),
                encoding="utf-8",
            )
            outputs.append(("tau independence audit", table_tau))
            table_robust_grad.write_text(
                make_robust_training_gradient_table(frequency_df),
                encoding="utf-8",
            )
            outputs.append(("robust-training gradient audit", table_robust_grad))
            table_jpeg_aware.write_text(
                make_jpeg_aware_boundary_table(frequency_df, source_data_df),
                encoding="utf-8",
            )
            outputs.append(("JPEG-aware boundary audit", table_jpeg_aware))
    elif args.frequency_csv is not None:
        table_tau.write_text(
            make_tau_independence_table(frequency_df),
            encoding="utf-8",
        )
        outputs.append(("tau independence audit", table_tau))
        table_robust_grad.write_text(
            make_robust_training_gradient_table(frequency_df),
            encoding="utf-8",
        )
        outputs.append(("robust-training gradient audit", table_robust_grad))
    print("Generated table snippets:")
    for label, path in outputs:
        print(f"- {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
