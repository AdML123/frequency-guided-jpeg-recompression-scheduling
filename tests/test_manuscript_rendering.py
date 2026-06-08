from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) in sys.path:
    sys.path.remove(str(SRC_ROOT))
sys.path.insert(0, str(SRC_ROOT))


def _write_metrics_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset,attack,range_name,start_qf,end_qf,schedule,generations,qfs,clean_accuracy,clean_correct,no_defense_asr,no_defense_successes,asr,ci_low,ci_high,successes,total",
                "cifar10,pgd,R1,75,50,geometric,1,75,0.65,333,1.0,333,0.86,0.81,0.89,285,333",
                "cifar10,pgd,R1,75,50,geometric,3,\"75,65,50\",0.65,333,1.0,333,0.82,0.77,0.86,273,333",
                "cifar10,pgd,R1,75,50,front_loaded,1,50,0.65,333,1.0,333,0.71,0.66,0.76,237,333",
                "cifar10,pgd,R1,75,50,front_loaded,3,\"50,65,75\",0.65,333,1.0,333,0.63,0.58,0.68,211,333",
                "cifar10,fgsm,R2,85,55,front_loaded,3,\"55,68,85\",0.66,658,1.0,657,0.61,0.57,0.65,401,658",
                "cifar10,fgsm,R2,85,55,geometric,3,\"85,69,55\",0.66,658,1.0,657,0.75,0.71,0.78,494,658",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_frequency_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset,model,attack,samples,omega_delta,tau,omega_delta_relation,tau_source,tau_resnet18,tau_all,tau_delta,tau_classification_stable,jpeg_aware_attack,predicted_best_schedule,prediction_rule",
                "cifar10,cifar_resnet18,fgsm,16,3.125,2.73,>,dataset_default,2.73,2.73,0.00,True,False,front_loaded,nonadaptive: front_loaded if omega_delta > tau else fixed",
                "cifar10,cifar_resnet18,pgd,16,2.650,2.73,<=,dataset_default,2.73,2.73,0.00,True,False,fixed,nonadaptive: front_loaded if omega_delta > tau else fixed",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_source_data_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset,model,attack,range_name,schedule,generations,sample_index,clean_correct,attack_success",
                "cifar10,cifar_resnet18,fgsm,R1,front_loaded,3,0,True,False",
                "cifar10,cifar_resnet18,fgsm,R1,front_loaded,3,1,True,False",
                "cifar10,cifar_resnet18,fgsm,R1,front_loaded,3,2,True,False",
                "cifar10,cifar_resnet18,fgsm,R1,front_loaded,3,3,True,True",
                "cifar10,cifar_resnet18,fgsm,R1,fixed,3,0,True,True",
                "cifar10,cifar_resnet18,fgsm,R1,fixed,3,1,True,True",
                "cifar10,cifar_resnet18,fgsm,R1,fixed,3,2,True,False",
                "cifar10,cifar_resnet18,fgsm,R1,fixed,3,3,True,True",
                "cifar10,cifar_resnet18,fgsm,R1,front_loaded,1,0,True,True",
                "cifar10,cifar_resnet18,fgsm,R1,fixed,1,0,True,True",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("placeholder", ["[X]", "[XX]", "[X.X]", "[Y]", "[0.9"])
def test_assert_no_placeholders_catches_known_placeholders(placeholder):
    from jpeg_defense.manuscript import assert_no_placeholders

    with pytest.raises(ValueError, match="placeholder"):
        assert_no_placeholders(f"Remaining value {placeholder} should fail")


def test_assert_no_placeholders_allows_normal_numeric_confidence_interval():
    from jpeg_defense.manuscript import assert_no_placeholders

    assert_no_placeholders("The observed interval was [0.35, 0.72].")


@pytest.mark.parametrize("placeholder", ["[0.9\\text{X}]", "[0.9X]"])
def test_assert_no_placeholders_rejects_decimal_placeholders_with_marker(
    placeholder,
):
    from jpeg_defense.manuscript import assert_no_placeholders

    with pytest.raises(ValueError, match="placeholder"):
        assert_no_placeholders(f"Remaining value {placeholder} should fail")


def test_latex_escape_handles_simple_table_text():
    from jpeg_defense.manuscript import latex_escape

    assert latex_escape("jpeg_q75 & ASR_1 50%") == "jpeg\\_q75 \\& ASR\\_1 50\\%"


def test_table_snippets_include_schedules_qfs_asr_and_no_placeholders(tmp_path):
    from jpeg_defense.manuscript import make_table_ii, make_table_iii
    from jpeg_defense.plotting import load_metrics_csv

    metrics = load_metrics_csv(_write_metrics_csv(tmp_path / "metrics.csv"))

    table_ii = make_table_ii(metrics)
    table_iii = make_table_iii(metrics)

    combined = table_ii + table_iii
    assert "front-loaded" in combined
    assert "geometric" in combined
    assert "R1" in combined
    assert "PGD" in combined or "pgd" in combined
    assert "0.630" in combined
    assert "1.000" in combined
    assert "[XX]" not in combined


def test_frequency_tables_include_prediction_rules_and_generated_diagnostics(tmp_path):
    from jpeg_defense.manuscript import (
        make_frequency_diagnostics_table,
        make_prediction_rule_table,
    )
    from jpeg_defense.plotting import load_metrics_csv

    frequency = load_metrics_csv(_write_frequency_csv(tmp_path / "frequency.csv"))

    prediction_table = make_prediction_rule_table()
    frequency_table = make_frequency_diagnostics_table(frequency)

    combined = prediction_table + frequency_table
    assert "\\(\\omega_\\delta\\)" in combined
    assert "\\(\\tau\\)" in combined
    assert "\\(\\omega_\\delta>\\tau\\)" in prediction_table
    assert "\\(\\omega_\\delta\\le\\tau\\)" in prediction_table
    assert "\\(\\eta" not in prediction_table
    assert "Image adaptive" not in prediction_table
    assert "\\kappa>0" not in prediction_table
    assert "\\kappa\\le 0" not in prediction_table
    assert "classifier-gradient centroid" not in frequency_table
    assert "kappa difference" not in frequency_table
    assert "front-loaded" in combined
    assert "\\(\\eta" not in frequency_table
    assert "eta" not in frequency_table.lower()
    assert "\\begin{table*}" not in combined
    assert "[XX]" not in combined


def test_frequency_diagnostics_table_recomputes_legacy_frequency_predictions():
    from jpeg_defense.manuscript import make_frequency_diagnostics_table

    legacy = pd.DataFrame(
        {
            "dataset": ["cifar10", "cifar10", "cifar10"],
            "model": [
                "robustbench_wong2020fast",
                "cifar_resnet18",
                "robustbench_wong2020fast",
            ],
            "attack": ["fgsm", "jpeg_aware_pgd", "jpeg_aware_pgd"],
            "samples": [64, 64, 64],
            "omega_delta": [2.986, 3.456, 2.680],
            "omega_f": [3.377, 4.976, 3.377],
            "kappa": [-0.391, -1.520, -0.697],
            "adaptive_attack": [False, True, True],
            "predicted_best_schedule": ["fixed", "geometric", "geometric"],
            "prediction_rule": [
                "nonadaptive: front_loaded if kappa > 0 else fixed",
                "adaptive: fixed if kappa > 0 else geometric",
                "adaptive: fixed if kappa > 0 else geometric",
            ],
        }
    )

    table = make_frequency_diagnostics_table(legacy)

    assert "C10 Wong20 FGSM & 64 & 2.986 & \\(>\\)" in table
    assert "& FL" in table
    assert "C10 ResNet-18 JPEG-aware PGD & 64 & 3.456 & \\(>\\)" in table
    assert "& Fix" in table
    assert "C10 Wong20 JPEG-aware PGD & 64 & 2.680 & \\(\\le\\)" in table
    assert "& Geo" in table
    assert "kappa" not in table.lower()


def test_frequency_diagnostics_table_compacts_thirteen_condition_rows():
    from jpeg_defense.manuscript import make_frequency_diagnostics_table

    rows = []
    conditions = [
        ("cifar10", "cifar_resnet18", "fgsm"),
        ("cifar10", "cifar_resnet18", "pgd"),
        ("cifar10", "cifar_resnet18", "jpeg_aware_pgd"),
        ("cifar10", "robustbench_wong2020fast", "fgsm"),
        ("cifar10", "robustbench_wong2020fast", "pgd"),
        ("cifar10", "robustbench_wong2020fast", "jpeg_aware_pgd"),
        ("cifar10", "robustbench_engstrom2019", "fgsm"),
        ("cifar10", "robustbench_engstrom2019", "pgd"),
        ("cifar10", "robustbench_rice2020", "fgsm"),
        ("cifar10", "robustbench_rice2020", "pgd"),
        ("imagenet", "imagenet_vit_b_16", "fgsm"),
        ("imagenet", "imagenet_swin_t", "fgsm"),
        ("imagenet", "imagenet_deit_tiny", "fgsm"),
    ]
    for index, (dataset, model, attack) in enumerate(conditions):
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "attack": attack,
                "samples": 64,
                "omega_delta": 2.0 if index == 0 else 4.0 + index * 0.1,
                "tau": 2.73 if dataset == "cifar10" else 3.01,
                "omega_delta_relation": "<=" if index == 0 else ">",
                "predicted_best_schedule": "front_loaded",
            }
        )

    table = make_frequency_diagnostics_table(pd.DataFrame(rows))

    assert "\\tiny" in table
    assert "C10 Rice20 PGD" in table
    assert "IN ViT-B/16 FGSM" not in table
    assert "\\(\\tau\\)" in table
    assert "Rel." in table
    assert "\\(>\\)" in table
    assert "\\(\\le\\)" in table
    assert " & > & " not in table
    assert " & <= & " not in table
    assert "\\(\\eta\\)" not in table
    assert table.count("\\\\") >= 11
    assert "\\begin{table*}" not in table


def test_mcnemar_audit_table_uses_paired_source_data():
    from jpeg_defense.manuscript import make_mcnemar_audit_table

    source_data = pd.DataFrame(
        {
            "dataset": ["cifar10"] * 9,
            "model": ["cifar_resnet18"] * 9,
            "attack": ["fgsm"] * 9,
            "range_name": ["R1"] * 9,
            "schedule": [
                "front_loaded",
                "front_loaded",
                "front_loaded",
                "front_loaded",
                "fixed",
                "fixed",
                "fixed",
                "fixed",
                "geometric",
            ],
            "generations": [3] * 9,
            "sample_index": [0, 1, 2, 3, 0, 1, 2, 3, 0],
            "clean_correct": [True] * 9,
            "attack_success": [
                False,
                False,
                False,
                True,
                True,
                True,
                False,
                True,
                True,
            ],
        }
    )

    table = make_mcnemar_audit_table(source_data)

    assert "Paired McNemar" in table
    assert "C10 ResNet-18 FGSM" in table
    assert "-50.0" in table
    assert "0/2" in table
    assert "0.500" in table
    assert "Sig." in table
    assert "\\begin{table*}" not in table


def test_mcnemar_audit_table_formats_tiny_pvalues_as_inequality():
    from jpeg_defense.manuscript import make_mcnemar_audit_table

    rows = []
    for sample_index in range(12):
        rows.append(
            {
                "dataset": "cifar10",
                "model": "cifar_resnet18",
                "attack": "fgsm",
                "range_name": "R1",
                "schedule": "front_loaded",
                "generations": 5,
                "sample_index": sample_index,
                "clean_correct": True,
                "attack_success": False,
            }
        )
        rows.append(
            {
                "dataset": "cifar10",
                "model": "cifar_resnet18",
                "attack": "fgsm",
                "range_name": "R1",
                "schedule": "fixed",
                "generations": 5,
                "sample_index": sample_index,
                "clean_correct": True,
                "attack_success": True,
            }
        )

    table = make_mcnemar_audit_table(pd.DataFrame(rows))

    assert "<0.001" in table


def test_threshold_margin_audit_table_marks_rule_mismatches():
    from jpeg_defense.manuscript import make_threshold_margin_audit_table

    frequency = pd.DataFrame(
        {
            "dataset": ["cifar10", "cifar10"],
            "model": ["cifar_resnet18", "robustbench_wong2020fast"],
            "attack": ["fgsm", "fgsm"],
            "samples": [10, 10],
            "omega_delta": [5.0, 2.89],
            "tau": [2.73, 2.73],
            "omega_delta_relation": [">", ">"],
            "predicted_best_schedule": ["front_loaded", "front_loaded"],
            "jpeg_aware_attack": [False, False],
        }
    )
    source_rows = []
    for model, fixed_successes in [
        ("cifar_resnet18", 5),
        ("robustbench_wong2020fast", 1),
    ]:
        for sample_index in range(6):
            source_rows.append(
                {
                    "dataset": "cifar10",
                    "model": model,
                    "attack": "fgsm",
                    "range_name": "R1",
                    "schedule": "front_loaded",
                    "generations": 5,
                    "sample_index": sample_index,
                    "clean_correct": True,
                    "attack_success": sample_index < 2,
                }
            )
            source_rows.append(
                {
                    "dataset": "cifar10",
                    "model": model,
                    "attack": "fgsm",
                    "range_name": "R1",
                    "schedule": "fixed",
                    "generations": 5,
                    "sample_index": sample_index,
                    "clean_correct": True,
                    "attack_success": sample_index < fixed_successes,
                }
            )

    table = make_threshold_margin_audit_table(frequency, pd.DataFrame(source_rows))

    assert "Threshold-margin audit" in table
    assert "C10 ResNet-18 FGSM" in table
    assert "C10 Wong20 FGSM" in table
    assert "yes" in table
    assert "no" in table
    assert "tab:threshold-margin-audit" in table


def test_direct_claim_tables_do_not_emit_zone_columns():
    from jpeg_defense.manuscript import (
        make_full_four_schedule_asr_table,
        make_jpeg_aware_boundary_table,
        make_mcnemar_audit_table,
        make_robust_training_gradient_table,
        make_tau_independence_table,
        make_frequency_diagnostics_table,
    )

    frequency = pd.DataFrame(
        {
            "dataset": ["cifar10", "cifar10", "cifar10"],
            "model": [
                "cifar_resnet18",
                "robustbench_rice2020",
                "cifar_resnet18",
            ],
            "attack": ["fgsm", "pgd", "jpeg_aware_pgd"],
            "samples": [10000, 10000, 10000],
            "omega_delta": [5.157, 2.466, 3.456],
            "tau": [2.73, 2.73, 2.73],
            "omega_delta_relation": [">", "<=", ">"],
            "jpeg_aware_attack": [False, False, True],
            "predicted_best_schedule": ["front_loaded", "fixed", "fixed"],
        }
    )
    source_rows = []
    for model, attack, fl_successes, fixed_successes in [
        ("cifar_resnet18", "fgsm", 2, 5),
        ("robustbench_rice2020", "pgd", 5, 2),
        ("cifar_resnet18", "jpeg_aware_pgd", 5, 2),
    ]:
        for sample_index in range(8):
            source_rows.append(
                {
                    "dataset": "cifar10",
                    "model": model,
                    "attack": attack,
                    "range_name": "R1",
                    "schedule": "front_loaded",
                    "generations": 5,
                    "sample_index": sample_index,
                    "clean_correct": True,
                    "attack_success": sample_index < fl_successes,
                }
            )
            source_rows.append(
                {
                    "dataset": "cifar10",
                    "model": model,
                    "attack": attack,
                    "range_name": "R1",
                    "schedule": "fixed",
                    "generations": 5,
                    "sample_index": sample_index,
                    "clean_correct": True,
                    "attack_success": sample_index < fixed_successes,
                }
            )
    source_data = pd.DataFrame(source_rows)
    metrics = pd.DataFrame(
        {
            "dataset": ["cifar10"] * 4,
            "model": ["cifar_resnet18"] * 4,
            "attack": ["fgsm"] * 4,
            "range_name": ["R1"] * 4,
            "schedule": ["front_loaded", "geometric", "arithmetic", "fixed"],
            "generations": [5] * 4,
            "qfs": ["50,58,65,70,75", "75,70,65,58,50", "75,69,63,56,50", "62,62,62,62,62"],
            "no_defense_asr": [1.0] * 4,
            "asr": [0.660, 0.722, 0.715, 0.788],
        }
    )

    frequency_table = make_frequency_diagnostics_table(frequency)
    mcnemar_table = make_mcnemar_audit_table(source_data, frequency)
    robust_table = make_robust_training_gradient_table(frequency)
    jpeg_aware_table = make_jpeg_aware_boundary_table(frequency, source_data)

    assert "Margin" in frequency_table
    assert "Rel." in frequency_table
    assert "Zone" not in frequency_table
    assert "Zone" not in mcnemar_table
    assert "Zone" not in robust_table
    assert "Zone" not in jpeg_aware_table
    assert "\\(\\Delta\\)" in mcnemar_table
    assert "\\(\\Delta\\)" in jpeg_aware_table
    assert "tab:tau-audit" in make_tau_independence_table(frequency)
    assert "tab:robust-grad" in make_robust_training_gradient_table(frequency)
    assert "tab:jpeg-aware" in make_jpeg_aware_boundary_table(frequency, source_data)
    assert "tab:full-asr" in make_full_four_schedule_asr_table(metrics)


def test_make_table_iii_limits_rows_to_largest_generation_per_attack_range(tmp_path):
    from jpeg_defense.manuscript import make_table_iii
    from jpeg_defense.plotting import load_metrics_csv

    metrics = load_metrics_csv(_write_metrics_csv(tmp_path / "metrics.csv"))

    table_iii = make_table_iii(metrics)

    assert "75,65,50" in table_iii
    assert "50,65,75" in table_iii
    assert " & 1 & " not in table_iii


def test_make_threat_model_boundary_table_marks_best_schedule_per_group():
    from jpeg_defense.manuscript import make_threat_model_boundary_table

    metrics = pd.DataFrame(
        {
            "attack": ["fgsm", "fgsm", "jpeg_aware_pgd", "jpeg_aware_pgd"],
            "range_name": ["R1", "R1", "R1", "R1"],
            "schedule": ["front_loaded", "geometric", "front_loaded", "fixed"],
            "generations": [5, 5, 5, 5],
            "qfs": [
                "50,58,65,70,75",
                "75,70,65,58,50",
                "50,58,65,70,75",
                "62,62,62,62,62",
            ],
            "no_defense_asr": [0.6, 0.7, 0.4, 0.3],
            "asr": [0.2, 0.5, 0.9, 0.5],
        }
    )

    table = make_threat_model_boundary_table(metrics)

    assert "Schedule selected by condition and threat model" in table
    assert "tab:model-boundary" in table
    assert "FGSM" in table
    assert "JPEG-aware PGD" in table
    assert "front-loaded" in table
    assert "fixed" in table
    assert "0.200" in table
    assert "0.500" in table
    assert "\\begin{table*}" not in table
    assert "[XX]" not in table


def test_make_threat_model_boundary_table_limits_to_largest_generation_per_attack_range():
    from jpeg_defense.manuscript import make_threat_model_boundary_table

    metrics = pd.DataFrame(
        {
            "attack": ["fgsm", "fgsm"],
            "range_name": ["R1", "R1"],
            "schedule": ["geometric", "front_loaded"],
            "generations": [3, 5],
            "qfs": ["75,65,50", "50,58,65,70,75"],
            "no_defense_asr": [1.0, 1.0],
            "asr": [0.3, 0.5],
        }
    )

    table = make_threat_model_boundary_table(metrics)

    assert "G5" in table
    assert "G3" not in table


def test_make_threat_model_boundary_table_keeps_models_separate():
    from jpeg_defense.manuscript import make_threat_model_boundary_table

    metrics = pd.DataFrame(
        {
            "dataset": ["cifar10", "cifar10", "cifar10", "cifar10"],
            "model": ["model_a", "model_a", "model_b", "model_b"],
            "attack": ["fgsm", "fgsm", "fgsm", "fgsm"],
            "range_name": ["R1", "R1", "R1", "R1"],
            "schedule": ["front_loaded", "fixed", "front_loaded", "fixed"],
            "generations": [3, 3, 3, 3],
            "qfs": ["50,65,75", "62,62,62", "50,65,75", "62,62,62"],
            "no_defense_asr": [1.0, 1.0, 1.0, 1.0],
            "asr": [0.2, 0.3, 0.4, 0.1],
        }
    )

    table = make_threat_model_boundary_table(metrics)

    assert "model-a" in table
    assert "model-b" in table
    assert table.count("FGSM") == 2
    assert "front-loaded" in table
    assert "fixed" in table
    assert "\\begin{table*}" not in table


def test_render_latex_manuscript_writes_ieee_tex_without_placeholders(tmp_path):
    from jpeg_defense.manuscript import render_latex_manuscript

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    source_data_csv = _write_source_data_csv(tmp_path / "source_data.csv")
    draft = tmp_path / "draft.md"
    draft.write_text(
        "\n".join(
            [
                "# Threat-Model-Aware Scheduling in JPEG Recompression Defense",
                "",
                "## Abstract",
                "Joint Photographic Experts Group (JPEG) recompression can change attack success rate (ASR).",
                "",
                "## Keywords",
                "adversarial examples, JPEG recompression, schedule selection",
                "",
                "## Introduction",
                "Joint Photographic Experts Group (JPEG) recompression is a common image operation [cite:jpeg].",
                "Fast Gradient Sign Method and Projected Gradient Descent attacks motivate schedule tests [cite:goodfellow], [cite:madry].",
                "Adaptive evaluation guards against misleading compression defenses [cite:athalye].",
                "",
                "## Methods",
                "{{FREQUENCY_MODEL}}",
                "{{MICROSTRUCTURE_MODEL}}",
                "{{SCHEDULE_DEFINITION}}",
                "{{ASR_DEFINITION}}",
                "{{PREDICTION_RULE_TABLE}}",
                "",
                "## Results",
                "The schedule trajectory is summarized in [fig:asr-generation].",
                "{{MECHANISM_FIGURE}}",
                "The frequency diagnostic in [tab:frequency-diagnostics] reports the model variables.",
                "{{FREQUENCY_DIAGNOSTIC_FIGURE}}",
                "{{FREQUENCY_DIAGNOSTIC_TABLE}}",
                "The schedule map in [fig:schedule-boundary] and the model summary in [tab:model-boundary] report the lowest-ASR schedule.",
                "{{SCHEDULE_BOUNDARY_FIGURE}}",
                "{{MODEL_BOUNDARY_TABLE}}",
            ]
        ),
        encoding="utf-8",
    )
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "figure1_mechanism_schematic.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure1_asr_by_generation.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_first_qf_effect.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_eta_distribution.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_frequency_matching.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure3_attack_range_summary.pdf").write_bytes(b"%PDF-1.4\n")
    tables_dir = tmp_path / "tables"
    out_path = tmp_path / "main.tex"

    render_latex_manuscript(
        draft_path=draft,
        metrics_csv=metrics_csv,
        frequency_csv=frequency_csv,
        source_data_csv=source_data_csv,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        out_path=out_path,
    )

    tex = out_path.read_text(encoding="utf-8")
    assert "\\documentclass[journal]{IEEEtran}" in tex
    assert "\\begin{abstract}" in tex
    assert "\\begin{IEEEkeywords}" in tex
    assert "\\title{Threat-Model-Aware Scheduling in JPEG Recompression Defense}" in tex
    assert "\\begin{equation}" in tex
    assert "\\mu_1" in tex
    assert "\\Delta_1" in tex
    assert "\\omega_\\delta" in tex
    assert "\\tau" in tex
    assert "elimination threshold" in tex
    assert "\\omega_f" not in tex
    assert "\\kappa" not in tex
    assert "\\includegraphics" in tex
    assert "\\begin{figure}[t]" in tex
    assert "\\includegraphics[width=\\linewidth]" in tex
    assert "\\detokenize{figures/figure1_asr_by_generation.pdf}" in tex
    assert "\\detokenize{figures/figure2_frequency_matching.pdf}" not in tex
    assert "\\detokenize{figures/figure2_first_qf_effect.pdf}" not in tex
    assert "\\detokenize{figures/figure1_mechanism_schematic.pdf}" not in tex
    assert "Fig.~\\ref{fig:asr-generation}" in tex
    assert "fig:frequency-diagnostics" not in tex
    assert "fig:mechanism-explanation" not in tex
    assert "Table~\\ref{tab:frequency-diagnostics}" in tex
    assert "Fig.~\\ref{fig:schedule-boundary}" in tex
    assert "Table~\\ref{tab:model-boundary}" in tex
    assert "Figure Fig." not in tex
    assert "Table Table" not in tex
    assert "\\begin{thebibliography}" in tex
    assert "\\clearpage\n\\begin{thebibliography}" not in tex
    assert "Feature squeezing" not in tex
    assert "W. Xu, D. Evans" not in tex
    assert "G. Xu, J. Li, and S. Liu" in tex
    assert "tab:mcnemar-audit" in tex
    assert "tab:model-boundary" in tex
    assert "tab:schedule-asr" not in tex
    assert "tab:generation-asr" not in tex
    assert "\\cite{jpeg}" in tex
    assert "\\cite{goodfellow}" in tex
    assert "\\cite{madry}" in tex
    assert "\\cite{athalye}" in tex
    assert "Joint Photographic Experts Group" in tex
    assert "\\begin{table*}" not in tex
    assert "reviewer" not in tex.lower()
    assert "[X]" not in tex
    assert "[XX]" not in tex


def test_render_latex_manuscript_uses_direct_claim_story(tmp_path):
    from jpeg_defense.manuscript import render_latex_manuscript

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    source_data_csv = _write_source_data_csv(tmp_path / "source_data.csv")
    draft = tmp_path / "draft.md"
    draft.write_text(
        "\n".join(
            [
                "# Perturbation Centroid Predicts JPEG Recompression Schedule for Adversarial Defense",
                "",
                "## Abstract",
                "The [math:\\omega_\\delta-\\tau] sign rule gives the lower-ASR FL-vs-Fix winner in 5/8 nonadaptive CIFAR-10 conditions. In the other 3/8 conditions, FL and Fix differ by [math:|\\Delta|<1] percentage point, making the choice practically equivalent.",
                "",
                "## Keywords",
                "adversarial examples, attack success rate, discrete cosine transform, JPEG recompression, schedule selection",
                "",
                "## Results",
                "{{MECHANISM_FIGURE}}",
                "{{FREQUENCY_DIAGNOSTIC_TABLE}}",
                "{{CENTROID_DELTA_FIGURE}}",
                "{{FULL_FOUR_SCHEDULE_ASR_TABLE}}",
                "{{MCNEMAR_AUDIT_TABLE}}",
                "{{ROBUST_TRAINING_GRADIENT_TABLE}}",
                "{{JPEG_AWARE_BOUNDARY_TABLE}}",
                "{{SCHEDULE_BOUNDARY_FIGURE}}",
            ]
        ),
        encoding="utf-8",
    )
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "figure1_asr_by_generation.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_centroid_vs_delta.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure3_attack_range_summary.pdf").write_bytes(b"%PDF-1.4\n")
    out_path = tmp_path / "main.tex"

    render_latex_manuscript(
        draft_path=draft,
        metrics_csv=metrics_csv,
        frequency_csv=frequency_csv,
        source_data_csv=source_data_csv,
        figures_dir=figures_dir,
        tables_dir=tmp_path / "tables",
        out_path=out_path,
    )

    tex = out_path.read_text(encoding="utf-8")
    assert "###" not in tex
    assert "Full-Audit Interpretation" not in tex
    assert "Table Table" not in tex
    assert "Feature squeezing" not in tex
    assert "W. Xu, D. Evans" not in tex
    assert "G. Xu, J. Li, and S. Liu" in tex
    assert "\\clearpage\n\\begin{thebibliography}" not in tex
    assert "5/8" in tex
    assert "3/8" in tex
    assert "\\Delta" in tex
    assert "practically equivalent" in tex
    assert "Zone" not in tex
    assert "tab:accuracy" not in tex
    assert "high-margin" not in tex
    assert "low-margin" not in tex
    assert "|\\omega_\\delta-\\tau|>0.3" not in tex
    assert "figure2_centroid_vs_delta.pdf" in tex


def test_render_latex_manuscript_renders_real_draft_and_preserves_caption_math(tmp_path):
    from jpeg_defense.manuscript import render_latex_manuscript
    from jpeg_defense.plotting import (
        make_asr_by_generation,
        make_attack_range_summary,
        make_centroid_vs_delta_scatter,
    )

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    source_data_csv = _write_source_data_csv(tmp_path / "source_data.csv")
    draft = Path(__file__).resolve().parents[1] / "paper" / "manuscript_draft.md"
    figures_dir = tmp_path / "figures"
    make_asr_by_generation(metrics_csv, figures_dir)
    make_attack_range_summary(metrics_csv, figures_dir)
    make_centroid_vs_delta_scatter(frequency_csv, source_data_csv, figures_dir)
    tables_dir = tmp_path / "tables"
    out_path = tmp_path / "main.tex"

    render_latex_manuscript(
        draft_path=draft,
        metrics_csv=metrics_csv,
        frequency_csv=frequency_csv,
        source_data_csv=source_data_csv,
        figures_dir=figures_dir,
        tables_dir=tables_dir,
        out_path=out_path,
    )

    tex = out_path.read_text(encoding="utf-8")
    assert "\\caption{ASR by JPEG generation and schedule family" in tex
    assert "tab:frequency-diagnostics" in tex
    assert "figure2_frequency_matching" not in tex
    assert "\\detokenize{figures/figure1_mechanism_schematic.pdf}" not in tex
    assert "fig:mechanism-explanation" not in tex
    assert "\\eta" not in tex
    assert "\\textbackslash{}(\\textbackslash{}omega" not in tex
    assert "\\omega_f" not in tex
    assert "\\kappa" not in tex
    assert "classifier-gradient" not in tex.lower()
    assert "frequency-matching index" not in tex.lower()
    assert "5/8" in tex
    assert "3/8" in tex
    assert "practically equivalent" in tex
    assert "Zone" not in tex
    assert "tab:accuracy" not in tex
    assert "high-margin" not in tex
    assert "low-margin" not in tex
    assert "|\\omega_\\delta-\\tau|>0.3" not in tex


def test_assert_manuscript_style_guardrails_rejects_ai_style_markers():
    from jpeg_defense.manuscript import assert_manuscript_style_guardrails

    for text in [
        "This sentence uses an em dash — and should fail.",
        "Taken together, these results shed light on the problem.",
        "This paragraph sounds like a reviewer response.",
    ]:
        with pytest.raises(ValueError):
            assert_manuscript_style_guardrails(text)


def test_assert_citation_order_matches_bibliography_order():
    from jpeg_defense.manuscript import assert_citation_order

    good_tex = (
        "\\cite{jpeg} text \\cite{goodfellow,madry} text "
        "\\begin{thebibliography}{12}"
        "\\bibitem{jpeg} one"
        "\\bibitem{goodfellow} two"
        "\\bibitem{madry} three"
        "\\end{thebibliography}"
    )
    assert_citation_order(good_tex)

    bad_tex = (
        "\\cite{madry} text \\cite{jpeg} text "
        "\\begin{thebibliography}{12}"
        "\\bibitem{jpeg} one"
        "\\bibitem{madry} two"
        "\\end{thebibliography}"
    )
    with pytest.raises(ValueError, match="citation order"):
        assert_citation_order(bad_tex)


def test_render_latex_manuscript_keeps_includegraphics_paths_usable(tmp_path):
    from jpeg_defense.manuscript import render_latex_manuscript

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    draft = tmp_path / "draft.md"
    draft.write_text("# JPEG Defense Evaluation", encoding="utf-8")
    figures_dir = tmp_path / "figures_with_underscores"
    figures_dir.mkdir()
    (figures_dir / "figure1_mechanism_schematic.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure1_asr_by_generation.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_eta_distribution.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_first_qf_effect.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure3_attack_range_summary.pdf").write_bytes(b"%PDF-1.4\n")
    out_path = tmp_path / "main.tex"

    render_latex_manuscript(
        draft_path=draft,
        metrics_csv=metrics_csv,
        figures_dir=figures_dir,
        tables_dir=tmp_path / "tables",
        out_path=out_path,
    )

    tex = out_path.read_text(encoding="utf-8")
    assert "\\detokenize{figures_with_underscores/figure3_attack_range_summary.pdf}" in tex
    assert "figure3\\_attack\\_range\\_summary.pdf" not in tex


def test_render_latex_manuscript_uses_paths_relative_to_output_tex_dir(tmp_path):
    from jpeg_defense.manuscript import render_latex_manuscript

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    draft = tmp_path / "draft.md"
    draft.write_text("# JPEG Defense Evaluation", encoding="utf-8")
    paper_dir = tmp_path / "paper-latex"
    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True)
    (figures_dir / "figure1_mechanism_schematic.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure1_asr_by_generation.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_eta_distribution.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_first_qf_effect.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure3_attack_range_summary.pdf").write_bytes(b"%PDF-1.4\n")
    out_path = paper_dir / "main.tex"

    render_latex_manuscript(
        draft_path=draft,
        metrics_csv=metrics_csv,
        figures_dir=figures_dir,
        tables_dir=paper_dir / "tables",
        out_path=out_path,
    )

    tex = out_path.read_text(encoding="utf-8")
    assert "\\detokenize{figures/figure3_attack_range_summary.pdf}" in tex
    assert "\\detokenize{figures/figure1_asr_by_generation.pdf}" in tex
    assert "\\detokenize{figures/figure2_first_qf_effect.pdf}" not in tex
    assert "\\detokenize{figures/figure1_mechanism_schematic.pdf}" not in tex
    assert "\\detokenize{figures/figure2_eta_distribution.pdf}" not in tex
    assert "paper-latex/figures" not in tex


def test_render_latex_manuscript_uses_relative_paths_for_sibling_figures_dir(
    tmp_path,
):
    from jpeg_defense.manuscript import render_latex_manuscript

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    draft = tmp_path / "draft.md"
    draft.write_text("# JPEG Defense Evaluation", encoding="utf-8")
    paper_dir = tmp_path / "paper-latex"
    figures_dir = tmp_path / "generated-figures"
    figures_dir.mkdir()
    (figures_dir / "figure1_mechanism_schematic.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure1_asr_by_generation.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_eta_distribution.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_first_qf_effect.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure3_attack_range_summary.pdf").write_bytes(b"%PDF-1.4\n")
    out_path = paper_dir / "main.tex"

    render_latex_manuscript(
        draft_path=draft,
        metrics_csv=metrics_csv,
        figures_dir=figures_dir,
        tables_dir=paper_dir / "tables",
        out_path=out_path,
    )

    tex = out_path.read_text(encoding="utf-8")
    assert "\\detokenize{../generated-figures/figure3_attack_range_summary.pdf}" in tex
    assert "\\detokenize{../generated-figures/figure1_asr_by_generation.pdf}" in tex
    assert "\\detokenize{../generated-figures/figure2_first_qf_effect.pdf}" not in tex
    assert "\\detokenize{../generated-figures/figure1_mechanism_schematic.pdf}" not in tex
    assert "\\detokenize{../generated-figures/figure2_eta_distribution.pdf}" not in tex
    assert "\\detokenize{generated-figures/figure1_asr_by_generation.pdf}" not in tex
    assert str(figures_dir).replace("\\", "/") not in tex


def test_render_latex_manuscript_fails_early_when_required_figures_are_missing(
    tmp_path,
):
    from jpeg_defense.manuscript import render_latex_manuscript

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    draft = tmp_path / "draft.md"
    draft.write_text("# JPEG Defense Evaluation", encoding="utf-8")
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    out_path = tmp_path / "main.tex"

    with pytest.raises(FileNotFoundError, match="figure1_asr_by_generation.pdf"):
        render_latex_manuscript(
            draft_path=draft,
            metrics_csv=metrics_csv,
            figures_dir=figures_dir,
            tables_dir=tmp_path / "tables",
            out_path=out_path,
        )

    assert not out_path.exists()


def test_make_tables_script_writes_table_files(tmp_path):
    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    source_data_csv = _write_source_data_csv(tmp_path / "source_data.csv")
    out_dir = tmp_path / "tables"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_tables.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metrics-csv",
            str(metrics_csv),
            "--frequency-csv",
            str(frequency_csv),
            "--source-data-csv",
            str(source_data_csv),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    prediction_table = (out_dir / "table_i_prediction_rules.tex").read_text(encoding="utf-8")
    frequency_table = (out_dir / "table_frequency_diagnostics.tex").read_text(encoding="utf-8")
    assert "\\(\\omega_\\delta>\\tau\\)" in prediction_table
    assert "\\kappa>0" not in prediction_table
    assert "\\(\\omega_\\delta\\)" in frequency_table
    assert "\\(\\tau\\)" in frequency_table
    assert not (out_dir / "table_eta_adaptive_audit.tex").exists()
    assert not (out_dir / "table_prediction_accuracy_by_zone.tex").exists()
    assert not (out_dir / "table_threshold_margin_audit.tex").exists()
    assert "\\(\\eta\\)" not in prediction_table + frequency_table
    assert "Zone" not in frequency_table
    assert "Zone" not in (out_dir / "table_mcnemar_audit.tex").read_text(encoding="utf-8")
    assert "Zone" not in (out_dir / "table_robust_training_gradient.tex").read_text(encoding="utf-8")
    assert "Zone" not in (out_dir / "table_jpeg_aware_boundary.tex").read_text(encoding="utf-8")
    assert "front-loaded" in (out_dir / "table_ii.tex").read_text(encoding="utf-8")
    assert "0.630" in (out_dir / "table_iii.tex").read_text(encoding="utf-8")
    assert "\\begin{table*}" not in prediction_table
    assert "\\begin{table*}" not in frequency_table


def test_render_manuscript_script_writes_output_tex(tmp_path):
    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    source_data_csv = _write_source_data_csv(tmp_path / "source_data.csv")
    draft = tmp_path / "draft.md"
    draft.write_text(
        "# JPEG Defense Evaluation\n\n## Introduction\nSmoke manuscript rendering.",
        encoding="utf-8",
    )
    figures_dir = tmp_path / "figures"
    figures_dir.mkdir()
    (figures_dir / "figure1_mechanism_schematic.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure1_asr_by_generation.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_eta_distribution.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_first_qf_effect.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure2_frequency_matching.pdf").write_bytes(b"%PDF-1.4\n")
    (figures_dir / "figure3_attack_range_summary.pdf").write_bytes(b"%PDF-1.4\n")
    tables_dir = tmp_path / "tables"
    out_path = tmp_path / "main.tex"
    script = Path(__file__).resolve().parents[1] / "scripts" / "render_manuscript.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--draft",
            str(draft),
            "--metrics-csv",
            str(metrics_csv),
            "--frequency-csv",
            str(frequency_csv),
            "--source-data-csv",
            str(source_data_csv),
            "--figures-dir",
            str(figures_dir),
            "--tables-dir",
            str(tables_dir),
            "--out",
            str(out_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "\\title{JPEG Defense Evaluation}" in out_path.read_text(encoding="utf-8")
