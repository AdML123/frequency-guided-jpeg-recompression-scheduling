from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
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


def test_load_metrics_csv_returns_dataframe_with_metric_rows(tmp_path):
    from jpeg_defense.plotting import load_metrics_csv

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")

    df = load_metrics_csv(metrics_csv)

    assert list(df["schedule"])[:2] == ["geometric", "geometric"]
    assert list(df["generations"])[:2] == [1, 3]
    assert list(df["asr"])[:2] == [0.86, 0.82]


def test_make_asr_by_generation_writes_pdf_and_png(tmp_path):
    from jpeg_defense.plotting import make_asr_by_generation

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    out_dir = tmp_path / "figures"

    outputs = make_asr_by_generation(metrics_csv, out_dir)

    assert outputs == (
        out_dir / "figure2_asr_by_generation.pdf",
        out_dir / "figure2_asr_by_generation.png",
    )
    assert outputs[0].is_file()
    assert outputs[1].is_file()


def test_make_asr_by_generation_facets_multiple_attacks(tmp_path):
    from jpeg_defense.plotting import _asr_generation_panel_count, load_metrics_csv

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    df = load_metrics_csv(metrics_csv)

    assert _asr_generation_panel_count(df) == 2


def test_make_first_qf_effect_writes_pdf_and_png(tmp_path):
    from jpeg_defense.plotting import make_first_qf_effect

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    out_dir = tmp_path / "figures"

    outputs = make_first_qf_effect(metrics_csv, out_dir)

    assert outputs == (
        out_dir / "figure2_first_qf_effect.pdf",
        out_dir / "figure2_first_qf_effect.png",
    )
    assert outputs[0].is_file()
    assert outputs[1].is_file()


def _write_frequency_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset,model,attack,samples,omega_delta,tau,omega_delta_relation,jpeg_aware_attack,predicted_best_schedule,prediction_rule",
                "cifar10,cifar_resnet18,fgsm,16,6.1,2.73,>,False,front_loaded,nonadaptive: front_loaded if omega_delta > tau=2.73 else fixed",
                "cifar10,cifar_resnet18,pgd,16,2.6,2.73,<=,False,fixed,nonadaptive: front_loaded if omega_delta > tau=2.73 else fixed",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_source_data_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "dataset,model,attack,range_name,schedule,generations,sample_index,attack_success",
                "cifar10,cifar_resnet18,fgsm,R1,front_loaded,3,0,True",
                "cifar10,cifar_resnet18,fgsm,R1,front_loaded,3,1,False",
                "cifar10,cifar_resnet18,fgsm,R1,fixed,3,0,False",
                "cifar10,cifar_resnet18,fgsm,R1,fixed,3,1,False",
                "cifar10,cifar_resnet18,pgd,R1,front_loaded,3,0,False",
                "cifar10,cifar_resnet18,pgd,R1,front_loaded,3,1,False",
                "cifar10,cifar_resnet18,pgd,R1,fixed,3,0,False",
                "cifar10,cifar_resnet18,pgd,R1,fixed,3,1,True",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_make_centroid_vs_delta_scatter_writes_pdf_and_png(tmp_path):
    from jpeg_defense.plotting import make_centroid_vs_delta_scatter

    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    source_data_csv = _write_source_data_csv(tmp_path / "source_data.csv")
    out_dir = tmp_path / "figures"

    outputs = make_centroid_vs_delta_scatter(frequency_csv, source_data_csv, out_dir)

    assert outputs == (
        out_dir / "figure4_centroid_vs_delta.pdf",
        out_dir / "figure4_centroid_vs_delta.png",
    )
    assert outputs[0].is_file()
    assert outputs[1].is_file()


def test_centroid_vs_delta_scatter_uses_direct_claim_visual_labels():
    import inspect

    from jpeg_defense.plotting import make_centroid_vs_delta_scatter

    source = inspect.getsource(make_centroid_vs_delta_scatter)

    assert "low margin" not in source
    assert "Centroid margin" not in source
    assert "visual reference" in source
    assert "practical equivalence" in source


def test_frequency_matching_figure_source_uses_threshold_rule():
    import inspect

    from jpeg_defense.plotting import make_frequency_matching_figure

    source = inspect.getsource(make_frequency_matching_figure)

    assert "omega_delta_relation" in source
    assert "tau" in source
    assert "predict_schedule_from_omega_delta" in source
    assert "Perturbation-threshold schedule diagnostics" in source
    assert "Frequency-matched schedule diagnostics" not in source
    assert "omega_f" not in source
    assert "kappa" not in source


def test_make_frequency_matching_figure_writes_pdf_and_png(tmp_path):
    from jpeg_defense.plotting import make_frequency_matching_figure

    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    out_dir = tmp_path / "figures"

    outputs = make_frequency_matching_figure(frequency_csv, out_dir)

    assert outputs == (
        out_dir / "figure2_frequency_matching.pdf",
        out_dir / "figure2_frequency_matching.png",
    )
    assert outputs[0].is_file()
    assert outputs[1].is_file()


def test_make_attack_range_summary_writes_pdf_and_png(tmp_path):
    from jpeg_defense.plotting import make_attack_range_summary

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    out_dir = tmp_path / "figures"

    outputs = make_attack_range_summary(metrics_csv, out_dir)

    assert outputs == (
        out_dir / "figure3_attack_range_summary.pdf",
        out_dir / "figure3_attack_range_summary.png",
    )
    assert outputs[0].is_file()
    assert outputs[1].is_file()


def test_model_group_column_is_used_when_metrics_include_model(tmp_path):
    from jpeg_defense.plotting import _model_group_columns, load_metrics_csv

    metrics_csv = tmp_path / "metrics.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                "model,model_family,dataset,attack,range_name,start_qf,end_qf,schedule,generations,qfs,clean_accuracy,clean_correct,no_defense_asr,no_defense_successes,asr,ci_low,ci_high,successes,total",
                "model_a,fam,cifar10,fgsm,R1,75,50,fixed,3,\"62,62,62\",0.9,9,1.0,9,0.8,0,1,8,9",
                "model_b,fam,cifar10,fgsm,R1,75,50,fixed,3,\"62,62,62\",0.9,9,1.0,9,0.2,0,1,2,9",
            ]
        ),
        encoding="utf-8",
    )

    df = load_metrics_csv(metrics_csv)

    assert _model_group_columns(df) == ["dataset", "model", "attack"]


def test_panel_grid_shape_wraps_many_model_panels():
    from jpeg_defense.plotting import _panel_grid_shape

    assert _panel_grid_shape(1, max_cols=4) == (1, 1)
    assert _panel_grid_shape(7, max_cols=4) == (2, 4)
    assert _panel_grid_shape(13, max_cols=4) == (4, 4)


def test_best_schedule_counts_keeps_model_attack_groups_separate(tmp_path):
    from jpeg_defense.plotting import _best_schedule_counts, load_metrics_csv

    metrics_csv = tmp_path / "metrics.csv"
    metrics_csv.write_text(
        "\n".join(
            [
                "model,model_family,dataset,attack,range_name,start_qf,end_qf,schedule,generations,qfs,clean_accuracy,clean_correct,no_defense_asr,no_defense_successes,asr,ci_low,ci_high,successes,total",
                "model_a,fam,cifar10,fgsm,R1,75,50,front_loaded,3,\"50,65,75\",0.9,9,1.0,9,0.2,0,1,2,9",
                "model_a,fam,cifar10,fgsm,R1,75,50,fixed,3,\"62,62,62\",0.9,9,1.0,9,0.5,0,1,5,9",
                "model_b,fam,cifar10,fgsm,R1,75,50,front_loaded,3,\"50,65,75\",0.9,9,1.0,9,0.6,0,1,6,9",
                "model_b,fam,cifar10,fgsm,R1,75,50,fixed,3,\"62,62,62\",0.9,9,1.0,9,0.1,0,1,1,9",
            ]
        ),
        encoding="utf-8",
    )

    counts = _best_schedule_counts(load_metrics_csv(metrics_csv))

    assert set(counts["label"]) == {"cifar10 model_a fgsm", "cifar10 model_b fgsm"}
    assert counts.set_index(["label", "schedule"])["count"].to_dict() == {
        ("cifar10 model_a fgsm", "front_loaded"): 1,
        ("cifar10 model_b fgsm", "fixed"): 1,
    }


def test_schedule_display_name_uses_paper_readable_labels():
    from jpeg_defense.plotting import _schedule_display_name

    assert _schedule_display_name("front_loaded") == "front-loaded"
    assert _schedule_display_name("geometric") == "geometric"


def test_condition_display_label_shortens_model_attack_names_for_figures():
    from jpeg_defense.plotting import _condition_display_label

    assert (
        _condition_display_label("cifar10 robustbench_wong2020fast jpeg_aware_pgd")
        == "C10 Wong20 JPEG-aware PGD"
    )
    assert (
        _condition_display_label("imagenet imagenet_vit_b_16 fgsm")
        == "IN ViT-B/16 FGSM"
    )


def test_make_figures_script_writes_expected_outputs(tmp_path):
    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    out_dir = tmp_path / "figures"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metrics-csv",
            str(metrics_csv),
            "--frequency-csv",
            str(frequency_csv),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "figure2_asr_by_generation.pdf").is_file()
    assert (out_dir / "figure2_asr_by_generation.png").is_file()
    assert not (out_dir / "figure1_mechanism_schematic.pdf").exists()
    assert not (out_dir / "figure2_eta_distribution.pdf").exists()
    assert not (out_dir / "figure2_frequency_matching.pdf").exists()
    assert not (out_dir / "figure2_frequency_matching.png").exists()
    assert not (out_dir / "figure2_first_qf_effect.pdf").exists()
    assert not (out_dir / "figure2_first_qf_effect.png").exists()
    assert (out_dir / "figure3_attack_range_summary.pdf").is_file()
    assert (out_dir / "figure3_attack_range_summary.png").is_file()


def test_make_figures_script_writes_centroid_delta_when_source_data_is_supplied(
    tmp_path,
):
    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency_metrics.csv")
    source_data_csv = _write_source_data_csv(tmp_path / "source_data.csv")
    out_dir = tmp_path / "figures"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"

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
    assert (out_dir / "figure4_centroid_vs_delta.pdf").is_file()
    assert (out_dir / "figure4_centroid_vs_delta.png").is_file()


def test_make_figures_script_accepts_separate_asr_metrics_csv(tmp_path):
    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    asr_metrics_csv = _write_metrics_csv(tmp_path / "figure_metrics.csv")
    out_dir = tmp_path / "figures"
    script = Path(__file__).resolve().parents[1] / "scripts" / "make_figures.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--metrics-csv",
            str(metrics_csv),
            "--asr-metrics-csv",
            str(asr_metrics_csv),
            "--out-dir",
            str(out_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (out_dir / "figure2_asr_by_generation.pdf").is_file()
    assert (out_dir / "figure3_attack_range_summary.pdf").is_file()
