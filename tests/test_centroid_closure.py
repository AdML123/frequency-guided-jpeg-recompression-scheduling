from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


def _load_run_experiments_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
    spec = importlib.util.spec_from_file_location("run_experiments_for_centroid_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_for_centroid_test", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_metrics_csv(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "model,model_family,dataset,attack,range_name,start_qf,end_qf,schedule,generations,qfs,clean_accuracy,clean_correct,no_defense_asr,no_defense_successes,asr,ci_low,ci_high,successes,total",
                "cifar_resnet18,standard_cifar,cifar10,pgd,R1,75,50,geometric,1,75,0.65,333,1.0,333,0.86,0.81,0.89,285,333",
                "cifar_resnet18,standard_cifar,cifar10,pgd,R1,75,50,geometric,3,\"75,65,50\",0.65,333,1.0,333,0.82,0.77,0.86,273,333",
                "cifar_resnet18,standard_cifar,cifar10,pgd,R1,75,50,front_loaded,1,50,0.65,333,1.0,333,0.71,0.66,0.76,237,333",
                "cifar_resnet18,standard_cifar,cifar10,pgd,R1,75,50,front_loaded,3,\"50,65,75\",0.65,333,1.0,333,0.63,0.58,0.68,211,333",
                "cifar_resnet18,standard_cifar,cifar10,pgd,R1,75,50,fixed,3,\"62,62,62\",0.65,333,1.0,333,0.67,0.62,0.72,223,333",
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
                "cifar10,cifar_resnet18,fgsm,64,5.178,2.73,>,dataset_default,2.73,2.73,0.00,True,False,front_loaded,legacy",
                "cifar10,cifar_resnet18,pgd,64,5.024,2.73,>,dataset_default,2.73,2.73,0.00,True,False,front_loaded,legacy",
                "cifar10,cifar_resnet18,jpeg_aware_pgd,64,3.456,2.73,>,dataset_default,2.73,2.73,0.00,True,True,fix,legacy",
                "cifar10,robustbench_wong2020fast,fgsm,64,2.986,2.73,>,dataset_default,2.73,2.73,0.00,True,False,front_loaded,legacy",
                "cifar10,robustbench_wong2020fast,pgd,64,2.809,2.73,>,dataset_default,2.73,2.73,0.00,True,False,front_loaded,legacy",
                "cifar10,robustbench_wong2020fast,jpeg_aware_pgd,64,2.680,2.73,<=,dataset_default,2.73,2.73,0.00,True,True,geo,legacy",
                "cifar10,robustbench_engstrom2019,fgsm,64,2.976,2.73,>,dataset_default,2.73,2.73,0.00,True,False,front_loaded,legacy",
                "cifar10,robustbench_engstrom2019,pgd,64,2.781,2.73,>,dataset_default,2.73,2.73,0.00,True,False,front_loaded,legacy",
                "cifar10,robustbench_rice2020,fgsm,64,2.690,2.73,<=,dataset_default,2.73,2.73,0.00,True,False,fixed,legacy",
                "cifar10,robustbench_rice2020,pgd,64,2.515,2.73,<=,dataset_default,2.73,2.73,0.00,True,False,fixed,legacy",
                "imagenet,imagenet_vit_b_16,fgsm,64,3.457,3.01,>,dataset_default,3.01,3.01,0.00,True,False,front_loaded,legacy",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_prediction_and_frequency_tables_are_centroid_only(tmp_path):
    from jpeg_defense.manuscript import (
        make_frequency_diagnostics_table,
        make_prediction_rule_table,
    )
    from jpeg_defense.plotting import load_metrics_csv

    frequency = load_metrics_csv(_write_frequency_csv(tmp_path / "frequency.csv"))

    prediction_table = make_prediction_rule_table()
    diagnostics_table = make_frequency_diagnostics_table(frequency)
    combined = prediction_table + diagnostics_table

    assert "Nonadaptive" in prediction_table
    assert "JPEG-aware" in prediction_table
    assert "Image adaptive" not in prediction_table
    assert "eta" not in combined.lower()
    assert "\\eta" not in combined
    assert "kappa" not in combined.lower()
    assert "rho_block" not in combined
    assert "learning" not in combined.lower()
    assert "IN ViT-B/16 FGSM" not in diagnostics_table
    assert "C10 Rice20 PGD" in diagnostics_table
    assert diagnostics_table.count("C10 ") == 10
    assert " & 64 & " in diagnostics_table


def test_make_figures_script_does_not_emit_removed_eta_or_mechanism_figures(tmp_path):
    make_figures = _load_script("make_figures")

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency.csv")
    out_dir = tmp_path / "figures"

    exit_code = make_figures.main(
        [
            "--metrics-csv",
            str(metrics_csv),
            "--frequency-csv",
            str(frequency_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "figure2_asr_by_generation.pdf").is_file()
    assert not (out_dir / "figure1_mechanism_schematic.pdf").exists()
    assert not (out_dir / "figure2_eta_distribution.pdf").exists()
    assert not (out_dir / "figure2_frequency_matching.pdf").exists()
    assert not (out_dir / "figure2_first_qf_effect.pdf").exists()


def test_make_tables_script_does_not_emit_eta_adaptive_table(tmp_path):
    make_tables = _load_script("make_tables")

    metrics_csv = _write_metrics_csv(tmp_path / "metrics.csv")
    frequency_csv = _write_frequency_csv(tmp_path / "frequency.csv")
    out_dir = tmp_path / "tables"

    exit_code = make_tables.main(
        [
            "--metrics-csv",
            str(metrics_csv),
            "--frequency-csv",
            str(frequency_csv),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "table_frequency_diagnostics.tex").is_file()
    assert not (out_dir / "table_eta_adaptive_audit.tex").exists()
    combined = "\n".join(path.read_text(encoding="utf-8") for path in out_dir.glob("*.tex"))
    assert "eta" not in combined.lower()
    assert "\\eta" not in combined


def test_frequency_diagnostics_csv_header_omits_eta_and_kappa(monkeypatch, tmp_path):
    from jpeg_defense import experiments

    rows = []

    def fake_write_csv(path, row_values, fields):
        rows.extend(row_values)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row_values)

    class Spec:
        dataset = "cifar10"
        name = "cifar_resnet18"
        family = "standard"
        input_size = 32

    class DummyModel:
        def to(self, _device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(experiments.models, "get_model_spec", lambda _name: Spec())
    monkeypatch.setattr(
        experiments,
        "_load_inputs_for_spec",
        lambda *_args, **_kwargs: (
            __import__("torch").zeros((2, 3, 32, 32)),
            __import__("torch").tensor([0, 1]),
        ),
    )
    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda *_args, **_kwargs: (DummyModel(), None),
    )
    monkeypatch.setattr(
        experiments,
        "_make_adversarial_examples",
        lambda _model, images, *_args, **_kwargs: images + 0.01,
    )
    monkeypatch.setattr(
        experiments.frequency,
        "perturbation_frequency_centroid",
        lambda *_args, **_kwargs: 3.0,
    )
    monkeypatch.setattr(experiments, "_write_csv", fake_write_csv)

    summary = experiments.run_frequency_diagnostics(
        data_root=tmp_path,
        results_dir=tmp_path / "results",
        max_samples=2,
        model_names=("cifar_resnet18",),
        attacks=("fgsm",),
    )

    header = (tmp_path / "results" / "frequency_metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert summary["rows"] == 1
    assert "mean_eta" not in header
    assert "eta_high_fraction" not in header
    assert "kappa" not in header
    assert "adaptive_attack" not in header
    assert "jpeg_aware_attack" in header
    assert rows[0]["predicted_best_schedule"] == "front_loaded"


def test_run_experiments_full_mode_routes_to_cifar_multimodel(monkeypatch, tmp_path, capsys):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_multimodel_sweep(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "full",
            "rows": 1,
            "metrics_csv": str(tmp_path / "metrics.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(script.experiments, "run_multimodel_sweep", fake_run_multimodel_sweep)

    exit_code = script.main(
        [
            "--mode",
            "full",
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            "10000",
            "--attack",
            "pgd",
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls
    assert calls[0]["model_names"] == (
        "cifar_resnet18",
        "robustbench_engstrom2019",
        "robustbench_wong2020fast",
        "robustbench_rice2020",
    )
    assert calls[0]["max_samples"] == 10000
    assert "full complete" in captured.out
