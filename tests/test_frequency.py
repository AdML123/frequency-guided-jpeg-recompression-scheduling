from __future__ import annotations

import csv

import pytest
import torch


def test_frequency_centroid_is_zero_for_dc_only_coefficients():
    from jpeg_defense.frequency import frequency_energy_centroid

    coefficients = torch.zeros(1, 1, 1, 1, 8, 8)
    coefficients[..., 0, 0] = 3.0

    assert frequency_energy_centroid(coefficients) == pytest.approx(0.0)


def test_frequency_centroid_increases_for_high_frequency_energy():
    from jpeg_defense.frequency import frequency_energy_centroid

    low = torch.zeros(1, 1, 1, 1, 8, 8)
    high = torch.zeros(1, 1, 1, 1, 8, 8)
    low[..., 1, 0] = 1.0
    high[..., 7, 7] = 1.0

    assert frequency_energy_centroid(high) > frequency_energy_centroid(low)


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("cifar10", 2.73),
        ("imagenet", 3.01),
    ],
)
def test_elimination_threshold_for_dataset_known_datasets(dataset, expected):
    from jpeg_defense.frequency import elimination_threshold_for_dataset

    assert elimination_threshold_for_dataset(dataset) == pytest.approx(expected)


def test_elimination_threshold_for_dataset_rejects_unsupported_dataset():
    from jpeg_defense.frequency import elimination_threshold_for_dataset

    with pytest.raises(ValueError, match="Unsupported dataset"):
        elimination_threshold_for_dataset("mnist")


@pytest.mark.parametrize(
    ("omega_delta", "tau", "expected"),
    [
        (2.731, 2.73, ">"),
        (2.730, 2.73, "<="),
        (2.700, 2.73, "<="),
    ],
)
def test_omega_delta_relation_treats_equality_as_not_high(
    omega_delta,
    tau,
    expected,
):
    from jpeg_defense.frequency import omega_delta_relation

    assert omega_delta_relation(omega_delta, tau) == expected


@pytest.mark.parametrize(
    ("omega_delta", "tau", "jpeg_aware", "expected"),
    [
        (2.731, 2.73, False, "front_loaded"),
        (2.730, 2.73, False, "fixed"),
        (3.200, 2.73, True, "fixed"),
        (2.700, 2.73, True, "geometric"),
    ],
)
def test_predict_schedule_from_omega_delta_threshold(
    omega_delta,
    tau,
    jpeg_aware,
    expected,
):
    from jpeg_defense.frequency import predict_schedule_from_omega_delta

    assert (
        predict_schedule_from_omega_delta(
            omega_delta,
            tau=tau,
            jpeg_aware_attack=jpeg_aware,
        )
        == expected
    )


def test_block_dct_coefficients_crop_to_complete_jpeg_blocks():
    from jpeg_defense.frequency import block_dct_coefficients

    images = torch.zeros(2, 3, 10, 17)

    coefficients = block_dct_coefficients(images)

    assert coefficients.shape == (2, 3, 1, 2, 8, 8)


def test_run_frequency_diagnostics_writes_csv(monkeypatch, tmp_path):
    from jpeg_defense import experiments

    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    archive_path = data_root / "cifar10" / "cifar-10-python.tar.gz"
    checkpoint_path = (
        data_root / "checkpoints" / "cifar" / "CIFAR10_ResNet18_epoch_20.pt"
    )
    archive_path.parent.mkdir(parents=True)
    checkpoint_path.parent.mkdir(parents=True)
    archive_path.write_bytes(b"fake")
    checkpoint_path.write_bytes(b"fake")

    images = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])
    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None, offset=0: (
            images[offset : offset + limit],
            labels[offset : offset + limit],
        ),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            pooled = x.flatten(1).mean(dim=1)
            logits = torch.zeros(x.shape[0], 10, device=x.device)
            logits[:, 0] = pooled
            logits[:, 1] = -pooled
            return logits

    monkeypatch.setattr(
        experiments.models,
        "get_model_spec",
        lambda name: experiments.models.ModelSpec(
            name="cifar_resnet18",
            dataset="cifar10",
            family="standard_cifar",
            checkpoint_relpath="unused.pt",
            input_size=32,
            builder="unit",
        ),
    )
    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda spec, data_root, device: (DummyModel().eval(), None),
    )
    monkeypatch.setattr(
        experiments,
        "_make_adversarial_examples",
        lambda model, images, labels, attack, **kwargs: torch.clamp(
            images + 0.01, 0.0, 1.0
        ),
    )

    summary = experiments.run_frequency_diagnostics(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="fgsm",
        device="cpu",
    )

    rows = list(csv.DictReader((results_dir / "frequency_metrics.csv").open()))

    assert summary["frequency_metrics_csv"] == str(results_dir / "frequency_metrics.csv")
    assert len(rows) == 1
    assert rows[0]["dataset"] == "cifar10"
    assert rows[0]["model"] == "cifar_resnet18"
    assert rows[0]["attack"] == "fgsm"
    assert rows[0]["tau"] == "2.73"
    assert rows[0]["omega_delta_relation"] in {">", "<="}
    assert rows[0]["predicted_best_schedule"] in {
        "front_loaded",
        "fixed",
        "geometric",
    }
    assert "kappa" not in rows[0]
    assert "adaptive_attack" not in rows[0]
    assert "jpeg_aware_attack" in rows[0]
    assert "tau_source" in rows[0]
    assert "tau_resnet18" in rows[0]
    assert "tau_all" in rows[0]
    assert "tau_delta" in rows[0]
    assert "tau_classification_stable" in rows[0]


def test_run_frequency_diagnostics_writes_requested_model_attack_pairs(
    monkeypatch,
    tmp_path,
):
    from jpeg_defense import experiments

    images = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    specs = {
        name: experiments.models.ModelSpec(
            name=name,
            dataset="cifar10",
            family="unit",
            checkpoint_relpath="unused.pt",
            input_size=32,
            builder="unit",
        )
        for name in ("model_a", "model_b")
    }

    monkeypatch.setattr(experiments.models, "get_model_spec", specs.__getitem__)
    monkeypatch.setattr(
        experiments.frequency,
        "perturbation_frequency_centroid",
        lambda clean, adversarial: 3.2,
    )
    monkeypatch.setattr(
        experiments,
        "_load_inputs_for_spec",
        lambda spec, data_root, max_samples, imagenet_root=None, sample_offset=0: (
            images[:max_samples],
            labels[:max_samples],
        ),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            pooled = x.flatten(1).mean(dim=1)
            logits = torch.zeros(x.shape[0], 10, device=x.device)
            logits[:, 0] = pooled
            logits[:, 1] = -pooled
            return logits

    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda spec, data_root, device: (DummyModel().eval(), None),
    )
    seen_attacks = []

    def fake_adversarial(model, images, labels, attack, **kwargs):
        seen_attacks.append((attack, tuple(kwargs.get("qfs") or ())))
        return torch.clamp(images + 0.01, 0.0, 1.0)

    monkeypatch.setattr(experiments, "_make_adversarial_examples", fake_adversarial)

    summary = experiments.run_frequency_diagnostics(
        data_root=tmp_path / "data",
        results_dir=tmp_path / "results",
        max_samples=4,
        model_attack_pairs=(
            ("model_a", "fgsm"),
            ("model_b", "jpeg_aware_pgd"),
        ),
        attack_batch_size=2,
        device="cpu",
    )

    rows = list(csv.DictReader((tmp_path / "results" / "frequency_metrics.csv").open()))

    assert summary["rows"] == 2
    assert [(row["model"], row["attack"]) for row in rows] == [
        ("model_a", "fgsm"),
        ("model_b", "jpeg_aware_pgd"),
    ]
    assert rows[0]["jpeg_aware_attack"] == "False"
    assert rows[1]["jpeg_aware_attack"] == "True"
    assert [row["tau"] for row in rows] == ["2.73", "2.73"]
    assert all(float(row["energy_sum"]) > 0.0 for row in rows)
    assert all(float(row["energy_weighted_sum"]) > 0.0 for row in rows)
    assert [row["omega_delta_relation"] for row in rows] in ([">", ">"], ["<=", "<="])
    assert rows[0]["predicted_best_schedule"] in {"front_loaded", "fixed"}
    assert rows[1]["predicted_best_schedule"] in {"fixed", "geometric"}
    assert all("kappa" not in row for row in rows)
    assert all("adaptive_attack" not in row for row in rows)
    assert all(row["tau_classification_stable"] in {"True", "False"} for row in rows)
    assert seen_attacks == [
        ("fgsm", ()),
        ("jpeg_aware_pgd", (75, 65, 50)),
    ]


def test_run_frequency_diagnostics_checkpoints_rows_before_later_failure(
    monkeypatch,
    tmp_path,
):
    from jpeg_defense import experiments

    images = torch.rand(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 0, 1])
    specs = {
        name: experiments.models.ModelSpec(
            name=name,
            dataset="cifar10",
            family="unit",
            checkpoint_relpath="unused.pt",
            input_size=32,
            builder="unit",
        )
        for name in ("model_a", "model_b")
    }

    monkeypatch.setattr(experiments.models, "get_model_spec", specs.__getitem__)
    monkeypatch.setattr(
        experiments,
        "_load_inputs_for_spec",
        lambda spec, data_root, max_samples, imagenet_root=None, sample_offset=0: (
            images[:max_samples],
            labels[:max_samples],
        ),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            logits = torch.zeros(x.shape[0], 10, device=x.device)
            logits[:, 0] = 1.0
            return logits

    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda spec, data_root, device: (DummyModel().eval(), None),
    )

    calls = []

    def fake_adversarial(model, images, labels, attack, **kwargs):
        calls.append(attack)
        if len(calls) == 2:
            raise RuntimeError("second condition failed")
        return torch.clamp(images + 0.01, 0.0, 1.0)

    monkeypatch.setattr(experiments, "_make_adversarial_examples", fake_adversarial)
    monkeypatch.setattr(
        experiments.frequency,
        "perturbation_frequency_centroid",
        lambda clean, adversarial: 3.2,
    )

    with pytest.raises(RuntimeError, match="second condition failed"):
        experiments.run_frequency_diagnostics(
            data_root=tmp_path / "data",
            results_dir=tmp_path / "results",
            max_samples=4,
            model_attack_pairs=(("model_a", "fgsm"), ("model_b", "pgd")),
            device="cpu",
        )

    rows = list(csv.DictReader((tmp_path / "results" / "frequency_metrics.csv").open()))

    assert [(row["model"], row["attack"]) for row in rows] == [("model_a", "fgsm")]
