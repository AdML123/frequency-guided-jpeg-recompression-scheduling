from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _experiments_module():
    try:
        return importlib.import_module("jpeg_defense.experiments")
    except ModuleNotFoundError as exc:
        pytest.fail(f"jpeg_defense.experiments module is missing: {exc}")


def _load_run_experiments_script():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_experiments.py"
    )
    spec = importlib.util.spec_from_file_location("run_experiments", script_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_cifar_smoke_resources(data_root: Path):
    archive_path = data_root / "cifar10" / "cifar-10-python.tar.gz"
    checkpoint_path = (
        data_root / "checkpoints" / "cifar" / "CIFAR10_ResNet18_epoch_20.pt"
    )
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(b"fake cifar archive")
    checkpoint_path.write_bytes(b"fake cifar checkpoint")
    return archive_path, checkpoint_path


def _create_all_expected_resources(data_root: Path):
    _create_cifar_smoke_resources(data_root)
    imagenet_dir = data_root / "hf_mirror" / "imagenet1k_val_1k"
    imagenet_dir.mkdir(parents=True, exist_ok=True)
    weights_path = (
        data_root / "checkpoints" / "torchvision" / "resnet50-11ad3fa6.pth"
    )
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"fake resnet50 weights")


def test_tensor_pil_rgb_conversion_preserves_shape_range_and_channel_order():
    experiments = _experiments_module()
    tensor = (torch.arange(3 * 4 * 5, dtype=torch.float32) / 255.0).view(3, 4, 5)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        image = experiments.tensor_to_pil_rgb(tensor)
        roundtrip = experiments.pil_rgb_to_tensor(image)

    assert image.mode == "RGB"
    assert image.size == (5, 4)
    assert roundtrip.shape == tensor.shape
    assert roundtrip.dtype == torch.float32
    assert torch.all(roundtrip >= 0.0)
    assert torch.all(roundtrip <= 1.0)
    assert roundtrip == pytest.approx(tensor)


def test_recompress_tensor_batch_uses_sequential_jpeg_helper_and_preserves_range(
    monkeypatch,
):
    experiments = _experiments_module()
    seen_qualities: list[tuple[int, ...]] = []

    def fake_recompress(image: Image.Image, qualities):
        seen_qualities.append(tuple(qualities))
        return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    monkeypatch.setattr(
        experiments.jpeg, "recompress_many_generations", fake_recompress
    )
    batch = torch.linspace(0.0, 1.0, 2 * 3 * 4 * 5).view(2, 3, 4, 5)

    recompressed = experiments.recompress_tensor_batch(batch, [75, 50])

    assert seen_qualities == [(75, 50), (75, 50)]
    assert recompressed.shape == batch.shape
    assert recompressed.dtype == torch.float32
    assert torch.all(recompressed >= 0.0)
    assert torch.all(recompressed <= 1.0)


def test_evaluate_predictions_batches_images_and_applies_normalization():
    experiments = _experiments_module()
    seen_batches: list[torch.Tensor] = []

    class MeanClassifier(torch.nn.Module):
        def forward(self, x):
            seen_batches.append(x.detach().cpu())
            means = x.flatten(1).mean(dim=1)
            return torch.stack([means, -means], dim=1)

    images = torch.stack(
        [
            torch.zeros(3, 4, 4),
            torch.ones(3, 4, 4) * 0.25,
            torch.ones(3, 4, 4) * 0.5,
        ]
    )

    predictions = experiments.evaluate_predictions(
        MeanClassifier(),
        images,
        batch_size=2,
        device="cpu",
        normalize_fn=lambda x: x + 1.0,
    )

    assert np.array_equal(predictions, np.array([0, 0, 0]))
    assert len(seen_batches) == 2
    assert seen_batches[0] == pytest.approx(images[:2] + 1.0)
    assert seen_batches[1] == pytest.approx(images[2:] + 1.0)


def test_run_cifar_smoke_writes_metrics_csv_and_summary_json(
    monkeypatch, tmp_path
):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    images = torch.zeros(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 3])

    _create_cifar_smoke_resources(data_root)
    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    monkeypatch.setattr(experiments.models, "build_cifar_resnet18", DummyModel)
    monkeypatch.setattr(
        experiments.models,
        "load_state_dict_file",
        lambda model, path, device: model.eval(),
    )
    normalize_calls = []

    def fake_fgsm_attack(model, inputs, labels, epsilon, normalize_fn=None):
        normalize_calls.append(("attack", normalize_fn))
        return inputs

    monkeypatch.setattr(experiments.attacks, "fgsm_attack", fake_fgsm_attack)

    prediction_calls = [np.array([0, 1, 2, 0]), np.array([1, 1, 0, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(12)
    ]

    def fake_evaluate_predictions(*args, **kwargs):
        normalize_calls.append(("predict", kwargs.get("normalize_fn")))
        return prediction_calls.pop(0)

    seen_qfs: list[tuple[int, ...]] = []

    def fake_recompress_tensor_batch(batch, qualities):
        seen_qfs.append(tuple(qualities))
        return batch

    monkeypatch.setattr(
        experiments, "evaluate_predictions", fake_evaluate_predictions
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", fake_recompress_tensor_batch
    )

    summary = experiments.run_cifar_smoke(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="fgsm",
        device="cpu",
    )

    metrics_path = results_dir / "metrics.csv"
    summary_path = results_dir / "summary.json"
    rows = list(csv.DictReader(metrics_path.open(newline="")))
    saved_summary = json.loads(summary_path.read_text())

    assert len(rows) == 12
    assert rows[0].keys() == {
        "dataset",
        "attack",
        "range_name",
        "start_qf",
        "end_qf",
        "schedule",
        "generations",
        "qfs",
        "clean_accuracy",
        "clean_correct",
        "no_defense_asr",
        "no_defense_successes",
        "asr",
        "ci_low",
        "ci_high",
        "successes",
        "total",
    }
    assert {row["schedule"] for row in rows} == {
        "geometric",
        "arithmetic",
        "fixed",
        "front_loaded",
    }
    assert {int(row["generations"]) for row in rows} == {1, 2, 3}
    assert rows[0]["dataset"] == "cifar10"
    assert rows[0]["attack"] == "fgsm"
    assert rows[0]["range_name"] == "R1"
    assert rows[0]["start_qf"] == "75"
    assert rows[0]["end_qf"] == "50"
    assert rows[0]["clean_correct"] == "3"
    assert float(rows[0]["clean_accuracy"]) == pytest.approx(3 / 4)
    assert rows[0]["no_defense_successes"] == "2"
    assert float(rows[0]["no_defense_asr"]) == pytest.approx(2 / 3)
    assert rows[0]["successes"] == "2"
    assert rows[0]["total"] == "3"
    assert float(rows[0]["asr"]) == pytest.approx(2 / 3)
    assert (75, 65, 50) in seen_qfs
    assert summary["rows"] == 12
    assert saved_summary["metrics_csv"] == str(metrics_path)
    assert not prediction_calls
    assert normalize_calls == [("predict", None), ("attack", None), ("predict", None)] + [
        ("predict", None) for _ in range(12)
    ]


def test_run_cifar_smoke_accepts_only_cifar_smoke_resources(monkeypatch, tmp_path):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    _create_cifar_smoke_resources(data_root)
    images = torch.zeros(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 3])

    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    monkeypatch.setattr(experiments.models, "build_cifar_resnet18", DummyModel)
    monkeypatch.setattr(
        experiments.models,
        "load_state_dict_file",
        lambda model, path, device: model.eval(),
    )
    monkeypatch.setattr(
        experiments.attacks,
        "fgsm_attack",
        lambda model, inputs, labels, epsilon, normalize_fn=None: inputs,
    )

    prediction_calls = [np.array([0, 1, 2, 0]), np.array([1, 1, 0, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(12)
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    summary = experiments.run_cifar_smoke(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="fgsm",
        device="cpu",
    )

    assert summary["rows"] == 12
    assert (results_dir / "metrics.csv").is_file()
    assert (results_dir / "summary.json").is_file()
    assert not prediction_calls


def test_run_cifar_smoke_accepts_ranges_and_generation_grid(monkeypatch, tmp_path):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    _create_cifar_smoke_resources(data_root)
    images = torch.zeros(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 3])

    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    monkeypatch.setattr(experiments.models, "build_cifar_resnet18", DummyModel)
    monkeypatch.setattr(
        experiments.models,
        "load_state_dict_file",
        lambda model, path, device: model.eval(),
    )
    monkeypatch.setattr(
        experiments.attacks,
        "fgsm_attack",
        lambda model, inputs, labels, epsilon, normalize_fn=None: inputs,
    )

    prediction_calls = [np.array([0, 1, 2, 0]), np.array([1, 1, 0, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(24)
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    summary = experiments.run_cifar_smoke(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="fgsm",
        device="cpu",
        ranges=[("R1", 75, 50), ("R2", 85, 55)],
        generations=(1, 3, 5),
    )

    rows = list(csv.DictReader((results_dir / "metrics.csv").open(newline="")))

    assert summary["rows"] == 24
    assert {row["range_name"] for row in rows} == {"R1", "R2"}
    assert {int(row["start_qf"]) for row in rows} == {75, 85}
    assert {int(row["end_qf"]) for row in rows} == {50, 55}
    assert {int(row["generations"]) for row in rows} == {1, 3, 5}
    assert not prediction_calls


def test_run_cifar_smoke_uses_pgd20_for_paper_aligned_pgd(monkeypatch, tmp_path):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    images = torch.zeros(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 3])

    _create_cifar_smoke_resources(data_root)
    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    monkeypatch.setattr(experiments.models, "build_cifar_resnet18", DummyModel)
    monkeypatch.setattr(
        experiments.models,
        "load_state_dict_file",
        lambda model, path, device: model.eval(),
    )

    seen_steps = []

    def fake_pgd_attack(
        model, inputs, labels, epsilon, step_size, steps, normalize_fn=None
    ):
        seen_steps.append(steps)
        return inputs

    monkeypatch.setattr(experiments.attacks, "pgd_attack", fake_pgd_attack)
    prediction_calls = [np.array([0, 1, 2, 0]), np.array([1, 1, 0, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(12)
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    experiments.run_cifar_smoke(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="pgd",
        device="cpu",
    )

    assert seen_steps == [20]
    assert not prediction_calls


def test_run_cifar_smoke_writes_per_sample_source_data(monkeypatch, tmp_path):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    _create_cifar_smoke_resources(data_root)
    images = torch.zeros(3, 3, 4, 4)
    labels = torch.tensor([0, 1, 2])

    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    monkeypatch.setattr(experiments.models, "build_cifar_resnet18", DummyModel)
    monkeypatch.setattr(
        experiments.models,
        "load_state_dict_file",
        lambda model, path, device: model.eval(),
    )
    monkeypatch.setattr(
        experiments.attacks,
        "fgsm_attack",
        lambda model, inputs, labels, epsilon, normalize_fn=None: inputs,
    )

    prediction_calls = [
        np.array([0, 1, 0]),
        np.array([1, 1, 0]),
        np.array([1, 1, 0]),
        np.array([1, 1, 0]),
        np.array([1, 1, 0]),
        np.array([1, 1, 0]),
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    summary = experiments.run_cifar_smoke(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=3,
        attack="fgsm",
        device="cpu",
        ranges=[("R1", 75, 50)],
        generations=(1,),
    )

    source_data_path = results_dir / "source_data.csv"
    source_rows = list(csv.DictReader(source_data_path.open(newline="")))

    assert summary["source_data_csv"] == str(source_data_path)
    assert len(source_rows) == 12
    assert source_rows[0].keys() == {
        "dataset",
        "attack",
        "range_name",
        "start_qf",
        "end_qf",
        "schedule",
        "generations",
        "qfs",
        "sample_index",
        "label",
        "clean_pred",
        "defended_pred",
        "clean_correct",
        "attack_success",
    }
    assert source_rows[0]["sample_index"] == "0"
    assert source_rows[0]["clean_correct"] == "True"
    assert source_rows[0]["attack_success"] == "True"
    assert not prediction_calls


def test_run_cifar_smoke_generates_schedule_specific_jpeg_aware_pgd(
    monkeypatch, tmp_path
):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    _create_cifar_smoke_resources(data_root)
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.tensor([0, 1, 2, 3])

    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    monkeypatch.setattr(experiments.models, "build_cifar_resnet18", DummyModel)
    monkeypatch.setattr(
        experiments.models,
        "load_state_dict_file",
        lambda model, path, device: model.eval(),
    )

    seen_qfs = []

    def fake_build_diffjpeg_schedule(height, width, qfs, device, **kwargs):
        seen_qfs.append(tuple(qfs))
        return torch.nn.Identity()

    def fake_jpeg_aware_pgd_attack(
        model,
        inputs,
        labels,
        epsilon,
        step_size,
        steps,
        differentiable_defense,
        normalize_fn=None,
    ):
        return inputs

    monkeypatch.setattr(
        experiments.diffjpeg_adapter,
        "build_diffjpeg_schedule",
        fake_build_diffjpeg_schedule,
    )
    monkeypatch.setattr(
        experiments.attacks,
        "jpeg_aware_pgd_attack",
        fake_jpeg_aware_pgd_attack,
    )

    prediction_calls = [np.array([0, 1, 2, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(8)
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    experiments.run_cifar_smoke(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="jpeg_aware_pgd",
        device="cpu",
        ranges=[("R1", 75, 50)],
        generations=(3,),
    )

    assert seen_qfs == [(75, 65, 50), (75, 63, 50), (62, 62, 62), (50, 65, 75)]
    assert not prediction_calls


def test_run_multimodel_sweep_writes_model_columns_and_propagates_normalization(
    monkeypatch, tmp_path
):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    _create_cifar_smoke_resources(data_root)
    images = torch.zeros(4, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 3])

    spec = experiments.models.ModelSpec(
        name="mock_cifar",
        dataset="cifar10",
        family="unit_family",
        checkpoint_relpath="unused.pt",
        input_size=32,
        builder="unit",
    )
    monkeypatch.setattr(experiments.models, "get_model_spec", lambda name: spec)
    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    normalize_fn = lambda x: x + 0.25
    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda model_spec, data_root, device: (DummyModel().eval(), normalize_fn),
    )
    attack_normalizers = []

    def fake_fgsm_attack(model, inputs, labels, epsilon, normalize_fn=None):
        attack_normalizers.append(normalize_fn)
        return inputs

    monkeypatch.setattr(experiments.attacks, "fgsm_attack", fake_fgsm_attack)
    prediction_calls = [np.array([0, 1, 2, 0]), np.array([1, 1, 0, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(4)
    ]
    prediction_normalizers = []

    def fake_evaluate_predictions(*args, **kwargs):
        prediction_normalizers.append(kwargs.get("normalize_fn"))
        return prediction_calls.pop(0)

    monkeypatch.setattr(experiments, "evaluate_predictions", fake_evaluate_predictions)
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    summary = experiments.run_multimodel_sweep(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="fgsm",
        device="cpu",
        model_names=("mock_cifar",),
        ranges=[("R1", 75, 50)],
        generations=(3,),
    )

    rows = list(csv.DictReader((results_dir / "metrics.csv").open(newline="")))

    assert summary["rows"] == 4
    assert {row["model"] for row in rows} == {"mock_cifar"}
    assert {row["model_family"] for row in rows} == {"unit_family"}
    assert attack_normalizers == [normalize_fn]
    assert all(value is normalize_fn for value in prediction_normalizers)
    assert not prediction_calls


def test_run_multimodel_sweep_batches_static_attack_generation(monkeypatch, tmp_path):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    _create_cifar_smoke_resources(data_root)
    images = torch.zeros(5, 3, 4, 4)
    labels = torch.tensor([0, 1, 2, 3, 4])

    spec = experiments.models.ModelSpec(
        name="mock_cifar",
        dataset="cifar10",
        family="unit_family",
        checkpoint_relpath="unused.pt",
        input_size=32,
        builder="unit",
    )
    monkeypatch.setattr(experiments.models, "get_model_spec", lambda name: spec)
    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda tar_path, limit=None: (images[:limit], labels[:limit]),
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10, device=x.device)

    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda model_spec, data_root, device: (DummyModel().eval(), None),
    )
    attack_batch_sizes = []

    def fake_fgsm_attack(model, inputs, labels, epsilon, normalize_fn=None):
        attack_batch_sizes.append(inputs.shape[0])
        return inputs

    monkeypatch.setattr(experiments.attacks, "fgsm_attack", fake_fgsm_attack)
    prediction_calls = [np.array([0, 1, 2, 3, 0]), np.array([1, 1, 0, 0, 0])] + [
        np.array([1, 1, 0, 0, 0]) for _ in range(4)
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    experiments.run_multimodel_sweep(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=5,
        attack="fgsm",
        device="cpu",
        model_names=("mock_cifar",),
        ranges=[("R1", 75, 50)],
        generations=(3,),
        attack_batch_size=2,
    )

    assert attack_batch_sizes == [2, 2, 1]
    assert not prediction_calls


def test_run_multimodel_sweep_uses_imagenet_loader_for_imagenet_specs(
    monkeypatch, tmp_path
):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    images = torch.zeros(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])

    spec = experiments.models.ModelSpec(
        name="mock_imagenet",
        dataset="imagenet",
        family="unit_imagenet",
        checkpoint_relpath="unused.pt",
        input_size=224,
        builder="unit",
    )
    monkeypatch.setattr(experiments.models, "get_model_spec", lambda name: spec)

    seen_loader_args = []

    def fake_load_imagenet(root, limit=None, image_size=224):
        seen_loader_args.append((Path(root), limit, image_size))
        return images[:limit], labels[:limit]

    monkeypatch.setattr(
        experiments.data, "load_imagenet_mirror_tensors", fake_load_imagenet
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1000, device=x.device)

    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda model_spec, data_root, device: (DummyModel().eval(), None),
    )
    monkeypatch.setattr(
        experiments.attacks,
        "fgsm_attack",
        lambda model, inputs, labels, epsilon, normalize_fn=None: inputs,
    )
    prediction_calls = [np.array([0, 1, 2, 0]), np.array([1, 1, 0, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(4)
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    experiments.run_multimodel_sweep(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="fgsm",
        device="cpu",
        model_names=("mock_imagenet",),
        ranges=[("R1", 75, 50)],
        generations=(3,),
    )

    assert seen_loader_args == [
        (data_root / "hf_mirror" / "imagenet1k_val_1k", 4, 224)
    ]
    assert not prediction_calls


def test_run_multimodel_sweep_accepts_custom_imagenet_root(monkeypatch, tmp_path):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    results_dir = tmp_path / "results"
    imagenet_root = tmp_path / "custom_imagenet"
    images = torch.zeros(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 2, 3])

    spec = experiments.models.ModelSpec(
        name="mock_imagenet",
        dataset="imagenet",
        family="unit_imagenet",
        checkpoint_relpath="unused.pt",
        input_size=224,
        builder="unit",
    )
    monkeypatch.setattr(experiments.models, "get_model_spec", lambda name: spec)

    seen_roots = []

    def fake_load_imagenet(root, limit=None, image_size=224):
        seen_roots.append(Path(root))
        return images[:limit], labels[:limit]

    monkeypatch.setattr(
        experiments.data, "load_imagenet_mirror_tensors", fake_load_imagenet
    )

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1000, device=x.device)

    monkeypatch.setattr(
        experiments.models,
        "build_model_from_spec",
        lambda model_spec, data_root, device: (DummyModel().eval(), None),
    )
    monkeypatch.setattr(
        experiments.attacks,
        "fgsm_attack",
        lambda model, inputs, labels, epsilon, normalize_fn=None: inputs,
    )
    prediction_calls = [np.array([0, 1, 2, 0]), np.array([1, 1, 0, 0])] + [
        np.array([1, 1, 0, 0]) for _ in range(4)
    ]
    monkeypatch.setattr(
        experiments,
        "evaluate_predictions",
        lambda *args, **kwargs: prediction_calls.pop(0),
    )
    monkeypatch.setattr(
        experiments, "recompress_tensor_batch", lambda batch, qualities: batch
    )

    experiments.run_multimodel_sweep(
        data_root=data_root,
        results_dir=results_dir,
        max_samples=4,
        attack="fgsm",
        device="cpu",
        model_names=("mock_imagenet",),
        ranges=[("R1", 75, 50)],
        generations=(3,),
        imagenet_root=imagenet_root,
    )

    assert seen_roots == [imagenet_root]
    assert not prediction_calls


@pytest.mark.parametrize("max_samples", [0, -1])
def test_run_cifar_smoke_rejects_non_positive_max_samples(
    monkeypatch, tmp_path, max_samples
):
    experiments = _experiments_module()
    data_root = tmp_path / "data"
    _create_all_expected_resources(data_root)

    monkeypatch.setattr(
        experiments.data,
        "load_cifar10_test_from_tar",
        lambda *args, **kwargs: pytest.fail(
            "run_cifar_smoke should validate max_samples before loading data"
        ),
    )

    with pytest.raises(ValueError, match="max_samples.*>= 1"):
        experiments.run_cifar_smoke(
            data_root=data_root,
            results_dir=tmp_path / "results",
            max_samples=max_samples,
            attack="fgsm",
            device="cpu",
        )


@pytest.mark.parametrize("mode", ["smoke", "cifar"])
def test_run_experiments_cli_cifar_modes_invoke_smoke_runner(
    monkeypatch, tmp_path, capsys, mode
):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_cifar_smoke(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "smoke",
            "rows": 12,
            "metrics_csv": str(tmp_path / "metrics.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(script.experiments, "run_cifar_smoke", fake_run_cifar_smoke)

    exit_code = script.main(
        [
            "--mode",
            mode,
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            "3",
            "--device",
            "cpu",
            "--attack",
            "pgd",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [
        {
            "data_root": tmp_path / "data",
            "results_dir": tmp_path / "results",
            "max_samples": 3,
            "attack": "pgd",
            "device": "cpu",
            "ranges": None,
            "generations": (1, 2, 3),
        }
    ]
    assert mode in captured.out
    assert "12 rows" in captured.out


def test_run_experiments_cli_passes_ranges_and_generations(monkeypatch, tmp_path):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_cifar_smoke(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "smoke",
            "rows": 48,
            "metrics_csv": str(tmp_path / "metrics.csv"),
            "source_data_csv": str(tmp_path / "source_data.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(script.experiments, "run_cifar_smoke", fake_run_cifar_smoke)

    exit_code = script.main(
        [
            "--mode",
            "smoke",
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            "3",
            "--device",
            "cpu",
            "--attack",
            "fgsm",
            "--ranges",
            "R1:75:50,R2:85:55",
            "--generations",
            "1,3,5",
        ]
    )

    assert exit_code == 0
    assert calls[0]["ranges"] == [("R1", 75, 50), ("R2", 85, 55)]
    assert calls[0]["generations"] == (1, 3, 5)


def test_run_experiments_cli_accepts_jpeg_aware_pgd_attack(monkeypatch, tmp_path):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_cifar_smoke(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "smoke",
            "rows": 4,
            "metrics_csv": str(tmp_path / "metrics.csv"),
            "source_data_csv": str(tmp_path / "source_data.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(script.experiments, "run_cifar_smoke", fake_run_cifar_smoke)

    exit_code = script.main(
        [
            "--mode",
            "smoke",
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            "3",
            "--device",
            "cpu",
            "--attack",
            "jpeg-aware-pgd",
        ]
    )

    assert exit_code == 0
    assert calls[0]["attack"] == "jpeg_aware_pgd"


def test_run_experiments_cli_multimodel_mode_invokes_multimodel_runner(
    monkeypatch,
    tmp_path,
    capsys,
):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_multimodel_sweep(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "multimodel",
            "rows": 8,
            "metrics_csv": str(tmp_path / "metrics.csv"),
            "source_data_csv": str(tmp_path / "source_data.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(
        script.experiments,
        "run_multimodel_sweep",
        fake_run_multimodel_sweep,
    )

    exit_code = script.main(
        [
            "--mode",
            "multimodel",
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            "5",
            "--device",
            "cpu",
            "--attack",
            "fgsm",
            "--models",
            "cifar_resnet18,imagenet_vit_b_16",
            "--ranges",
            "R1:75:50",
            "--generations",
            "3",
            "--imagenet-root",
            str(tmp_path / "imagenet"),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [
        {
            "data_root": tmp_path / "data",
            "results_dir": tmp_path / "results",
            "max_samples": 5,
            "attack": "fgsm",
            "device": "cpu",
            "model_names": ("cifar_resnet18", "imagenet_vit_b_16"),
            "ranges": [("R1", 75, 50)],
            "generations": (3,),
            "imagenet_root": tmp_path / "imagenet",
            "attack_batch_size": 64,
        }
    ]
    assert "multimodel" in captured.out
    assert "8 rows" in captured.out


def test_run_experiments_cli_frequency_mode_passes_model_attack_pairs(
    monkeypatch,
    tmp_path,
    capsys,
):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_frequency_diagnostics(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "frequency",
            "rows": 2,
            "frequency_metrics_csv": str(tmp_path / "frequency_metrics.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(
        script.experiments,
        "run_frequency_diagnostics",
        fake_run_frequency_diagnostics,
    )

    exit_code = script.main(
        [
            "--mode",
            "frequency",
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            "5",
            "--device",
            "cpu",
            "--model-attacks",
            "cifar_resnet18:fgsm,robustbench_wong2020fast:jpeg-aware-pgd",
            "--attack-batch-size",
            "2",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert calls == [
        {
            "data_root": tmp_path / "data",
            "results_dir": tmp_path / "results",
            "max_samples": 5,
            "attack": "fgsm",
            "attacks": None,
            "device": "cpu",
            "model_names": None,
            "model_attack_pairs": (
                ("cifar_resnet18", "fgsm"),
                ("robustbench_wong2020fast", "jpeg_aware_pgd"),
            ),
            "imagenet_root": None,
            "attack_batch_size": 2,
            "sample_offset": 0,
        }
    ]
    assert "frequency" in captured.out
    assert "diagnostics written" in captured.out


def test_run_experiments_cli_imagenet_smoke_fails_without_cifar_dispatch(
    monkeypatch, tmp_path, capsys
):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_cifar_smoke(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "smoke",
            "rows": 12,
            "metrics_csv": str(tmp_path / "metrics.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(script.experiments, "run_cifar_smoke", fake_run_cifar_smoke)

    exit_code = script.main(
        [
            "--mode",
            "imagenet-smoke",
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            "3",
            "--device",
            "cpu",
            "--attack",
            "fgsm",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert calls == []
    assert "imagenet-smoke mode is not implemented yet" in captured.err
    assert "No downloads are attempted" in captured.err


@pytest.mark.parametrize("max_samples", ["0", "-1"])
def test_run_experiments_cli_rejects_non_positive_max_samples(
    monkeypatch, tmp_path, capsys, max_samples
):
    script = _load_run_experiments_script()
    calls = []

    def fake_run_cifar_smoke(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "smoke",
            "rows": 12,
            "metrics_csv": str(tmp_path / "metrics.csv"),
            "summary_json": str(tmp_path / "summary.json"),
        }

    monkeypatch.setattr(script.experiments, "run_cifar_smoke", fake_run_cifar_smoke)

    exit_code = script.main(
        [
            "--mode",
            "smoke",
            "--data-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
            "--max-samples",
            max_samples,
            "--device",
            "cpu",
            "--attack",
            "fgsm",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 2
    assert calls == []
    assert "max-samples" in captured.err
    assert ">= 1" in captured.err
