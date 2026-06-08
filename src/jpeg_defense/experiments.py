"""Experiment helpers and CIFAR-10 smoke-run orchestration."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from jpeg_defense import (
    attacks,
    data,
    diffjpeg_adapter,
    frequency,
    jpeg,
    metrics,
    models,
    schedules,
)
from jpeg_defense.paths import ensure_results_dir, required_file


_CSV_FIELDS = [
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
]


_SOURCE_DATA_FIELDS = [
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
]

_MULTIMODEL_CSV_FIELDS = [
    "model",
    "model_family",
    *_CSV_FIELDS,
]

_MULTIMODEL_SOURCE_DATA_FIELDS = [
    "model",
    "model_family",
    *_SOURCE_DATA_FIELDS,
]

_FREQUENCY_CSV_FIELDS = [
    "dataset",
    "model",
    "attack",
    "samples",
    "omega_delta",
    "energy_weighted_sum",
    "energy_sum",
    "tau",
    "omega_delta_relation",
    "tau_source",
    "tau_resnet18",
    "tau_all",
    "tau_delta",
    "tau_classification_stable",
    "jpeg_aware_attack",
    "predicted_best_schedule",
    "prediction_rule",
]


def tensor_to_pil_rgb(image_tensor):
    """Convert a 3xHxW float tensor in [0, 1] to a PIL RGB image."""
    if image_tensor.ndim != 3 or image_tensor.shape[0] != 3:
        raise ValueError(
            f"image_tensor must have shape 3xHxW, got {tuple(image_tensor.shape)}"
        )

    array = (
        image_tensor.detach()
        .cpu()
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(array, mode="RGB")


def pil_rgb_to_tensor(image):
    """Convert a PIL image to a 3xHxW float32 RGB tensor in [0, 1]."""
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float32)
    return tensor.div(255.0)


def recompress_tensor_batch(batch, qualities):
    """Apply sequential PIL JPEG recompression to each tensor image in a batch."""
    qualities = list(qualities)
    recompressed = []
    for image_tensor in batch.detach().cpu():
        image = tensor_to_pil_rgb(image_tensor)
        recompressed_image = jpeg.recompress_many_generations(image, qualities)
        recompressed.append(pil_rgb_to_tensor(recompressed_image))

    if not recompressed:
        return batch.detach().clone()
    return torch.stack(recompressed).to(device=batch.device, dtype=batch.dtype)


def evaluate_predictions(model, images, batch_size, device, normalize_fn=None):
    """Return predicted class ids for images as a NumPy array."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size!r}")

    device = torch.device(device)
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, images.shape[0], int(batch_size)):
            batch = images[start : start + int(batch_size)].to(device)
            model_batch = normalize_fn(batch) if normalize_fn is not None else batch
            logits = model(model_batch)
            predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    if not predictions:
        return np.asarray([], dtype=np.int64)
    return np.concatenate(predictions)


def run_cifar_smoke(
    data_root,
    results_dir,
    max_samples=8,
    attack="fgsm",
    device="cpu",
    ranges=None,
    generations=(1, 2, 3),
):
    """Run a small offline CIFAR-10 attack/recompression smoke experiment."""
    max_samples = _require_positive_max_samples(max_samples)
    attack = _canonical_attack_name(attack)
    qf_ranges = _normalize_ranges(ranges)
    generation_grid = tuple(int(value) for value in generations)
    data_root = Path(data_root)
    results_dir = ensure_results_dir(results_dir)
    cifar_archive, checkpoint = _require_cifar_smoke_resources(data_root)
    images, labels = data.load_cifar10_test_from_tar(
        cifar_archive, limit=max_samples
    )

    device = torch.device(device)
    model = models.build_cifar_resnet18()
    model = models.load_state_dict_file(model, checkpoint, device=device)
    model.to(device)
    labels = labels.to(device)
    images = images.to(device)

    clean_predictions = evaluate_predictions(
        model,
        images,
        batch_size=min(64, max_samples),
        device=device,
        normalize_fn=None,
    )
    label_array = labels.detach().cpu().numpy()
    clean_correct = int(np.count_nonzero(clean_predictions == label_array))
    clean_accuracy = float(clean_correct / len(label_array)) if len(label_array) else 0.0
    static_attack = not _requires_schedule_specific_attack(attack)
    static_adversarial = None
    static_no_defense_successes = None
    static_no_defense_asr = None
    if static_attack:
        static_adversarial = _make_adversarial_examples(model, images, labels, attack)
        static_no_defense_predictions = evaluate_predictions(
            model,
            static_adversarial,
            batch_size=min(64, max_samples),
            device=device,
            normalize_fn=None,
        )
        static_no_defense_successes, _ = _attack_success_counts(
            clean_predictions, static_no_defense_predictions, label_array
        )
        static_no_defense_asr = metrics.attack_success_rate(
            clean_predictions, static_no_defense_predictions, label_array
        )

    rows = []
    source_rows = []
    for range_name, start_qf, end_qf in qf_ranges:
        for schedule_name, schedule_fn in _schedule_functions().items():
            for generation_count in generation_grid:
                qfs = schedule_fn(start_qf, end_qf, generation_count)
                if static_attack:
                    adversarial = static_adversarial
                    no_defense_successes = static_no_defense_successes
                    no_defense_asr = static_no_defense_asr
                else:
                    adversarial = _make_adversarial_examples(
                        model,
                        images,
                        labels,
                        attack,
                        qfs=qfs,
                        device=device,
                    )
                    no_defense_predictions = evaluate_predictions(
                        model,
                        adversarial,
                        batch_size=min(64, max_samples),
                        device=device,
                        normalize_fn=None,
                    )
                    no_defense_successes, _ = _attack_success_counts(
                        clean_predictions, no_defense_predictions, label_array
                    )
                    no_defense_asr = metrics.attack_success_rate(
                        clean_predictions, no_defense_predictions, label_array
                    )
                defended = recompress_tensor_batch(adversarial, qfs)
                defended_predictions = evaluate_predictions(
                    model,
                    defended,
                    batch_size=min(64, max_samples),
                    device=device,
                    normalize_fn=None,
                )
                successes, total = _attack_success_counts(
                    clean_predictions, defended_predictions, label_array
                )
                ci_low, ci_high = metrics.binomial_ci(successes, total)
                asr = metrics.attack_success_rate(
                    clean_predictions, defended_predictions, label_array
                )
                qfs_text = ",".join(str(qf) for qf in qfs)
                rows.append(
                    {
                        "dataset": "cifar10",
                        "attack": attack,
                        "range_name": range_name,
                        "start_qf": start_qf,
                        "end_qf": end_qf,
                        "schedule": schedule_name,
                        "generations": generation_count,
                        "qfs": qfs_text,
                        "clean_accuracy": clean_accuracy,
                        "clean_correct": clean_correct,
                        "no_defense_asr": no_defense_asr,
                        "no_defense_successes": no_defense_successes,
                        "asr": asr,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "successes": successes,
                        "total": total,
                    }
                )
                source_rows.extend(
                    _source_data_rows(
                        dataset="cifar10",
                        attack=attack,
                        range_name=range_name,
                        start_qf=start_qf,
                        end_qf=end_qf,
                        schedule=schedule_name,
                        generations=generation_count,
                        qfs=qfs_text,
                        labels=label_array,
                        clean_predictions=clean_predictions,
                        defended_predictions=defended_predictions,
                    )
                )

    metrics_path = results_dir / "metrics.csv"
    source_data_path = results_dir / "source_data.csv"
    summary_path = results_dir / "summary.json"
    _write_metrics_csv(metrics_path, rows)
    _write_source_data_csv(source_data_path, source_rows)

    summary = {
        "mode": "smoke",
        "dataset": "cifar10",
        "attack": attack,
        "max_samples": max_samples,
        "rows": len(rows),
        "metrics_csv": str(metrics_path),
        "source_data_csv": str(source_data_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_multimodel_sweep(
    data_root,
    results_dir,
    max_samples=8,
    attack="fgsm",
    device="cpu",
    model_names=None,
    ranges=None,
    generations=(1, 2, 3),
    imagenet_root=None,
    attack_batch_size=64,
):
    """Run the schedule grid across registered CIFAR-10 and ImageNet models."""
    max_samples = _require_positive_max_samples(max_samples)
    attack = _canonical_attack_name(attack)
    qf_ranges = _normalize_ranges(ranges)
    generation_grid = tuple(int(value) for value in generations)
    data_root = Path(data_root)
    results_dir = ensure_results_dir(results_dir)
    selected_names = _normalize_model_names(model_names)
    device = torch.device(device)

    rows = []
    source_rows = []
    for model_name in selected_names:
        spec = models.get_model_spec(model_name)
        model, normalize_fn = models.build_model_from_spec(
            spec,
            data_root,
            device=device,
        )
        images, labels = _load_inputs_for_spec(
            spec,
            data_root,
            max_samples,
            imagenet_root=imagenet_root,
        )
        images = images.to(device)
        labels = labels.to(device)

        model_rows, model_source_rows = _run_schedule_grid(
            dataset=spec.dataset,
            model_name=spec.name,
            model_family=spec.family,
            model=model,
            images=images,
            labels=labels,
            attack=attack,
            device=device,
            qf_ranges=qf_ranges,
            generation_grid=generation_grid,
            max_samples=max_samples,
            normalize_fn=normalize_fn,
            attack_batch_size=attack_batch_size,
        )
        rows.extend(model_rows)
        source_rows.extend(model_source_rows)

    metrics_path = results_dir / "metrics.csv"
    source_data_path = results_dir / "source_data.csv"
    summary_path = results_dir / "summary.json"
    _write_csv(metrics_path, rows, _MULTIMODEL_CSV_FIELDS)
    _write_csv(source_data_path, source_rows, _MULTIMODEL_SOURCE_DATA_FIELDS)

    summary = {
        "mode": "multimodel",
        "models": list(selected_names),
        "attack": attack,
        "max_samples": max_samples,
        "rows": len(rows),
        "metrics_csv": str(metrics_path),
        "source_data_csv": str(source_data_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_frequency_diagnostics(
    data_root,
    results_dir,
    max_samples=16,
    attack="fgsm",
    attacks=None,
    device="cpu",
    model_names=None,
    model_attack_pairs=None,
    imagenet_root=None,
    attack_batch_size=64,
    sample_offset=0,
):
    """Measure DCT-frequency centroids for requested model/attack conditions."""
    max_samples = _require_positive_max_samples(max_samples)
    data_root = Path(data_root)
    results_dir = ensure_results_dir(results_dir)
    device = torch.device(device)
    condition_pairs = _normalize_frequency_pairs(
        attack=attack,
        attacks=attacks,
        model_names=model_names,
        model_attack_pairs=model_attack_pairs,
    )

    rows = []
    metrics_path = results_dir / "frequency_metrics.csv"
    summary_path = results_dir / "summary.json"
    _write_csv(metrics_path, rows, _FREQUENCY_CSV_FIELDS)
    for model_name, attack_names in _group_attacks_by_model(condition_pairs).items():
        spec = models.get_model_spec(model_name)
        images, labels = _load_inputs_for_spec(
            spec,
            data_root,
            max_samples,
            imagenet_root=imagenet_root,
            sample_offset=sample_offset,
        )
        model, normalize_fn = models.build_model_from_spec(
            spec,
            data_root,
            device=device,
        )
        model.to(device).eval()
        images = images.to(device)
        labels = labels.to(device)
        tau = frequency.elimination_threshold_for_dataset(spec.dataset)

        for attack_name in attack_names:
            qfs = None
            if _requires_schedule_specific_attack(attack_name):
                qfs = schedules.geometric_schedule(75, 50, 3)
            adversarial = _make_adversarial_examples(
                model,
                images,
                labels,
                attack_name,
                qfs=qfs,
                device=device,
                normalize_fn=normalize_fn,
                batch_size=attack_batch_size,
            )
            clean_cpu = images.detach().cpu()
            adversarial_cpu = adversarial.detach().cpu()
            weighted_sum, energy_sum = frequency.frequency_energy_moments(
                frequency.block_dct_coefficients(adversarial_cpu - clean_cpu)
            )
            omega_delta = frequency.perturbation_frequency_centroid(
                clean_cpu,
                adversarial_cpu,
            )
            jpeg_aware_attack = _requires_schedule_specific_attack(attack_name)
            predicted_schedule = frequency.predict_schedule_from_omega_delta(
                omega_delta,
                tau=tau,
                jpeg_aware_attack=jpeg_aware_attack,
            )
            rows.append(
                {
                    "dataset": spec.dataset,
                    "model": spec.name,
                    "attack": attack_name,
                    "samples": int(images.shape[0]),
                    "omega_delta": omega_delta,
                    "energy_weighted_sum": weighted_sum,
                    "energy_sum": energy_sum,
                    "tau": tau,
                    "omega_delta_relation": frequency.omega_delta_relation(
                        omega_delta,
                        tau,
                    ),
                    "tau_source": "dataset_default",
                    "tau_resnet18": tau,
                    "tau_all": tau,
                    "tau_delta": 0.0,
                    "tau_classification_stable": True,
                    "jpeg_aware_attack": jpeg_aware_attack,
                    "predicted_best_schedule": predicted_schedule,
                    "prediction_rule": frequency.prediction_rule_label(
                        jpeg_aware_attack,
                        tau=tau,
                    ),
                }
            )
            _write_csv(metrics_path, rows, _FREQUENCY_CSV_FIELDS)
            _write_frequency_summary(
                summary_path,
                condition_pairs=condition_pairs,
                max_samples=max_samples,
                sample_offset=sample_offset,
                row_count=len(rows),
                metrics_path=metrics_path,
                completed=False,
            )

    summary = _write_frequency_summary(
        summary_path,
        condition_pairs=condition_pairs,
        max_samples=max_samples,
        sample_offset=sample_offset,
        row_count=len(rows),
        metrics_path=metrics_path,
        completed=True,
    )
    return summary


def _write_frequency_summary(
    summary_path,
    *,
    condition_pairs,
    max_samples,
    sample_offset,
    row_count,
    metrics_path,
    completed,
):
    summary = {
        "mode": "frequency",
        "models": list(_group_attacks_by_model(condition_pairs).keys()),
        "attacks": sorted({pair_attack for _model, pair_attack in condition_pairs}),
        "max_samples": max_samples,
        "sample_offset": int(sample_offset or 0),
        "rows": int(row_count),
        "completed": bool(completed),
        "frequency_metrics_csv": str(metrics_path),
        "summary_json": str(summary_path),
    }
    Path(summary_path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_mcnemar_audit_sweep(
    data_root,
    results_dir,
    max_samples=10000,
    attacks=("fgsm", "pgd"),
    device="cpu",
    model_names=None,
    ranges=None,
    generation=5,
    imagenet_root=None,
    attack_batch_size=64,
    sample_offset=0,
):
    """Run the FL-vs-Fix McNemar audit with incremental CSV writes."""
    max_samples = _require_positive_max_samples(max_samples)
    selected_attacks = tuple(_canonical_attack_name(value) for value in attacks)
    qf_ranges = _normalize_ranges(ranges)
    data_root = Path(data_root)
    results_dir = ensure_results_dir(results_dir)
    selected_names = _normalize_model_names(model_names)
    device = torch.device(device)
    generation = int(generation)
    if generation < 1:
        raise ValueError(f"generation must be >= 1, got {generation!r}")

    metrics_path = results_dir / "metrics.csv"
    source_data_path = results_dir / "source_data.csv"
    summary_path = results_dir / "summary.json"
    rows = []
    source_rows = []
    completed = []
    _write_csv(metrics_path, rows, _MULTIMODEL_CSV_FIELDS)
    _write_csv(source_data_path, source_rows, _MULTIMODEL_SOURCE_DATA_FIELDS)

    for model_name in selected_names:
        spec = models.get_model_spec(model_name)
        model, normalize_fn = models.build_model_from_spec(
            spec,
            data_root,
            device=device,
        )
        images, labels = _load_inputs_for_spec(
            spec,
            data_root,
            max_samples,
            imagenet_root=imagenet_root,
            sample_offset=sample_offset,
        )
        images = images.to(device)
        labels = labels.to(device)

        for attack in selected_attacks:
            for range_entry in qf_ranges:
                model_rows, model_source_rows = _run_schedule_grid(
                    dataset=spec.dataset,
                    model_name=spec.name,
                    model_family=spec.family,
                    model=model,
                    images=images,
                    labels=labels,
                    attack=attack,
                    device=device,
                    qf_ranges=(range_entry,),
                    generation_grid=(generation,),
                    max_samples=max_samples,
                    normalize_fn=normalize_fn,
                    attack_batch_size=attack_batch_size,
                    schedule_names=("front_loaded", "fixed"),
                    sample_offset=sample_offset,
                )
                rows.extend(model_rows)
                source_rows.extend(model_source_rows)
                completed.append(
                    {
                        "model": spec.name,
                        "attack": attack,
                        "range": range_entry[0],
                    }
                )
                _write_csv(metrics_path, rows, _MULTIMODEL_CSV_FIELDS)
                _write_csv(source_data_path, source_rows, _MULTIMODEL_SOURCE_DATA_FIELDS)
                summary_path.write_text(
                    json.dumps(
                        {
                            "mode": "mcnemar-audit",
                            "models": list(selected_names),
                            "attacks": list(selected_attacks),
                            "max_samples": max_samples,
                        "generation": generation,
                        "sample_offset": sample_offset,
                            "rows": len(rows),
                            "source_rows": len(source_rows),
                            "completed": completed,
                            "metrics_csv": str(metrics_path),
                            "source_data_csv": str(source_data_path),
                            "summary_json": str(summary_path),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    return json.loads(summary_path.read_text(encoding="utf-8"))


def _normalize_frequency_pairs(
    *,
    attack,
    attacks,
    model_names,
    model_attack_pairs,
):
    if model_attack_pairs is not None:
        pairs = []
        for model_name, pair_attack in model_attack_pairs:
            pairs.append((str(model_name), _canonical_attack_name(pair_attack)))
        if not pairs:
            raise ValueError("model_attack_pairs must contain at least one pair")
        return tuple(pairs)

    if model_names is None:
        selected_models = ("cifar_resnet18",)
    else:
        selected_models = _normalize_model_names(model_names)
    selected_attacks = _normalize_attack_names(attacks, attack)
    return tuple(
        (model_name, attack_name)
        for model_name in selected_models
        for attack_name in selected_attacks
    )


def _normalize_attack_names(attacks, attack):
    if attacks is None:
        values = (attack,)
    elif isinstance(attacks, str):
        values = (attacks,)
    else:
        values = tuple(attacks)
    if not values:
        raise ValueError("attacks must contain at least one attack")
    return tuple(_canonical_attack_name(value) for value in values)


def _group_attacks_by_model(condition_pairs):
    grouped = {}
    for model_name, attack_name in condition_pairs:
        grouped.setdefault(model_name, []).append(attack_name)
    return grouped


def _require_positive_max_samples(max_samples):
    try:
        value = int(max_samples)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"max_samples must be an integer >= 1, got {max_samples!r}"
        ) from exc
    if value < 1:
        raise ValueError(f"max_samples must be >= 1, got {max_samples!r}")
    return value


def _normalize_ranges(ranges):
    if ranges is None:
        return (("R1", 75, 50),)
    normalized = []
    for range_item in ranges:
        if len(range_item) != 3:
            raise ValueError(f"ranges entries must be (name, start_qf, end_qf), got {range_item!r}")
        name, start_qf, end_qf = range_item
        normalized.append((str(name), int(start_qf), int(end_qf)))
    return tuple(normalized)


def _require_cifar_smoke_resources(data_root):
    root = Path(data_root)
    cifar_archive = required_file(
        root / "cifar10" / "cifar-10-python.tar.gz",
        "CIFAR-10 archive",
    )
    checkpoint = required_file(
        root / "checkpoints" / "cifar" / "CIFAR10_ResNet18_epoch_20.pt",
        "CIFAR-10 checkpoint",
    )
    return cifar_archive, checkpoint


def _normalize_model_names(model_names):
    if model_names is None:
        return tuple(spec.name for spec in models.list_model_specs())
    names = tuple(str(name) for name in model_names)
    if not names:
        raise ValueError("model_names must contain at least one model")
    return names


def _load_inputs_for_spec(spec, data_root, max_samples, imagenet_root=None, sample_offset=0):
    if spec.dataset == "cifar10":
        cifar_archive, _checkpoint = _require_cifar_smoke_resources(data_root)
        kwargs = {"limit": max_samples}
        if int(sample_offset or 0):
            kwargs["offset"] = sample_offset
        return data.load_cifar10_test_from_tar(cifar_archive, **kwargs)
    if spec.dataset == "imagenet":
        if imagenet_root is None:
            imagenet_root = Path(data_root) / "hf_mirror" / "imagenet1k_val_1k"
        else:
            imagenet_root = Path(imagenet_root)
        return data.load_imagenet_mirror_tensors(
            imagenet_root,
            limit=max_samples,
            image_size=spec.input_size,
        )
    raise ValueError(f"Unsupported model dataset {spec.dataset!r}")


def _run_schedule_grid(
    *,
    dataset,
    model_name,
    model_family,
    model,
    images,
    labels,
    attack,
    device,
    qf_ranges,
    generation_grid,
    max_samples,
    normalize_fn,
    attack_batch_size,
    schedule_names=None,
    sample_offset=0,
):
    clean_predictions = evaluate_predictions(
        model,
        images,
        batch_size=min(64, max_samples),
        device=device,
        normalize_fn=normalize_fn,
    )
    label_array = labels.detach().cpu().numpy()
    clean_correct = int(np.count_nonzero(clean_predictions == label_array))
    clean_accuracy = float(clean_correct / len(label_array)) if len(label_array) else 0.0
    static_attack = not _requires_schedule_specific_attack(attack)
    static_adversarial = None
    static_no_defense_successes = None
    static_no_defense_asr = None
    if static_attack:
        static_adversarial = _make_adversarial_examples(
            model,
            images,
            labels,
            attack,
            normalize_fn=normalize_fn,
            batch_size=attack_batch_size,
        )
        static_no_defense_predictions = evaluate_predictions(
            model,
            static_adversarial,
            batch_size=min(64, max_samples),
            device=device,
            normalize_fn=normalize_fn,
        )
        static_no_defense_successes, _ = _attack_success_counts(
            clean_predictions, static_no_defense_predictions, label_array
        )
        static_no_defense_asr = metrics.attack_success_rate(
            clean_predictions, static_no_defense_predictions, label_array
        )

    rows = []
    source_rows = []
    schedule_functions = _schedule_functions()
    selected_schedule_names = tuple(schedule_names or schedule_functions.keys())
    for range_name, start_qf, end_qf in qf_ranges:
        for schedule_name in selected_schedule_names:
            schedule_fn = schedule_functions[schedule_name]
            for generation_count in generation_grid:
                qfs = schedule_fn(start_qf, end_qf, generation_count)
                if static_attack:
                    adversarial = static_adversarial
                    no_defense_successes = static_no_defense_successes
                    no_defense_asr = static_no_defense_asr
                else:
                    adversarial = _make_adversarial_examples(
                        model,
                        images,
                        labels,
                        attack,
                        qfs=qfs,
                        device=device,
                        normalize_fn=normalize_fn,
                        batch_size=attack_batch_size,
                    )
                    no_defense_predictions = evaluate_predictions(
                        model,
                        adversarial,
                        batch_size=min(64, max_samples),
                        device=device,
                        normalize_fn=normalize_fn,
                    )
                    no_defense_successes, _ = _attack_success_counts(
                        clean_predictions, no_defense_predictions, label_array
                    )
                    no_defense_asr = metrics.attack_success_rate(
                        clean_predictions, no_defense_predictions, label_array
                    )
                defended = recompress_tensor_batch(adversarial, qfs)
                defended_predictions = evaluate_predictions(
                    model,
                    defended,
                    batch_size=min(64, max_samples),
                    device=device,
                    normalize_fn=normalize_fn,
                )
                successes, total = _attack_success_counts(
                    clean_predictions, defended_predictions, label_array
                )
                ci_low, ci_high = metrics.binomial_ci(successes, total)
                asr = metrics.attack_success_rate(
                    clean_predictions, defended_predictions, label_array
                )
                qfs_text = ",".join(str(qf) for qf in qfs)
                rows.append(
                    {
                        "model": model_name,
                        "model_family": model_family,
                        "dataset": dataset,
                        "attack": attack,
                        "range_name": range_name,
                        "start_qf": start_qf,
                        "end_qf": end_qf,
                        "schedule": schedule_name,
                        "generations": generation_count,
                        "qfs": qfs_text,
                        "clean_accuracy": clean_accuracy,
                        "clean_correct": clean_correct,
                        "no_defense_asr": no_defense_asr,
                        "no_defense_successes": no_defense_successes,
                        "asr": asr,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "successes": successes,
                        "total": total,
                    }
                )
                for source_row in _source_data_rows(
                    dataset=dataset,
                    attack=attack,
                    range_name=range_name,
                    start_qf=start_qf,
                    end_qf=end_qf,
                    schedule=schedule_name,
                    generations=generation_count,
                    qfs=qfs_text,
                    labels=label_array,
                    clean_predictions=clean_predictions,
                    defended_predictions=defended_predictions,
                    sample_offset=sample_offset,
                ):
                    source_rows.append(
                        {
                            "model": model_name,
                            "model_family": model_family,
                            **source_row,
                        }
                    )
    return rows, source_rows


def _make_adversarial_examples(
    model,
    images,
    labels,
    attack,
    qfs=None,
    device=None,
    normalize_fn=None,
    batch_size=None,
):
    if batch_size is not None and int(batch_size) < images.shape[0]:
        chunks = []
        for start in range(0, images.shape[0], int(batch_size)):
            chunks.append(
                _make_adversarial_examples(
                    model,
                    images[start : start + int(batch_size)],
                    labels[start : start + int(batch_size)],
                    attack,
                    qfs=qfs,
                    device=device,
                    normalize_fn=normalize_fn,
                    batch_size=None,
                )
            )
        return torch.cat(chunks, dim=0)

    attack = _canonical_attack_name(attack)
    if attack == "fgsm":
        return attacks.fgsm_attack(
            model,
            images,
            labels,
            epsilon=8.0 / 255.0,
            normalize_fn=normalize_fn,
        )
    if attack == "pgd":
        return attacks.pgd_attack(
            model,
            images,
            labels,
            epsilon=8.0 / 255.0,
            step_size=2.0 / 255.0,
            steps=20,
            normalize_fn=normalize_fn,
        )
    if attack == "jpeg_aware_pgd":
        if qfs is None:
            raise ValueError("jpeg_aware_pgd requires a JPEG quality schedule")
        _, _, height, width = images.shape
        attack_device = device if device is not None else images.device
        differentiable_defense = diffjpeg_adapter.build_diffjpeg_schedule(
            height=height,
            width=width,
            qfs=qfs,
            device=attack_device,
        )
        return attacks.jpeg_aware_pgd_attack(
            model,
            images,
            labels,
            epsilon=8.0 / 255.0,
            step_size=2.0 / 255.0,
            steps=20,
            differentiable_defense=differentiable_defense,
            normalize_fn=normalize_fn,
        )
    raise ValueError(
        f"Unsupported attack {attack!r}; expected 'fgsm', 'pgd', or 'jpeg_aware_pgd'"
    )


def _canonical_attack_name(attack):
    normalized = str(attack).replace("-", "_")
    if normalized in {"fgsm", "pgd", "jpeg_aware_pgd"}:
        return normalized
    raise ValueError(
        f"Unsupported attack {attack!r}; expected 'fgsm', 'pgd', or 'jpeg_aware_pgd'"
    )


def _requires_schedule_specific_attack(attack):
    return _canonical_attack_name(attack) == "jpeg_aware_pgd"


def _schedule_functions():
    return {
        "geometric": schedules.geometric_schedule,
        "arithmetic": schedules.arithmetic_schedule,
        "fixed": schedules.fixed_schedule,
        "front_loaded": schedules.reverse_geometric_schedule,
    }


def _attack_success_counts(clean_pred, adv_pred, labels):
    clean = np.asarray(clean_pred)
    adv = np.asarray(adv_pred)
    label_array = np.asarray(labels)
    clean_correct = clean == label_array
    attack_success = clean_correct & (adv != label_array)
    return int(np.count_nonzero(attack_success)), int(np.count_nonzero(clean_correct))


def _source_data_rows(
    *,
    dataset,
    attack,
    range_name,
    start_qf,
    end_qf,
    schedule,
    generations,
    qfs,
    labels,
    clean_predictions,
    defended_predictions,
    sample_offset=0,
):
    rows = []
    sample_offset = int(sample_offset or 0)
    for sample_index, (label, clean_pred, defended_pred) in enumerate(
        zip(labels, clean_predictions, defended_predictions)
    ):
        clean_correct = int(clean_pred) == int(label)
        attack_success = clean_correct and int(defended_pred) != int(label)
        rows.append(
            {
                "dataset": dataset,
                "attack": attack,
                "range_name": range_name,
                "start_qf": start_qf,
                "end_qf": end_qf,
                "schedule": schedule,
                "generations": generations,
                "qfs": qfs,
                "sample_index": sample_offset + sample_index,
                "label": int(label),
                "clean_pred": int(clean_pred),
                "defended_pred": int(defended_pred),
                "clean_correct": clean_correct,
                "attack_success": attack_success,
            }
        )
    return rows


def _write_metrics_csv(path, rows):
    _write_csv(path, rows, _CSV_FIELDS)


def _write_source_data_csv(path, rows):
    _write_csv(path, rows, _SOURCE_DATA_FIELDS)


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
