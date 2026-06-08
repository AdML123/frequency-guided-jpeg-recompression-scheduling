from __future__ import annotations

import importlib

import pytest
import torch


def _attacks_module():
    try:
        return importlib.import_module("jpeg_defense.attacks")
    except ModuleNotFoundError as exc:
        pytest.fail(f"jpeg_defense.attacks module is missing: {exc}")


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3 * 4 * 4, 2, bias=False)
        weights = torch.linspace(-1.0, 1.0, 3 * 4 * 4)
        with torch.no_grad():
            self.linear.weight[0].copy_(weights)
            self.linear.weight[1].copy_(-weights)

    def forward(self, x):
        return self.linear(x.flatten(1))


@pytest.mark.parametrize(
    ("attack_name", "kwargs"),
    [
        ("fgsm_attack", {"epsilon": 0.05}),
        ("pgd_attack", {"epsilon": 0.05, "step_size": 0.02, "steps": 3}),
    ],
)
def test_attacks_preserve_shape_range_and_epsilon_bound(attack_name, kwargs):
    attacks = _attacks_module()
    model = TinyClassifier()
    inputs = torch.linspace(0.05, 0.95, 2 * 3 * 4 * 4).view(2, 3, 4, 4)
    labels = torch.tensor([0, 1])

    adversarial = getattr(attacks, attack_name)(model, inputs, labels, **kwargs)

    assert adversarial.shape == inputs.shape
    assert torch.all(adversarial >= 0.0)
    assert torch.all(adversarial <= 1.0)
    assert torch.max(torch.abs(adversarial - inputs)) <= kwargs["epsilon"] + 1e-6
    assert not adversarial.requires_grad


@pytest.mark.parametrize(
    ("attack_name", "kwargs"),
    [
        ("fgsm_attack", {"epsilon": 0.05}),
        ("pgd_attack", {"epsilon": 0.05, "step_size": 0.02, "steps": 1}),
    ],
)
def test_attacks_evaluate_model_with_normalized_inputs(attack_name, kwargs):
    attacks = _attacks_module()
    model = TinyClassifier()
    seen_batches: list[torch.Tensor] = []

    def recording_forward(x):
        seen_batches.append(x.detach().clone())
        return TinyClassifier.forward(model, x)

    model.forward = recording_forward
    inputs = torch.linspace(0.1, 0.9, 2 * 3 * 4 * 4).view(2, 3, 4, 4)
    labels = torch.tensor([0, 1])

    def normalize_fn(x):
        return x * 2.0 - 0.5

    getattr(attacks, attack_name)(
        model, inputs, labels, normalize_fn=normalize_fn, **kwargs
    )

    assert seen_batches
    assert seen_batches[0] == pytest.approx(normalize_fn(inputs))


def test_jpeg_aware_pgd_preserves_shape_range_and_epsilon_bound():
    attacks = _attacks_module()
    model = TinyClassifier()
    inputs = torch.linspace(0.05, 0.95, 2 * 3 * 4 * 4).view(2, 3, 4, 4)
    labels = torch.tensor([0, 1])

    adversarial = attacks.jpeg_aware_pgd_attack(
        model,
        inputs,
        labels,
        epsilon=0.05,
        step_size=0.02,
        steps=3,
        differentiable_defense=torch.nn.Identity(),
    )

    assert adversarial.shape == inputs.shape
    assert torch.all(adversarial >= 0.0)
    assert torch.all(adversarial <= 1.0)
    assert torch.max(torch.abs(adversarial - inputs)) <= 0.05 + 1e-6
    assert not adversarial.requires_grad


def test_jpeg_aware_pgd_optimizes_through_differentiable_defense():
    attacks = _attacks_module()
    model = TinyClassifier()
    inputs = torch.linspace(0.1, 0.9, 2 * 3 * 4 * 4).view(2, 3, 4, 4)
    labels = torch.tensor([0, 1])
    calls = []

    class RecordingDefense(torch.nn.Module):
        def forward(self, x):
            calls.append(x.detach().clone())
            return x

    attacks.jpeg_aware_pgd_attack(
        model,
        inputs,
        labels,
        epsilon=0.05,
        step_size=0.02,
        steps=2,
        differentiable_defense=RecordingDefense(),
    )

    assert len(calls) == 2
