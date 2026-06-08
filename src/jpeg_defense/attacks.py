"""Gradient-based adversarial attacks for image tensors in [0, 1]."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fgsm_attack(model, inputs, labels, epsilon, normalize_fn=None):
    """Return FGSM adversarial examples clipped to the input image range."""
    adversarial = inputs.detach().clone().requires_grad_(True)
    logits = model(_model_inputs(adversarial, normalize_fn))
    loss = F.cross_entropy(logits, labels.to(adversarial.device))
    loss.backward()

    perturbed = adversarial + float(epsilon) * adversarial.grad.sign()
    return _project(perturbed, inputs, float(epsilon)).detach()


def pgd_attack(model, inputs, labels, epsilon, step_size, steps, normalize_fn=None):
    """Return projected-gradient adversarial examples clipped to [0, 1]."""
    epsilon = float(epsilon)
    step_size = float(step_size)
    adversarial = inputs.detach().clone()

    for _step in range(int(steps)):
        adversarial = adversarial.detach().requires_grad_(True)
        logits = model(_model_inputs(adversarial, normalize_fn))
        loss = F.cross_entropy(logits, labels.to(adversarial.device))
        loss.backward()

        updated = adversarial + step_size * adversarial.grad.sign()
        adversarial = _project(updated, inputs, epsilon)

    return adversarial.detach()


def jpeg_aware_pgd_attack(
    model,
    inputs,
    labels,
    epsilon,
    step_size,
    steps,
    differentiable_defense,
    normalize_fn=None,
):
    """Return PGD examples optimized through a differentiable JPEG defense."""
    epsilon = float(epsilon)
    step_size = float(step_size)
    adversarial = inputs.detach().clone()
    differentiable_defense.to(adversarial.device)
    differentiable_defense.eval()

    for _step in range(int(steps)):
        adversarial = adversarial.detach().requires_grad_(True)
        defended = differentiable_defense(adversarial)
        logits = model(_model_inputs(defended, normalize_fn))
        loss = F.cross_entropy(logits, labels.to(adversarial.device))
        loss.backward()

        updated = adversarial + step_size * adversarial.grad.sign()
        adversarial = _project(updated, inputs, epsilon)

    return adversarial.detach()


def _model_inputs(inputs, normalize_fn):
    if normalize_fn is None:
        return inputs
    return normalize_fn(inputs)


def _project(candidate, reference, epsilon):
    delta = torch.clamp(candidate - reference, min=-epsilon, max=epsilon)
    return torch.clamp(reference + delta, min=0.0, max=1.0)
