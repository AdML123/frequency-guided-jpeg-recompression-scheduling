"""Survival-bound estimator for JPEG perturbation schedules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def survival_bound(initial_mu: float, ratios: Iterable[float]) -> float:
    """Unroll mu_{i+1} <= r_i * mu_i + 1 for a scalar residual."""
    bound = float(initial_mu)
    for ratio in ratios:
        bound = float(ratio) * bound + 1.0
    return bound


def survival_score(
    initial_mu: np.ndarray,
    ratios: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Return weighted fraction of frequencies with final bound >= 1."""
    mu = np.asarray(initial_mu, dtype=float)
    ratio_values = np.asarray(ratios, dtype=float)
    if ratio_values.shape == mu.shape:
        final_bound = ratio_values * mu + 1.0
    elif ratio_values.ndim == mu.ndim + 1 and ratio_values.shape[1:] == mu.shape:
        final_bound = mu.copy()
        for stage_ratios in ratio_values:
            final_bound = stage_ratios * final_bound + 1.0
    else:
        raise ValueError(
            "ratios must have the same shape as initial_mu or be a stage-by-frequency array, "
            f"got {ratio_values.shape} for initial_mu shape {mu.shape}"
        )

    if weights is None:
        weight_values = np.full(mu.shape, 1.0 / mu.size, dtype=float)
    else:
        weight_values = np.asarray(weights, dtype=float)
        if weight_values.shape != mu.shape:
            raise ValueError(
                f"weights must have shape {mu.shape}, got {weight_values.shape}"
            )
        total = weight_values.sum()
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        weight_values = weight_values / total

    return float(weight_values[final_bound >= 1.0].sum())
