"""Evaluation metrics for JPEG defense experiments."""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
from numpy.typing import ArrayLike


def attack_success_rate(clean_pred: ArrayLike, adv_pred: ArrayLike, labels: ArrayLike) -> float:
    """Return the fraction of clean-correct examples misclassified after attack."""
    clean = np.asarray(clean_pred)
    adv = np.asarray(adv_pred)
    label_array = np.asarray(labels)
    _require_same_shape(
        ("clean_pred", clean), ("adv_pred", adv), ("labels", label_array)
    )

    clean_correct = clean == label_array
    clean_correct_count = int(np.count_nonzero(clean_correct))
    if clean_correct_count == 0:
        return 0.0

    attack_success = clean_correct & (adv != label_array)
    return float(np.count_nonzero(attack_success) / clean_correct_count)


def binomial_ci(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial proportion."""
    _require_finite_integer("successes", successes)
    _require_finite_integer("total", total)
    if not math.isfinite(float(confidence)):
        raise ValueError(f"confidence must be finite, got {confidence!r}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total!r}")
    if successes < 0:
        raise ValueError(f"successes must be non-negative, got {successes!r}")
    if successes > total:
        raise ValueError(
            f"successes must be less than or equal to total; got {successes!r} > {total!r}"
        )
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if total == 0:
        return (0.0, 0.0)

    proportion = successes / total
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) + z_squared / (4.0 * total)) / total
    )

    return ((center - margin) / denominator, (center + margin) / denominator)


def mcnemar_table(a_success: ArrayLike, b_success: ArrayLike) -> tuple[int, int]:
    """Return paired discordant counts for McNemar's exact test."""
    a = np.asarray(a_success, dtype=bool)
    b = np.asarray(b_success, dtype=bool)
    _require_same_shape(("a_success", a), ("b_success", b))

    n01 = int(np.count_nonzero(~a & b))
    n10 = int(np.count_nonzero(a & ~b))
    return n01, n10


def mcnemar_pvalue(n01: int, n10: int) -> float:
    """Return the exact two-sided McNemar binomial p-value."""
    discordant = n01 + n10
    if discordant == 0:
        return 1.0

    try:
        from scipy.stats import binomtest

        return float(binomtest(min(n01, n10), discordant, p=0.5).pvalue)
    except ImportError:
        return _exact_two_sided_binomial_pvalue(min(n01, n10), discordant)


def _exact_two_sided_binomial_pvalue(successes: int, total: int) -> float:
    observed_probability = _binomial_probability(successes, total)
    pvalue = 0.0
    for count in range(total + 1):
        probability = _binomial_probability(count, total)
        if probability <= observed_probability + 1e-15:
            pvalue += probability
    return min(1.0, pvalue)


def _binomial_probability(successes: int, total: int) -> float:
    return math.comb(total, successes) * math.pow(0.5, total)


def _require_finite_integer(name: str, value: int) -> None:
    value_as_float = float(value)
    if not math.isfinite(value_as_float) or not value_as_float.is_integer():
        raise ValueError(f"{name} must be a finite integer, got {value!r}")


def _require_same_shape(*named_arrays: tuple[str, np.ndarray]) -> None:
    first_array = named_arrays[0][1]
    mismatched = [
        (name, array.shape)
        for name, array in named_arrays
        if array.shape != first_array.shape
    ]
    if mismatched:
        shapes = ", ".join(
            f"{name}.shape={array.shape}" for name, array in named_arrays
        )
        names = ", ".join(name for name, _array in named_arrays)
        raise ValueError(f"{names} must have matching shapes; got {shapes}")
