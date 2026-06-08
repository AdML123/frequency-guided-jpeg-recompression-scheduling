from __future__ import annotations

import importlib

import numpy as np
import pytest


def _metrics_module():
    try:
        return importlib.import_module("jpeg_defense.metrics")
    except ModuleNotFoundError as exc:
        pytest.fail(f"jpeg_defense.metrics module is missing: {exc}")


def test_attack_success_rate_counts_clean_correct_then_adv_wrong_examples():
    metrics = _metrics_module()
    labels = np.array([0, 1, 2, 1, 0])
    clean_pred = np.array([0, 1, 1, 1, 0])
    adv_pred = np.array([1, 1, 1, 2, 0])

    asr = metrics.attack_success_rate(clean_pred, adv_pred, labels)

    assert asr == pytest.approx(2 / 4)


def test_attack_success_rate_returns_zero_when_no_clean_correct_examples():
    metrics = _metrics_module()

    asr = metrics.attack_success_rate([1, 0], [0, 1], [0, 1])

    assert asr == 0.0


@pytest.mark.parametrize(
    ("clean_pred", "adv_pred", "labels", "message"),
    [
        ([[0], [1]], [1, 0], [0, 1], "clean_pred"),
        ([0, 1], [[1], [0]], [0, 1], "adv_pred"),
        ([0, 1], [1, 0], [[0], [1]], "labels"),
    ],
)
def test_attack_success_rate_rejects_mismatched_shapes(
    clean_pred, adv_pred, labels, message
):
    metrics = _metrics_module()

    with pytest.raises(ValueError, match=message):
        metrics.attack_success_rate(clean_pred, adv_pred, labels)


def test_binomial_ci_uses_wilson_interval_for_default_confidence():
    metrics = _metrics_module()

    low, high = metrics.binomial_ci(2, 10)

    assert low == pytest.approx(0.0567, abs=1e-4)
    assert high == pytest.approx(0.5098, abs=1e-4)


def test_binomial_ci_returns_zeros_for_empty_total():
    metrics = _metrics_module()

    assert metrics.binomial_ci(0, 0) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("successes", "total", "confidence", "message"),
    [
        (11, 10, 0.95, "successes"),
        (-1, 10, 0.95, "successes"),
        (2.5, 10, 0.95, "successes"),
        (0, -1, 0.95, "total"),
        (2, 10.5, 0.95, "total"),
        (2, 10, 0.0, "confidence"),
        (2, 10, 1.0, "confidence"),
        (2, 10, float("nan"), "confidence"),
        (2, 10, float("inf"), "confidence"),
    ],
)
def test_binomial_ci_rejects_invalid_counts_and_confidence(
    successes, total, confidence, message
):
    metrics = _metrics_module()

    with pytest.raises(ValueError, match=message):
        metrics.binomial_ci(successes, total, confidence=confidence)


def test_mcnemar_table_counts_discordant_paired_successes():
    metrics = _metrics_module()
    a_success = np.array([True, False, True, False, True])
    b_success = np.array([True, True, False, False, False])

    n01, n10 = metrics.mcnemar_table(a_success, b_success)

    assert (n01, n10) == (1, 2)


def test_mcnemar_table_rejects_mismatched_shapes():
    metrics = _metrics_module()

    with pytest.raises(ValueError, match="a_success.*b_success"):
        metrics.mcnemar_table([[True], [False]], [True, False])


def test_mcnemar_pvalue_returns_one_when_no_discordant_pairs():
    metrics = _metrics_module()

    assert metrics.mcnemar_pvalue(0, 0) == 1.0


def test_mcnemar_pvalue_uses_exact_two_sided_binomial_test():
    metrics = _metrics_module()

    assert metrics.mcnemar_pvalue(1, 4) == pytest.approx(0.375)
