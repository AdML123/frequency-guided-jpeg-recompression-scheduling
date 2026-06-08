import numpy as np
import pytest

from jpeg_defense.estimator import survival_bound, survival_score


def test_survival_bound_unrolls_recursive_ratio_bound():
    ratios = [0.5, 0.25]

    assert survival_bound(4.0, ratios) == pytest.approx(1.75)


def test_survival_bound_with_no_ratios_returns_initial_mu():
    assert survival_bound(2.25, []) == pytest.approx(2.25)


def test_survival_score_counts_surviving_frequencies_with_uniform_weights():
    mu = np.array([0.0, 0.5, 1.0, 3.0])
    ratios = np.empty((0, 4))

    assert survival_score(mu, ratios) == pytest.approx(0.5)


def test_survival_score_accepts_explicit_weights():
    mu = np.array([0.0, 0.5, 1.0, 3.0])
    ratios = np.empty((0, 4))
    weights = np.array([0.1, 0.2, 0.3, 0.4])

    assert survival_score(mu, ratios, weights=weights) == pytest.approx(0.7)
