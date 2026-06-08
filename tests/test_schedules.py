import pytest

from jpeg_defense.schedules import (
    arithmetic_schedule,
    fixed_schedule,
    geometric_schedule,
    jpeg_scale_from_qf,
    reverse_geometric_schedule,
)


@pytest.mark.parametrize(
    ("qf", "expected"),
    [
        (50, 100.0),
        (75, 50.0),
        (90, 20.0),
        (25, 200.0),
    ],
)
def test_jpeg_scale_from_qf_matches_standard_piecewise_rule(qf, expected):
    assert jpeg_scale_from_qf(qf) == pytest.approx(expected)


def test_geometric_schedule_for_r1_three_generations_uses_jpeg_scale_space():
    assert geometric_schedule(75, 50, 3) == [75, 65, 50]


def test_geometric_schedule_for_r1_five_generations_uses_jpeg_scale_space():
    assert geometric_schedule(75, 50, 5) == [75, 70, 65, 58, 50]


def test_arithmetic_schedule_for_r1_three_generations_matches_paper_definition():
    assert arithmetic_schedule(75, 50, 3) == [75, 63, 50]


def test_fixed_schedule_uses_rounded_midpoint():
    assert fixed_schedule(75, 50, 5) == [62, 62, 62, 62, 62]


def test_reverse_geometric_swaps_start_and_end_quality_factors():
    assert reverse_geometric_schedule(75, 50, 3) == [50, 65, 75]


def test_single_generation_schedules_return_start_quality():
    assert geometric_schedule(75, 50, 1) == [75]
    assert arithmetic_schedule(75, 50, 1) == [75]
    assert fixed_schedule(75, 50, 1) == [62]
    assert reverse_geometric_schedule(75, 50, 1) == [50]
