"""Quality-factor schedules for multi-generation JPEG recompression."""

from __future__ import annotations

import math


def jpeg_scale_from_qf(qf: int | float) -> float:
    """Return the standard JPEG quantization scale for a quality factor."""
    qf_float = float(qf)
    if qf_float <= 0 or qf_float > 100:
        raise ValueError(f"qf must be in (0, 100], got {qf!r}")
    if qf_float < 50:
        return 5000.0 / qf_float
    return 200.0 - 2.0 * qf_float


def qf_from_jpeg_scale(scale: int | float) -> int:
    """Return the nearest JPEG quality factor for a quantization scale."""
    scale_float = float(scale)
    if scale_float <= 0:
        raise ValueError(f"scale must be positive, got {scale!r}")
    if scale_float < 100.0:
        qf = (200.0 - scale_float) / 2.0
    else:
        qf = 5000.0 / scale_float
    return min(100, max(1, _round_qf(qf)))


def _round_qf(value: float) -> int:
    return int(math.floor(value + 0.5))


def _require_generation_count(generations: int) -> None:
    if generations < 1:
        raise ValueError(f"generations must be >= 1, got {generations}")


def arithmetic_schedule(start_qf: int, end_qf: int, generations: int) -> list[int]:
    """Linearly interpolate quality factors from start to end."""
    _require_generation_count(generations)
    if generations == 1:
        return [int(start_qf)]
    step = (end_qf - start_qf) / (generations - 1)
    return [_round_qf(start_qf + step * index) for index in range(generations)]


def fixed_schedule(start_qf: int, end_qf: int, generations: int) -> list[int]:
    """Use the rounded midpoint quality factor for every generation."""
    _require_generation_count(generations)
    midpoint = int(math.floor((start_qf + end_qf) / 2.0))
    return [midpoint for _ in range(generations)]


def geometric_schedule(start_qf: int, end_qf: int, generations: int) -> list[int]:
    """Geometrically interpolate JPEG quantization scale from start to end."""
    _require_generation_count(generations)
    if start_qf <= 0 or end_qf <= 0:
        raise ValueError("geometric schedules require positive quality factors")
    if generations == 1:
        return [int(start_qf)]
    start_scale = jpeg_scale_from_qf(start_qf)
    end_scale = jpeg_scale_from_qf(end_qf)
    ratio = end_scale / start_scale
    return [
        qf_from_jpeg_scale(start_scale * math.pow(ratio, index / (generations - 1)))
        for index in range(generations)
    ]


def reverse_geometric_schedule(start_qf: int, end_qf: int, generations: int) -> list[int]:
    """Geometrically interpolate from the lower-quality endpoint back upward."""
    return geometric_schedule(end_qf, start_qf, generations)
