"""Benchmark JPEG recompression schedules on a synthetic RGB image."""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jpeg_defense.schedules import (  # noqa: E402
    arithmetic_schedule,
    fixed_schedule,
    geometric_schedule,
    reverse_geometric_schedule,
)


ScheduleFn = Callable[[int, int, int], list[int]]
SCHEDULES: tuple[tuple[str, ScheduleFn], ...] = (
    ("front_loaded", reverse_geometric_schedule),
    ("fixed", fixed_schedule),
    ("geometric", geometric_schedule),
    ("arithmetic", arithmetic_schedule),
)


def _synthetic_image(size: int) -> Image.Image:
    rng = random.Random(0)
    data = bytes(rng.randrange(256) for _ in range(size * size * 3))
    return Image.frombytes("RGB", (size, size), data)


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _run_schedule(image: Image.Image, qfs: list[int]) -> Image.Image:
    out = image
    for qf in qfs:
        out = _jpeg_roundtrip(out, qf)
    return out


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark JPEG schedule runtime on a synthetic RGB image."
    )
    parser.add_argument("--image-size", type=_positive_int, default=32)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--generations", type=_positive_int, default=5)
    parser.add_argument("--q-min", type=_positive_int, default=50)
    parser.add_argument("--q-max", type=_positive_int, default=75)
    args = parser.parse_args(argv)

    if args.q_min > args.q_max:
        parser.error("--q-min must be <= --q-max")

    image = _synthetic_image(args.image_size)
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["schedule", "generations", "ms_per_image"])

    for name, schedule_fn in SCHEDULES:
        qfs = schedule_fn(args.q_max, args.q_min, args.generations)
        start = time.perf_counter()
        for _ in range(args.repeats):
            _run_schedule(image, qfs)
        elapsed = time.perf_counter() - start
        writer.writerow(
            [name, args.generations, f"{elapsed * 1000.0 / args.repeats:.3f}"]
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
