"""In-memory JPEG recompression transforms."""

from __future__ import annotations

from io import BytesIO
from typing import Iterable

from PIL import Image


def recompress_image(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip an image through JPEG compression at the given quality."""
    if quality < 1 or quality > 100:
        raise ValueError(f"quality must be in [1, 100], got {quality!r}")

    rgb_image = image.convert("RGB")
    buffer = BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)

    with Image.open(buffer) as recompressed:
        return recompressed.convert("RGB")


def recompress_many_generations(
    image: Image.Image, qualities: Iterable[int]
) -> Image.Image:
    """Apply JPEG recompression sequentially for each quality."""
    current = image
    for quality in qualities:
        current = recompress_image(current, quality)
    return current
