from __future__ import annotations

import importlib

import numpy as np
import pytest
from PIL import Image


def _jpeg_module():
    try:
        return importlib.import_module("jpeg_defense.jpeg")
    except ModuleNotFoundError as exc:
        pytest.fail(f"jpeg_defense.jpeg module is missing: {exc}")


def _rgb_gradient_image() -> Image.Image:
    data = np.zeros((8, 8, 3), dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            data[row, col] = (row * 31, col * 29, (row + col) * 13)
    return Image.fromarray(data, mode="RGB")


def test_recompress_image_round_trips_through_jpeg_and_preserves_rgb_mode():
    jpeg = _jpeg_module()
    image = _rgb_gradient_image()

    recompressed = jpeg.recompress_image(image, quality=75)

    assert recompressed.mode == "RGB"
    assert recompressed.size == image.size
    assert recompressed is not image
    assert np.asarray(recompressed).dtype == np.uint8


def test_recompress_image_converts_non_rgb_inputs_to_rgb_by_default():
    jpeg = _jpeg_module()
    image = Image.fromarray(np.arange(64, dtype=np.uint8).reshape(8, 8), mode="L")

    recompressed = jpeg.recompress_image(image, quality=90)

    assert recompressed.mode == "RGB"
    assert recompressed.size == image.size


@pytest.mark.parametrize("quality", [0, -1, 101])
def test_recompress_image_rejects_quality_outside_jpeg_range(quality):
    jpeg = _jpeg_module()

    with pytest.raises(ValueError, match="quality"):
        jpeg.recompress_image(_rgb_gradient_image(), quality=quality)


def test_recompress_many_generations_applies_each_quality_in_order(monkeypatch):
    jpeg = _jpeg_module()
    image = _rgb_gradient_image()
    qualities_seen: list[int] = []

    def fake_recompress(current: Image.Image, quality: int) -> Image.Image:
        qualities_seen.append(quality)
        return current.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    monkeypatch.setattr(jpeg, "recompress_image", fake_recompress)

    result = jpeg.recompress_many_generations(image, [90, 75, 60])

    assert qualities_seen == [90, 75, 60]
    assert result.size == image.size
