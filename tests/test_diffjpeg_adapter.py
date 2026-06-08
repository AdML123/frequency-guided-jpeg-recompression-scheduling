from __future__ import annotations

import torch


def test_build_diffjpeg_schedule_preserves_shape_range_and_gradients():
    from jpeg_defense.diffjpeg_adapter import build_diffjpeg_schedule

    schedule = build_diffjpeg_schedule(
        height=32,
        width=32,
        qfs=[75, 50],
        device="cpu",
    )
    inputs = torch.rand(2, 3, 32, 32, requires_grad=True)

    outputs = schedule(inputs)
    outputs.mean().backward()

    assert outputs.shape == inputs.shape
    assert torch.all(outputs >= 0.0)
    assert torch.all(outputs <= 1.0)
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_capsule_vendored_diffjpeg_root_is_packaged():
    from jpeg_defense.diffjpeg_adapter import CAPSULE_DIFFJPEG_ROOT

    assert CAPSULE_DIFFJPEG_ROOT.is_dir()
    assert (CAPSULE_DIFFJPEG_ROOT / "DiffJPEG.py").is_file()
    assert (CAPSULE_DIFFJPEG_ROOT / "modules" / "compression.py").is_file()
