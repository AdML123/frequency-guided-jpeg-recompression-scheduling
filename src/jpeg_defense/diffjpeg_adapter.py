"""Adapter for local DiffJPEG attack-generation layers."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_DIFFJPEG_ROOT = REPO_ROOT / "third_party" / "diffjpeg"
CAPSULE_DIFFJPEG_ROOT = VENDORED_DIFFJPEG_ROOT


class DiffJPEGSchedule(torch.nn.Module):
    """Apply a sequence of differentiable JPEG layers."""

    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs).clamp(0.0, 1.0)
        return outputs


def build_diffjpeg_schedule(
    *,
    height,
    width,
    qfs,
    device,
    diffjpeg_root=None,
):
    """Build a differentiable JPEG schedule from the local DiffJPEG package."""
    diffjpeg_cls = _load_diffjpeg_class(diffjpeg_root)
    layers = [
        diffjpeg_cls(
            height=int(height),
            width=int(width),
            differentiable=True,
            quality=int(qf),
        )
        for qf in qfs
    ]
    return DiffJPEGSchedule(layers).to(torch.device(device))


def _load_diffjpeg_class(diffjpeg_root=None):
    candidates = _candidate_roots(diffjpeg_root)
    root = next((candidate for candidate in candidates if candidate.is_dir()), None)
    if root is None:
        roots = ", ".join(str(candidate) for candidate in _candidate_roots(diffjpeg_root))
        raise FileNotFoundError(f"DiffJPEG root not found in any candidate path: {roots}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("DiffJPEG")
    return module.DiffJPEG


def _candidate_roots(diffjpeg_root=None):
    candidates = []
    if diffjpeg_root is not None:
        candidates.append(Path(diffjpeg_root))
    env_root = os.environ.get("DIFFJPEG_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(VENDORED_DIFFJPEG_ROOT)
    return candidates
