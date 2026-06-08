"""Path helpers for repository smoke tests and full local experiment reruns."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CapsulePaths:
    """Resolved roots for code, data, and results locations."""

    code_root: Path
    data_root: Path
    results_root: Path

    @classmethod
    def from_env(cls) -> "CapsulePaths":
        """Build paths from local override env vars or container-style defaults."""
        return cls(
            code_root=_env_path("CODE_DIR", "/code"),
            data_root=_env_path("DATA_DIR", "/data"),
            results_root=_env_path("RESULTS_DIR", "/results"),
        )


EXPECTED_RESOURCE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("CIFAR-10 archive", ("cifar10/cifar-10-python.tar.gz",)),
    (
        "CIFAR-10 checkpoint",
        ("checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt",),
    ),
    (
        "ImageNet validation input",
        (
            "imagenet_raw/ILSVRC2012_img_val.tar",
            "hf_mirror/imagenet1k_val_1k",
        ),
    ),
    (
        "RobustBench Wong2020Fast weights",
        (
            "checkpoints/robustbench/cifar10/Linf/Wong2020Fast.pt",
        ),
    ),
    (
        "RobustBench Engstrom2019 weights",
        ("checkpoints/robustbench/cifar10/Linf/Engstrom2019Robustness.pt",),
    ),
    (
        "RobustBench Rice2020 weights",
        ("checkpoints/robustbench/cifar10/Linf/Rice2020Overfitting.pt",),
    ),
    (
        "ImageNet ViT-B/16 weights",
        ("checkpoints/torch_home/hub/checkpoints/vit_b_16-c867db91.pth",),
    ),
    (
        "ImageNet Swin-T weights",
        ("checkpoints/torch_home/hub/checkpoints/swin_t-704ceda3.pth",),
    ),
    (
        "ImageNet DeiT-Tiny weights",
        ("checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k/model.safetensors",),
    ),
)

DIRECTORY_RESOURCE_PATHS = frozenset({"hf_mirror/imagenet1k_val_1k"})


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def required_file(path: str | Path, label: str) -> Path:
    """Return an existing file path, or raise a clear missing-resource error."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing {label}: {file_path}")
    return file_path


def ensure_results_dir(path: str | Path) -> Path:
    """Create and return the results directory."""
    results_dir = Path(path)
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def check_expected_resources(data_root: str | Path) -> dict[str, Path]:
    """Validate expected capsule resources under the data root.

    Alternative resource groups pass when any listed path exists.
    """
    root = Path(data_root)
    found: dict[str, Path] = {}
    missing_messages: list[str] = []

    for label, relative_options in EXPECTED_RESOURCE_GROUPS:
        existing = _first_existing(root, relative_options)
        if existing is not None:
            found[label] = existing
            continue

        if len(relative_options) == 1:
            missing_messages.append(f"- {label}: {root / relative_options[0]}")
        else:
            options = " OR ".join(relative_options)
            missing_messages.append(f"- {label}: expected one of {options}")

    if missing_messages:
        raise FileNotFoundError(
            "Missing required resources under "
            f"{root}:\n" + "\n".join(missing_messages)
        )

    return found


def _first_existing(root: Path, relative_options: tuple[str, ...]) -> Path | None:
    for relative_path in relative_options:
        candidate = root / relative_path
        if relative_path in DIRECTORY_RESOURCE_PATHS and candidate.is_dir():
            return candidate
        if relative_path not in DIRECTORY_RESOURCE_PATHS and candidate.is_file():
            return candidate
    return None
