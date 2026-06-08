"""Create the local resource layout for full experiment reruns.

The script does not download or redistribute third-party datasets or weights.
It prepares ignored directories and can copy or link already-authorized local
resources into the layout expected by the experiment code.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RESOURCE_DIRS = (
    "cifar10",
    "imagenet_raw",
    "hf_mirror",
    "checkpoints/cifar",
    "checkpoints/robustbench/cifar10/Linf",
    "checkpoints/torch_home/hub/checkpoints",
    "checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k",
)

EXPECTED_LINKS = (
    ("cifar10", "cifar10/cifar-10-python.tar.gz"),
    ("imagenet-val-tar", "imagenet_raw/ILSVRC2012_img_val.tar"),
    ("imagenet-val-dir", "hf_mirror/imagenet1k_val_1k"),
    ("cifar-resnet18", "checkpoints/cifar/CIFAR10_ResNet18_epoch_20.pt"),
    (
        "robustbench-wong2020",
        "checkpoints/robustbench/cifar10/Linf/Wong2020Fast.pt",
    ),
    (
        "robustbench-engstrom2019",
        "checkpoints/robustbench/cifar10/Linf/Engstrom2019Robustness.pt",
    ),
    (
        "robustbench-rice2020",
        "checkpoints/robustbench/cifar10/Linf/Rice2020Overfitting.pt",
    ),
    (
        "torchvision-vit-b16",
        "checkpoints/torch_home/hub/checkpoints/vit_b_16-c867db91.pth",
    ),
    (
        "torchvision-swin-t",
        "checkpoints/torch_home/hub/checkpoints/swin_t-704ceda3.pth",
    ),
    (
        "timm-deit-tiny",
        "checkpoints/imagenet/deit_tiny_patch16_224.fb_in1k/model.safetensors",
    ),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/data")),
        help="Ignored local resource root used by the experiment pipeline.",
    )
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Copy or link an authorized local resource into the expected layout.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy resources instead of creating links.",
    )
    args = parser.parse_args(argv)

    data_root = args.data_root.expanduser()
    create_layout(data_root)
    linked = install_resources(data_root, args.resource, copy=args.copy)

    print(f"Prepared resource layout under {data_root}")
    if linked:
        for name, target in linked:
            print(f"{name}: {target}")
    print("No downloads were attempted.")
    return 0


def create_layout(data_root: Path) -> None:
    for relative in RESOURCE_DIRS:
        (data_root / relative).mkdir(parents=True, exist_ok=True)


def install_resources(
    data_root: Path,
    resource_args: list[str],
    *,
    copy: bool,
) -> list[tuple[str, Path]]:
    installed: list[tuple[str, Path]] = []
    target_map = dict(EXPECTED_LINKS)
    for raw_arg in resource_args:
        name, source = _parse_resource_arg(raw_arg)
        if name not in target_map:
            valid = ", ".join(sorted(target_map))
            raise SystemExit(f"Unknown resource name {name!r}. Valid names: {valid}")
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise SystemExit(f"Resource source does not exist: {source_path}")
        target = data_root / target_map[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        _replace_with_resource(source_path, target, copy=copy)
        installed.append((name, target))
    return installed


def _parse_resource_arg(raw_arg: str) -> tuple[str, str]:
    if "=" not in raw_arg:
        raise SystemExit("--resource expects NAME=PATH")
    name, source = raw_arg.split("=", maxsplit=1)
    if not name or not source:
        raise SystemExit("--resource expects NAME=PATH")
    return name, source


def _replace_with_resource(source: Path, target: Path, *, copy: bool) -> None:
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    if copy:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        return

    try:
        target.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)


if __name__ == "__main__":
    raise SystemExit(main())
