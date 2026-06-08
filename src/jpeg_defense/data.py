import pickle
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps


_IMAGENET_MIRROR_LABEL_RE = re.compile(r"^\d+_label_(\d+)\.JPEG$")


def parse_imagenet_mirror_label(path):
    name = Path(path).name
    match = _IMAGENET_MIRROR_LABEL_RE.match(name)
    if not match:
        raise ValueError(f"Could not parse ImageNet mirror label from {path!s}")
    return int(match.group(1))


def list_imagenet_mirror(root, limit=None):
    root = Path(root)
    examples = [(path, parse_imagenet_mirror_label(path)) for path in sorted(root.glob("*.JPEG"))]
    if limit is not None:
        examples = examples[:limit]
    return examples


def load_cifar10_test_from_tar(tar_path, limit=None, offset=0):
    with tarfile.open(tar_path, "r:gz") as archive:
        member = next(
            (
                archive_member
                for archive_member in archive.getmembers()
                if Path(archive_member.name).name == "test_batch"
            ),
            None,
        )
        if member is None:
            raise FileNotFoundError(f"test_batch not found in {tar_path!s}")

        extracted = archive.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"test_batch is not a regular file in {tar_path!s}")
        batch = pickle.load(extracted, encoding="latin1")

    data = batch.get(b"data", batch.get("data"))
    labels = batch.get(b"labels", batch.get("labels"))
    if data is None or labels is None:
        raise KeyError("CIFAR-10 test_batch must contain data and labels")

    offset = max(0, int(offset or 0))
    if offset:
        data = data[offset:]
        labels = labels[offset:]
    if limit is not None:
        data = data[:limit]
        labels = labels[:limit]

    images_np = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32).astype(np.float32)
    images = torch.from_numpy(images_np).div(255.0)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return images, label_tensor


def load_imagenet_mirror_tensors(root, limit=None, image_size=224):
    examples = list_imagenet_mirror(root, limit=limit)
    tensors = []
    labels = []
    for path, label in examples:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            resized = ImageOps.fit(
                rgb,
                (int(image_size), int(image_size)),
                method=Image.Resampling.BICUBIC,
                centering=(0.5, 0.5),
            )
            array = np.asarray(resized, dtype=np.uint8).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1).to(torch.float32).div(255.0))
        labels.append(int(label))
    if not tensors:
        return (
            torch.empty(0, 3, int(image_size), int(image_size), dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )
    return torch.stack(tensors), torch.tensor(labels, dtype=torch.long)
