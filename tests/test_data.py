import io
import pickle
import tarfile

import numpy as np
import pytest
import torch
from PIL import Image


def _write_cifar_test_tar(tar_path, images, labels):
    payload = pickle.dumps({b"data": images, b"labels": labels})
    info = tarfile.TarInfo("cifar-10-batches-py/test_batch")
    info.size = len(payload)
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.addfile(info, fileobj=io.BytesIO(payload))


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("000123_label_456.JPEG", 456),
        ("nested/000001_label_7.JPEG", 7),
    ],
)
def test_parse_imagenet_mirror_label_extracts_label_integer(name, expected):
    from jpeg_defense.data import parse_imagenet_mirror_label

    assert parse_imagenet_mirror_label(name) == expected


def test_parse_imagenet_mirror_label_rejects_unexpected_names():
    from jpeg_defense.data import parse_imagenet_mirror_label

    with pytest.raises(ValueError, match="ImageNet mirror label"):
        parse_imagenet_mirror_label("000123.JPEG")


def test_list_imagenet_mirror_returns_sorted_paths_and_labels(tmp_path):
    from jpeg_defense.data import list_imagenet_mirror

    root = tmp_path / "imagenet"
    root.mkdir()
    (root / "000002_label_9.JPEG").write_bytes(b"")
    (root / "000001_label_4.JPEG").write_bytes(b"")

    examples = list_imagenet_mirror(root)

    assert examples == [
        (root / "000001_label_4.JPEG", 4),
        (root / "000002_label_9.JPEG", 9),
    ]


def test_list_imagenet_mirror_applies_limit_after_sort(tmp_path):
    from jpeg_defense.data import list_imagenet_mirror

    root = tmp_path / "imagenet"
    root.mkdir()
    (root / "000002_label_9.JPEG").write_bytes(b"")
    (root / "000001_label_4.JPEG").write_bytes(b"")

    assert list_imagenet_mirror(root, limit=1) == [(root / "000001_label_4.JPEG", 4)]


def test_load_cifar10_test_from_tar_reads_test_batch_without_extracting(tmp_path):
    from jpeg_defense.data import load_cifar10_test_from_tar

    tar_path = tmp_path / "cifar-10-python.tar.gz"
    flat_images = np.stack(
        [
            np.arange(3072, dtype=np.uint8),
            np.full(3072, 255, dtype=np.uint8),
        ]
    )
    _write_cifar_test_tar(tar_path, flat_images, [3, 5])

    images, labels = load_cifar10_test_from_tar(tar_path)

    assert images.shape == (2, 3, 32, 32)
    assert images.dtype == torch.float32
    assert images[0] == pytest.approx(torch.from_numpy(flat_images[0].reshape(3, 32, 32) / 255.0))
    assert labels.dtype == torch.long
    assert labels.tolist() == [3, 5]
    assert not (tmp_path / "cifar-10-batches-py").exists()


def test_load_cifar10_test_from_tar_applies_limit(tmp_path):
    from jpeg_defense.data import load_cifar10_test_from_tar

    tar_path = tmp_path / "cifar-10-python.tar.gz"
    flat_images = np.stack(
        [
            np.full(3072, 10, dtype=np.uint8),
            np.full(3072, 20, dtype=np.uint8),
            np.full(3072, 30, dtype=np.uint8),
        ]
    )
    _write_cifar_test_tar(tar_path, flat_images, [1, 2, 3])

    images, labels = load_cifar10_test_from_tar(tar_path, limit=2)

    assert images.shape == (2, 3, 32, 32)
    assert labels.tolist() == [1, 2]


def test_load_cifar10_test_from_tar_applies_offset_before_limit(tmp_path):
    from jpeg_defense.data import load_cifar10_test_from_tar

    tar_path = tmp_path / "cifar-10-python.tar.gz"
    flat_images = np.stack(
        [
            np.full(3072, 10, dtype=np.uint8),
            np.full(3072, 20, dtype=np.uint8),
            np.full(3072, 30, dtype=np.uint8),
            np.full(3072, 40, dtype=np.uint8),
        ]
    )
    _write_cifar_test_tar(tar_path, flat_images, [1, 2, 3, 4])

    images, labels = load_cifar10_test_from_tar(tar_path, limit=2, offset=1)

    assert images.shape == (2, 3, 32, 32)
    assert labels.tolist() == [2, 3]
    assert images[0].mean() == pytest.approx(20 / 255.0)


def test_load_imagenet_mirror_tensors_resizes_and_returns_labels(tmp_path):
    from jpeg_defense.data import load_imagenet_mirror_tensors

    root = tmp_path / "imagenet"
    root.mkdir()
    Image.fromarray(np.full((12, 16, 3), 64, dtype=np.uint8), mode="RGB").save(
        root / "000002_label_9.JPEG"
    )
    Image.fromarray(np.full((16, 12, 3), 192, dtype=np.uint8), mode="RGB").save(
        root / "000001_label_4.JPEG"
    )

    images, labels = load_imagenet_mirror_tensors(root, limit=1, image_size=8)

    assert images.shape == (1, 3, 8, 8)
    assert images.dtype == torch.float32
    assert torch.all(images >= 0.0)
    assert torch.all(images <= 1.0)
    assert labels.dtype == torch.long
    assert labels.tolist() == [4]
