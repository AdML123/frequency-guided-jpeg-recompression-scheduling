import os
import subprocess
import sys
from pathlib import Path

import pytest


def _create_resource_set(data_root: Path) -> None:
    files = [
        data_root / "cifar10" / "cifar-10-python.tar.gz",
        data_root / "checkpoints" / "cifar" / "CIFAR10_ResNet18_epoch_20.pt",
        data_root
        / "checkpoints"
        / "torch_home"
        / "hub"
        / "checkpoints"
        / "vit_b_16-c867db91.pth",
        data_root
        / "checkpoints"
        / "torch_home"
        / "hub"
        / "checkpoints"
        / "swin_t-704ceda3.pth",
        data_root
        / "checkpoints"
        / "imagenet"
        / "deit_tiny_patch16_224.fb_in1k"
        / "model.safetensors",
        data_root
        / "checkpoints"
        / "robustbench"
        / "cifar10"
        / "Linf"
        / "Wong2020Fast.pt",
        data_root
        / "checkpoints"
        / "robustbench"
        / "cifar10"
        / "Linf"
        / "Engstrom2019Robustness.pt",
        data_root
        / "checkpoints"
        / "robustbench"
        / "cifar10"
        / "Linf"
        / "Rice2020Overfitting.pt",
    ]
    for file_path in files:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(b"test resource")

    (data_root / "hf_mirror" / "imagenet1k_val_1k").mkdir(parents=True)


def test_paths_uses_container_defaults(monkeypatch):
    monkeypatch.delenv("CODE_DIR", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    monkeypatch.delenv("RESULTS_DIR", raising=False)

    from jpeg_defense.paths import CapsulePaths

    paths = CapsulePaths.from_env()

    assert paths.code_root == Path("/code")
    assert paths.data_root == Path("/data")
    assert paths.results_root == Path("/results")


def test_paths_uses_local_environment_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("CODE_DIR", str(tmp_path / "code"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))

    from jpeg_defense.paths import CapsulePaths

    paths = CapsulePaths.from_env()

    assert paths.code_root == tmp_path / "code"
    assert paths.data_root == tmp_path / "data"
    assert paths.results_root == tmp_path / "results"


def test_required_file_returns_existing_path(tmp_path):
    from jpeg_defense.paths import required_file

    target = tmp_path / "resource.bin"
    target.write_bytes(b"present")

    assert required_file(target, "test resource") == target


def test_required_file_raises_with_label_and_path_when_missing(tmp_path):
    from jpeg_defense.paths import required_file

    target = tmp_path / "missing.bin"

    with pytest.raises(FileNotFoundError) as exc_info:
        required_file(target, "test resource")

    message = str(exc_info.value)
    assert "test resource" in message
    assert str(target) in message


def test_ensure_results_dir_creates_directory(tmp_path):
    from jpeg_defense.paths import ensure_results_dir

    results_dir = tmp_path / "nested" / "results"

    assert ensure_results_dir(results_dir) == results_dir
    assert results_dir.is_dir()


def test_check_expected_resources_accepts_required_files_and_alternatives(tmp_path):
    from jpeg_defense.paths import check_expected_resources

    data_root = tmp_path / "data"
    _create_resource_set(data_root)

    found = check_expected_resources(data_root)

    assert found["CIFAR-10 archive"] == data_root / "cifar10" / "cifar-10-python.tar.gz"
    assert (
        found["CIFAR-10 checkpoint"]
        == data_root / "checkpoints" / "cifar" / "CIFAR10_ResNet18_epoch_20.pt"
    )
    assert found["ImageNet validation input"] == data_root / "hf_mirror" / "imagenet1k_val_1k"
    assert found["RobustBench Wong2020Fast weights"].name == "Wong2020Fast.pt"
    assert (
        found["RobustBench Engstrom2019 weights"].name
        == "Engstrom2019Robustness.pt"
    )
    assert found["RobustBench Rice2020 weights"].name == "Rice2020Overfitting.pt"
    assert found["ImageNet ViT-B/16 weights"].name == "vit_b_16-c867db91.pth"
    assert found["ImageNet Swin-T weights"].name == "swin_t-704ceda3.pth"
    assert found["ImageNet DeiT-Tiny weights"].name == "model.safetensors"


def test_check_expected_resources_reports_missing_required_and_alternative_groups(tmp_path):
    from jpeg_defense.paths import check_expected_resources

    data_root = tmp_path / "data"
    (data_root / "cifar10").mkdir(parents=True)
    (data_root / "cifar10" / "cifar-10-python.tar.gz").write_bytes(b"present")

    with pytest.raises(FileNotFoundError) as exc_info:
        check_expected_resources(data_root)

    message = str(exc_info.value)
    assert "CIFAR-10 checkpoint" in message
    assert "ImageNet validation input" in message
    assert "imagenet_raw/ILSVRC2012_img_val.tar" in message
    assert "hf_mirror/imagenet1k_val_1k" in message
    assert "ImageNet ViT-B/16 weights" in message
    assert "vit_b_16-c867db91.pth" in message
    assert "ImageNet Swin-T weights" in message
    assert "swin_t-704ceda3.pth" in message
    assert "ImageNet DeiT-Tiny weights" in message
    assert "model.safetensors" in message
    assert "RobustBench Wong2020Fast weights" in message
    assert "Wong2020Fast.pt" in message
    assert "RobustBench Engstrom2019 weights" in message
    assert "Engstrom2019Robustness.pt" in message
    assert "RobustBench Rice2020 weights" in message
    assert "Rice2020Overfitting.pt" in message


def test_check_expected_resources_rejects_directories_for_file_resources(tmp_path):
    from jpeg_defense.paths import check_expected_resources

    data_root = tmp_path / "data"
    directories_that_should_be_files = [
        data_root / "cifar10" / "cifar-10-python.tar.gz",
        data_root / "checkpoints" / "cifar" / "CIFAR10_ResNet18_epoch_20.pt",
        data_root
        / "checkpoints"
        / "torch_home"
        / "hub"
        / "checkpoints"
        / "vit_b_16-c867db91.pth",
        data_root
        / "checkpoints"
        / "torch_home"
        / "hub"
        / "checkpoints"
        / "swin_t-704ceda3.pth",
        data_root
        / "checkpoints"
        / "imagenet"
        / "deit_tiny_patch16_224.fb_in1k"
        / "model.safetensors",
        data_root
        / "checkpoints"
        / "robustbench"
        / "cifar10"
        / "Linf"
        / "Wong2020Fast.pt",
    ]
    for directory_path in directories_that_should_be_files:
        directory_path.mkdir(parents=True)
    (data_root / "hf_mirror" / "imagenet1k_val_1k").mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as exc_info:
        check_expected_resources(data_root)

    message = str(exc_info.value)
    assert "CIFAR-10 archive" in message
    assert "CIFAR-10 checkpoint" in message
    assert "ImageNet ViT-B/16 weights" in message
    assert "ImageNet Swin-T weights" in message
    assert "ImageNet DeiT-Tiny weights" in message
    assert "RobustBench Wong2020Fast weights" in message
    assert "RobustBench Engstrom2019 weights" in message
    assert "RobustBench Rice2020 weights" in message
    assert "ImageNet validation input" not in message


def test_check_resources_script_succeeds_with_temp_resources(tmp_path):
    data_root = tmp_path / "data"
    results_root = tmp_path / "results"
    code_root = tmp_path / "code"
    _create_resource_set(data_root)

    env = os.environ.copy()
    env.update(
        {
            "CODE_DIR": str(code_root),
            "DATA_DIR": str(data_root),
            "RESULTS_DIR": str(results_root),
        }
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_resources.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "All required resources present" in result.stdout
    assert "9 resource groups" in result.stdout
    assert results_root.is_dir()


def test_check_resources_script_fails_clearly_without_downloads(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "CODE_DIR": str(tmp_path / "code"),
            "DATA_DIR": str(tmp_path / "data"),
            "RESULTS_DIR": str(tmp_path / "results"),
        }
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_resources.py"

    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.returncode == 1
    assert "Missing required resources" in result.stderr
    assert "No downloads are attempted" in result.stderr
