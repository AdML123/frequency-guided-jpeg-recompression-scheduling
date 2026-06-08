from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_github_ready_repository_has_required_files():
    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        ".gitignore",
        ".github/workflows/ci.yml",
    ]
    for rel in required:
        assert (REPO / rel).is_file(), rel

    assert (REPO / "src" / "jpeg_defense").is_dir()
    assert (REPO / "tests").is_dir()
    assert (REPO / "scripts").is_dir()
    assert (REPO / "results" / "figures").is_dir()
    assert (REPO / "results" / "tables").is_dir()
    assert (REPO / "results" / "data").is_dir()
    assert (REPO / "resources_manifest.yml").is_file()
    assert not (REPO / "metadata").exists()
    assert not (REPO / "environment").exists()


def test_paper_derived_data_is_available_for_reproduction():
    data_dir = REPO / "data" / "derived"
    for name in ["metrics.csv", "figure_metrics.csv", "frequency_metrics.csv"]:
        assert (data_dir / name).is_file(), name
    assert (data_dir / "source_data.csv.gz").is_file()
    assert not (data_dir / "source_data.csv").exists()


def test_repository_contains_no_secret_placeholders_or_local_paths():
    forbidden = [
        "D:" + "\\paper33",
        "C:" + "\\Users" + "\\yuefe",
        "paper20" + "-cu128",
        "paper" + "-latex",
        ".." + "\\" + ".." + "\\" + "tmp",
        "Signal Processing " + "Letters",
        "render_latex_" + "manuscript",
        "Template class " + "path",
        "fill-after-local-" + "verification",
    ]
    text_suffixes = {
        ".bib",
        ".cff",
        ".md",
        ".py",
        ".sh",
        ".tex",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in forbidden:
            assert token not in text, f"{token} found in {path}"


def test_heavy_external_resources_are_not_committed_to_git_tree():
    forbidden_dirs = [
        "data/raw",
        "data/external",
        "data/imagenet",
        "data/cifar-10-batches-py",
        "checkpoints",
        "weights",
        ".cache",
    ]
    for rel in forbidden_dirs:
        assert not (REPO / rel).exists(), rel

    forbidden_suffixes = {
        ".pth",
        ".pt",
        ".ckpt",
        ".onnx",
        ".tar",
        ".tgz",
        ".zip",
        ".JPEG",
        ".jpg",
        ".png",
    }
    allowed_pngs = (REPO / "results" / "figures",)
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in forbidden_suffixes:
            continue
        if any(path.is_relative_to(root) for root in allowed_pngs):
            continue
        assert False, f"Unexpected bundled external asset: {path}"
