from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_paper_assets_are_regenerated_from_repo_data():
    for rel in [
        "data/derived/metrics.csv",
        "data/derived/figure_metrics.csv",
        "data/derived/frequency_metrics.csv",
        "data/derived/source_data.csv",
        "data/derived/summary.json",
        "data/derived/full_audit_summary.json",
        "data/derived/frequency_full8_summary.json",
        "scripts/make_figures.py",
        "scripts/make_tables.py",
    ]:
        assert (REPO / rel).is_file(), rel


def test_readme_documents_verified_reproduction_commands():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    required = [
        "python -m pytest tests -q",
        "scripts/make_figures.py",
        "scripts/make_tables.py",
        "Data",
        "No credentials",
    ]
    for token in required:
        assert token in readme


def test_repository_scope_is_data_result_reproduction_not_full_paper_rendering():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert "reproduce the paper data results" in readme
    forbidden = [
        "Rebuild LaTeX Paper",
        "latexmk",
        "copy the regenerated",
        "compile from the LaTeX source root",
    ]
    for token in forbidden:
        assert token not in readme


def test_ieee_conference_artifact_best_practice_files_exist():
    required = [
        "environment.yml",
        "REPRODUCIBILITY.md",
        "ARTIFACTS.md",
        "SECURITY.md",
        "results/README.md",
        "scripts/benchmark_runtime.py",
    ]
    for rel in required:
        assert (REPO / rel).is_file(), rel


def test_readme_identifies_icsip2026_artifact_and_safe_auth():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    required = [
        "2026 11th International Conference on Signal and Image Processing",
        "GitHub CLI",
        "gh auth login --web",
        "personal access token",
        "SSH",
        "No password authentication",
    ]
    for token in required:
        assert token in readme
