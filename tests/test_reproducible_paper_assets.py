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


def test_latex_uses_figure_assets_written_by_repository_scripts():
    tex = (REPO.parent / "latex_source" / "main.tex").read_text(encoding="utf-8")
    for figure in [
        "figure1_asr_by_generation.pdf",
        "figure2_centroid_vs_delta.pdf",
        "figure3_attack_range_summary.pdf",
    ]:
        assert figure in tex
        assert (REPO.parent / "latex_source" / "figures" / figure).is_file()
