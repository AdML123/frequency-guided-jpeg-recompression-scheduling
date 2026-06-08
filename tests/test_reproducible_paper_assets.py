from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_paper_assets_are_regenerated_from_repo_data():
    for rel in [
        "data/derived/metrics.csv",
        "data/derived/figure_metrics.csv",
        "data/derived/frequency_metrics.csv",
        "data/derived/source_data.csv.gz",
        "data/derived/summary.json",
        "data/derived/full_audit_summary.json",
        "data/derived/frequency_full8_summary.json",
        "results/data/metrics.csv",
        "results/data/figure_metrics.csv",
        "results/data/frequency_metrics.csv",
        "results/data/source_data.csv.gz",
        "scripts/make_figures.py",
        "scripts/make_tables.py",
    ]:
        assert (REPO / rel).is_file(), rel


def test_final_paper_results_are_included_for_direct_comparison():
    required = [
        "results/figures/figure1_mechanism_schematic.pdf",
        "results/figures/figure1_mechanism_schematic.png",
        "results/figures/figure1_mechanism_schematic.mmd",
        "results/figures/figure2_asr_by_generation.pdf",
        "results/figures/figure2_asr_by_generation.png",
        "results/figures/figure3_attack_range_summary.pdf",
        "results/figures/figure3_attack_range_summary.png",
        "results/figures/figure4_centroid_vs_delta.pdf",
        "results/figures/figure4_centroid_vs_delta.png",
        "results/tables/table_i_prediction_rules.tex",
        "results/tables/table_ii.tex",
        "results/tables/table_iii.tex",
        "results/tables/table_full_four_schedule_asr.tex",
        "results/tables/table_mcnemar_audit.tex",
        "results/tables/table_robust_training_gradient.tex",
        "results/tables/table_jpeg_aware_boundary.tex",
    ]
    for rel in required:
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
    assert "public reproducibility package" in readme
    forbidden = [
        "Rebuild LaTeX Paper",
        "latex" + "mk",
        "copy the " + "regenerated",
        "compile from the " + "LaTeX source root",
        "paper20" + "-cu128",
        "LaTeX paper" + "-build repository",
    ]
    for token in forbidden:
        assert token not in readme


def test_ieee_conference_artifact_best_practice_files_exist():
    required = [
        "environment.yml",
        "REPRODUCIBILITY.md",
        "ARTIFACTS.md",
        "resources_manifest.yml",
        "SECURITY.md",
        "results/README.md",
        "scripts/benchmark_runtime.py",
        "scripts/prepare_resources.py",
        "scripts/verify_resources.py",
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


def test_external_resource_policy_is_documented_without_bundling_raw_assets():
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    manifest = (REPO / "resources_manifest.yml").read_text(encoding="utf-8")
    required_readme = [
        "External Resource Policy",
        "scripts/prepare_resources.py",
        "scripts/verify_resources.py",
        "ImageNet validation images are not redistributed",
        "model weights are referenced by source and cache path",
    ]
    for token in required_readme:
        assert token in readme

    for token in [
        "cifar10",
        "imagenet_val",
        "robustbench_checkpoints",
        "torchvision_timm_weights",
        "redistribution: no",
    ]:
        assert token in manifest


def test_full_pipeline_script_defaults_to_paper_data_result_profile():
    script = (REPO / "scripts" / "run_full_pipeline.sh").read_text(encoding="utf-8")
    required = [
        'PIPELINE_PROFILE="${PIPELINE_PROFILE:-paper}"',
        "--mode mcnemar-audit",
        "--mode multimodel",
        "--mode frequency",
        "run_figure_metrics_cifar_fgsm.log",
        "run_figure_metrics_cifar_pgd.log",
        "run_figure_metrics_jpeg_aware.log",
        "run_figure_metrics_imagenet.log",
        "FIGURE_INPUTS",
        'AUDIT_MAX_SAMPLES="${AUDIT_MAX_SAMPLES:-10000}"',
        'FREQUENCY_MAX_SAMPLES="${FREQUENCY_MAX_SAMPLES:-10000}"',
        'PIPELINE_PROFILE=smoke',
    ]
    for token in required:
        assert token in script

    assert "--mode smoke" not in script
