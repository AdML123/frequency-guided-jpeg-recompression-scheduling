from __future__ import annotations

import subprocess
from pathlib import Path


DELIVERY = Path(__file__).resolve().parents[1].parent
LATEX = DELIVERY / "latex_source"


def test_icsip_main_tex_uses_conference_template_and_is_double_blind():
    tex = (LATEX / "main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[conference]{IEEEtran}" in tex
    assert r"\markboth" not in tex
    assert "Anonymous Authors" in tex or r"\IEEEauthorblockN" in tex
    assert "Code Ocean" not in tex
    assert "D:\\" not in tex
    assert "C:\\Users" not in tex
    assert "High zone" not in tex
    assert "Low zone" not in tex
    assert "margin-zone" not in tex


def test_final_pdf_is_icsip_safe_and_uses_full_five_page_budget():
    pdf = DELIVERY / "paper.pdf"
    assert pdf.is_file()

    info = subprocess.run(
        ["pdfinfo", str(pdf)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True,
    ).stdout
    pages = int(
        next(
            line.split(":", 1)[1].strip()
            for line in info.splitlines()
            if line.startswith("Pages:")
        )
    )
    assert pages == 5

    text = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True,
    ).stdout

    assert "5/8" in text or "5 of 8" in text
    assert "p < 0.001" in text
    for token in ["Code Ocean", "D:\\", "C:\\Users", "High zone", "Low zone"]:
        assert token not in text

    page_five = subprocess.run(
        ["pdftotext", "-f", "5", "-l", "5", str(pdf), "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True,
    ).stdout
    assert "REFERENCES" in page_five or "R EFERENCES" in page_five
    assert len(page_five.split()) >= 180
