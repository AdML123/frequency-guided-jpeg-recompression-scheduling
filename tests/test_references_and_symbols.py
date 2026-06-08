from __future__ import annotations

import re
import subprocess
from pathlib import Path


DELIVERY = Path(__file__).resolve().parents[1].parent
LATEX = DELIVERY / "latex_source"


def _tex() -> str:
    return (LATEX / "main.tex").read_text(encoding="utf-8")


def test_required_symbols_are_defined_near_methods():
    tex = _tex()
    required_snippets = [
        r"\omega_\delta",
        r"\rho_{u,v}",
        r"\tau",
        r"\Delta_{\bar q}",
        r"\mu_1",
        r"\mu_2",
        r"s=(q_1,\ldots,q_T)",
        r"D_s(x)",
        r"N_c",
        r"\alpha",
        r"\Delta=\mathrm{ASR}_{\mathrm{FL}}-",
    ]
    for snippet in required_snippets:
        assert snippet in tex


def test_abbreviations_are_expanded_on_first_use():
    tex = _tex()
    required_phrases = [
        "quality factor (QF)",
        "discrete cosine transform (DCT)",
        "attack success rate",
        "front-loaded (FL)",
        "fixed (Fix)",
        "geometric (Geo)",
        "arithmetic (Arith)",
        "projected gradient descent",
        "fast gradient sign method",
        "Vision Transformer",
    ]
    for phrase in required_phrases:
        assert phrase in tex


def test_citation_numbers_are_monotone_on_first_use_after_compile():
    pdf = DELIVERY / "paper.pdf"
    text = subprocess.run(
        ["pdftotext", str(pdf), "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        check=True,
    ).stdout
    marker = "REFERENCES" if "REFERENCES" in text else "R EFERENCES"
    body = text.split(marker)[0]
    seen: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", body):
        number = int(match.group(1))
        if number not in seen:
            seen.append(number)
    assert seen == sorted(seen), seen
