from __future__ import annotations

from pathlib import Path


DELIVERY = Path(__file__).resolve().parents[1].parent
TEX_PATH = DELIVERY / "latex_source" / "main.tex"


def _tex() -> str:
    return TEX_PATH.read_text(encoding="utf-8")


def test_manuscript_avoids_ai_style_markers():
    tex = _tex()
    forbidden = [
        "Taken together",
        "Overall,",
        "delve into",
        "shed light on",
        "advance understanding",
        "not only",
        "but also",
        "In response",
        "reviewer",
        "---",
    ]
    for token in forbidden:
        assert token not in tex
    assert tex.count(":") < 45
    assert tex.count(";") < 18


def test_manuscript_contains_required_revision_content():
    tex = _tex()
    required = [
        "quality factor",
        "discrete cosine transform",
        "attack success rate",
        "front-loaded",
        "geometric",
        "arithmetic",
        "McNemar",
        "practical",
        "fully adaptive",
        "gradient mismatch",
        "SHIELD",
        "feature squeezing",
        "ImageNet",
        "Transformer",
    ]
    for token in required:
        assert token in tex


def test_required_sections_are_present_without_response_tone():
    tex = _tex()
    for section in [
        r"\section{Introduction}",
        r"\section{Proposed Approach}",
        r"\section{Experimental Setup}",
        r"\section{Results}",
        r"\section{Discussion}",
    ]:
        assert section in tex
    defensive_phrases = [
        "we acknowledge",
        "to address this concern",
        "this limitation is acceptable",
        "although this was not tested",
    ]
    lowered = tex.lower()
    for phrase in defensive_phrases:
        assert phrase not in lowered
