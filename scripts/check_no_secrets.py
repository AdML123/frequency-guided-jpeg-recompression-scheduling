"""Scan repository text files for likely secrets and private local paths."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".cfg",
    ".csv",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github_classic_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("github_fine_grained_token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    (
        "password_assignment",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd)\b\s*[:=]\s*[\"']?[^\"'\s]{6,}"
        ),
    ),
    (
        "email_password_assignment",
        re.compile(
            r"(?i)\b(?:email|account|username|user)\b\s*[:=]\s*"
            r"[\"']?[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}[^\\n\\r]*"
            r"\b(?:password|passwd|pwd)\b\s*[:=]"
        ),
    ),
    ("local_project_path", re.compile(r"(?i)\b[A-Z]:\\paper33\b")),
    ("local_user_path", re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s]+")),
)


def scan_text(text: str) -> list[str]:
    """Return finding names for likely secrets or private paths in text."""
    findings: list[str] = []
    for name, pattern in PATTERNS:
        if pattern.search(text):
            findings.append(name)
    for secret in _extra_secret_values():
        if secret and secret in text:
            findings.append("configured_secret_value")
            break
    return findings


def scan_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ["unreadable_file"]
    return scan_text(text)


def iter_text_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.name in {".gitignore"} or path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def _extra_secret_values() -> list[str]:
    raw = os.environ.get("SECRET_SCAN_DENYLIST", "")
    return [value for value in raw.split(os.pathsep) if len(value) >= 6]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=REPO_ROOT,
        help="Repository root to scan.",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    failures: list[tuple[Path, list[str]]] = []
    for path in iter_text_files(root):
        findings = scan_file(path)
        if findings:
            failures.append((path, findings))

    if failures:
        for path, findings in failures:
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            print(f"{rel}: {', '.join(findings)}")
        return 1

    print(f"No likely secrets found under {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
