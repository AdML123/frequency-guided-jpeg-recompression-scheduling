"""Print a local inventory for resources needed by full experiment reruns."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jpeg_defense.paths import check_expected_resources  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "/data")),
        help="Local resource root to validate.",
    )
    parser.add_argument(
        "--hash",
        action="store_true",
        help="Also print SHA-256 hashes for file resources.",
    )
    args = parser.parse_args(argv)

    data_root = args.data_root.expanduser()
    try:
        found = check_expected_resources(data_root)
    except FileNotFoundError as exc:
        print(f"Missing required resources:\n{exc}", file=sys.stderr)
        return 1

    print(f"Resource inventory under {data_root}:")
    for label, path in sorted(found.items()):
        rel_path = path.relative_to(data_root) if path.is_relative_to(data_root) else path
        kind = "dir" if path.is_dir() else "file"
        size = _display_size(path)
        line = f"- {label}: {rel_path} ({kind}, {size})"
        if args.hash and path.is_file():
            line += f", sha256={_sha256(path)}"
        print(line)
    return 0


def _display_size(path: Path) -> str:
    if path.is_dir():
        count = sum(1 for child in path.rglob("*") if child.is_file())
        return f"{count} files"
    size = path.stat().st_size
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{path.stat().st_size} B"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
