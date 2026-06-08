"""Validate expected local resources for full experiment reruns."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jpeg_defense.paths import (  # noqa: E402
    CapsulePaths,
    check_expected_resources,
    ensure_results_dir,
)


def main() -> int:
    paths = CapsulePaths.from_env()

    try:
        found = check_expected_resources(paths.data_root)
    except FileNotFoundError as exc:
        print(f"Missing required resources:\n{exc}", file=sys.stderr)
        print("No downloads are attempted by this check.", file=sys.stderr)
        return 1

    ensure_results_dir(paths.results_root)
    print(
        "All required resources present: "
        f"{len(found)} resource groups. "
        f"data={paths.data_root} results={paths.results_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
