"""Validate expected local resources for full experiment reruns."""

from __future__ import annotations

import argparse
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Local resource root to validate. Defaults to DATA_DIR or /data.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Results root to create. Defaults to RESULTS_DIR or /results.",
    )
    args = parser.parse_args(argv)

    paths = CapsulePaths.from_env()
    data_root = args.data_root.expanduser() if args.data_root else paths.data_root
    results_root = (
        args.results_root.expanduser() if args.results_root else paths.results_root
    )

    try:
        found = check_expected_resources(data_root)
    except FileNotFoundError as exc:
        print(f"Missing required resources:\n{exc}", file=sys.stderr)
        print("No downloads are attempted by this check.", file=sys.stderr)
        return 1

    ensure_results_dir(results_root)
    print(
        "All required resources present: "
        f"{len(found)} resource groups. "
        f"data={data_root} results={results_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
