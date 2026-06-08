from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_runtime_benchmark_smoke_runs_without_raw_data():
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "benchmark_runtime.py"),
            "--image-size",
            "32",
            "--repeats",
            "1",
            "--generations",
            "3",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "front_loaded" in result.stdout
    assert "fixed" in result.stdout
    assert "ms_per_image" in result.stdout
