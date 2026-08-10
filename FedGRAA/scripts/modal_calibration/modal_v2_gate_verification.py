"""Re-run FedGRAA-v2's two hard-gate test files on the actual Modal image,
per docs/research_c/REMAINING_COMPUTE_WORK.md §3.2's "cheap insurance"
recommendation: the Modal container's CPU lacks AVX512 (confirmed by
COMPUTE_INFRASTRUCTURE_2026-07-27.md §4.4 -- one v1 gradient test missed its
tolerance by 0.00537 on non-AVX512 hardware), so before trusting a G3/G4 pass
produced on Modal specifically, tests/test_aggregation_v2.py (T-4/T-4a/T-4b,
G3) and tests/test_trainer_v2.py (T-5/T-11/T-12/T-13, G4) need to actually be
executed there, not just assumed to transfer from the local Windows/AVX512
run recorded in REMAINING_COMPUTE_WORK.md §7.

Both files only use fedgraa.data.make_synthetic_dataset (confirmed by grep:
no load_dataset/data_dir call with a real CSV path in either file), so `data/`
is excluded from the image upload -- this is a code-correctness gate, not a
real-data campaign run. This script runs no compute-campaign cells; it only
runs pytest against these two files.

Usage::

    modal run scripts/modal_calibration/modal_v2_gate_verification.py
"""

from __future__ import annotations

import modal

app = modal.App("fedgraa-v2-gate-verification")

EXCLUDE_DIRS = {".venv313", ".cache", "data", ".pytest_cache", "__pycache__", "archive", ".git"}


def _ignore(path):
    from pathlib import Path

    return any(part in EXCLUDE_DIRS for part in Path(path).parts)


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "tensorflow>=2.21.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
        "pytest>=7.0",
        "matplotlib>=3.7",
    )
    .add_local_dir("C:/Dev/FedGRAA/FedGRAA", remote_path="/repo", copy=True, ignore=_ignore)
)


@app.function(image=image, cpu=1.0, memory=4096, timeout=1200)
def run_v2_gate_tests() -> dict:
    import multiprocessing

    # Same fork-vs-spawn finding as COMPUTE_INFRASTRUCTURE_2026-07-27.md §4.3
    # -- forced defensively even though neither test file exercises
    # ProcessPoolExecutor directly, since it must be set before any pool
    # could be constructed during test collection.
    multiprocessing.set_start_method("spawn", force=True)

    import os
    import platform
    import subprocess
    import sys

    sys.path.insert(0, "/repo")
    os.chdir("/repo")

    import fedgraa  # noqa: F401
    print("mp OK / fedgraa OK")

    cpu_flags = subprocess.run(
        ["grep", "-m", "1", "flags", "/proc/cpuinfo"], capture_output=True, text=True,
    ).stdout
    has_avx512 = "avx512" in cpu_flags.lower()
    print(f"AVX512 present on this container: {has_avx512}")

    import pytest

    exit_code = pytest.main([
        "tests/test_aggregation_v2.py", "tests/test_trainer_v2.py", "-v",
    ])
    return {
        "exit_code": int(exit_code),
        "platform": platform.platform(),
        "avx512": has_avx512,
    }


@app.local_entrypoint()
def main():
    import json
    import os

    result = run_v2_gate_tests.remote()
    print("\n" + "=" * 72)
    print(json.dumps(result, indent=2))
    print("=" * 72)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modal_v2_gate_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
