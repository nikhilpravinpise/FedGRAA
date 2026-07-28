"""Real, measured verification of B6's fix (docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md).

Runs the actual bounded-memory loader (`fedgraa.data.load_nf_uq_nids_v2_silos`
with `max_per_class` set) against the real 12.79 GB / 75,987,976-row
NF-UQ-NIDS-v2 release, and reports the process's true peak working set --
not an estimate, not an extrapolation. This is precisely the run
`docs/audit/nf_uq_nids_v2_provenance_audit.md`'s own §6 flagged as never
having been done.

Windows has no `resource.getrusage` (the Linux/Modal calibration scripts'
own measurement method, `scripts/modal_calibration/*.py`), so this uses the
OS's own peak-working-set counter via `ctypes` -- no new dependency, and a
number Windows tracks continuously rather than by polling, so no sampling
gap can hide a transient spike.

Usage::

    .venv313/Scripts/python.exe scripts/verify_nf_uq_nids_v2_memory.py \\
        --max-per-class 10000 --seed 0
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)

_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.GetCurrentProcess.argtypes = []

_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
    wintypes.DWORD,
]


def peak_working_set_mb() -> float:
    """Current process's peak working set, in MB, per the Windows kernel's
    own continuously-tracked counter (``GetProcessMemoryInfo``) -- not a
    polling-based estimate. Explicit ``restype``/``argtypes`` are required:
    without them ctypes assumes 32-bit ``int`` args/return on a 64-bit
    HANDLE and truncates it, which is what made the naive
    ``ctypes.windll.psapi.GetProcessMemoryInfo(...)`` call fail outright."""
    counters = _PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(_PROCESS_MEMORY_COUNTERS)
    handle = _kernel32.GetCurrentProcess()
    ok = _psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize / (1024.0 * 1024.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv-path", default="data/NF-UQ-NIDS-v2/NF-UQ-NIDS-v2.csv",
    )
    ap.add_argument("--max-per-class", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    print(f"peak working set before import : {peak_working_set_mb():8.1f} MB",
          flush=True)

    import numpy as np
    from fedgraa.data import load_nf_uq_nids_v2_silos

    print(f"peak working set after import  : {peak_working_set_mb():8.1f} MB",
          flush=True)
    print(f"csv_path       = {args.csv_path}")
    print(f"max_per_class  = {args.max_per_class:,}")
    print(f"seed           = {args.seed}")
    print(f"chunksize      = {args.chunksize:,}")
    print("=" * 78, flush=True)

    t0 = time.time()
    silos, labels = load_nf_uq_nids_v2_silos(
        args.csv_path,
        max_per_class=args.max_per_class,
        seed=args.seed,
        chunksize=args.chunksize,
        dtype=np.float32,
    )
    elapsed = time.time() - t0
    peak_mb = peak_working_set_mb()

    total_rows = sum(len(y) for _X, y, _n in silos)
    per_silo = {name: len(y) for _X, y, name in silos}
    n_features = silos[0][0].shape[1]

    print("=" * 78)
    print(f"elapsed              : {elapsed:8.1f} s")
    print(f"PEAK WORKING SET     : {peak_mb:8.1f} MB  ({peak_mb / 1024:.2f} GB)")
    print(f"labels (n={len(labels)}) : {labels}")
    print(f"n_features           : {n_features}")
    print(f"total selected rows  : {total_rows:,}")
    for name, n in per_silo.items():
        print(f"  {name:24s}: {n:,} rows")

    result = {
        "csv_path": args.csv_path,
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "chunksize": args.chunksize,
        "elapsed_s": elapsed,
        "peak_working_set_mb": peak_mb,
        "labels": labels,
        "n_features": n_features,
        "total_selected_rows": total_rows,
        "per_silo_rows": per_silo,
    }
    out_path = os.path.join(
        os.path.dirname(__file__), "modal_calibration",
        "nf_uq_nids_v2_memory_result.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
