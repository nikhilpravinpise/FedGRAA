"""Full-fidelity CIC-IDS-2017 memory calibration: 3 gammas x 60 rounds,
matching exactly what scripts/run_ablation_seed_sweep.py's default (--rounds
60 --eval-every 1) invokes per (arm, seed) via diagnose_fedgraa.py. This is
the fully-faithful version -- an earlier single-gamma/3-round probe measured
the same ~8.63 GB peak, but could not rule out memory accumulating across the
three sequential load_dataset/partition_non_iid calls the real per-unit job
makes back-to-back in one process.

Measured result (2026-07-28, see modal_cic_memory_result.json):
  total_elapsed_sec = 1235.9s (~20.6 min) for the full 3-gamma unit -- matches
  FEDGRAA_V2_MASTER_PLAN.md Sec 7.2's "15-20 min per (arm, seed)" estimate.
  peak_mem_mb_after_each_gamma = 8629.875 MB, IDENTICAL after gamma 0.0, 0.5,
  and 1.0 -- confirms memory does NOT compound across sequential gammas in one
  process; ~8.63 GB is the stable per-unit peak for CIC-IDS-2017, not a
  lower bound that keeps climbing.

Recommendation drawn from this: memory=12288 (12 GiB) for CIC-IDS-2017 runs
in the real campaign (~40% headroom over the measured 8.63 GB peak);
memory=4096 (4 GiB) is enough for UNSW-NB15 (measured ~2.1 GB peak, see
modal_cpu_scaling_result.json).

Usage: modal run scripts/modal_calibration/modal_cic_memory_calibration.py
"""

from __future__ import annotations

import modal

app = modal.App("fedgraa-mem-calibration-cic-full")


def _ignore(path):
    from pathlib import Path

    p = Path(path)
    parts = p.parts
    if "UNSW-NB15" in parts:
        return True
    return any(
        part in {".venv313", ".cache", ".pytest_cache", "__pycache__", "archive", ".git"}
        for part in parts
    )


image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        "tensorflow>=2.21.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "scipy>=1.10",
    )
    .add_local_dir("C:/Dev/Research/PAKDD", remote_path="/repo", copy=True, ignore=_ignore)
)


@app.function(image=image, cpu=1.0, memory=16384, timeout=2400)
def run_cic_full() -> dict:
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    import os
    import resource
    import sys
    import time

    sys.path.insert(0, "/repo")
    os.chdir("/repo")

    from fedgraa.config import FedConfig
    from fedgraa.data import load_dataset
    from fedgraa.partition import partition_non_iid
    from fedgraa.trainer import FederatedTrainer

    def _run_one_gamma(gamma):
        X, y, spec = load_dataset("CIC-IDS-2017", data_dir="/repo/data/CIC-IDS-2017", seed=0, cache_dir="/tmp/.cache")
        cfg = FedConfig(rounds=60, gamma=gamma, n_rare=3, seed=0)
        split = partition_non_iid(X, y, spec, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction, seed=cfg.seed)
        trainer = FederatedTrainer(split, cfg, method="fedgraa")
        return trainer.train(eval_every=1, verbose=False)

    per_gamma = {}
    peak_after_each = {}
    t_total0 = time.perf_counter()
    for gamma in (0.0, 0.5, 1.0):
        t0 = time.perf_counter()
        history = _run_one_gamma(gamma)
        per_gamma[gamma] = {
            "elapsed_sec": time.perf_counter() - t0,
            "final_all_acc": history.final_all_acc,
            "final_last_n_acc": history.final_last_n_acc,
        }
        peak_after_each[gamma] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    return {
        "dataset": "CIC-IDS-2017",
        "total_elapsed_sec": time.perf_counter() - t_total0,
        "per_gamma": per_gamma,
        "peak_mem_mb_after_each_gamma": peak_after_each,
        "final_peak_mem_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }


@app.local_entrypoint()
def main():
    import json
    import os

    print("Running full-fidelity CIC-IDS-2017 calibration (3 gammas x 60 rounds) ...")
    result = run_cic_full.remote()
    print(json.dumps(result, indent=2))

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modal_cic_memory_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")
