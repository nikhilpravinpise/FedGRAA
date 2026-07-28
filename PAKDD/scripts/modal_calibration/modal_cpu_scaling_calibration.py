"""Modal calibration: measure real wall-clock + peak memory for one (arm, seed)
unit of work (dataset=UNSW-NB15, arm=literal, gammas=[0,0.5,1.0], rounds=60,
eval_every=1, seed=0 -- exactly what scripts/run_ablation_seed_sweep.py's
default --rounds 60 --eval-every 1 invokes per (arm, seed) via diagnose_fedgraa.py's
run_fine_grained) at cpu=1.0, cpu=2.0, cpu=4.0.

This exists to replace the *assumed* core-scaling behaviour in
docs/research_c/COMPUTE_INFRASTRUCTURE_2026-07-27.md Sec 6.2 with a
measurement, per this project's own "measure, don't assume" discipline
(FEDGRAA_V2_MASTER_PLAN.md F3/F12).

Uses existing v1 code only (FedGRAA / ProjectedFedGRAA) -- no v2 code exists
yet, so this is safe to run today.

Measured result (2026-07-28, see modal_cpu_scaling_result.json):
cpu=1.0 -> 681.5s total (3 gammas), peak 2086 MB
cpu=2.0 -> 790.1s total,             peak 2068 MB
cpu=4.0 -> 617.2s total,             peak 2117 MB
No reliable speedup from more cores -- consistent with fedgraa/trainer.py
hard-coding TF_NUM_INTRAOP_THREADS=1 at import time, which caps TensorFlow to
single-threaded matmul regardless of the cpu= request. Use cpu=1.0.

Usage: modal run scripts/modal_calibration/modal_cpu_scaling_calibration.py
"""

from __future__ import annotations

import modal

app = modal.App("fedgraa-cpu-calibration")


def _ignore(path):
    from pathlib import Path

    p = Path(path)
    parts = p.parts
    if "CIC-IDS-2017" in parts:  # skip the 844 MB dir -- not needed for this test
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


def _run_calibration(cpu_label: str) -> dict:
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)

    import os
    import resource
    import sys
    import time

    sys.path.insert(0, "/repo")
    os.chdir("/repo")

    # Mirrors diagnose_fedgraa.run_fine_grained exactly, but with verbose=False --
    # run_fine_grained hardcodes verbose=True, which prints unicode progress
    # characters that crash a Windows cp1252 local console log stream.
    from fedgraa.config import FedConfig
    from fedgraa.data import load_dataset
    from fedgraa.partition import partition_non_iid
    from fedgraa.trainer import FederatedTrainer

    def _run_fine_grained_quiet(gamma):
        X, y, spec = load_dataset("UNSW-NB15", data_dir="/repo/data/UNSW-NB15", seed=0, cache_dir="/tmp/.cache")
        cfg = FedConfig(rounds=60, gamma=gamma, n_rare=3, seed=0, eta_prime=0.1, weight_epochs=5)
        split = partition_non_iid(X, y, spec, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction, seed=cfg.seed)
        trainer = FederatedTrainer(split, cfg, method="fedgraa")
        return trainer.train(eval_every=1, verbose=False)

    per_gamma = {}
    t_total0 = time.perf_counter()
    for gamma in (0.0, 0.5, 1.0):
        t0 = time.perf_counter()
        history = _run_fine_grained_quiet(gamma)
        per_gamma[gamma] = {
            "elapsed_sec": time.perf_counter() - t0,
            "final_all_acc": history.final_all_acc,
            "final_last_n_acc": history.final_last_n_acc,
        }
    total_elapsed = time.perf_counter() - t_total0
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
    return {
        "cpu_label": cpu_label,
        "cpu_count_visible": os.cpu_count(),
        "total_elapsed_sec": total_elapsed,
        "per_gamma": per_gamma,
        "peak_mem_mb": peak_kb / 1024.0,
        "env_threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }


@app.function(image=image, cpu=1.0, memory=4096, timeout=1800)
def run_cpu1() -> dict:
    return _run_calibration("cpu=1.0")


@app.function(image=image, cpu=2.0, memory=4096, timeout=1800)
def run_cpu2() -> dict:
    return _run_calibration("cpu=2.0")


@app.function(image=image, cpu=4.0, memory=4096, timeout=1800)
def run_cpu4() -> dict:
    return _run_calibration("cpu=4.0")


@app.local_entrypoint()
def main():
    import json
    import os
    import time

    print("Launching cpu=1.0 / cpu=2.0 / cpu=4.0 calibration runs in parallel on Modal ...")
    t0 = time.time()
    h1 = run_cpu1.spawn()
    h2 = run_cpu2.spawn()
    h4 = run_cpu4.spawn()
    results = [h1.get(), h2.get(), h4.get()]
    wall = time.time() - t0

    print("\n" + "=" * 72)
    for r in results:
        print(json.dumps(r, indent=2))
    print("=" * 72)
    print(f"All three finished (parallel wall-clock): {wall:.1f}s")

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modal_cpu_scaling_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
