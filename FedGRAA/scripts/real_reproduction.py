"""Real reproduction test: gamma sweep on real CIC-IDS-2017 data."""

import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from fedgraa.experiment import (
    run_gamma_sweep,
    run_gamma_sweep_parallel,
    run_n_sweep,
    summarize_gamma_sweep,
    save_results,
)
from fedgraa.config import FedConfig

DATA_DIR = "data"
CACHE_DIR = ".cache"

print("=" * 72)
print("  FedGRAA Reproduction Test — Real Data (CIC-IDS-2017)")
print("  System: Windows CPU only (TF 2.21, oneDNN, XLA)")
print("=" * 72)

cfg = FedConfig(rounds=200, n_rare=3, seed=0)

# --- Gamma sweep on real CIC-IDS-2017 data ---
print("\n>>> Gamma sweep: CIC-IDS-2017 (200 rounds, n=3, real data)")
t0 = time.time()
results = run_gamma_sweep(
    "CIC-IDS-2017",
    n_rare=3,
    rounds=200,
    eval_every=20,
    seed=0,
    data_dir=DATA_DIR,
    cache_dir=CACHE_DIR,
    cfg=cfg,
)
t1 = time.time()
dt = t1 - t0
hours = int(dt // 3600)
mins = int((dt % 3600) // 60)
secs = int(dt % 60)
print(f"\n=== Gamma sweep finished: {hours}h {mins}m {secs}s ===")
print(summarize_gamma_sweep(results))

# Save results
save_results(results, ".cache/gamma_sweep_cic_results.json")

# --- Print per-round details ---
print("\n=== Per-round detail ===")
for method, r in results.items():
    h = r.history
    final_all = h.final_all_acc * 100
    final_last = h.final_last_n_acc * 100
    total_rounds = len(h.rounds)
    print(f"\n{method}:")
    print(f"  Rounds evaluated: {total_rounds}")
    print(f"  Final all-acc: {final_all:.2f}%")
    print(f"  Final last-n-acc: {final_last:.2f}%")
    for rnd, all_a, last_a in zip(h.rounds, h.all_acc, h.last_n_acc):
        print(f"    round {rnd:3d}: all={all_a*100:5.2f}%  last-n={last_a*100:5.2f}%")
