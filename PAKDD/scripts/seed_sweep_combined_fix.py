"""Seed-variance band for the combined fix (simplex projection + eta'=0.01).

docs/fedgraa_divergence_diagnosis.md Section 8 showed that projection *and* a
reduced eta' together (unlike either alone) recover FedGRAA's rare-class
accuracy on real CIC-IDS-2017 -- but that result, as first reported, was a
single seed (seed=0), 60 rounds. Section 6.2 already gave FedAvg and Static
Reparam. a 3-seed variance band; the fix deserves the same treatment before
its headline numbers are taken as more than a single lucky draw.

This script does not re-run training itself -- it reads the three per-seed
diagnostic JSON files already produced by
``scripts/diagnose_fedgraa.py --project --eta-prime 0.01 --weight-epochs 5``
(seed 0: ``CIC_IDS_2017_projected_combined_eta0.01_E5_r60.json``; seed 1/2:
the same with a ``_seed{n}`` tag) and aggregates ``final_all_acc`` /
``final_last_n_acc`` per gamma into a mean +/- std, in the same format as
``run_seed_sweep.py``'s summary JSON.

Usage::

    python scripts/seed_sweep_combined_fix.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIAG_DIR = "results/diagnostics"
GAMMAS = [0.0, 0.5, 1.0]
SEED_FILES = {
    0: "CIC_IDS_2017_projected_combined_eta0.01_E5_r60.json",
    1: "CIC_IDS_2017_projected_combined_eta0.01_E5_seed1_r60.json",
    2: "CIC_IDS_2017_projected_combined_eta0.01_E5_seed2_r60.json",
}


def main() -> None:
    per_seed = {}
    for seed, fname in SEED_FILES.items():
        path = os.path.join(DIAG_DIR, fname)
        with open(path, encoding="utf-8") as f:
            per_seed[seed] = json.load(f)

    summary = {}
    print("=" * 88)
    print("  Combined-fix (projection + eta'=0.01, E'=5) seed-variance band, CIC-IDS-2017")
    print("=" * 88)
    print(f"{'gamma':<8} | {'all-acc mean+/-std':<22} | {'rare-acc mean+/-std':<22} | per-seed rare-acc")
    print("-" * 88)

    for gamma in GAMMAS:
        key = f"g{gamma}"
        all_accs = [per_seed[s][key]["diagnosis"]["final_all_acc"] for s in SEED_FILES]
        rare_accs = [per_seed[s][key]["diagnosis"]["final_last_n_acc"] for s in SEED_FILES]
        blowups = [per_seed[s][key]["diagnosis"]["composite_loss_blowup_ratio"] for s in SEED_FILES]

        entry = {
            "gamma": gamma,
            "seeds": list(SEED_FILES.keys()),
            "all_acc_per_seed": all_accs,
            "last_n_acc_per_seed": rare_accs,
            "blowup_ratio_per_seed": blowups,
            "all_acc_mean": float(np.mean(all_accs)),
            "all_acc_std": float(np.std(all_accs)),
            "last_n_acc_mean": float(np.mean(rare_accs)),
            "last_n_acc_std": float(np.std(rare_accs)),
        }
        summary[key] = entry

        print(
            f"{gamma:<8} | {entry['all_acc_mean']*100:6.2f}% +/- {entry['all_acc_std']*100:4.2f}%    "
            f"| {entry['last_n_acc_mean']*100:6.2f}% +/- {entry['last_n_acc_std']*100:4.2f}%    "
            f"| {[f'{a*100:.1f}%' for a in rare_accs]}"
        )

    out_path = os.path.join(DIAG_DIR, "seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
