"""Seed-variance band for the combined fix (simplex projection + eta'=0.01).

docs/fedgraa_divergence_diagnosis.md Section 8 showed that projection *and* a
reduced eta' together (unlike either alone) recover FedGRAA's rare-class
accuracy on real CIC-IDS-2017 -- but that result, as first reported, was a
single seed (seed=0), 60 rounds. Section 6.2 already gave FedAvg and Static
Reparam. a 3-seed variance band; the fix deserves the same treatment before
its headline numbers are taken as more than a single lucky draw.

This script does not re-run training itself -- it reads the per-seed
diagnostic JSON files already produced by
``scripts/diagnose_fedgraa.py --project --eta-prime 0.01 --weight-epochs 5``
(seed 0: ``{DS}_projected_combined_eta{eta}_E{E}_r{rounds}.json``; seed>0:
the same with a ``_seed{n}`` tag) and aggregates ``final_all_acc`` /
``final_last_n_acc`` / ``composite_loss_blowup_ratio`` per gamma into a
mean +/- std, in the same format as ``run_seed_sweep.py``'s summary JSON.

Usage (CIC, defaults -- matches the original single-dataset behavior)::

    python scripts/seed_sweep_combined_fix.py

Usage (any dataset, stage_a_plan.md §6.3)::

    python scripts/seed_sweep_combined_fix.py --dataset UNSW-NB15
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIAG_DIR = "results/diagnostics"
GAMMAS = [0.0, 0.5, 1.0]


def _seed_filename(dataset_tag: str, eta_prime: str, weight_epochs: int, rounds: int, seed: int) -> str:
    suffix = f"_seed{seed}" if seed > 0 else ""
    return f"{dataset_tag}_projected_combined_eta{eta_prime}_E{weight_epochs}{suffix}_r{rounds}.json"


def aggregate(per_seed: dict, gammas=GAMMAS) -> dict:
    """Aggregate per-seed diagnostic dicts (keyed by seed int) into a mean/std
    summary keyed by ``g{gamma}``, mirroring ``run_seed_sweep.py``'s schema.
    """
    summary = {}
    for gamma in gammas:
        key = f"g{gamma}"
        all_accs = [per_seed[s][key]["diagnosis"]["final_all_acc"] for s in per_seed]
        rare_accs = [per_seed[s][key]["diagnosis"]["final_last_n_acc"] for s in per_seed]
        blowups = [per_seed[s][key]["diagnosis"]["composite_loss_blowup_ratio"] for s in per_seed]

        summary[key] = {
            "gamma": gamma,
            "seeds": list(per_seed.keys()),
            "all_acc_per_seed": all_accs,
            "last_n_acc_per_seed": rare_accs,
            "blowup_ratio_per_seed": blowups,
            "all_acc_mean": float(np.mean(all_accs)),
            "all_acc_std": float(np.std(all_accs)),
            "last_n_acc_mean": float(np.mean(rare_accs)),
            "last_n_acc_std": float(np.std(rare_accs)),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="CIC-IDS-2017")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--eta-prime", default="0.01")
    ap.add_argument("--weight-epochs", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=60)
    args = ap.parse_args()

    dataset_tag = args.dataset.replace("-", "_")
    seed_files = {
        seed: _seed_filename(dataset_tag, args.eta_prime, args.weight_epochs, args.rounds, seed)
        for seed in args.seeds
    }

    per_seed = {}
    for seed, fname in seed_files.items():
        path = os.path.join(DIAG_DIR, fname)
        with open(path, encoding="utf-8") as f:
            per_seed[seed] = json.load(f)

    summary = aggregate(per_seed, GAMMAS)
    print("=" * 88)
    print(f"  Combined-fix (projection + eta'={args.eta_prime}, E'={args.weight_epochs}) "
          f"seed-variance band, {args.dataset}")
    print("=" * 88)
    print(f"{'gamma':<8} | {'all-acc mean+/-std':<22} | {'rare-acc mean+/-std':<22} | per-seed rare-acc")
    print("-" * 88)

    for gamma in GAMMAS:
        entry = summary[f"g{gamma}"]
        print(
            f"{gamma:<8} | {entry['all_acc_mean']*100:6.2f}% +/- {entry['all_acc_std']*100:4.2f}%    "
            f"| {entry['last_n_acc_mean']*100:6.2f}% +/- {entry['last_n_acc_std']*100:4.2f}%    "
            f"| {[f'{a*100:.1f}%' for a in entry['last_n_acc_per_seed']]}"
        )

    out_path = os.path.join(DIAG_DIR, f"seed_sweep_{dataset_tag}_fedgraa_combined_fix.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
