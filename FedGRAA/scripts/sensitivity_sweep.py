"""Hyperparameter sensitivity sweep for the FedGRAA weight learner.

P0 of ``docs/GPU_DECISION_AND_NEXT_STEPS.md``: the paper's defaults
(``eta' = 0.1``, ``E' = 5``) are the prime suspects for the observed weight
divergence. This sweeps ``eta' in {0.001, 0.01, 0.1}`` and ``E' in {1, 5}``
first on the fast synthetic path (cheap enough to run every combination for
the full 200 rounds), then re-runs the two most promising / most informative
combinations on real CIC-IDS-2017 data to confirm the finding transfers.

Usage::

    python scripts/sensitivity_sweep.py --stage synthetic
    python scripts/sensitivity_sweep.py --stage real --etas 0.001 --epochs 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_fedgraa import analyze_history, run_fine_grained  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--dataset", default="CIC-IDS-2017")
    ap.add_argument("--etas", nargs="+", type=float, default=[0.001, 0.01, 0.1])
    ap.add_argument("--epochs", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--gammas", nargs="+", type=float, default=[1.0, 0.0])
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.stage == "synthetic":
        data_dir, cache_dir = None, None
        rounds = args.rounds or 200
        eval_every = args.eval_every or 20
    else:
        data_dir, cache_dir = "data", ".cache"
        rounds = args.rounds or 100
        eval_every = args.eval_every or 10

    out_path = args.out or f"results/diagnostics/sensitivity_{args.stage}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 72)
    print(f"  FedGRAA eta'/E' sensitivity sweep — stage={args.stage} "
          f"dataset={args.dataset} rounds={rounds} eval_every={eval_every}")
    print("=" * 72)

    results = {}
    for gamma, eta_prime, weight_epochs in itertools.product(
        args.gammas, args.etas, args.epochs
    ):
        key = f"g{gamma}_eta{eta_prime}_E{weight_epochs}"
        print(f"\n>>> {key}")
        t0 = time.time()
        history = run_fine_grained(
            args.dataset, gamma, rounds, eval_every, args.seed,
            data_dir, cache_dir, eta_prime=eta_prime, weight_epochs=weight_epochs,
        )
        dt = time.time() - t0
        diag = analyze_history(history)
        print(
            f"  done in {dt:.0f}s | monotone={diag['monotone_lemma1']} "
            f"| blowup_ratio={diag['composite_loss_blowup_ratio']} "
            f"| final_weight_sum={diag['final_weight_sum']} "
            f"| final all={diag['final_all_acc']:.3f} last-n={diag['final_last_n_acc']:.3f}"
        )
        results[key] = {
            "gamma": gamma,
            "eta_prime": eta_prime,
            "weight_epochs": weight_epochs,
            "wall_time_s": dt,
            "diagnosis": diag,
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sensitivity sweep to {out_path}")

    # Summary table
    print("\n" + "-" * 92)
    print(f"{'config':<28} | {'monotone':>8} | {'blowup':>10} | {'|w|sum-1|':>10} "
          f"| {'all-acc':>8} | {'last-n':>8}")
    print("-" * 92)
    for key, r in results.items():
        d = r["diagnosis"]
        blowup = d["composite_loss_blowup_ratio"]
        wsum = d["final_weight_sum"]
        wdev = abs(wsum - 1.0) if wsum is not None else float("nan")
        print(
            f"{key:<28} | {str(d['monotone_lemma1']):>8} | "
            f"{blowup if blowup is None else round(blowup, 1):>10} | {wdev:>10.4f} "
            f"| {d['final_all_acc']*100:7.2f}% | {d['final_last_n_acc']*100:7.2f}%"
        )


if __name__ == "__main__":
    main()
