"""P3: seed-variance sweep — report variance bands instead of a single point.

The paper discloses no random seed (PAPER_AMBIGUITY_REPORT.md #16). This runs
the same configuration across multiple seeds and reports mean +/- std, so the
reproduction can state an honest variance band rather than imply a single run
is "the" answer.

Usage::

    python scripts/run_seed_sweep.py --dataset CIC-IDS-2017 --method static \\
        --seeds 0 1 2 3 4 --rounds 200 --eval-every 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from fedgraa.experiment import run_seed_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="CIC-IDS-2017")
    ap.add_argument("--method", default="static", choices=["fedavg", "static", "fedgraa"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--n-rare", type=int, default=3)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or (
        f"results/diagnostics/seed_sweep_{args.dataset.replace('-', '_')}"
        f"_{args.method}{f'_g{args.gamma}' if args.method == 'fedgraa' else ''}.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Seed sweep: {args.dataset} {args.method} "
          f"{'g=' + str(args.gamma) if args.method == 'fedgraa' else ''} "
          f"seeds={args.seeds} rounds={args.rounds}")
    t0 = time.time()
    results = run_seed_sweep(
        args.dataset, method=args.method, gamma=args.gamma, n_rare=args.n_rare,
        seeds=args.seeds, rounds=args.rounds, eval_every=args.eval_every,
        data_dir=args.data_dir, cache_dir=args.cache_dir,
    )
    dt = time.time() - t0

    all_accs = [r.final_all_acc for r in results.values()]
    last_n_accs = [r.final_last_n_acc for r in results.values()]

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "gamma": args.gamma if args.method == "fedgraa" else None,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "wall_time_s": dt,
        "all_acc_per_seed": all_accs,
        "last_n_acc_per_seed": last_n_accs,
        "all_acc_mean": float(np.mean(all_accs)),
        "all_acc_std": float(np.std(all_accs)),
        "last_n_acc_mean": float(np.mean(last_n_accs)),
        "last_n_acc_std": float(np.std(last_n_accs)),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {dt:.0f}s. Saved to {out_path}")
    print(f"All-acc:   {summary['all_acc_mean']*100:.2f}% +/- {summary['all_acc_std']*100:.2f}% "
          f"(per-seed: {[f'{a*100:.1f}%' for a in all_accs]})")
    print(f"Rare-acc:  {summary['last_n_acc_mean']*100:.2f}% +/- {summary['last_n_acc_std']*100:.2f}% "
          f"(per-seed: {[f'{a*100:.1f}%' for a in last_n_accs]})")


if __name__ == "__main__":
    main()
