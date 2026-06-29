"""Reproduce the paper's headline tables (Figures 1-3) end to end.

Run with the bundled synthetic data (no downloads)::

    python run_experiments.py

Use real data by passing a directory of CSV exports::

    python run_experiments.py --data-dir data/CIC-IDS-2017 --datasets CIC-IDS-2017

Cache preprocessed arrays so subsequent runs skip the ~30 s CSV-load step::

    python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache \\
        --datasets CIC-IDS-2017

For a fast smoke-test with real data (verifies the pipeline end-to-end)::

    python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache \\
        --datasets CIC-IDS-2017 --rounds 30 --eval-every 10 --sweep gamma

For the full reproducibility run (slow -- expect 30-120 min on CPU)::

    python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache \\
        --datasets CIC-IDS-2017
"""

from __future__ import annotations

import argparse
import time

from fedgraa.experiment import run_gamma_sweep, run_n_sweep, summarize_gamma_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce FedGRAA Figures 1-3")
    parser.add_argument("--rounds", type=int, default=200,
                        help="Federated training rounds (default 200; use 30-50 for quick tests)")
    parser.add_argument("--eval-every", type=int, default=20,
                        help="Evaluate every N rounds (default 20)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory with CIC-IDS-2017 or UNSW-NB15 CSV files")
    parser.add_argument("--cache-dir", default=None,
                        help="Cache preprocessed .npy arrays here (reused on subsequent runs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CIC-IDS-2017", "UNSW-NB15"],
        help="Datasets to run (default: both). Use 'CIC-IDS-2017' to run only CIC.",
    )
    parser.add_argument(
        "--sweep",
        choices=["gamma", "n", "both"],
        default="both",
        help="Which sweep to run: 'gamma' (Fig 1), 'n' (Figs 2-3), or 'both' (default)",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Legacy flag (progress is now always printed)")
    args = parser.parse_args()

    t_total = time.time()

    for dataset in args.datasets:
        print("\n" + "=" * 64)
        print(f"  {dataset}")
        print("=" * 64)

        if args.sweep in ("gamma", "both"):
            print(f"\n--- Figure 1: gamma sweep (n=3) ---")
            t0 = time.time()
            gamma_results = run_gamma_sweep(
                dataset,
                n_rare=3,
                rounds=args.rounds,
                data_dir=args.data_dir,
                eval_every=args.eval_every,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )
            print(f"\nGamma sweep done in {time.time()-t0:.0f}s")
            print(summarize_gamma_sweep(gamma_results))

        if args.sweep in ("n", "both"):
            print(f"\n--- Figures 2-3: n sweep (gamma=1) ---")
            t0 = time.time()
            n_results = run_n_sweep(
                dataset,
                ns=(3, 5, 7),
                gamma=1.0,
                rounds=args.rounds,
                data_dir=args.data_dir,
                eval_every=args.eval_every,
                seed=args.seed,
                cache_dir=args.cache_dir,
            )
            print(f"\nn sweep done in {time.time()-t0:.0f}s")
            print("n | FedAvg(last-n) | Static(last-n) | FedGRAA(last-n)")
            print("--+----------------+----------------+----------------")
            for n, res in n_results.items():
                print(
                    f"{n} | {res['fedavg'].final_last_n_acc*100:13.2f}  | "
                    f"{res['static'].final_last_n_acc*100:13.2f}  | "
                    f"{res['fedgraa'].final_last_n_acc*100:13.2f}"
                )

    print(f"\nTotal wall time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
