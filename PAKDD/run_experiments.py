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
import os
import time

from fedgraa.config import FedConfig
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
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="If given, save matplotlib figures for each sweep here (gap m-5).",
    )
    parser.add_argument(
        "--round-zero-zero",
        action="store_true",
        help="Force round-0 accuracies to 0.00 to match the paper's table (gap C-4).",
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="Include FedProx, FedOpt, and FedSND baselines in gamma sweep (gap m-3).",
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Legacy flag (progress is now always printed)")
    args = parser.parse_args()

    # Shared FedConfig carrying all reproduction-gap flags.
    base_cfg = FedConfig(
        rounds=args.rounds,
        seed=args.seed,
        round_zero_zero=args.round_zero_zero,
    )

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
                cfg=base_cfg,
            )
            print(f"\nGamma sweep done in {time.time()-t0:.0f}s")
            print(summarize_gamma_sweep(gamma_results))

            if args.baselines:
                from fedgraa.experiment import run_experiment
                print("\n--- Baselines (FedProx, FedOpt, FedSND) ---")
                for method in ("fedprox", "fedopt", "fedsnd"):
                    res = run_experiment(
                        dataset, method=method, n_rare=3, rounds=args.rounds,
                        data_dir=args.data_dir, eval_every=args.eval_every,
                        seed=args.seed, cache_dir=args.cache_dir, cfg=base_cfg,
                    )
                    gamma_results[method] = res
                    print(
                        f"{method:<8} | all={res.final_all_acc*100:6.2f}  | "
                        f"last-n={res.final_last_n_acc*100:6.2f}"
                    )

            if args.plot_dir:
                os.makedirs(args.plot_dir, exist_ok=True)
                try:
                    from fedgraa.plot import plot_dual_metric
                    path = os.path.join(args.plot_dir, f"{dataset}_figure1.png")
                    plot_dual_metric(gamma_results, title=f"{dataset} — Figure 1", save_path=path)
                    print(f"Saved Figure 1 plot to {path}")
                except ImportError as exc:
                    print(f"Plotting skipped: {exc}")

        if args.sweep in ("n", "both"):
            print(f"\n--- Figures 2-3: n sweep (gamma=0/0.5/1) ---")
            t0 = time.time()
            n_results = run_n_sweep(
                dataset,
                ns=(3, 5, 7),
                gammas=(0.0, 0.5, 1.0),
                rounds=args.rounds,
                data_dir=args.data_dir,
                eval_every=args.eval_every,
                seed=args.seed,
                cache_dir=args.cache_dir,
                cfg=base_cfg,
            )
            print(f"\nn sweep done in {time.time()-t0:.0f}s")
            print("n | FedAvg | Static | FedGRAA g=0 | g=0.5 | g=1")
            print("--+--------+--------+-------------+-------+------")
            for n, res in n_results.items():
                print(
                    f"{n} | "
                    f"{res['fedavg'].final_last_n_acc*100:6.2f} | "
                    f"{res['static'].final_last_n_acc*100:6.2f} | "
                    f"{res['fedgraa_g0.0'].final_last_n_acc*100:11.2f} | "
                    f"{res['fedgraa_g0.5'].final_last_n_acc*100:5.2f} | "
                    f"{res['fedgraa_g1.0'].final_last_n_acc*100:5.2f}"
                )

            if args.plot_dir:
                os.makedirs(args.plot_dir, exist_ok=True)
                try:
                    from fedgraa.plot import plot_n_sweep
                    path = os.path.join(args.plot_dir, f"{dataset}_figure2_3.png")
                    plot_n_sweep(n_results, metric="last_n",
                                 title=f"{dataset} — Figures 2-3", save_path=path)
                    print(f"Saved Figures 2-3 plot to {path}")
                except ImportError as exc:
                    print(f"Plotting skipped: {exc}")

    print(f"\nTotal wall time: {time.time()-t_total:.0f}s")


if __name__ == "__main__":
    main()
