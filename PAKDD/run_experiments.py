"""Reproduce the paper's headline tables (Figures 1-3) end to end.

Run with the bundled synthetic data (no downloads)::

    python run_experiments.py

Use real data by passing a directory of CSV exports::

    python run_experiments.py --data-dir path/to/CIC-IDS-2017

For a fast smoke run, lower the round count::

    python run_experiments.py --rounds 40 --eval-every 10
"""

from __future__ import annotations

import argparse

from fedgraa.experiment import run_gamma_sweep, run_n_sweep, summarize_gamma_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce FedGRAA Figures 1-3")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CIC-IDS-2017", "UNSW-NB15"],
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    for dataset in args.datasets:
        print("\n" + "=" * 64)
        print(f"  {dataset}")
        print("=" * 64)

        print(f"\n--- Figure 1: gamma sweep (n=3) ---")
        gamma_results = run_gamma_sweep(
            dataset,
            n_rare=3,
            rounds=args.rounds,
            data_dir=args.data_dir,
            eval_every=args.eval_every,
            seed=args.seed,
            verbose=args.verbose,
        )
        print(summarize_gamma_sweep(gamma_results))

        print(f"\n--- Figures 2-3: n sweep (gamma=1) ---")
        n_results = run_n_sweep(
            dataset,
            ns=(3, 5, 7),
            gamma=1.0,
            rounds=args.rounds,
            data_dir=args.data_dir,
            eval_every=args.eval_every,
            seed=args.seed,
            verbose=args.verbose,
        )
        print("n | FedAvg(last-n) | Static(last-n) | FedGRAA(last-n)")
        print("--+----------------+----------------+----------------")
        for n, res in n_results.items():
            print(
                f"{n} | {res['fedavg'].final_last_n_acc*100:13.2f}  | "
                f"{res['static'].final_last_n_acc*100:13.2f}  | "
                f"{res['fedgraa'].final_last_n_acc*100:13.2f}"
            )


if __name__ == "__main__":
    main()
