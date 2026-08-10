"""Command-line interface: ``python -m fedgraa ...``.

Examples
--------
Reproduce the Figure-1 gamma sweep on synthetic CIC-IDS-2017::

    python -m fedgraa gamma-sweep --dataset CIC-IDS-2017 --rounds 200

Run a single FedGRAA configuration::

    python -m fedgraa run --dataset UNSW-NB15 --method fedgraa --gamma 1 --n 3

Point at real CSV exports with ``--data-dir``.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .config import DATASETS
from .experiment import (
    run_experiment,
    run_gamma_sweep,
    run_gamma_sweep_parallel,
    run_n_sweep,
    run_n_sweep_parallel,
    summarize_gamma_sweep,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fedgraa", description="FedGRAA reproduction")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--dataset", default="CIC-IDS-2017", choices=sorted(DATASETS)
    )
    common.add_argument("--rounds", type=int, default=200)
    common.add_argument("--data-dir", default=None)
    common.add_argument("--eval-every", type=int, default=20)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--verbose", action="store_true")
    common.add_argument(
        "--cache-dir", default=None,
        help="Directory to cache preprocessed .npy arrays (skips CSV re-processing)",
    )
    common.add_argument(
        "--parallel", action="store_true",
        help="Run sweep configurations in parallel via ProcessPoolExecutor (Phase 9)",
    )
    common.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory for training checkpoints (Phase 14)",
    )

    run_p = sub.add_parser("run", parents=[common], help="single configuration")
    run_p.add_argument(
        "--method", default="fedgraa", choices=["fedavg", "static", "fedgraa"]
    )
    run_p.add_argument("--gamma", type=float, default=1.0)
    run_p.add_argument("--n", type=int, default=3, dest="n_rare")

    gs_p = sub.add_parser(
        "gamma-sweep", parents=[common], help="Figure 1 reproduction"
    )
    gs_p.add_argument("--n", type=int, default=3, dest="n_rare")

    ns_p = sub.add_parser(
        "n-sweep", parents=[common], help="Figures 2-3 reproduction"
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        res = run_experiment(
            args.dataset,
            method=args.method,
            gamma=args.gamma,
            n_rare=args.n_rare,
            rounds=args.rounds,
            data_dir=args.data_dir,
            eval_every=args.eval_every,
            seed=args.seed,
            verbose=args.verbose,
            cache_dir=args.cache_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(
            f"\n[{res.dataset}] method={res.method} gamma={res.gamma} "
            f"n={res.n_rare}\n"
            f"  final all-class accuracy : {res.final_all_acc*100:.2f}%\n"
            f"  final last-n  accuracy   : {res.final_last_n_acc*100:.2f}%"
        )

    elif args.command == "gamma-sweep":
        if args.parallel:
            results = run_gamma_sweep_parallel(
                args.dataset,
                n_rare=args.n_rare,
                rounds=args.rounds,
                data_dir=args.data_dir,
                eval_every=args.eval_every,
                seed=args.seed,
                verbose=args.verbose,
                cache_dir=args.cache_dir,
                checkpoint_dir=args.checkpoint_dir,
            )
        else:
            results = run_gamma_sweep(
                args.dataset,
                n_rare=args.n_rare,
                rounds=args.rounds,
                data_dir=args.data_dir,
                eval_every=args.eval_every,
                seed=args.seed,
                verbose=args.verbose,
                cache_dir=args.cache_dir,
            )
        print(f"\n=== Figure 1 reproduction — {args.dataset} (n={args.n_rare}) ===")
        print(summarize_gamma_sweep(results))

    elif args.command == "n-sweep":
        if args.parallel:
            out = run_n_sweep_parallel(
                args.dataset,
                gammas=(0.0, 0.5, 1.0),
                rounds=args.rounds,
                data_dir=args.data_dir,
                eval_every=args.eval_every,
                seed=args.seed,
                verbose=args.verbose,
                cache_dir=args.cache_dir,
                checkpoint_dir=args.checkpoint_dir,
            )
        else:
            out = run_n_sweep(
                args.dataset,
                gammas=(0.0, 0.5, 1.0),
                rounds=args.rounds,
                data_dir=args.data_dir,
                eval_every=args.eval_every,
                seed=args.seed,
                verbose=args.verbose,
                cache_dir=args.cache_dir,
            )
        print(f"\n=== Figures 2-3 reproduction — {args.dataset} ===")
        print("n | FedAvg(last-n) | Static(last-n) | FedGRAA g=0 | g=0.5 | g=1")
        print("--+----------------+----------------+-------------+-------+------")
        for n, res in out.items():
            print(
                f"{n} | {res['fedavg'].final_last_n_acc*100:13.2f}  | "
                f"{res['static'].final_last_n_acc*100:13.2f}  | "
                f"{res['fedgraa_g0.0'].final_last_n_acc*100:11.2f} | "
                f"{res['fedgraa_g0.5'].final_last_n_acc*100:5.2f} | "
                f"{res['fedgraa_g1.0'].final_last_n_acc*100:5.2f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
