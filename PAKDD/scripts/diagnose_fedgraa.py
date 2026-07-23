"""Diagnose FedGRAA's weight-learner divergence on real data.

This is the instrumentation called for by ``docs/GPU_DECISION_AND_NEXT_STEPS.md``
P0: dump the composite loss ``L'(w^t)`` and the aggregation-weight vector
``w^t`` every round (not just every ``eval_every`` rounds) so we can see
*exactly* where Lemma 1 (non-increasing ``L'``) breaks down, and whether it
correlates with ``w`` leaving the probability simplex (``sum(w) != 1`` or
``min(w) < 0``) — the one place Algorithm 2, taken literally, has no
safety rail (see ``docs/strict_reproduction_report.md`` §2.2).

Two run modes:

* **Default** (``--project`` not given): the unmodified, strictly-literal
  ``fedgraa.aggregation.FedGRAA`` — exactly what ``fedgraa/`` ships. This is
  the number that goes in the paper comparison.
* **``--project``**: a *diagnostic-only* ablation that Euclidean-projects
  ``w`` back onto the probability simplex after every ``FedGRAA.update()``
  call. This class lives only in this script — it is never imported by
  ``fedgraa/`` and never changes the strict-reproduction default. It exists
  solely to test the hypothesis that unconstrained weight drift, not the
  paper's algorithm per se, is what's blowing up ``L'``.

Usage::

    # Strict default, fine-grained, real CIC-IDS-2017, all three gammas
    python scripts/diagnose_fedgraa.py --dataset CIC-IDS-2017 \\
        --gammas 0 0.5 1.0 --rounds 50 --eval-every 1

    # Simplex-projected ablation, same config, for comparison
    python scripts/diagnose_fedgraa.py --dataset CIC-IDS-2017 \\
        --gammas 0 0.5 1.0 --rounds 50 --eval-every 1 --project
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

import numpy as np

from fedgraa.aggregation import FedGRAA
from fedgraa.config import FedConfig
from fedgraa.data import load_dataset
from fedgraa.partition import partition_non_iid
from fedgraa.trainer import FederatedTrainer, TrainHistory


# --------------------------------------------------------------------------- #
# Diagnostic-only ablation: Euclidean projection onto the probability simplex.
# Not part of fedgraa/ — see module docstring.
# --------------------------------------------------------------------------- #
def project_to_simplex(w: np.ndarray) -> np.ndarray:
    """Euclidean projection of ``w`` onto ``{x : x >= 0, sum(x) = 1}``.

    Standard algorithm (Duchi et al., 2008, "Efficient Projections onto the
    L1-Ball for Learning in High Dimensions", Fig. 1).
    """
    w = np.asarray(w, dtype=np.float64)
    n = len(w)
    u = np.sort(w)[::-1]
    css = np.cumsum(u) - 1.0
    idx = np.arange(1, n + 1)
    cond = u - css / idx > 0
    if not np.any(cond):
        return np.full(n, 1.0 / n, dtype=np.float32)
    rho = idx[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(w - theta, 0.0).astype(np.float32)


class ProjectedFedGRAA(FedGRAA):
    """Diagnostic-only ablation of :class:`fedgraa.aggregation.FedGRAA`.

    Identical to the strict-reproduction ``FedGRAA`` except that the weight
    vector is re-projected onto the probability simplex after the ``E'``-epoch
    Adam update. Used only by this script to isolate whether unconstrained
    weight drift (not present in this variant) is the cause of the composite
    loss divergence observed with the literal Algorithm 2.
    """

    def update(self, client_params, weights_prev, val):
        w = super().update(client_params, weights_prev, val)
        return project_to_simplex(w)


# --------------------------------------------------------------------------- #
# Fine-grained run
# --------------------------------------------------------------------------- #
def run_fine_grained(
    dataset: str,
    gamma: float,
    rounds: int,
    eval_every: int,
    seed: int,
    data_dir: str,
    cache_dir: str,
    n_rare: int = 3,
    project: bool = False,
    eta_prime: float = 0.1,
    weight_epochs: int = 5,
) -> TrainHistory:
    spec_dir = (
        os.path.join(data_dir, dataset)
        if data_dir and os.path.isdir(os.path.join(data_dir, dataset))
        else data_dir
    )
    X, y, spec = load_dataset(dataset, data_dir=spec_dir, seed=seed, cache_dir=cache_dir)
    cfg = FedConfig(
        rounds=rounds, gamma=gamma, n_rare=n_rare, seed=seed,
        eta_prime=eta_prime, weight_epochs=weight_epochs,
    )
    split = partition_non_iid(
        X, y, spec, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    trainer = FederatedTrainer(split, cfg, method="fedgraa")
    if project:
        layer_sizes = spec.model_config().layer_sizes
        trainer.fedgraa = ProjectedFedGRAA(
            gamma=cfg.gamma, eta_prime=cfg.eta_prime,
            weight_epochs=cfg.weight_epochs, layer_sizes=layer_sizes,
        )
    return trainer.train(eval_every=eval_every, verbose=True)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze_history(history: TrainHistory) -> dict:
    """Derive the Lemma-1 / simplex-drift diagnostics from a ``TrainHistory``."""
    comp = history.composite_loss
    finite_pairs = [(r, l) for r, l in zip(history.rounds, comp) if np.isfinite(l)]

    monotone = history.is_composite_loss_monotone
    first_violation_round = None
    for (r0, l0), (r1, l1) in zip(finite_pairs, finite_pairs[1:]):
        if l1 > l0:
            first_violation_round = r1
            break

    finite_losses = [l for _, l in finite_pairs]
    blowup_ratio = (
        max(finite_losses) / finite_losses[0]
        if finite_losses and finite_losses[0] > 0 else None
    )

    sums = [float(w.sum()) for w in history.weights]
    mins = [float(w.min()) for w in history.weights]
    maxs = [float(np.abs(w).max()) for w in history.weights]

    simplex_violation_round = None
    for r, s, m in zip(history.rounds, sums, mins):
        if abs(s - 1.0) > 1e-3 or m < -1e-6:
            simplex_violation_round = r
            break

    return {
        "monotone_lemma1": bool(monotone),
        "first_monotonicity_violation_round": first_violation_round,
        "composite_loss_blowup_ratio": blowup_ratio,
        "composite_loss_final": finite_losses[-1] if finite_losses else None,
        "composite_loss_initial": finite_losses[0] if finite_losses else None,
        "weight_sum_trajectory": sums,
        "weight_min_trajectory": mins,
        "weight_max_abs_trajectory": maxs,
        "first_simplex_violation_round": simplex_violation_round,
        "final_weight_sum": sums[-1] if sums else None,
        "final_weight_min": mins[-1] if mins else None,
        "final_all_acc": history.final_all_acc,
        "final_last_n_acc": history.final_last_n_acc,
    }


def _history_to_dict(history: TrainHistory) -> dict:
    return {
        "method": history.method,
        "gamma": history.gamma,
        "n_rare": history.n_rare,
        "rounds": history.rounds,
        "all_acc": history.all_acc,
        "last_n_acc": history.last_n_acc,
        "composite_loss": [v if np.isfinite(v) else None for v in history.composite_loss],
        "weights": [w.tolist() for w in history.weights],
    }


def dict_to_history(d: dict) -> TrainHistory:
    """Inverse of :func:`_history_to_dict` — reconstruct a ``TrainHistory``
    from a saved diagnostic JSON entry, for post-hoc plotting/analysis."""
    h = TrainHistory(method=d["method"], gamma=d.get("gamma"), n_rare=d["n_rare"])
    h.rounds = list(d["rounds"])
    h.all_acc = list(d["all_acc"])
    h.last_n_acc = list(d["last_n_acc"])
    h.composite_loss = [
        float(v) if v is not None else float("nan") for v in d["composite_loss"]
    ]
    h.weights = [np.asarray(w, dtype=np.float32) for w in d["weights"]]
    return h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="CIC-IDS-2017")
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-rare", type=int, default=3)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out-dir", default="results/diagnostics")
    ap.add_argument("--project", action="store_true",
                     help="Run the diagnostic-only simplex-projected ablation.")
    ap.add_argument("--eta-prime", type=float, default=0.1)
    ap.add_argument("--weight-epochs", type=int, default=5)
    ap.add_argument("--tag", default=None,
                     help="Extra suffix for output filenames (e.g. sensitivity sweep tag).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    variant = "projected" if args.project else "literal"
    tag = f"_{args.tag}" if args.tag else ""

    print("=" * 72)
    print(f"  FedGRAA divergence diagnosis — {args.dataset} — variant={variant}{tag}")
    print(f"  rounds={args.rounds} eval_every={args.eval_every} "
          f"eta'={args.eta_prime} E'={args.weight_epochs}")
    print("=" * 72)

    all_results = {}
    for gamma in args.gammas:
        print(f"\n>>> gamma={gamma}")
        t0 = time.time()
        history = run_fine_grained(
            args.dataset, gamma, args.rounds, args.eval_every, args.seed,
            args.data_dir, args.cache_dir, n_rare=args.n_rare,
            project=args.project, eta_prime=args.eta_prime,
            weight_epochs=args.weight_epochs,
        )
        dt = time.time() - t0
        diag = analyze_history(history)
        print(f"  done in {dt:.0f}s | monotone={diag['monotone_lemma1']} "
              f"| first_violation_round={diag['first_monotonicity_violation_round']} "
              f"| blowup_ratio={diag['composite_loss_blowup_ratio']} "
              f"| first_simplex_violation_round={diag['first_simplex_violation_round']} "
              f"| final all={diag['final_all_acc']:.3f} last-n={diag['final_last_n_acc']:.3f}")

        key = f"g{gamma}"
        all_results[key] = {
            "history": _history_to_dict(history),
            "diagnosis": diag,
            "config": {
                "eta_prime": args.eta_prime,
                "weight_epochs": args.weight_epochs,
                "rounds": args.rounds,
                "eval_every": args.eval_every,
                "seed": args.seed,
                "n_rare": args.n_rare,
                "project": args.project,
            },
        }

    out_path = os.path.join(
        args.out_dir,
        f"{args.dataset.replace('-', '_')}_{variant}{tag}_r{args.rounds}.json",
    )
    # Merge with any existing file at this path instead of clobbering it, so
    # that running one gamma per process (e.g. for wall-clock parallelism)
    # accumulates into a single combined file rather than each process
    # overwriting the others' results.
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved diagnostic results to {out_path}")


if __name__ == "__main__":
    main()
