"""Experiment orchestration: reproduce Figures 1-3.

* :func:`run_experiment` runs a single (dataset, method, gamma, n) configuration.
* :func:`run_gamma_sweep` reproduces Figure 1: FedAvg vs Static vs FedGRAA across
  ``gamma in {0, 0.5, 1}`` at ``n = 3``.
* :func:`run_n_sweep` reproduces Figures 2-3: FedGRAA robustness as the number of
  protected rare clients grows (``n in {3, 5, 7}``).
* :func:`tabulate_history` produces the full per-round accuracy table (gap 3.2).
* :func:`save_results` / :func:`load_results` persist results to JSON (gap 3.4).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

from .baselines import FedOptServer, FedProxTrainer, FedSNDTrainer
from .config import DatasetSpec, FedConfig
from .data import load_dataset
from .partition import partition_non_iid
from .trainer import FederatedTrainer, TrainHistory


@dataclass
class ExperimentResult:
    dataset: str
    method: str
    gamma: Optional[float]
    n_rare: int
    history: TrainHistory

    @property
    def final_all_acc(self) -> float:
        return self.history.final_all_acc

    @property
    def final_last_n_acc(self) -> float:
        return self.history.final_last_n_acc


def run_experiment(
    dataset: str,
    method: str = "fedgraa",
    gamma: float = 1.0,
    n_rare: int = 3,
    rounds: int = 200,
    data_dir: Optional[str] = None,
    eval_every: int = 20,
    seed: int = 0,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    cfg: Optional[FedConfig] = None,
) -> ExperimentResult:
    """Run one configuration end-to-end and return its history.

    Supports the three main methods ("fedavg", "static", "fedgraa") and the
    additional baselines from :mod:`fedgraa.baselines` ("fedprox", "fedopt",
    "fedsnd"), addressing gap m-3.
    """
    dataset_data_dir = (
        os.path.join(data_dir, dataset) if data_dir and os.path.isdir(os.path.join(data_dir, dataset))
        else data_dir
    )
    X, y, spec = load_dataset(
        dataset, data_dir=dataset_data_dir, seed=seed, cache_dir=cache_dir
    )
    if cfg is None:
        cfg = FedConfig(
            rounds=rounds,
            gamma=gamma,
            n_rare=n_rare,
            seed=seed,
        )
    else:
        # Sweep callers pass a base config; the explicit sweep arguments must
        # override its gamma / n_rare / rounds / seed.
        cfg = replace(
            cfg,
            rounds=rounds,
            gamma=gamma,
            n_rare=n_rare,
            seed=seed,
        )
    split = partition_non_iid(
        X, y, spec, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction, seed=cfg.seed
    )

    method = method.lower()
    if method == "fedprox":
        trainer: object = FedProxTrainer(split, cfg)
    elif method == "fedopt":
        trainer = FedOptServer(split, cfg)
    elif method == "fedsnd":
        trainer = FedSNDTrainer(split, cfg)
    else:
        trainer = FederatedTrainer(split, cfg, method=method)

    history = trainer.train(eval_every=eval_every, verbose=verbose)
    return ExperimentResult(
        dataset=spec.name,
        method=method,
        gamma=gamma if method == "fedgraa" else None,
        n_rare=n_rare,
        history=history,
    )


def run_gamma_sweep(
    dataset: str,
    gammas: Sequence[float] = (0.0, 0.5, 1.0),
    n_rare: int = 3,
    rounds: int = 200,
    data_dir: Optional[str] = None,
    eval_every: int = 20,
    seed: int = 0,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    cfg: Optional[FedConfig] = None,
) -> Dict[str, ExperimentResult]:
    """Reproduce Figure 1: the three methods for each gamma."""
    results: Dict[str, ExperimentResult] = {}

    if cfg is None:
        cfg = FedConfig(rounds=rounds, gamma=1.0, n_rare=n_rare, seed=seed)

    results["fedavg"] = run_experiment(
        dataset, "fedavg", gamma=0.0, n_rare=n_rare, rounds=rounds,
        data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
        cache_dir=cache_dir, cfg=cfg,
    )
    results["static"] = run_experiment(
        dataset, "static", gamma=0.0, n_rare=n_rare, rounds=rounds,
        data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
        cache_dir=cache_dir, cfg=cfg,
    )
    for g in gammas:
        results[f"fedgraa_g{g}"] = run_experiment(
            dataset, "fedgraa", gamma=g, n_rare=n_rare, rounds=rounds,
            data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
            cache_dir=cache_dir, cfg=cfg,
        )
    return results


def run_n_sweep(
    dataset: str,
    ns: Sequence[int] = (3, 5, 7),
    gammas: Sequence[float] = (0.0, 0.5, 1.0),
    rounds: int = 200,
    data_dir: Optional[str] = None,
    eval_every: int = 20,
    seed: int = 0,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
    cfg: Optional[FedConfig] = None,
) -> Dict[int, Dict[str, ExperimentResult]]:
    """Reproduce Figures 2-3: FedGRAA vs baselines as ``n`` grows.

    Gap C-3 fix: the paper's Figures 2 and 3 contain six sub-figures each,
    sweeping ``gamma in {0, 0.5, 1}`` for every ``n``. The previous version
    only ran FedGRAA at a single fixed gamma. Now all three gamma values are
    run for each ``n``.
    """
    if cfg is None:
        cfg = FedConfig(rounds=rounds, gamma=1.0, n_rare=3, seed=seed)

    out: Dict[int, Dict[str, ExperimentResult]] = {}
    for n in ns:
        out[n] = {
            "fedavg": run_experiment(
                dataset, "fedavg", gamma=0.0, n_rare=n, rounds=rounds,
                data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
                cache_dir=cache_dir, cfg=cfg,
            ),
            "static": run_experiment(
                dataset, "static", gamma=0.0, n_rare=n, rounds=rounds,
                data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
                cache_dir=cache_dir, cfg=cfg,
            ),
        }
        for g in gammas:
            out[n][f"fedgraa_g{g}"] = run_experiment(
                dataset, "fedgraa", gamma=g, n_rare=n, rounds=rounds,
                data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
                cache_dir=cache_dir, cfg=cfg,
            )
    return out


def summarize_gamma_sweep(results: Dict[str, ExperimentResult]) -> str:
    """Render a Figure-1-style summary table of final rare-class accuracy."""
    lines = [
        "Method            | All-acc | Last-n acc",
        "------------------+---------+-----------",
    ]
    order = [
        ("fedavg", "FedAvg"),
        ("static", "Static Reparam."),
        ("fedgraa_g1.0", "FedGRAA (g=1)"),
        ("fedgraa_g0.5", "FedGRAA (g=0.5)"),
        ("fedgraa_g0.0", "FedGRAA (g=0)"),
    ]
    for key, label in order:
        if key in results:
            r = results[key]
            lines.append(
                f"{label:<17} | {r.final_all_acc*100:6.2f}  | "
                f"{r.final_last_n_acc*100:6.2f}"
            )
    return "\n".join(lines)


def tabulate_history(history: "TrainHistory") -> str:  # noqa: F821 (forward ref)
    """Full per-round accuracy table matching the paper’s Figure 1 layout.

    Gap 3.2 fix: ``summarize_gamma_sweep`` only shows the final round; this
    function reproduces every evaluation checkpoint (rounds 0, 20, 40, …, 200
    at the default ``eval_every=20``) so the numbers can be compared directly
    against the paper’s tables.
    """
    lines = [
        f"{'Round':>5s} | {'All-acc':>7s} | {'Last-n acc':>10s} | {'Composite L\'':>12s}",
        "-" * 47,
    ]
    for rnd, all_a, last_a, comp in zip(
        history.rounds,
        history.all_acc,
        history.last_n_acc,
        history.composite_loss,
    ):
        comp_s = f"{comp:12.6f}" if np.isfinite(comp) else f"{'N/A':>12s}"
        lines.append(
            f"{rnd:5d} | {all_a:7.5f} | {last_a:10.8f} | {comp_s}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Result persistence  (gap 3.4 fix)
# --------------------------------------------------------------------------- #

def save_results(results: Dict[str, ExperimentResult], path: str) -> None:
    """Persist a results dict (keyed by method string) to a JSON file.

    The file can be reloaded with :func:`load_results` without re-running the
    full 200-round experiment.
    """
    out: Dict = {}
    for key, r in results.items():
        h = r.history
        out[key] = {
            "dataset": r.dataset,
            "method": r.method,
            "gamma": r.gamma,
            "n_rare": r.n_rare,
            "rounds": h.rounds,
            "all_acc": h.all_acc,
            "last_n_acc": h.last_n_acc,
            "composite_loss": [
                v if np.isfinite(v) else None for v in h.composite_loss
            ],
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {path}", flush=True)


def run_seed_sweep(
    dataset: str,
    method: str = "fedgraa",
    gamma: float = 1.0,
    n_rare: int = 3,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    rounds: int = 200,
    data_dir: Optional[str] = None,
    eval_every: int = 20,
    verbose: bool = False,
    cache_dir: Optional[str] = None,
) -> Dict[int, ExperimentResult]:
    """Run the same configuration across multiple random seeds (gap B-3).

    The paper does not report its random seed, so sweeping seeds lets us
    estimate the variance of the reported numbers and identify seeds that
    reproduce the paper's headline results.
    """
    return {
        seed: run_experiment(
            dataset, method=method, gamma=gamma, n_rare=n_rare, rounds=rounds,
            data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
            cache_dir=cache_dir,
        )
        for seed in seeds
    }


def load_results(path: str) -> Dict[str, ExperimentResult]:
    """Reload results previously saved with :func:`save_results`."""
    from .trainer import TrainHistory

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    results: Dict[str, ExperimentResult] = {}
    for key, d in raw.items():
        h = TrainHistory(
            method=d["method"],
            gamma=d.get("gamma"),
            n_rare=d["n_rare"],
        )
        h.rounds = d["rounds"]
        h.all_acc = d["all_acc"]
        h.last_n_acc = d["last_n_acc"]
        raw_comp = d.get("composite_loss", [])
        h.composite_loss = [
            float(v) if v is not None else float("nan") for v in raw_comp
        ] or [float("nan")] * len(h.rounds)
        results[key] = ExperimentResult(
            dataset=d["dataset"],
            method=d["method"],
            gamma=d.get("gamma"),
            n_rare=d["n_rare"],
            history=h,
        )
    return results
