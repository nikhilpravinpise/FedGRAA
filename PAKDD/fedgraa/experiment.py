"""Experiment orchestration: reproduce Figures 1-3.

* :func:`run_experiment` runs a single (dataset, method, gamma, n) configuration.
* :func:`run_gamma_sweep` reproduces Figure 1: FedAvg vs Static vs FedGRAA across
  ``gamma in {0, 0.5, 1}`` at ``n = 3``.
* :func:`run_n_sweep` reproduces Figures 2-3: FedGRAA robustness as the number of
  protected rare clients grows (``n in {3, 5, 7}``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

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
) -> ExperimentResult:
    """Run one configuration end-to-end and return its history."""
    X, y, spec = load_dataset(dataset, data_dir=data_dir, seed=seed)
    cfg = FedConfig(rounds=rounds, gamma=gamma, n_rare=n_rare, seed=seed)
    split = partition_non_iid(
        X, y, spec, n_rare=n_rare, val_fraction=cfg.val_fraction, seed=seed
    )
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
) -> Dict[str, ExperimentResult]:
    """Reproduce Figure 1: the three methods for each gamma."""
    results: Dict[str, ExperimentResult] = {}

    results["fedavg"] = run_experiment(
        dataset, "fedavg", gamma=0.0, n_rare=n_rare, rounds=rounds,
        data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
    )
    results["static"] = run_experiment(
        dataset, "static", gamma=0.0, n_rare=n_rare, rounds=rounds,
        data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
    )
    for g in gammas:
        results[f"fedgraa_g{g}"] = run_experiment(
            dataset, "fedgraa", gamma=g, n_rare=n_rare, rounds=rounds,
            data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
        )
    return results


def run_n_sweep(
    dataset: str,
    ns: Sequence[int] = (3, 5, 7),
    gamma: float = 1.0,
    rounds: int = 200,
    data_dir: Optional[str] = None,
    eval_every: int = 20,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[int, Dict[str, ExperimentResult]]:
    """Reproduce Figures 2-3: FedGRAA vs baselines as ``n`` grows."""
    out: Dict[int, Dict[str, ExperimentResult]] = {}
    for n in ns:
        out[n] = {
            "fedavg": run_experiment(
                dataset, "fedavg", gamma=0.0, n_rare=n, rounds=rounds,
                data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
            ),
            "static": run_experiment(
                dataset, "static", gamma=0.0, n_rare=n, rounds=rounds,
                data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
            ),
            "fedgraa": run_experiment(
                dataset, "fedgraa", gamma=gamma, n_rare=n, rounds=rounds,
                data_dir=data_dir, eval_every=eval_every, seed=seed, verbose=verbose,
            ),
        }
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
