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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
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
    checkpoint_dir: Optional[str] = None,
) -> ExperimentResult:
    """Run one configuration end-to-end and return its history.

    Supports the three main methods ("fedavg", "static", "fedgraa") and the
    additional baselines from :mod:`fedgraa.baselines` ("fedprox", "fedopt",
    "fedsnd"), addressing gap m-3.

    If ``checkpoint_dir`` is given, the trainer saves checkpoints after each
    eval round; if a checkpoint already exists, training resumes from the
    saved round (Phase 14).
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

    history = trainer.train(eval_every=eval_every, verbose=verbose, checkpoint_dir=checkpoint_dir)
    return ExperimentResult(
        dataset=spec.name,
        method=method,
        gamma=gamma if method == "fedgraa" else None,
        n_rare=n_rare,
        history=history,
    )


# --------------------------------------------------------------------------- #
# Phase 9: Sweep-Level Parallelism
# --------------------------------------------------------------------------- #

def _run_single_config(args: tuple) -> tuple:
    """Top-level function for multiprocessing — runs one experiment config.

    Must be at module level (not nested or lambda) for Windows ``spawn``.
    Returns ``(key, result_or_exception)``.
    """
    key, dataset, method, gamma, n_rare, rounds, data_dir, \
        eval_every, seed, verbose, cache_dir, cfg_dict, checkpoint_dir = args
    try:
        cfg = FedConfig(**cfg_dict) if cfg_dict else None
        res = run_experiment(
            dataset, method=method, gamma=gamma, n_rare=n_rare,
            rounds=rounds, data_dir=data_dir, eval_every=eval_every,
            seed=seed, verbose=verbose, cache_dir=cache_dir, cfg=cfg,
            checkpoint_dir=checkpoint_dir,
        )
        return (key, res)
    except Exception as exc:
        warnings.warn(f"Worker for {key} failed: {exc}", RuntimeWarning, stacklevel=1)
        return (key, exc)


def run_gamma_sweep_parallel(
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
    workers: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> Dict[str, ExperimentResult]:
    """Reproduce Figure 1 with parallel workers (Phase 9)."""
    if workers is None:
        workers = min(5, os.cpu_count() // 2)
    if cfg is None:
        cfg = FedConfig(rounds=rounds, gamma=1.0, n_rare=n_rare, seed=seed)

    cfg_dict = {
        "rounds": rounds, "gamma": cfg.gamma, "n_rare": cfg.n_rare,
        "seed": cfg.seed, "eta": cfg.eta, "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size, "eta_prime": cfg.eta_prime,
        "weight_epochs": cfg.weight_epochs, "val_fraction": cfg.val_fraction,
        "round_zero_zero": cfg.round_zero_zero,
    }

    configs = [
        ("fedavg", dataset, "fedavg", 0.0, n_rare, rounds, data_dir,
         eval_every, seed, verbose, cache_dir, cfg_dict,
         str(checkpoint_dir) + "/fedavg" if checkpoint_dir else None),
        ("static", dataset, "static", 0.0, n_rare, rounds, data_dir,
         eval_every, seed, verbose, cache_dir, cfg_dict,
         str(checkpoint_dir) + "/static" if checkpoint_dir else None),
    ]
    for g in gammas:
        key = f"fedgraa_g{g}"
        configs.append(
            (key, dataset, "fedgraa", g, n_rare, rounds, data_dir,
             eval_every, seed, verbose, cache_dir, cfg_dict,
             str(checkpoint_dir) + f"/fedgraa_g{g}" if checkpoint_dir else None),
        )

    results: Dict[str, ExperimentResult] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_single_config, c): c[0] for c in configs}
        for future in as_completed(futures):
            key = futures[future]
            try:
                val = future.result()
                result_key, result_or_exc = val
                if isinstance(result_or_exc, Exception):
                    print(f"  [parallel] {result_key} failed: {result_or_exc}", flush=True)
                else:
                    results[result_key] = result_or_exc
                    print(f"  [parallel] {result_key} done", flush=True)
            except Exception as exc:
                print(f"  [parallel] {key} raised: {exc}", flush=True)

    # Sort results by method name for deterministic ordering
    sorted_keys = sorted(results.keys(), key=lambda k: (k != "fedavg", k != "static", k))
    return {k: results[k] for k in sorted_keys}


def run_n_sweep_parallel(
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
    workers: Optional[int] = None,
    checkpoint_dir: Optional[str] = None,
) -> Dict[int, Dict[str, ExperimentResult]]:
    """Reproduce Figures 2-3 with parallel workers (Phase 9)."""
    if workers is None:
        workers = min(8, os.cpu_count() // 2)
    if cfg is None:
        cfg = FedConfig(rounds=rounds, gamma=1.0, n_rare=3, seed=seed)

    cfg_dict = {
        "rounds": rounds, "gamma": cfg.gamma, "n_rare": cfg.n_rare,
        "seed": cfg.seed, "eta": cfg.eta, "local_epochs": cfg.local_epochs,
        "batch_size": cfg.batch_size, "eta_prime": cfg.eta_prime,
        "weight_epochs": cfg.weight_epochs, "val_fraction": cfg.val_fraction,
        "round_zero_zero": cfg.round_zero_zero,
    }

    configs = []
    for n in ns:
        for g in gammas:
            key = (n, f"fedgraa_g{g}")
            ckpt = (
                str(Path(checkpoint_dir) / f"n{n}" / f"fedgraa_g{g}")
                if checkpoint_dir else None
            )
            configs.append(
                (key, dataset, "fedgraa", g, n, rounds, data_dir,
                 eval_every, seed, verbose, cache_dir, cfg_dict, ckpt),
            )
        for method in ("fedavg", "static"):
            key = (n, method)
            ckpt = (
                str(Path(checkpoint_dir) / f"n{n}" / method)
                if checkpoint_dir else None
            )
            configs.append(
                (key, dataset, method, 0.0, n, rounds, data_dir,
                 eval_every, seed, verbose, cache_dir, cfg_dict, ckpt),
            )

    raw: Dict[int, Dict[str, ExperimentResult]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_single_config, c): c[0] for c in configs}
        for future in as_completed(futures):
            n_key, method_key = futures[future]
            try:
                val = future.result()
                result_key, result_or_exc = val
                if isinstance(result_or_exc, Exception):
                    print(f"  [parallel] n={n_key} {method_key} failed: {result_or_exc}", flush=True)
                else:
                    raw.setdefault(n_key, {})[method_key] = result_or_exc
                    print(f"  [parallel] n={n_key} {method_key} done", flush=True)
            except Exception as exc:
                print(f"  [parallel] n={n_key} {method_key} raised: {exc}", flush=True)

    out: Dict[int, Dict[str, ExperimentResult]] = {}
    for n in sorted(raw.keys()):
        methods = raw[n]
        order = ["fedavg", "static"] + [f"fedgraa_g{g}" for g in gammas]
        out[n] = {k: methods[k] for k in order if k in methods}
    return out


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


def save_n_sweep_results(
    results: Dict[int, Dict[str, ExperimentResult]], path: str
) -> None:
    """Persist the nested ``n``-sweep results (Figures 2-3) to a JSON file.

    Mirrors :func:`save_results` but keys the outer level by ``n`` (as a
    string, since JSON object keys must be strings) and the inner level by
    method string.
    """
    out: Dict = {}
    for n, methods in results.items():
        out[str(n)] = {}
        for key, r in methods.items():
            h = r.history
            out[str(n)][key] = {
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


def load_n_sweep_results(path: str) -> Dict[int, Dict[str, ExperimentResult]]:
    """Reload results previously saved with :func:`save_n_sweep_results`."""
    from .trainer import TrainHistory

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[int, Dict[str, ExperimentResult]] = {}
    for n_str, methods in raw.items():
        n = int(n_str)
        out[n] = {}
        for key, d in methods.items():
            h = TrainHistory(method=d["method"], gamma=d.get("gamma"), n_rare=d["n_rare"])
            h.rounds = d["rounds"]
            h.all_acc = d["all_acc"]
            h.last_n_acc = d["last_n_acc"]
            raw_comp = d.get("composite_loss", [])
            h.composite_loss = [
                float(v) if v is not None else float("nan") for v in raw_comp
            ] or [float("nan")] * len(h.rounds)
            out[n][key] = ExperimentResult(
                dataset=d["dataset"], method=d["method"], gamma=d.get("gamma"),
                n_rare=d["n_rare"], history=h,
            )
    return out


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
