"""Smoke tests for the plotting module (gap 3.1 fix).

Imports matplotlib only if available; all tests are skipped gracefully when it
is not installed. The tests don't visually inspect figures — they verify that
the public API runs without error and returns the expected figure type.
"""

from __future__ import annotations

import pytest

import numpy as np

pytest.importorskip("matplotlib", reason="matplotlib not installed")

from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.experiment import run_gamma_sweep, run_n_sweep
from fedgraa.partition import partition_non_iid
from fedgraa.plot import (
    plot_composite_loss,
    plot_dual_metric,
    plot_gamma_sweep,
    plot_n_sweep,
    plot_weight_evolution,
)
from fedgraa.trainer import FederatedTrainer


@pytest.fixture(scope="module")
def tiny_gamma_results():
    spec = DatasetSpec(
        name="PLOT_TOY",
        n_features=5,
        labels=["Benign", "A", "B", "Rare"],
        used_counts=[200, 80, 60, 8],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, spec = make_synthetic_dataset(spec, seed=0, class_separation=3.0)
    split = partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.1, seed=0)
    results = {}
    for method in ("fedavg", "static"):
        cfg = FedConfig(rounds=6, n_rare=1, seed=0)
        hist = FederatedTrainer(split, cfg, method=method).train(eval_every=3)
        key = method
        from fedgraa.experiment import ExperimentResult
        results[key] = ExperimentResult(
            dataset="PLOT_TOY", method=method, gamma=None, n_rare=1, history=hist
        )
    for g in (0.0, 0.5, 1.0):
        cfg = FedConfig(rounds=6, gamma=g, n_rare=1, seed=0)
        hist = FederatedTrainer(split, cfg, method="fedgraa").train(eval_every=3)
        key = f"fedgraa_g{g}"
        from fedgraa.experiment import ExperimentResult
        results[key] = ExperimentResult(
            dataset="PLOT_TOY", method="fedgraa", gamma=g, n_rare=1, history=hist
        )
    return results, split


def test_plot_gamma_sweep_last_n(tiny_gamma_results):
    import matplotlib
    matplotlib.use("Agg")
    results, _ = tiny_gamma_results
    fig = plot_gamma_sweep(results, metric="last_n")
    assert fig is not None


def test_plot_gamma_sweep_all(tiny_gamma_results):
    import matplotlib
    matplotlib.use("Agg")
    results, _ = tiny_gamma_results
    fig = plot_gamma_sweep(results, metric="all")
    assert fig is not None


def test_plot_dual_metric(tiny_gamma_results):
    import matplotlib
    matplotlib.use("Agg")
    results, _ = tiny_gamma_results
    fig = plot_dual_metric(results, title="Test dual metric")
    assert fig is not None


def test_plot_weight_evolution(tiny_gamma_results):
    import matplotlib
    matplotlib.use("Agg")
    results, _ = tiny_gamma_results
    hist = results["fedgraa_g1.0"].history
    fig = plot_weight_evolution(hist, top_k=2)
    assert fig is not None


def test_plot_composite_loss(tiny_gamma_results):
    import matplotlib
    matplotlib.use("Agg")
    results, _ = tiny_gamma_results
    histories = {k: v.history for k, v in results.items() if "fedgraa" in k}
    fig = plot_composite_loss(histories)
    assert fig is not None


def test_plot_without_matplotlib(monkeypatch):
    """plot functions must raise ImportError when matplotlib is missing."""
    import fedgraa.plot as plot_mod
    monkeypatch.setattr(plot_mod, "_MPL_AVAILABLE", False)
    with pytest.raises(ImportError, match="matplotlib"):
        plot_mod.plot_gamma_sweep({})
