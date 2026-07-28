"""Tests for strict-reproduction defaults and remaining ambiguity flags.

These tests verify the behaviours that are required to reproduce the paper as
written.  Features that were configurable work-arounds for paper ambiguities
(and have since been removed) are no longer tested here; they are documented in
``docs/strict_reproduction_report.md`` instead.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.aggregation import FedGRAA, static_reparam_weights
from fedgraa.baselines import (
    FedOptServer,
    FedProxTrainer,
    StaticDistributionAwareTrainer,
)
from fedgraa.config import UNSW_NB15, DatasetSpec, FedConfig
from fedgraa.data import _cap_frequent_classes, make_synthetic_dataset
from fedgraa.model import init_params
from fedgraa.partition import partition_non_iid
from fedgraa.trainer import FederatedTrainer


# --------------------------------------------------------------------------- #
# Config / data gaps
# --------------------------------------------------------------------------- #

def test_unsw_rare_order_uses_genuinely_rare_classes():
    """The first five UNSW rare classes are the genuinely rare ones.

    n=7 is a paper ambiguity because UNSW only has four genuinely rare attack
    classes (the rest are capped at 10,000).  The strict implementation leaves
    the rare_label_order at five entries; requesting n>5 simply returns those
    five.  This is recorded as an ambiguity in the strict-reproduction report.
    """
    rare5 = UNSW_NB15.rare_client_ids(5)
    labels = [UNSW_NB15.labels[i] for i in rare5]
    assert "Fuzzers" not in labels
    assert "DoS" not in labels
    assert "Reconnaissance" not in labels
    assert "Generic" not in labels
    assert labels == ["Worms", "Shellcode", "Backdoor", "Analysis", "Exploits"]


def test_unsw_n7_warns_instead_of_silent_truncation():
    """2026-07-10 audit fix (F9): requesting n=7 on UNSW (which only defines
    5 genuinely rare classes) must raise a RuntimeWarning, not silently
    return 5 classes with no signal."""
    with pytest.warns(RuntimeWarning, match="only defines"):
        rare7 = UNSW_NB15.rare_client_ids(7)
    assert len(rare7) == 5


def test_cap_frequent_classes_skips_benign():
    """Strict reproduction: frequent-attack cap must not touch benign/normal."""
    spec = DatasetSpec(
        name="CAP_TEST",
        n_features=4,
        labels=["Benign", "Attack"],
        used_counts=[50_000, 50_000],
        benign_id=0,
        frequent_cap=10_000,
    )
    X = np.random.randn(100_000, 4)
    y = np.array([0] * 50_000 + [1] * 50_000)
    Xc, yc = _cap_frequent_classes(X, y, spec, seed=0)
    assert (yc == 0).sum() == 50_000
    assert (yc == 1).sum() == 10_000


# --------------------------------------------------------------------------- #
# Partition gaps
# --------------------------------------------------------------------------- #

def test_partition_does_not_rescale_features():
    """2026-07-10 audit fix (F8): partition_non_iid must not re-derive
    mean/std from the post-split validation pool and rescale a second time.
    Feature scaling happens exactly once, upstream, in data.py; every row
    handed to a client must be bit-identical to a row of the input X.
    """
    spec = DatasetSpec(
        name="NOSCALE_TEST",
        n_features=4,
        labels=["Benign", "A", "B"],
        used_counts=[40, 20, 20],
        benign_id=0,
    )
    X, y, _ = make_synthetic_dataset(spec, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=0)
    for client in split.clients:
        for row in np.concatenate([client.X_train, client.X_val]):
            assert np.any(np.all(np.isclose(X, row), axis=1)), (
                "client feature row does not match any row of the input X "
                "-- partition_non_iid appears to be transforming features"
            )


def test_benign_pool_shortage_warns():
    """Gap M-3: warn when benign demand exceeds benign pool."""
    spec = DatasetSpec(
        name="POOL_TEST",
        n_features=4,
        labels=["Benign", "A", "B"],
        used_counts=[15, 10, 10],  # demand=20, pool=15
        benign_id=0,
    )
    X, y, _ = make_synthetic_dataset(spec, seed=0)
    with pytest.warns(RuntimeWarning, match="Benign pool"):
        partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=0)


# --------------------------------------------------------------------------- #
# Aggregation gaps
# --------------------------------------------------------------------------- #

def _make_fg_setup(rng):
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    from fedgraa.partition import ValidationSplit
    val = ValidationSplit(
        X_h=rng.normal(size=(15, 6)).astype(np.float32),
        y_h=rng.integers(0, 4, size=15),
        X_m=rng.normal(size=(12, 6)).astype(np.float32),
        y_m=rng.integers(0, 4, size=12),
        X_n=rng.normal(size=(9, 6)).astype(np.float32),
        y_n=rng.integers(0, 4, size=9),
        rare_class_ids=[0],
    )
    return client_params, val


def test_fedgraa_update_runs_without_simplex(rng):
    """Strict reproduction: no simplex projection; weights are updated by Adam."""
    layer_sizes = (6, 7, 5, 4)
    client_params, val = _make_fg_setup(rng)
    fg = FedGRAA(
        gamma=0.5, eta_prime=0.1, weight_epochs=3, layer_sizes=layer_sizes
    )
    w0 = static_reparam_weights([100, 50, 10])
    w1 = fg.update(client_params, w0.copy(), val)
    assert w1.shape == w0.shape
    # Weights move because Adam minimizes the composite loss.
    assert not np.allclose(w0, w1)


# --------------------------------------------------------------------------- #
# Trainer gaps
# --------------------------------------------------------------------------- #

def _tiny_split(seed=0):
    spec = DatasetSpec(
        name="ROUND0_TEST",
        n_features=6,
        labels=["Benign", "A", "Rare"],
        used_counts=[200, 100, 20],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    split = partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)
    return split


def test_round_zero_zero_flag():
    """Gap C-4: round_zero_zero forces round-0 accuracies to 0."""
    split = _tiny_split()
    cfg_zero = FedConfig(rounds=5, gamma=1.0, n_rare=1, seed=0, round_zero_zero=True)
    hist_zero = FederatedTrainer(split, cfg_zero, method="fedgraa").train(eval_every=5)
    assert hist_zero.all_acc[0] == 0.0
    assert hist_zero.last_n_acc[0] == 0.0

    cfg_real = FedConfig(rounds=5, gamma=1.0, n_rare=1, seed=0, round_zero_zero=False)
    hist_real = FederatedTrainer(split, cfg_real, method="fedgraa").train(eval_every=5)
    assert hist_real.all_acc[0] > 0.0


# --------------------------------------------------------------------------- #
# Baselines (gap m-3)
# --------------------------------------------------------------------------- #

def test_all_methods_no_nan_collapse():
    """All methods must produce finite global params at every checkpoint.
    Regression test: ensures the np.empty→np.zeros fix + NaN-safe loss
    prevents the NaN collapse that affected all methods."""
    split = _tiny_split(seed=0)
    cfg = FedConfig(rounds=5, gamma=0.5, n_rare=1, seed=0, round_zero_zero=False)

    methods = {
        "fedavg": FederatedTrainer(split, cfg, method="fedavg"),
        "static": FederatedTrainer(split, cfg, method="static"),
        "fedgraa": FederatedTrainer(split, cfg, method="fedgraa"),
        "fedprox": FedProxTrainer(split, cfg, mu=0.01),
        "fedopt": FedOptServer(split, cfg, server_lr=0.01),
        "static_dist_aware": StaticDistributionAwareTrainer(split, cfg),
    }

    for name, trainer in methods.items():
        trainer.train(eval_every=5)
        params = trainer.final_params
        assert params is not None, f"{name}: final_params is None"
        for p in params:
            assert np.all(np.isfinite(p)), (
                f"{name}: final params contain NaN — padding bug still present\n"
                f"  param shape={p.shape}, num_nan={np.sum(~np.isfinite(p))}"
            )


def test_baseline_trainers_run():
    """Gap m-3: FedProx, FedOpt, and the static distribution-aware baseline
    trainers run end-to-end."""
    split = _tiny_split()
    cfg = FedConfig(rounds=5, gamma=1.0, n_rare=1, seed=0)
    for Trainer in (FedProxTrainer, FedOptServer, StaticDistributionAwareTrainer):
        hist = Trainer(split, cfg).train(eval_every=5)
        assert len(hist.all_acc) >= 1
        assert 0.0 <= hist.final_all_acc <= 1.0
