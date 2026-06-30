"""End-to-end smoke tests for the federated trainer.

Uses a small synthetic dataset so the full Algorithm-1 loop runs quickly while
still exhibiting the paper's qualitative result: FedGRAA lifts rare-class
accuracy above FedAvg.
"""

import numpy as np
import pytest

from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import partition_non_iid
from fedgraa.trainer import FederatedTrainer


def _small_spec() -> DatasetSpec:
    return DatasetSpec(
        name="TOY",
        n_features=6,
        labels=["Benign", "A", "B", "C", "Rare"],
        used_counts=[300, 120, 90, 60, 10],
        benign_id=0,
        rare_label_order=["Rare", "C", "B"],
    )


def _make_split(n_rare=1, seed=0):
    spec = _small_spec()
    X, y, spec = make_synthetic_dataset(spec, seed=seed, class_separation=3.0)
    split = partition_non_iid(X, y, spec, n_rare=n_rare, val_fraction=0.10, seed=seed)
    return split


def test_trainer_runs_and_metrics_in_range():
    split = _make_split()
    cfg = FedConfig(rounds=20, gamma=1.0, n_rare=1, seed=0)
    trainer = FederatedTrainer(split, cfg, method="fedgraa")
    hist = trainer.train(eval_every=10)
    assert len(hist.all_acc) >= 1
    assert 0.0 <= hist.final_all_acc <= 1.0
    assert 0.0 <= hist.final_last_n_acc <= 1.0


def test_fedgraa_beats_fedavg_on_rare_class():
    split = _make_split(n_rare=1, seed=0)

    cfg_avg = FedConfig(rounds=40, gamma=0.0, n_rare=1, seed=0)
    avg = FederatedTrainer(split, cfg_avg, method="fedavg").train(eval_every=20)

    cfg_graa = FedConfig(rounds=40, gamma=1.0, n_rare=1, seed=0)
    graa = FederatedTrainer(split, cfg_graa, method="fedgraa").train(eval_every=20)

    # The whole point of the paper: FedGRAA detects the rare class better.
    assert graa.final_last_n_acc >= avg.final_last_n_acc


def test_methods_all_constructible():
    split = _make_split()
    for method in ("fedavg", "static", "fedgraa"):
        cfg = FedConfig(rounds=4, gamma=0.5, n_rare=1, seed=0)
        FederatedTrainer(split, cfg, method=method).train(eval_every=2)


def test_unknown_method_raises():
    split = _make_split()
    cfg = FedConfig(rounds=2, seed=0)
    with pytest.raises(ValueError):
        FederatedTrainer(split, cfg, method="nope")


# --------------------------------------------------------------------------- #
# Gap 2.5: round 0 recorded
# --------------------------------------------------------------------------- #

def test_round_0_recorded_in_history():
    """Gap 2.5: history must start with round=0 matching the paper's Figure 1."""
    split = _make_split()
    cfg = FedConfig(rounds=20, gamma=1.0, n_rare=1, seed=0)
    hist = FederatedTrainer(split, cfg, method="fedgraa").train(eval_every=10)
    assert hist.rounds[0] == 0, f"Expected history to start at round 0, got {hist.rounds[0]}"
    # rounds 0, 10, 20
    assert len(hist.rounds) == 3


# --------------------------------------------------------------------------- #
# Gap 3.7: composite loss tracked for FedGRAA
# --------------------------------------------------------------------------- #

def test_composite_loss_tracked_for_fedgraa():
    """Gap 3.7: composite_loss list must be populated for FedGRAA runs."""
    split = _make_split()
    cfg = FedConfig(rounds=20, gamma=1.0, n_rare=1, seed=0)
    hist = FederatedTrainer(split, cfg, method="fedgraa").train(eval_every=10)
    assert len(hist.composite_loss) == len(hist.rounds)
    finite_vals = [v for v in hist.composite_loss if np.isfinite(v)]
    assert len(finite_vals) > 0, "No finite composite-loss values recorded"


def test_composite_loss_nan_for_non_fedgraa():
    """Gap 3.7: non-FedGRAA methods produce NaN composite loss."""
    split = _make_split()
    cfg = FedConfig(rounds=10, n_rare=1, seed=0)
    for method in ("fedavg", "static"):
        hist = FederatedTrainer(split, cfg, method=method).train(eval_every=5)
        assert all(not np.isfinite(v) for v in hist.composite_loss), (
            f"Expected NaN composite_loss for {method}"
        )


# --------------------------------------------------------------------------- #
# Gap 2.2: per-client optimizer persistence
# --------------------------------------------------------------------------- #

def test_client_optimizer_objects_created_and_used():
    """Gap 2.2: FederatedTrainer must maintain per-client Adam instances."""
    split = _make_split()
    cfg = FedConfig(rounds=10, gamma=1.0, n_rare=1, seed=0)
    trainer = FederatedTrainer(split, cfg, method="fedgraa")

    assert len(trainer._client_optimizers) == split.n_clients, (
        "Expected one Adam optimizer per client"
    )
    trainer.train(eval_every=5)

    # After training, every optimizer's step counter must be > 0.
    for client_id, opt in trainer._client_optimizers.items():
        assert opt._t > 0, (
            f"Client {client_id} optimizer step count is 0 — was it used?"
        )
