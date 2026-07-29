"""Tests for the FedPGO baseline (fedgraa/baselines_t5_fedpgo.py).

Conventions follow ``tests/test_baselines_t5_adaptive.py``: a tiny synthetic
``DatasetSpec`` built with ``make_synthetic_dataset`` + ``partition_non_iid``,
a handful of rounds, no-NaN / valid-simplex checks, plus a property test for
each of the mechanism's named pieces (AGO's decomposition, PGF's scoring,
the parameter-split personalization).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.baselines_t5_fedpgo import (
    FedPGOTrainer,
    fedpgo_decompose,
    fedpgo_scores,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import partition_non_iid


def _tiny_split(seed: int = 0):
    spec = DatasetSpec(
        name="T5_FEDPGO_TEST",
        n_features=6,
        labels=["Benign", "A", "B", "Rare"],
        used_counts=[300, 120, 90, 24],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _cfg(rounds: int = 4) -> FedConfig:
    return FedConfig(rounds=rounds, gamma=0.5, n_rare=1, seed=0)


# --------------------------------------------------------------------------- #
# End-to-end: no NaN collapse, valid simplex every recorded round
# --------------------------------------------------------------------------- #
def test_fedpgo_no_nan_collapse():
    split = _tiny_split(seed=0)
    trainer = FedPGOTrainer(split, _cfg(rounds=4))
    trainer.train(eval_every=2)
    params = trainer.final_params
    assert params is not None
    for p in params:
        assert np.all(np.isfinite(p)), (
            f"final params contain non-finite values "
            f"(shape={p.shape}, n_bad={int(np.sum(~np.isfinite(p)))})"
        )


def test_fedpgo_weights_are_valid_simplex():
    split = _tiny_split(seed=1)
    trainer = FedPGOTrainer(split, _cfg(rounds=4))
    hist = trainer.train(eval_every=2)
    assert len(hist.weights) >= 2
    K = split.n_clients
    for rnd, w in zip(hist.rounds, hist.weights):
        w = np.asarray(w, dtype=np.float64)
        assert w.shape == (K,), f"round {rnd} weight shape {w.shape} != ({K},)"
        assert np.all(np.isfinite(w)), f"round {rnd} weights non-finite"
        assert np.all(w >= -1e-9), f"round {rnd} has a negative weight: {w}"
        assert abs(w.sum() - 1.0) < 1e-5, f"round {rnd} weights sum to {w.sum()}"


def test_fedpgo_runs_end_to_end():
    split = _tiny_split()
    hist = FedPGOTrainer(split, _cfg(rounds=3)).train(eval_every=3)
    assert len(hist.all_acc) >= 1
    assert 0.0 <= hist.final_all_acc <= 1.0
    assert 0.0 <= hist.final_last_n_acc <= 1.0


def test_fedpgo_personalized_params_differ_across_clients():
    """The parameter-split's whole point: personalized layers are never
    aggregated, so distinct clients must end up with distinct personalized
    layers after training on their own (non-identical, non-IID) shards."""
    split = _tiny_split(seed=2)
    trainer = FedPGOTrainer(split, _cfg(rounds=4), personalized_layers=1)
    trainer.train(eval_every=4)
    personalized_last_layers = [p[-1] for p in trainer.personalized_params]
    assert not all(
        np.allclose(personalized_last_layers[0], p) for p in personalized_last_layers[1:]
    ), "every client's personalized layer is identical -- parameter-split had no effect"


def test_fedpgo_rejects_personalized_layers_out_of_range():
    split = _tiny_split()
    with pytest.raises(ValueError):
        FedPGOTrainer(split, _cfg(), personalized_layers=0)
    with pytest.raises(ValueError):
        # 2 layers total for this MLP config; requesting all of them shared
        # away leaves nothing to aggregate.
        n_layers = len(split.spec.model_config().layer_sizes) - 1
        FedPGOTrainer(split, _cfg(), personalized_layers=n_layers)


# --------------------------------------------------------------------------- #
# AGO: parallel/orthogonal decomposition
# --------------------------------------------------------------------------- #
def test_fedpgo_decompose_parallel_component_matches_projection_formula():
    tau_g = [np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)]
    tau_k = [np.array([[2.0, 3.0], [0.0, 0.0]], dtype=np.float32)]
    parallel, orthogonal, effective = fedpgo_decompose(tau_k, tau_g, orthogonal_weight=1.0)
    # coeff = (2*1) / (1*1) = 2.0 -> parallel = 2 * tau_g
    np.testing.assert_allclose(parallel[0], 2.0 * tau_g[0])
    np.testing.assert_allclose(orthogonal[0], tau_k[0] - parallel[0])
    # orthogonal_weight=1.0 -> effective == tau_k exactly (full reconstruction)
    np.testing.assert_allclose(effective[0], tau_k[0], atol=1e-6)


def test_fedpgo_decompose_zero_orthogonal_weight_drops_disagreement():
    """orthogonal_weight=0 must leave only the parallel (consensus-aligned)
    component -- the "damp disagreement" reading of AGO's reweighting."""
    tau_g = [np.array([1.0, 0.0], dtype=np.float32)]
    tau_k = [np.array([1.0, 5.0], dtype=np.float32)]  # mostly orthogonal
    _, _, effective = fedpgo_decompose(tau_k, tau_g, orthogonal_weight=0.0)
    np.testing.assert_allclose(effective[0], np.array([1.0, 0.0]), atol=1e-6)


def test_fedpgo_decompose_handles_zero_global_direction():
    """A degenerate round-0 case (every client's update is exactly zero, so
    tau_g is zero) must not divide by zero."""
    tau_g = [np.zeros((2, 2), dtype=np.float32)]
    tau_k = [np.ones((2, 2), dtype=np.float32)]
    parallel, orthogonal, effective = fedpgo_decompose(tau_k, tau_g, orthogonal_weight=0.5)
    assert np.all(np.isfinite(parallel[0]))
    assert np.all(np.isfinite(orthogonal[0]))
    assert np.all(np.isfinite(effective[0]))


# --------------------------------------------------------------------------- #
# PGF: directionality + orthogonality scoring
# --------------------------------------------------------------------------- #
def test_fedpgo_scores_favor_aligned_over_anti_aligned_client():
    """A client whose update points opposite the consensus direction must
    score at or below a perfectly-aligned one (directionality term is
    ReLU-clipped, so an anti-aligned client's directionality floors at 0)."""
    tau_g = [np.array([1.0, 0.0], dtype=np.float32)]
    aligned = [np.array([1.0, 0.0], dtype=np.float32)]
    anti_aligned = [np.array([-1.0, 0.0], dtype=np.float32)]
    scores = fedpgo_scores(
        [aligned, anti_aligned], tau_g, orthogonality_weight_score=0.0,
    )
    assert scores[0] > scores[1]


def test_fedpgo_scores_reward_orthogonality_when_weighted():
    """With orthogonality_weight_score > 0, a client whose update is a mix of
    aligned + orthogonal components scores higher than one that is purely
    aligned (same directionality, more orthogonality)."""
    tau_g = [np.array([1.0, 0.0], dtype=np.float32)]
    purely_aligned = [np.array([1.0, 0.0], dtype=np.float32)]
    mixed = [np.array([1.0, 1.0], dtype=np.float32)]
    scores = fedpgo_scores(
        [purely_aligned, mixed], tau_g, orthogonality_weight_score=2.0,
    )
    assert scores[1] > scores[0]


def test_fedpgo_scores_are_valid_simplex():
    tau_g = [np.array([1.0, 2.0], dtype=np.float32)]
    tau_list = [
        [np.array([1.0, 2.0], dtype=np.float32)],
        [np.array([-1.0, 0.5], dtype=np.float32)],
        [np.array([0.3, -0.2], dtype=np.float32)],
    ]
    scores = fedpgo_scores(tau_list, tau_g, orthogonality_weight_score=1.0)
    assert scores.shape == (3,)
    assert np.all(scores >= 0.0)
    assert abs(scores.sum() - 1.0) < 1e-5


def test_fedpgo_scores_degenerate_all_zero_falls_back_to_uniform():
    tau_g = [np.zeros(2, dtype=np.float32)]
    tau_list = [[np.zeros(2, dtype=np.float32)] for _ in range(4)]
    scores = fedpgo_scores(tau_list, tau_g, orthogonality_weight_score=1.0)
    np.testing.assert_allclose(scores, np.full(4, 0.25))
