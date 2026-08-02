"""Tests for the T7 defense reproductions (fedgraa/baselines_t7_adversarial.py),
per the master plan's §2.7 test spec items 5, 6, 9, 10
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.attacks import wrap_model_poison_attack
from fedgraa.baselines_t7_adversarial import (
    SecFedNIDSTrainer,
    WeiDetectTrainer,
    secfednids_consensus_filter,
    sos_outlier_probabilities,
    validation_f1_scores,
    weibull_top_t_selection,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.model import MLP, init_params
from fedgraa.partition import partition_non_iid


def _tiny_split(seed: int = 0):
    spec = DatasetSpec(
        name="T7_TEST",
        n_features=6,
        labels=["Benign", "A", "B", "Rare"],
        used_counts=[300, 120, 90, 24],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _cfg(rounds: int = 3) -> FedConfig:
    return FedConfig(rounds=rounds, gamma=0.5, n_rare=1, seed=0, round_zero_zero=False)


# --------------------------------------------------------------------------- #
# sos_outlier_probabilities
# --------------------------------------------------------------------------- #
def test_sos_flags_the_obvious_outlier():
    rng = np.random.default_rng(0)
    cluster = rng.normal(loc=0.0, scale=0.1, size=(9, 5))
    outlier = rng.normal(loc=20.0, scale=0.1, size=(1, 5))
    X = np.vstack([cluster, outlier])
    scores = sos_outlier_probabilities(X, perplexity=3.0)
    assert np.argmax(scores) == 9  # the last row, the injected outlier


def test_sos_handles_single_row_without_crashing():
    scores = sos_outlier_probabilities(np.zeros((1, 3)), perplexity=3.0)
    assert scores.shape == (1,)


def test_sos_scores_are_bounded_in_unit_interval():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 4))
    scores = sos_outlier_probabilities(X, perplexity=3.0)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0 + 1e-9)


# --------------------------------------------------------------------------- #
# secfednids_consensus_filter -- §2.7 item 5
# --------------------------------------------------------------------------- #
def _make_omega_delta(K, shapes=((4, 3), (3,)), poison_idx=None, seed=0):
    """Every client's omega peaks at flat-index 0 of each array (a fixed,
    shared "important coordinate," the realistic case this defense is built
    for -- correlated local data gives correlated importance rankings
    across honest clients) so top-1 selection is deterministic and a
    consensus mask always forms. The poisoned client's *delta* (not its
    omega ranking) is then an obvious outlier at exactly that consensus
    coordinate."""
    rng = np.random.default_rng(seed)
    omegas, deltas = [], []
    for i in range(K):
        omega = []
        for s in shapes:
            arr = np.abs(rng.normal(scale=0.01, size=s)).astype(np.float32)
            flat = arr.ravel()
            flat[0] = 100.0  # dominant, shared peak coordinate
            omega.append(flat.reshape(s))
        delta = [rng.normal(scale=0.1, size=s).astype(np.float32) for s in shapes]
        if poison_idx is not None and i == poison_idx:
            poisoned = []
            for d in delta:
                d2 = d.copy()
                d2.flat[0] = 50.0  # obvious outlier at the consensus coordinate
                poisoned.append(d2)
            delta = poisoned
        omegas.append(omega)
        deltas.append(delta)
    return omegas, deltas


def test_secfednids_consensus_filter_flags_known_poisoned_client():
    omegas, deltas = _make_omega_delta(10, poison_idx=3, seed=0)
    flagged = secfednids_consensus_filter(
        omegas, deltas, assumed_malicious_fraction=0.1, consensus_frac=0.5,
    )
    assert flagged[3] == True  # noqa: E712 -- explicit bool check reads clearer here


def test_secfednids_consensus_filter_zero_false_positives_on_all_benign():
    omegas, deltas = _make_omega_delta(10, poison_idx=None, seed=1)
    flagged = secfednids_consensus_filter(
        omegas, deltas, assumed_malicious_fraction=0.0, consensus_frac=0.5,
    )
    assert not np.any(flagged)


def test_secfednids_consensus_filter_empty_clients_returns_empty():
    assert secfednids_consensus_filter([], []).shape == (0,)


# --------------------------------------------------------------------------- #
# weibull_top_t_selection -- §2.7 item 6
# --------------------------------------------------------------------------- #
def test_weibull_top_t_converges_on_benign_only_fixture():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.85, 0.95, size=10)  # all high-F1, "benign"
    keep = weibull_top_t_selection(scores, assumed_malicious_fraction=0.3)
    assert keep.sum() == 7  # T = round(0.7 * 10)


def test_weibull_top_t_downweights_injected_low_scorer():
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.85, 0.95, size=9)
    scores = np.append(scores, 0.01)  # one obviously-poisoned low F1 score
    keep = weibull_top_t_selection(scores, assumed_malicious_fraction=0.3)
    assert keep[-1] == False  # noqa: E712


def test_weibull_top_t_degenerate_identical_scores_keeps_all():
    scores = np.full(5, 0.9)
    keep = weibull_top_t_selection(scores, assumed_malicious_fraction=0.3)
    assert keep.all()


# --------------------------------------------------------------------------- #
# validation_f1_scores
# --------------------------------------------------------------------------- #
def test_validation_f1_scores_shape_and_bounds():
    split = _tiny_split()
    layer_sizes = split.spec.model_config().layer_sizes
    model = MLP(layer_sizes, seed=0)
    params0 = init_params(layer_sizes, seed=0)
    client_params = [params0, params0]
    scores = validation_f1_scores(model, client_params, split.X_val, split.y_val)
    assert scores.shape == (2,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# --------------------------------------------------------------------------- #
# End-to-end trainer tests -- §2.7 items 9-10
# --------------------------------------------------------------------------- #
TRAINERS = {"secfednids": SecFedNIDSTrainer, "weidetect": WeiDetectTrainer}


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_t7_defense_trainers_no_nan_collapse(name):
    split = _tiny_split(seed=0)
    rng = np.random.default_rng(0)
    attack_fn = wrap_model_poison_attack(rng, n_classes=4, scale=5.0)
    trainer = TRAINERS[name](split, _cfg(rounds=2), attack_fn=attack_fn, attacker_fraction=0.3)
    trainer.train(eval_every=2)
    for p in trainer.final_params:
        assert np.all(np.isfinite(p))


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_t7_defense_trainers_run_end_to_end_with_valid_accuracy_range(name):
    split = _tiny_split(seed=1)
    hist = TRAINERS[name](split, _cfg(rounds=2)).train(eval_every=2)
    assert 0.0 <= hist.final_all_acc <= 1.0
    assert 0.0 <= hist.final_last_n_acc <= 1.0


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_t7_defense_trainers_deterministic_attacker_selection(name):
    split_a = _tiny_split(seed=2)
    split_b = _tiny_split(seed=2)
    cfg = _cfg(rounds=1)
    trainer_a = TRAINERS[name](split_a, cfg, attacker_fraction=0.3)
    trainer_b = TRAINERS[name](split_b, cfg, attacker_fraction=0.3)
    assert trainer_a.attacker_ids == trainer_b.attacker_ids


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_t7_defense_trainers_two_arg_construction(name):
    """Constructible as trainer_cls(split, cfg), matching every other
    BASELINE_ARM_CLASSES-dispatched trainer in this codebase."""
    split = _tiny_split()
    trainer = TRAINERS[name](split, _cfg(rounds=1))
    trainer.train(eval_every=1)
    assert trainer.final_params is not None


def test_t7_defense_trainers_zero_attacker_fraction_flags_nothing_spuriously():
    """With no attackers at all, detection should not systematically nuke
    the honest population -- final accuracy stays in a sane range and the
    attack-success-rate history is never populated (no attackers to track)."""
    split = _tiny_split(seed=3)
    trainer = SecFedNIDSTrainer(split, _cfg(rounds=2), attacker_fraction=0.0)
    trainer.train(eval_every=2)
    assert trainer.attack_success_rate_history == []
