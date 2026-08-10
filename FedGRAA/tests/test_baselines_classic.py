"""Tests for the classic-FL-optimizer baselines
(docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md §4):
FedAvgTrainer (adapter), SCAFFOLDTrainer, FedNovaTrainer.

Conventions follow ``tests/test_baselines_t5_adaptive.py``: a tiny synthetic
``DatasetSpec`` built with ``make_synthetic_dataset`` + ``partition_non_iid``,
end-to-end no-NaN checks, plus hand-computed regression tests for the two
new methods' pure math functions (``scaffold_option2_control_variate``,
``scaffold_server_update``, ``fednova_normalized_update``) -- mirroring this
project's own "regression test against a hand-computed value" bar
(build_v2_significance_table.py's Wilcoxon test being the most recent
precedent).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.baselines_classic import (
    FedAvgTrainer,
    FedNovaTrainer,
    SCAFFOLDTrainer,
    fednova_normalized_update,
    n_local_steps,
    scaffold_option2_control_variate,
    scaffold_server_update,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import partition_non_iid
from fedgraa.trainer import FederatedTrainer


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
def _tiny_split(seed: int = 0):
    spec = DatasetSpec(
        name="TC_TEST",
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


TRAINERS = {
    "fedavg": FedAvgTrainer,
    "scaffold": SCAFFOLDTrainer,
    "fednova": FedNovaTrainer,
}


# --------------------------------------------------------------------------- #
# n_local_steps
# --------------------------------------------------------------------------- #
def test_n_local_steps_zero_for_empty_shard():
    cfg = FedConfig(local_epochs=2, batch_size=64)
    assert n_local_steps(0, cfg) == 0


def test_n_local_steps_matches_ceil_formula():
    cfg = FedConfig(local_epochs=2, batch_size=64)
    # n=100, batch=64 -> ceil(100/64) = 2 batches/epoch * 2 epochs = 4 steps
    assert n_local_steps(100, cfg) == 4
    # n=64 exactly -> 1 batch/epoch
    assert n_local_steps(64, cfg) == 2
    # n=65 -> 2 batches/epoch (one partial)
    assert n_local_steps(65, cfg) == 4


def test_n_local_steps_heterogeneous_across_client_sizes():
    """The property FedNova's normalization needs to be non-degenerate:
    different client sizes genuinely produce different step counts."""
    cfg = FedConfig(local_epochs=1, batch_size=32)
    assert n_local_steps(32, cfg) != n_local_steps(200, cfg)


# --------------------------------------------------------------------------- #
# scaffold_option2_control_variate -- hand-computed regression test
# --------------------------------------------------------------------------- #
def test_scaffold_option2_control_variate_hand_computed():
    c_i_old = [np.array([0.1, 0.2], dtype=np.float32)]
    c_global = [np.array([0.05, 0.05], dtype=np.float32)]
    x = [np.array([1.0, 1.0], dtype=np.float32)]
    y_i = [np.array([0.8, 0.9], dtype=np.float32)]
    K = 4
    eta_l = 0.5

    result = scaffold_option2_control_variate(c_i_old, c_global, x, y_i, K, eta_l)

    # c_i^+ = c_i - c + (1/(K*eta_l))*(x - y_i)
    # (1/(4*0.5)) * ([1,1] - [0.8,0.9]) = 0.5 * [0.2, 0.1] = [0.1, 0.05]
    # c_i^+ = [0.1,0.2] - [0.05,0.05] + [0.1,0.05] = [0.15, 0.20]
    expected = np.array([0.15, 0.20], dtype=np.float32)
    np.testing.assert_allclose(result[0], expected, atol=1e-6)


def test_scaffold_option2_control_variate_guards_zero_steps():
    """K=0 must not raise a ZeroDivisionError (max(K,1) guard)."""
    c_i_old = [np.array([0.0], dtype=np.float32)]
    c_global = [np.array([0.0], dtype=np.float32)]
    x = [np.array([1.0], dtype=np.float32)]
    y_i = [np.array([1.0], dtype=np.float32)]
    result = scaffold_option2_control_variate(c_i_old, c_global, x, y_i, 0, 0.1)
    assert np.isfinite(result[0]).all()


# --------------------------------------------------------------------------- #
# scaffold_server_update -- hand-computed regression test
# --------------------------------------------------------------------------- #
def test_scaffold_server_update_hand_computed():
    global_params = [np.array([1.0, 1.0], dtype=np.float32)]
    y_list = [
        [np.array([2.0, 0.0], dtype=np.float32)],
        [np.array([0.0, 2.0], dtype=np.float32)],
    ]
    result = scaffold_server_update(global_params, y_list, eta_g=1.0)
    # mean(y_i - x) = mean([1,-1],[-1,1]) = [0,0] -> x unchanged
    np.testing.assert_allclose(result[0], [1.0, 1.0], atol=1e-6)


def test_scaffold_server_update_eta_g_scales_the_step():
    global_params = [np.array([0.0], dtype=np.float32)]
    y_list = [[np.array([2.0], dtype=np.float32)]]  # single client, delta=2
    result_full = scaffold_server_update(global_params, y_list, eta_g=1.0)
    result_half = scaffold_server_update(global_params, y_list, eta_g=0.5)
    assert result_full[0][0] == pytest.approx(2.0)
    assert result_half[0][0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# fednova_normalized_update -- hand-computed regression test
# --------------------------------------------------------------------------- #
def test_fednova_normalized_update_hand_computed():
    global_params = [np.array([0.0, 0.0], dtype=np.float32)]
    # Client 0: size 100, delta [2, 0], tau=2 -> d_0 = [1, 0]
    # Client 1: size 300, delta [0, 8], tau=8 -> d_1 = [0, 1]
    client_params = [
        [np.array([2.0, 0.0], dtype=np.float32)],
        [np.array([0.0, 8.0], dtype=np.float32)],
    ]
    client_sizes = [100, 300]
    taus = [2, 8]

    result = fednova_normalized_update(global_params, client_params, client_sizes, taus)

    # p = [0.25, 0.75]; tau_eff = 0.25*2 + 0.75*8 = 0.5 + 6.0 = 6.5
    # sum p_i*d_i = 0.25*[1,0] + 0.75*[0,1] = [0.25, 0.75]
    # update = tau_eff * [0.25, 0.75] = 6.5 * [0.25, 0.75] = [1.625, 4.875]
    expected = np.array([1.625, 4.875], dtype=np.float32)
    np.testing.assert_allclose(result[0], expected, atol=1e-4)


def test_fednova_normalized_update_excludes_zero_tau_clients():
    """A client with tau_i=0 (empty shard) must not contribute (and must not
    raise a division error) -- mirrors client_train_round's own n==0 contract."""
    global_params = [np.array([0.0], dtype=np.float32)]
    client_params = [
        [np.array([5.0], dtype=np.float32)],   # would-be huge contribution
        [np.array([1.0], dtype=np.float32)],
    ]
    client_sizes = [50, 50]
    taus = [0, 4]  # first client did nothing this round

    result = fednova_normalized_update(global_params, client_params, client_sizes, taus)
    assert np.isfinite(result[0]).all()
    # Only client 1 contributes: p=[0.5,0.5], tau_eff = 0.5*0+0.5*4=2.0
    # d_1 = (1-0)/4 = 0.25; sum p_i d_i = 0.5*0.25 = 0.125; update=2.0*0.125=0.25
    np.testing.assert_allclose(result[0], [0.25], atol=1e-4)


def test_fednova_reduces_to_fedavg_when_taus_are_uniform():
    """When every client takes the same number of local steps, tau_eff ==
    tau (constant) and the normalized update collapses to a plain
    size-weighted average of client deltas -- exactly FedAvg's own
    aggregation. This is the sanity check that FedNova's normalization is a
    generalization of FedAvg, not an unrelated formula.

    Known blind spot (noted 2026-08-03, independent Stage C audit, confirmed
    by mutation testing): because the buggy and correct per-client
    normalization formulas algebraically coincide whenever every tau_i is
    equal, this specific test cannot distinguish a correct implementation
    from one with a wrong per-client normalization -- that bug class is
    instead caught by the heterogeneous-tau hand-computed cases above
    (test_fednova_normalized_update_excludes_zero_tau_clients and its
    neighbors), which use genuinely different tau_i values per client."""
    global_params = [np.array([0.0, 0.0], dtype=np.float32)]
    client_params = [
        [np.array([2.0, 4.0], dtype=np.float32)],
        [np.array([6.0, 0.0], dtype=np.float32)],
    ]
    client_sizes = [100, 300]
    taus = [5, 5]  # uniform step count

    result = fednova_normalized_update(global_params, client_params, client_sizes, taus)
    # FedAvg equivalent: p=[0.25,0.75]; weighted mean of deltas directly
    expected = 0.25 * np.array([2.0, 4.0]) + 0.75 * np.array([6.0, 0.0])
    np.testing.assert_allclose(result[0], expected, atol=1e-4)


# --------------------------------------------------------------------------- #
# FedAvgTrainer: thin-adapter equivalence
# --------------------------------------------------------------------------- #
def test_fedavg_trainer_matches_federated_trainer_fedavg_exactly():
    """FedAvgTrainer is a pure adapter -- same seed must produce byte-for-byte
    identical history to calling FederatedTrainer(method='fedavg') directly."""
    split_a = _tiny_split(seed=2)
    split_b = _tiny_split(seed=2)
    cfg = _cfg(rounds=3)

    hist_adapter = FedAvgTrainer(split_a, cfg).train(eval_every=1)
    hist_direct = FederatedTrainer(split_b, cfg, method="fedavg").train(eval_every=1)

    np.testing.assert_allclose(hist_adapter.all_acc, hist_direct.all_acc)
    np.testing.assert_allclose(hist_adapter.last_n_acc, hist_direct.last_n_acc)
    assert hist_adapter.method == "fedavg"


def test_fedavg_trainer_two_arg_construction():
    """Must be constructible as trainer_cls(split, cfg) -- the exact shape
    BASELINE_ARM_CLASSES dispatch in run_v2_frontier_sweep.py uses."""
    split = _tiny_split()
    trainer = FedAvgTrainer(split, _cfg(rounds=1))
    hist = trainer.train(eval_every=1)
    assert trainer.final_params is not None
    assert len(hist.all_acc) >= 1


# --------------------------------------------------------------------------- #
# End-to-end: no NaN collapse (mirrors test_t5_adaptive_no_nan_collapse)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_tc_trainers_no_nan_collapse(name):
    split = _tiny_split(seed=0)
    trainer = TRAINERS[name](split, _cfg(rounds=3))
    trainer.train(eval_every=2)
    params = trainer.final_params
    assert params is not None, f"{name}: final_params is None"
    for p in params:
        assert np.all(np.isfinite(p)), (
            f"{name}: final params contain non-finite values "
            f"(shape={p.shape}, n_bad={int(np.sum(~np.isfinite(p)))})"
        )


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_tc_trainers_run_end_to_end(name):
    split = _tiny_split()
    hist = TRAINERS[name](split, _cfg(rounds=3)).train(eval_every=3)
    assert len(hist.all_acc) >= 1
    assert 0.0 <= hist.final_all_acc <= 1.0
    assert 0.0 <= hist.final_last_n_acc <= 1.0


@pytest.mark.parametrize("name", sorted(TRAINERS))
def test_tc_trainers_two_arg_dispatch_shape(name):
    """Every arm class in TC_ARM_CLASSES must be constructible as
    trainer_cls(split, cfg) with no other required args -- the exact call
    run_one_cell's BASELINE_ARM_CLASSES branch makes."""
    split = _tiny_split()
    trainer = TRAINERS[name](split, _cfg(rounds=1))
    assert trainer is not None


# --------------------------------------------------------------------------- #
# SCAFFOLD: control variates actually move away from zero init
# --------------------------------------------------------------------------- #
def test_scaffold_control_variates_are_nonzero_after_training():
    """A no-op SCAFFOLD implementation (correction always 0) would be
    indistinguishable from FedAvg -- assert the control variates actually
    accumulate nonzero state, i.e. the mechanism is doing something."""
    split = _tiny_split(seed=3)
    trainer = SCAFFOLDTrainer(split, _cfg(rounds=2))
    trainer.train(eval_every=2)
    total_abs = sum(float(np.sum(np.abs(c))) for c in trainer._c_global)
    assert total_abs > 0.0


# --------------------------------------------------------------------------- #
# TC arm registry wiring (run_v2_frontier_sweep.py)
# --------------------------------------------------------------------------- #
def test_tc_arm_classes_registered_and_importable():
    from scripts.run_v2_frontier_sweep import BASELINE_ARM_CLASSES, TC_ARM_CLASSES
    import importlib

    assert set(TC_ARM_CLASSES) == {"fedavg", "fedprox", "fedopt", "scaffold", "fednova"}
    assert set(TC_ARM_CLASSES).issubset(BASELINE_ARM_CLASSES)
    for module_name, class_name in TC_ARM_CLASSES.values():
        cls = getattr(importlib.import_module(module_name), class_name)
        assert cls is not None


def test_tc_tier_enumerates_expected_cells():
    from scripts.run_v2_frontier_sweep import enumerate_cells

    cells = enumerate_cells(["TC"], datasets=["DS_A"], seeds=[0, 1])
    arms = {c.arm for c in cells}
    assert arms == {"fedavg", "fedprox", "fedopt", "scaffold", "fednova"}
    assert len(cells) == 5 * 2  # 5 arms x 2 seeds, 1 dataset
    assert all(c.gamma == 1.0 for c in cells)
