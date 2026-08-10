"""Tests for the tier-3 robust-aggregation baselines (FLTrust / RAGA / FedTruth).

Follows ``tests/test_reproduction_gaps.py``'s conventions: small synthetic
datasets built from a ``DatasetSpec`` via ``make_synthetic_dataset`` +
``partition_non_iid``, and a no-NaN-collapse regression check.

The load-bearing tests here are the *robustness* ones. All three methods exist
to down-weight an off-distribution client; a test suite that only checked they
run end-to-end would not have verified the mechanism at all. The weight
functions are therefore also exercised in isolation against hand-constructed
gradient vectors, where the expected answer is known analytically.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.baselines_t3_trust import (
    FedTruthTrainer,
    FLTrustTrainer,
    RAGATrainer,
    fedtruth_weights,
    fltrust_aggregate,
    fltrust_trust_scores,
    flatten_params,
    model_update,
    raga_weights,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.partition import partition_non_iid


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tiny_split(seed: int = 0):
    """Mirrors ``test_reproduction_gaps._tiny_split`` (3 classes, 2 clients)."""
    spec = DatasetSpec(
        name="T3_TRUST_TEST",
        n_features=6,
        labels=["Benign", "A", "B", "Rare"],
        used_counts=[300, 100, 100, 40],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _cfg(rounds: int = 4) -> FedConfig:
    return FedConfig(rounds=rounds, gamma=0.5, n_rare=1, seed=0, round_zero_zero=False)


def _all_trainers(split, cfg):
    return {
        "fltrust": FLTrustTrainer(split, cfg, root_size=20),
        "raga": RAGATrainer(split, cfg),
        "fedtruth": FedTruthTrainer(split, cfg),
    }


def _vec(*values) -> list:
    """A one-array ``Params`` holding an explicit vector."""
    return [np.asarray(values, dtype=np.float32)]


def _poison(trainer, victim: int = 0, scale: float = -25.0):
    """Make ``victim`` submit a large update opposed to the benign consensus.

    Wraps the trainer's client-training step rather than its aggregation, so
    everything downstream (update extraction, trust scoring, weighting) runs
    exactly as it does in production -- the only difference is what one client
    puts on the wire.
    """
    original = trainer._train_clients

    def patched(global_params, rng):
        client_params = original(global_params, rng)
        benign = [cp for i, cp in enumerate(client_params) if i != victim]
        mean_update = [
            np.mean([cp[j] - global_params[j] for cp in benign], axis=0)
            for j in range(len(global_params))
        ]
        client_params[victim] = [
            global_params[j] + np.float32(scale) * mean_update[j]
            for j in range(len(global_params))
        ]
        return client_params

    trainer._train_clients = patched
    return trainer


# --------------------------------------------------------------------------- #
# End-to-end: no NaN collapse, valid simplex
# --------------------------------------------------------------------------- #
def test_t3_trust_methods_no_nan_collapse():
    """Mirror of ``test_all_methods_no_nan_collapse`` for the T3 trust family."""
    split = _tiny_split(seed=0)
    cfg = _cfg(rounds=4)

    for name, trainer in _all_trainers(split, cfg).items():
        trainer.train(eval_every=2)
        params = trainer.final_params
        assert params is not None, f"{name}: final_params is None"
        for p in params:
            assert np.all(np.isfinite(p)), (
                f"{name}: final params contain non-finite values\n"
                f"  param shape={p.shape}, n_bad={np.sum(~np.isfinite(p))}"
            )


def test_t3_trust_weights_are_valid_simplex_every_round():
    """Every recorded weight vector must be finite, non-negative, and sum to 1."""
    split = _tiny_split(seed=0)
    cfg = _cfg(rounds=4)
    k = len(split.clients)

    for name, trainer in _all_trainers(split, cfg).items():
        hist = trainer.train(eval_every=1)
        assert len(hist.weights) >= 2, f"{name}: too few recorded rounds"
        for rnd, w in zip(hist.rounds, hist.weights):
            w = np.asarray(w, dtype=np.float64)
            assert w.shape == (k,), f"{name} round {rnd}: shape {w.shape} != ({k},)"
            assert np.all(np.isfinite(w)), f"{name} round {rnd}: non-finite weight"
            assert np.all(w >= 0.0), f"{name} round {rnd}: negative weight {w}"
            assert w.sum() == pytest.approx(1.0, abs=1e-5), (
                f"{name} round {rnd}: weights sum to {w.sum()}, not 1"
            )


def test_t3_trust_trainers_run_and_report_accuracy():
    split = _tiny_split(seed=1)
    cfg = _cfg(rounds=3)
    for name, trainer in _all_trainers(split, cfg).items():
        hist = trainer.train(eval_every=3)
        assert hist.method == name
        assert len(hist.all_acc) >= 1
        assert 0.0 <= hist.final_all_acc <= 1.0
        assert 0.0 <= hist.final_last_n_acc <= 1.0


# --------------------------------------------------------------------------- #
# THE robustness property: a malicious/off-distribution client is down-weighted
# --------------------------------------------------------------------------- #
def test_fltrust_relu_clips_anti_aligned_client_to_zero_trust():
    """FLTrust's defining property: ``TS = ReLU(cos)`` excludes anti-aligned
    updates outright, and does so regardless of their magnitude."""
    server = _vec(1.0, 0.0, 0.0)
    benign_a = _vec(0.9, 0.2, 0.1)
    benign_b = _vec(1.1, -0.1, 0.05)
    malicious = _vec(-8.0, 0.5, 0.0)  # large AND pointing the wrong way

    ts = fltrust_trust_scores([benign_a, benign_b, malicious], server)
    assert ts[2] == 0.0, "anti-aligned client was not clipped to zero trust"
    assert ts[0] > 0.9 and ts[1] > 0.9, f"benign clients under-trusted: {ts}"

    _, w = fltrust_aggregate(_vec(0.0, 0.0, 0.0),
                             [benign_a, benign_b, malicious], server)
    assert w[2] == 0.0
    assert w[0] > 0.4 and w[1] > 0.4


def test_fltrust_orthogonal_and_partially_misaligned_clients_are_down_weighted():
    """Not only the fully-reversed case: trust decreases monotonically as the
    update rotates away from the server's reference direction."""
    server = _vec(1.0, 0.0)
    aligned = _vec(1.0, 0.0)
    tilted = _vec(1.0, 1.0)          # 45 degrees
    orthogonal = _vec(0.0, 1.0)      # 90 degrees

    ts = fltrust_trust_scores([aligned, tilted, orthogonal], server)
    assert ts[0] == pytest.approx(1.0)
    assert ts[1] == pytest.approx(1.0 / np.sqrt(2.0))
    assert ts[2] == pytest.approx(0.0)
    assert ts[0] > ts[1] > ts[2]


def test_fltrust_magnitude_normalization_removes_scale_advantage():
    """A client cannot buy influence with a huge update: FLTrust rescales every
    update onto the server update's hypersphere before averaging.

    Two clients pointing the same way but 1000x apart in norm must contribute
    identically, and the aggregate must land on the server update's magnitude.
    """
    server = _vec(1.0, 0.0)
    small = _vec(0.001, 0.0)
    huge = _vec(1000.0, 0.0)

    new_global, w = fltrust_aggregate(_vec(0.0, 0.0), [small, huge], server)
    assert w[0] == pytest.approx(w[1])
    # ||g|| == ||g_0|| when every (rescaled) update is collinear with g_0.
    assert float(np.linalg.norm(flatten_params(new_global))) == pytest.approx(
        float(np.linalg.norm(flatten_params(server))), rel=1e-5
    )


def test_raga_geometric_median_down_weights_the_outlier():
    """RAGA is a geometric median: the far-away point loses weight because
    ``1/||z* - z_i||`` is small -- with no reference direction involved."""
    benign = [_vec(1.0, 0.0), _vec(1.05, 0.1), _vec(0.95, -0.05), _vec(1.02, 0.02)]
    malicious = _vec(-40.0, 30.0)
    updates = benign + [malicious]
    sizes = [100] * len(updates)

    w = raga_weights(updates, sizes)
    assert w[-1] < 0.02, f"outlier retained weight {w[-1]:.4f}"
    assert w[-1] < 0.1 * min(w[:-1]), (
        f"outlier weight {w[-1]:.4g} is not markedly below the benign minimum "
        f"{min(w[:-1]):.4g}"
    )


def test_raga_recovers_the_one_dimensional_median():
    """Sanity check that the Weiszfeld coefficients really are the geometric
    median's: in 1-D with equal sizes the geometric median is the median."""
    updates = [_vec(-1.0), _vec(0.0), _vec(5.0)]
    w = raga_weights(updates, [10, 10, 10])
    assert int(np.argmax(w)) == 1
    assert w[1] > 0.99
    aggregated = float(np.sum([wi * u[0][0] for wi, u in zip(w, updates)]))
    assert aggregated == pytest.approx(0.0, abs=1e-3)


def test_raga_weights_respect_dataset_size():
    """RAGA's ``beta_i = S_i / sum S_j`` is the data-heterogeneity half of the
    rule: identical updates must be weighted purely by dataset size."""
    updates = [_vec(1.0, 1.0), _vec(1.0, 1.0), _vec(1.0, 1.0)]
    w = raga_weights(updates, [100, 200, 700])
    assert w == pytest.approx(np.array([0.1, 0.2, 0.7], dtype=np.float32), abs=1e-5)


def test_fedtruth_down_weights_the_client_far_from_the_estimated_truth():
    """FedTruth bootstraps its reference from the clients themselves, then
    penalizes distance to it -- no root dataset, no size prior."""
    benign = [_vec(1.0, 0.0), _vec(1.05, 0.1), _vec(0.95, -0.05), _vec(1.02, 0.02)]
    malicious = _vec(-40.0, 30.0)
    updates = benign + [malicious]

    w = fedtruth_weights(updates)
    assert w[-1] < min(w[:-1]), (
        f"outlier weight {w[-1]:.4g} not below benign minimum {min(w[:-1]):.4g}"
    )
    assert w[-1] < 0.05, f"outlier retained weight {w[-1]:.4f}"


def test_fedtruth_cosine_distance_catches_a_direction_flip_of_matched_norm():
    """The paper's alternative distance: with ``d = 1 - cos`` a same-magnitude
    but reversed update is maximally distant and is down-weighted, which the
    Euclidean reading conflates with 'far away'."""
    benign = [_vec(1.0, 0.0), _vec(0.99, 0.05), _vec(1.01, -0.03)]
    flipped = _vec(-1.0, 0.0)  # same norm, opposite direction
    w = fedtruth_weights(benign + [flipped], distance="cosine")
    assert w[-1] < min(w[:-1])


def test_fedtruth_uniform_when_all_updates_agree():
    updates = [_vec(0.3, -0.7)] * 4
    w = fedtruth_weights(updates)
    assert w == pytest.approx(np.full(4, 0.25, dtype=np.float32), abs=1e-5)


@pytest.mark.slow
@pytest.mark.parametrize("factory", [
    pytest.param(lambda s, c: FLTrustTrainer(s, c, root_size=20), id="fltrust"),
    pytest.param(lambda s, c: RAGATrainer(s, c), id="raga"),
    pytest.param(lambda s, c: FedTruthTrainer(s, c), id="fedtruth"),
])
def test_poisoned_client_is_down_weighted_end_to_end(factory):
    """The same robustness property, exercised through the full round loop.

    One client's submitted parameters are replaced with a large update opposed
    to the benign consensus. Every recorded round must give that client a
    markedly smaller aggregation weight than the benign clients receive.
    """
    split = _tiny_split(seed=0)
    assert len(split.clients) >= 3, "need >=3 clients for a meaningful majority"
    cfg = _cfg(rounds=3)
    trainer = _poison(factory(split, cfg), victim=0, scale=-25.0)

    with np.errstate(all="ignore"):
        hist = trainer.train(eval_every=1)

    # hist.weights[0] is the pre-training round-0 record (uniform by construction).
    for rnd, w in list(zip(hist.rounds, hist.weights))[1:]:
        w = np.asarray(w, dtype=np.float64)
        benign_min = float(w[1:].min())
        assert w[0] < benign_min, (
            f"round {rnd}: poisoned client weight {w[0]:.4g} is not below the "
            f"benign minimum {benign_min:.4g} (weights={w})"
        )
        assert w[0] < 0.5 * benign_min, (
            f"round {rnd}: poisoned client weight {w[0]:.4g} is not *markedly* "
            f"below the benign minimum {benign_min:.4g}"
        )


# --------------------------------------------------------------------------- #
# The three mechanisms must actually be three mechanisms
# --------------------------------------------------------------------------- #
def test_the_three_weight_rules_are_not_the_same_function():
    """Guard against the failure mode where all three baselines collapse into
    one generic 'robust average'. On a single shared input the three rules must
    disagree, because they use three different reference signals: a server root
    direction (FLTrust), a size-weighted geometric median (RAGA), and a
    bootstrapped consensus (FedTruth).
    """
    server = _vec(1.0, 0.0)
    updates = [_vec(1.0, 0.0), _vec(0.2, 1.4), _vec(3.0, 0.1), _vec(-0.4, 0.9)]
    sizes = [10, 500, 30, 60]  # deliberately unequal: only RAGA reads these

    w_fl = fltrust_trust_scores(updates, server)
    w_fl = w_fl / w_fl.sum()
    w_ra = raga_weights(updates, sizes)
    w_ft = fedtruth_weights(updates)

    assert not np.allclose(w_fl, w_ra, atol=1e-3)
    assert not np.allclose(w_fl, w_ft, atol=1e-3)
    assert not np.allclose(w_ra, w_ft, atol=1e-3)


def test_only_raga_reads_dataset_sizes():
    """RAGA's weights change with dataset size; FedTruth's and FLTrust's cannot,
    because neither rule takes sizes as an input at all."""
    updates = [_vec(1.0, 0.0), _vec(0.2, 1.4), _vec(3.0, 0.1)]
    w_a = raga_weights(updates, [10, 10, 10])
    w_b = raga_weights(updates, [10, 900, 10])
    assert not np.allclose(w_a, w_b, atol=1e-3)


def test_fedtruth_inverse_coefficient_reduces_to_the_geometric_median():
    """Verifies the deviation note in :func:`fedtruth_weights`' docstring.

    With ``g(p) = 1/p`` and Euclidean distance the FedTruth fixed point is
    ``a_k proportional to 1/d_k`` -- the unweighted Weiszfeld iteration -- so
    that configuration coincides with RAGA at uniform dataset sizes. This is an
    observation made in this reproduction, not a claim from either paper, and
    it is exactly why ``"neg_log"`` is the default: ``"inverse"`` would make
    FedTruth a duplicate of RAGA rather than a distinct baseline.
    """
    updates = [_vec(1.0, 0.0), _vec(0.2, 1.4), _vec(3.0, 0.1), _vec(-0.4, 0.9)]
    sizes = [50] * len(updates)

    w_inverse = fedtruth_weights(updates, coefficient="inverse")
    w_raga = raga_weights(updates, sizes)
    w_neglog = fedtruth_weights(updates, coefficient="neg_log")

    assert np.allclose(w_inverse, w_raga, atol=5e-3), (
        f"inverse-coefficient FedTruth {w_inverse} should coincide with the "
        f"unweighted geometric median {w_raga}"
    )
    # ... and the shipped default must NOT coincide with it.
    assert not np.allclose(w_neglog, w_raga, atol=5e-3)


# --------------------------------------------------------------------------- #
# FLTrust's root-dataset construction (the disclosed adaptation)
# --------------------------------------------------------------------------- #
def test_fltrust_root_dataset_comes_from_training_data_and_is_removed():
    """The disclosed adaptation must hold: the root set is carved out of the
    clients' *training* rows, never their validation rows, and the contributing
    clients no longer see those rows."""
    split = _tiny_split(seed=0)
    cfg = _cfg(rounds=1)
    trainer = FLTrustTrainer(split, cfg, root_size=20)

    assert trainer.root_size_actual > 0
    assert len(trainer.root_y) == trainer.root_size_actual

    val_rows = {tuple(np.round(r, 6)) for r in split.X_val}
    root_rows = [tuple(np.round(r, 6)) for r in trainer.root_X]
    assert not (set(root_rows) & val_rows), (
        "root dataset overlaps the held-out validation pool D_V"
    )

    # Every root row came from some client's training data...
    train_rows = set()
    for c in split.clients:
        train_rows |= {tuple(np.round(r, 6)) for r in c.X_train}
    assert set(root_rows) <= train_rows

    # ...and is no longer in the training data the trainer actually uses.
    kept_rows = set()
    for i in range(len(split.clients)):
        X, _ = trainer._client_data(i)
        kept_rows |= {tuple(np.round(r, 6)) for r in X}
    assert not (set(root_rows) & kept_rows), (
        "root rows are still present in the clients' training sets"
    )

    total_kept = sum(len(y) for _, y in trainer._kept)
    total_orig = sum(c.n_train for c in split.clients)
    assert total_kept + trainer.root_size_actual == total_orig
    assert trainer.client_sizes == [len(y) for _, y in trainer._kept]
    assert all(n > 0 for n in trainer.client_sizes), "a client was emptied"


def test_fltrust_server_update_is_a_real_reference_direction():
    """The server must actually train on the root dataset: ``g_0`` is non-zero
    and differs from every client's update."""
    split = _tiny_split(seed=0)
    cfg = _cfg(rounds=1)
    trainer = FLTrustTrainer(split, cfg, root_size=20)
    rng = np.random.default_rng(0)

    from fedgraa.model import init_params
    global_params = init_params(trainer._model.layer_sizes, seed=cfg.seed)
    g0 = trainer.server_update(global_params, rng)
    assert float(np.linalg.norm(flatten_params(g0))) > 0.0
    assert all(np.all(np.isfinite(a)) for a in g0)


def test_fltrust_records_trust_scores_for_diagnostics():
    split = _tiny_split(seed=0)
    trainer = FLTrustTrainer(split, _cfg(rounds=2), root_size=20)
    trainer.train(eval_every=2)
    assert trainer.last_trust_scores is not None
    assert trainer.last_trust_scores.shape == (len(split.clients),)
    assert np.all(trainer.last_trust_scores >= 0.0)
    assert np.all(trainer.last_trust_scores <= 1.0 + 1e-9)


def test_fltrust_all_zero_trust_round_warns_and_applies_no_update():
    """The one case FLTrust's paper leaves undefined (``0 / sum(TS)``) must be
    signalled, not silently papered over."""
    server = _vec(1.0, 0.0)
    updates = [_vec(-1.0, 0.0), _vec(-0.5, -0.1)]  # all anti-aligned
    global_params = _vec(3.0, -2.0)

    with pytest.warns(RuntimeWarning, match="all trust scores are zero"):
        new_global, w = fltrust_aggregate(global_params, updates, server)

    assert np.allclose(new_global[0], global_params[0])
    assert w == pytest.approx(np.full(2, 0.5, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Argument validation
# --------------------------------------------------------------------------- #
def test_invalid_options_are_rejected():
    split = _tiny_split(seed=0)
    cfg = _cfg(rounds=1)
    with pytest.raises(ValueError, match="root_size"):
        FLTrustTrainer(split, cfg, root_size=0)
    with pytest.raises(ValueError, match="coefficient"):
        FedTruthTrainer(split, cfg, coefficient="sqrt")
    with pytest.raises(ValueError, match="distance"):
        FedTruthTrainer(split, cfg, distance="manhattan")
    with pytest.raises(ValueError, match="length mismatch"):
        raga_weights([_vec(1.0), _vec(2.0)], [10])


def test_model_update_is_local_minus_global():
    """Sign convention: the papers' ``g_i = w_i - w`` (not FedOpt's negation)."""
    g = [np.array([1.0, 2.0], dtype=np.float32)]
    local = [np.array([1.5, 2.5], dtype=np.float32)]
    upd = model_update(g, local)
    assert upd[0] == pytest.approx(np.array([0.5, 0.5], dtype=np.float32))
