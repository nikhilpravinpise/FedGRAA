"""Tests for the T5 same-domain NIDS baselines (FedMADE / SBFedNIDS / FedAPA).

Conventions follow ``tests/test_reproduction_gaps.py``: small synthetic
``DatasetSpec`` -> ``make_synthetic_dataset`` -> ``partition_non_iid``, a
handful of rounds, and a NaN-collapse regression check mirroring
``test_all_methods_no_nan_collapse``.

Each baseline gets three kinds of test:

1. no-NaN collapse over a few rounds,
2. every recorded weight vector is a valid non-negative simplex point,
3. one test of that method's actual distinguishing behaviour -- CPM-based
   clustering for FedMADE, local-balance-driven rebalancing for SBFedNIDS, and
   the exact Eq.-7 weight-row arithmetic for FedAPA (hand-computable).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.baselines_t5_nids import (
    FedAPATrainer,
    FedMADETrainer,
    SBFedNIDSTrainer,
    build_server_auxiliary_set,
    class_probability_matrix,
    client_smote,
    dbscan_cluster_cpms,
    federated_cost_weights,
    fedapa_weight_row_update,
    fedmade_weights,
    solve_cluster_weights,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.model import MLP, init_params
from fedgraa.partition import partition_non_iid


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tiny_split(seed: int = 0):
    """Same shape as ``test_reproduction_gaps._tiny_split``."""
    spec = DatasetSpec(
        name="T5_NIDS_TEST",
        n_features=6,
        labels=["Benign", "A", "Rare"],
        used_counts=[200, 100, 20],
        benign_id=0,
        rare_label_order=["Rare"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _cfg(**kw) -> FedConfig:
    base = dict(rounds=4, gamma=0.5, n_rare=1, seed=0, round_zero_zero=False)
    base.update(kw)
    return FedConfig(**base)


def _assert_simplex(w, name: str):
    w = np.asarray(w, dtype=np.float64)
    assert np.all(np.isfinite(w)), f"{name}: weights contain non-finite entries"
    assert np.all(w >= -1e-9), f"{name}: negative weight present: {w}"
    assert np.isclose(w.sum(), 1.0, atol=1e-6), f"{name}: weights sum to {w.sum()}"


def _trainers(split, cfg):
    return {
        "fedmade": FedMADETrainer(split, cfg, alpha_steps=25),
        "sbfednids": SBFedNIDSTrainer(split, cfg),
        "fedapa": FedAPATrainer(split, cfg),
    }


# --------------------------------------------------------------------------- #
# 1. No-NaN collapse (mirrors test_all_methods_no_nan_collapse)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_t5_methods_no_nan_collapse():
    """All three T5 baselines must produce finite params at every checkpoint."""
    split = _tiny_split(seed=0)
    cfg = _cfg()
    for name, trainer in _trainers(split, cfg).items():
        hist = trainer.train(eval_every=2)
        params = trainer.final_params
        assert params is not None, f"{name}: final_params is None"
        for p in params:
            assert np.all(np.isfinite(p)), (
                f"{name}: final params contain non-finite values "
                f"(shape={p.shape}, n_bad={np.sum(~np.isfinite(p))})"
            )
        assert np.all(np.isfinite(hist.all_acc)), f"{name}: non-finite all_acc"
        assert np.all(np.isfinite(hist.last_n_acc)), f"{name}: non-finite last_n_acc"
        assert 0.0 <= hist.final_all_acc <= 1.0, f"{name}: accuracy out of range"


# --------------------------------------------------------------------------- #
# 2. Weight vectors are valid simplex points every recorded round
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_t5_methods_weights_are_simplex_every_round():
    split = _tiny_split(seed=1)
    cfg = _cfg(seed=1)
    for name, trainer in _trainers(split, cfg).items():
        hist = trainer.train(eval_every=1)
        assert len(hist.weights) == len(hist.rounds) >= 2, f"{name}: nothing recorded"
        for rnd, w in zip(hist.rounds, hist.weights):
            assert len(w) == split.n_clients, f"{name}: wrong weight length"
            _assert_simplex(w, f"{name} round {rnd}")


@pytest.mark.slow
def test_fedapa_every_weight_matrix_row_is_a_simplex():
    """FedAPA's per-client rows -- not just their mean -- must stay on the simplex."""
    split = _tiny_split(seed=2)
    trainer = FedAPATrainer(split, _cfg(seed=2))
    trainer.train(eval_every=1)
    for t, A in enumerate(trainer.weight_matrix_history[1:], start=1):
        for i, row in enumerate(A):
            _assert_simplex(row, f"fedapa A[{i}] at recorded step {t}")
        # Algorithm 1 line 12 fixes a_ii = 0.5 *before* line 13's normalization,
        # so the self weight is the largest entry unless peers saturate at 1.
        assert np.all(A >= 0.0)


# --------------------------------------------------------------------------- #
# 3a. FedMADE: distinguishing behaviour -- CPM clustering + within-cluster split
# --------------------------------------------------------------------------- #
def test_fedmade_clusters_similar_cpms_together():
    """Two near-identical CPMs cluster together; a visibly different one does not.

    This is the core of FedMADE's "cluster devices by their traffic patterns":
    the clustering acts on measured class-conditional model behaviour, so
    clients whose behaviour is nearly the same must land in one cluster while a
    client with a visibly different class-probability profile must not.
    """
    base = np.array([[0.9, 0.1], [0.15, 0.85]], dtype=np.float64)
    cpms = [
        base,
        base + 1e-4,                                   # essentially the same
        np.array([[0.2, 0.8], [0.85, 0.15]]),          # visibly different
    ]
    labels = dbscan_cluster_cpms(cpms)
    assert labels[0] == labels[1], "near-identical CPMs were not clustered together"
    assert labels[2] != labels[0], "a visibly different CPM was merged anyway"


def test_fedmade_cluster_members_share_weight_equally():
    """Eq. 4: ``beta_i = alpha_k / |C_k|`` -- members of one cluster tie."""
    spec = DatasetSpec(
        name="T5_MADE_BETA",
        n_features=5,
        labels=["Benign", "A", "B", "C"],
        used_counts=[240, 80, 80, 80],
        benign_id=0,
        rare_label_order=["C"],
    )
    X, y, _ = make_synthetic_dataset(spec, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=0)
    model = MLP(spec.model_config().layer_sizes, seed=0)
    X_aux, y_aux = build_server_auxiliary_set(
        split.X_val, split.y_val, spec.n_classes, per_class=10, seed=0
    )

    # Two clients share one parameter set exactly; the third is different.
    p_same = init_params(model.layer_sizes, seed=0)
    p_other = init_params(model.layer_sizes, seed=99)
    client_params = [p_same, [p.copy() for p in p_same], p_other]

    beta, labels = fedmade_weights(
        model, client_params, X_aux, y_aux, spec.n_classes,
        client_sizes=[100, 100, 100], alpha_steps=50,
    )
    _assert_simplex(beta, "fedmade beta")
    assert labels[0] == labels[1], "identical models were not clustered together"
    assert np.isclose(beta[0], beta[1]), "cluster members did not share alpha equally"


def test_fedmade_cpm_of_a_perfect_model_is_near_identity():
    """A model that classifies the auxiliary set correctly yields a CPM near I.

    That is the whole point of Eq. 3's ``||sum_k alpha_k P_{C_k} - I||``: the
    identity matrix is the CPM of a perfect classifier, so the optimization is
    a contribution score. This test pins the direction of the CPM statistic.
    """
    C = 3
    P_good = np.eye(C) * 0.9 + 0.05
    P_bad = np.full((C, C), 1.0 / C)
    I = np.eye(C)
    assert np.linalg.norm(P_good - I) < np.linalg.norm(P_bad - I)

    # And the solver prefers the closer-to-identity cluster.
    alpha = solve_cluster_weights([P_good, P_bad], n_steps=400, lr=0.05)
    assert np.all(alpha >= 0.0), "non-negativity constraint violated"
    assert alpha[0] > alpha[1], (
        "Eq. 3 solver did not favour the cluster whose CCPM is closer to I"
    )


def test_fedmade_auxiliary_set_is_class_balanced():
    """Verified detail: ten samples per class (up to availability)."""
    split = _tiny_split(seed=0)
    X_aux, y_aux = build_server_auxiliary_set(
        split.X_val, split.y_val, split.spec.n_classes, per_class=10, seed=0
    )
    counts = np.bincount(y_aux, minlength=split.spec.n_classes)
    present = counts[counts > 0]
    assert len(X_aux) == len(y_aux) > 0
    assert np.all(present <= 10), "auxiliary set took more than ten rows of a class"


def test_fedmade_cpm_rows_are_probability_vectors():
    split = _tiny_split(seed=0)
    spec = split.spec
    model = MLP(spec.model_config().layer_sizes, seed=0)
    X_aux, y_aux = build_server_auxiliary_set(
        split.X_val, split.y_val, spec.n_classes, per_class=10, seed=0
    )
    cpm = class_probability_matrix(
        model, init_params(model.layer_sizes, seed=0), X_aux, y_aux, spec.n_classes
    )
    assert cpm.shape == (spec.n_classes, spec.n_classes)
    present = np.unique(y_aux)
    for c in present:
        assert np.isclose(cpm[c].sum(), 1.0, atol=1e-5), (
            "CPM row for a present class is not a probability vector"
        )
    for c in set(range(spec.n_classes)) - set(present.tolist()):
        assert np.allclose(cpm[c], 0.0), "absent class did not get a zero CPM row"


# --------------------------------------------------------------------------- #
# 3b. SBFedNIDS: distinguishing behaviour -- balance-driven rebalancing
# --------------------------------------------------------------------------- #
def test_csmote_rebalances_an_imbalanced_client_and_leaves_a_balanced_one_alone():
    """A heavily imbalanced client is treated differently from a balanced one.

    This is the verified data-level half of SBFedNIDS (CSMOTE): the imbalanced
    client gains synthetic minority rows until its classes match its local
    majority count; the already-balanced client gains none.
    """
    rng = np.random.default_rng(0)
    n_classes = 2

    X_imb = np.vstack([rng.normal(size=(100, 4)), rng.normal(loc=5, size=(10, 4))])
    y_imb = np.array([0] * 100 + [1] * 10)
    X_bal = np.vstack([rng.normal(size=(50, 4)), rng.normal(loc=5, size=(50, 4))])
    y_bal = np.array([0] * 50 + [1] * 50)

    Xi, yi = client_smote(X_imb, y_imb, n_classes, np.random.default_rng(1))
    Xb, yb = client_smote(X_bal, y_bal, n_classes, np.random.default_rng(1))

    ci = np.bincount(yi, minlength=n_classes)
    cb = np.bincount(yb, minlength=n_classes)
    assert ci[0] == ci[1] == 100, f"imbalanced client not rebalanced: {ci}"
    assert len(yi) > len(y_imb), "no synthetic rows generated for the minority class"
    assert list(cb) == [50, 50], f"balanced client was altered: {cb}"
    assert len(yb) == len(y_bal), "balanced client gained rows it did not need"
    assert len(Xi) == len(yi) and len(Xb) == len(yb)


def test_csmote_synthetic_rows_lie_between_real_minority_rows():
    """SMOTE interpolates -- synthetic rows stay inside the minority hull's box."""
    rng = np.random.default_rng(0)
    X = np.vstack([
        np.zeros((20, 2)),                       # majority at the origin
        rng.uniform(1.0, 2.0, size=(5, 2)),      # minority in [1,2]^2
    ])
    y = np.array([0] * 20 + [1] * 5)
    Xs, ys = client_smote(X, y, 2, np.random.default_rng(3))
    synth = Xs[len(y):][ys[len(y):] == 1]
    assert len(synth) > 0
    assert np.all(synth >= 1.0 - 1e-9) and np.all(synth <= 2.0 + 1e-9), (
        "synthetic minority rows fell outside the real minority range -- "
        "that is extrapolation, not SMOTE interpolation"
    )


def test_sbfednids_weights_shift_toward_the_imbalanced_client():
    """Post-CSMOTE size weighting differs from FedAvg over the raw sizes.

    The aggregation rule itself is FedAvg (the abstract describes no other),
    but CSMOTE changes the sizes it is computed over, so a client whose local
    distribution was more imbalanced gains weight relative to plain FedAvg.
    """
    from fedgraa.aggregation import fedavg_weights

    raw_sizes = [110, 100]                       # imbalanced client, balanced client
    # imbalanced: 100/10 -> 100/100 = 200 rows;  balanced: 50/50 -> unchanged
    rebalanced_sizes = [200, 100]
    w_fedavg = fedavg_weights(raw_sizes)
    w_sb = fedavg_weights(rebalanced_sizes)
    _assert_simplex(w_sb, "sbfednids")
    assert w_sb[0] > w_fedavg[0], (
        "the rebalanced (formerly imbalanced) client did not gain weight"
    )


def test_federated_cost_weights_are_inverse_global_frequency():
    """FCSL derives its costs from the *global* histogram, inverse-frequency."""
    counts = np.array([900, 90, 10])
    w = federated_cost_weights(counts, n_classes=3)
    assert w[2] > w[1] > w[0], "cost weights are not inverse to global frequency"
    assert np.isclose(w.mean(), 1.0), "cost weights were not rescaled to mean 1"
    # Exact inverse-frequency ratio: c_k ∝ 1 / count_k.
    assert np.isclose(w[0] * counts[0], w[2] * counts[2])
    # Absent classes contribute nothing.
    w2 = federated_cost_weights(np.array([10, 0, 10]), n_classes=3)
    assert w2[1] == 0.0


def test_sbfednids_uses_global_not_local_counts():
    """The cost vector must match the pooled histogram, not any one client's."""
    split = _tiny_split(seed=0)
    trainer = SBFedNIDSTrainer(split, _cfg())
    C = split.spec.n_classes
    pooled = np.zeros(C, dtype=np.int64)
    for c in split.clients:
        pooled += np.bincount(c.y_train, minlength=C)[:C]
    assert np.array_equal(trainer.global_counts, pooled)
    assert np.allclose(
        trainer.class_weights, federated_cost_weights(pooled, C)
    )
    # And it is genuinely different from what a single client would compute.
    local = np.bincount(split.clients[0].y_train, minlength=C)[:C]
    assert not np.allclose(
        trainer.class_weights, federated_cost_weights(local, C)
    ), "FCSL weights coincide with a single client's local weights"


# --------------------------------------------------------------------------- #
# 3c. FedAPA: distinguishing behaviour -- exact Eq. 7 + lines 11-13 arithmetic
# --------------------------------------------------------------------------- #
def test_fedapa_weight_row_update_hand_computed():
    """Eq. 7 + Algorithm 1 lines 11-13 on a fully hand-computable toy case.

    Three clients, one scalar-ish parameter array each:
        theta_0 = [1, 0],  theta_1 = [0, 1],  theta_2 = [-1, 0]
        delta   = [2, 0]   (client 0's parameter change)
        eta     = 0.1,  starting row a = [1, 0, 0]

    <theta_j, delta> = [2, 0, -2].
    Eq. 7 (plus-sign reading): a = [1, 0, 0] + 0.1 * [2, 0, -2]
                                 = [1.2, 0.0, -0.2]
    line 11 clip to [0,1]        -> [1.0, 0.0, 0.0]
    line 12 a_00 <- 0.5          -> [0.5, 0.0, 0.0]
    line 13 normalize (sum=0.5)  -> [1.0, 0.0, 0.0]
    """
    thetas = [
        [np.array([1.0, 0.0], dtype=np.float32)],
        [np.array([0.0, 1.0], dtype=np.float32)],
        [np.array([-1.0, 0.0], dtype=np.float32)],
    ]
    delta = [np.array([2.0, 0.0], dtype=np.float32)]
    out = fedapa_weight_row_update(
        np.array([1.0, 0.0, 0.0]), 0, thetas, delta, eta=0.1
    )
    assert np.allclose(out, [1.0, 0.0, 0.0])
    _assert_simplex(out, "fedapa hand case")


def test_fedapa_weight_row_update_hand_computed_unsaturated():
    """Same arithmetic without clipping saturation, so every entry is checkable.

        a = [0.4, 0.2, 0.2], <theta_j, delta> = [2, 0, -2], eta = 0.05
        Eq. 7   -> [0.5, 0.2, 0.1]
        clip    -> [0.5, 0.2, 0.1]   (already inside [0,1])
        a_00=.5 -> [0.5, 0.2, 0.1]
        sum=0.8 -> [0.625, 0.25, 0.125]
    """
    thetas = [
        [np.array([1.0, 0.0], dtype=np.float32)],
        [np.array([0.0, 1.0], dtype=np.float32)],
        [np.array([-1.0, 0.0], dtype=np.float32)],
    ]
    delta = [np.array([2.0, 0.0], dtype=np.float32)]
    out = fedapa_weight_row_update(
        np.array([0.4, 0.2, 0.2]), 0, thetas, delta, eta=0.05
    )
    assert np.allclose(out, [0.625, 0.25, 0.125]), out
    _assert_simplex(out, "fedapa unsaturated hand case")


def test_fedapa_update_is_monotone_in_alignment_with_the_delta():
    """The distinguishing property: alignment with ``delta_theta_i`` drives the weight.

    Client ``j``'s weight must increase with ``<theta_j, delta_theta_i>`` --
    i.e. a peer whose parameters point where client ``i``'s local training
    wants to go is up-weighted, and an anti-aligned peer is down-weighted.
    This is what makes FedAPA a *parameter-change-gradient* method rather than
    a contribution-measurement (FedMADE) or balance-statistics (SBFedNIDS) one.
    """
    delta = [np.array([1.0, 0.0], dtype=np.float32)]
    prev = np.array([0.4, 0.3, 0.3])
    for scale in (0.5, 1.0, 2.0):
        thetas = [
            [np.array([0.0, 1.0], dtype=np.float32)],           # orthogonal
            [np.array([scale, 0.0], dtype=np.float32)],         # aligned
            [np.array([-scale, 0.0], dtype=np.float32)],        # anti-aligned
        ]
        out = fedapa_weight_row_update(prev, 0, thetas, delta, eta=0.1)
        assert out[1] > out[2], "aligned peer was not weighted above anti-aligned"
    # Stronger alignment -> strictly larger raw (pre-normalization) increment.
    inc = []
    for scale in (0.5, 1.0, 2.0):
        thetas = [
            [np.array([0.0, 1.0], dtype=np.float32)],
            [np.array([scale, 0.0], dtype=np.float32)],
            [np.array([0.0, 0.0], dtype=np.float32)],
        ]
        out = fedapa_weight_row_update(prev, 0, thetas, delta, eta=0.1)
        inc.append(out[1] / out[2])   # ratio against the zero-gradient peer
    assert inc[0] < inc[1] < inc[2], f"not monotone in alignment strength: {inc}"


def test_fedapa_sign_flag_reverses_the_update():
    """The documented sign ambiguity is reversible, not baked in silently."""
    thetas = [
        [np.array([1.0, 0.0], dtype=np.float32)],
        [np.array([0.0, 1.0], dtype=np.float32)],
        [np.array([-1.0, 0.0], dtype=np.float32)],
    ]
    delta = [np.array([2.0, 0.0], dtype=np.float32)]
    prev = np.array([0.4, 0.2, 0.2])
    plus = fedapa_weight_row_update(prev, 0, thetas, delta, eta=0.05)
    minus = fedapa_weight_row_update(
        prev, 0, thetas, delta, eta=0.05, update_sign=-1.0
    )
    assert not np.allclose(plus, minus)
    # Under the literal (minus) transcription the anti-aligned peer gains.
    assert minus[2] > plus[2]


def test_fedapa_initializes_with_identity_weight_matrix():
    """Paper's init: ``a_ii = 1``, ``a_ij = 0`` for ``j != i``."""
    split = _tiny_split(seed=0)
    trainer = FedAPATrainer(split, _cfg())
    assert trainer.A.shape == (split.n_clients, split.n_clients)
    assert np.allclose(trainer.A, np.eye(split.n_clients))


def test_fedapa_keeps_the_decision_layer_private():
    """Only the feature extractor is shared; the last Dense layer stays local."""
    split = _tiny_split(seed=0)
    trainer = FedAPATrainer(split, _cfg(rounds=2))
    trainer.train(eval_every=2)
    heads = trainer.final_heads
    assert len(heads) == split.n_clients
    assert all(len(h) == 2 for h in heads), "head is not the final Dense (W, b)"
    # Personalization means the private heads must not all be identical.
    assert any(
        not np.allclose(heads[0][0], h[0]) for h in heads[1:]
    ), "every client's decision layer is identical -- nothing was personalized"


# --------------------------------------------------------------------------- #
# Cross-method: the three are genuinely different mechanisms
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_t5_methods_produce_distinct_weight_vectors():
    """The three baselines must not collapse onto the same weighting rule."""
    split = _tiny_split(seed=0)
    cfg = _cfg(rounds=3, seed=0)
    final = {}
    for name, trainer in _trainers(split, cfg).items():
        hist = trainer.train(eval_every=3)
        final[name] = np.asarray(hist.weights[-1], dtype=np.float64)

    pairs = [("fedmade", "sbfednids"), ("fedmade", "fedapa"), ("sbfednids", "fedapa")]
    for a, b in pairs:
        assert not np.allclose(final[a], final[b], atol=1e-6), (
            f"{a} and {b} produced identical weight vectors -- the two "
            f"mechanisms are not actually distinct in this implementation"
        )


def test_t5_trainers_expose_the_shared_baseline_interface():
    """Same shape as ``fedgraa.baselines``: ``__init__(split, cfg)`` + ``train()``."""
    split = _tiny_split(seed=0)
    cfg = _cfg(rounds=1)
    for Trainer in (FedMADETrainer, SBFedNIDSTrainer, FedAPATrainer):
        t = Trainer(split, cfg)
        assert t.split is split and t.cfg is cfg
        assert t.rare_class_ids == split.spec.rare_client_ids(cfg.n_rare)
        assert t.client_sizes == [c.n_train for c in split.clients]
        assert hasattr(t, "train")
