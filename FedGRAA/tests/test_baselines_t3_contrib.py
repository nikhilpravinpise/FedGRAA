"""Tests for the T3/T5 data-free contribution-estimation baselines.

Covers :mod:`fedgraa.baselines_t3_contrib` -- CELM (arXiv:2605.18892),
SpectralFed (arXiv:2604.22562) and CoAst (arXiv:2409.02495).

Three kinds of test per method:

1. *it runs* -- a handful of rounds on a small synthetic dataset with no
   NaN/inf anywhere (mirroring ``test_all_methods_no_nan_collapse`` in
   ``tests/test_reproduction_gaps.py``);
2. *the weights are legal* -- every recorded weight vector is a probability
   simplex point;
3. *the mechanism is the paper's* -- a numeric check against the paper's own
   equations on a hand-computable case, or a checkable behavioural property
   the paper claims. These are the tests that would fail if the implementation
   were a plausible-looking placeholder rather than the published method.
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.aggregation import fedavg_weights
from fedgraa.baselines_t3_contrib import (
    CELMTrainer,
    CoAstTrainer,
    SpectralFedTrainer,
    celm_contribution_weights,
    coast_sign_agreement,
    logit_maximization_probe,
    spectralfed_weights,
    ternary_quantize_update,
    von_neumann_entropy,
)
from fedgraa.config import DatasetSpec, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.model import MLP, init_params
from fedgraa.partition import partition_non_iid


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _tiny_spec(n_attacks: int = 2) -> DatasetSpec:
    labels = ["Benign"] + [f"A{i}" for i in range(n_attacks - 1)] + ["Rare"]
    counts = [200] + [100] * (n_attacks - 1) + [20]
    return DatasetSpec(
        name="T3_CONTRIB_TEST",
        n_features=6,
        labels=labels,
        used_counts=counts,
        benign_id=0,
        rare_label_order=["Rare"],
    )


def _tiny_split(seed: int = 0, n_attacks: int = 2):
    spec = _tiny_spec(n_attacks)
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    return partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)


def _iid_split_with_one_corrupt_client(
    seed: int = 0, n_attacks: int = 3, corrupt_idx: int = 1
):
    """IID clients over all classes, one of which has cyclically shifted labels.

    The repo's own ``partition_non_iid`` gives every client a *different*
    attack class, so cross-client disagreement there is dominated by the label
    skew rather than by data quality -- useless for isolating a contribution
    estimator's response to a bad contributor. This helper reuses that split's
    spec/validation machinery but hands every client an IID slice of the same
    pool, then relabels one client ``y -> (y + 1) mod C``. Clients ``!=
    corrupt_idx`` are then statistically interchangeable, and the only thing
    that can separate the corrupt one is the estimator itself.
    """
    spec = _tiny_spec(n_attacks)
    X, y, _ = make_synthetic_dataset(spec, seed=seed)
    split = partition_non_iid(X, y, spec, n_rare=1, val_fraction=0.10, seed=seed)

    rng = np.random.default_rng(seed + 991)
    perm = rng.permutation(len(y))
    chunks = np.array_split(perm, len(split.clients))
    for i, (shard, idx) in enumerate(zip(split.clients, chunks)):
        shard.X_train = X[idx].astype(np.float32)
        yi = y[idx].astype(np.int32)
        if i == corrupt_idx:
            yi = ((yi + 1) % spec.n_classes).astype(np.int32)
        shard.y_train = yi
    return split


def _make_trainers(split, cfg):
    """One instance of each T3 baseline, cheap enough for a unit test."""
    return {
        # probe_steps is cut from the paper's 200 purely for test runtime; the
        # probe itself is exercised properly in test_logit_maximization_probe_*.
        "celm": CELMTrainer(split, cfg, warmup_frac=0.5, probe_steps=4),
        "spectralfed": SpectralFedTrainer(split, cfg),
        "coast": CoAstTrainer(split, cfg, k=2),
    }


def _assert_simplex(w, ctx: str = ""):
    w = np.asarray(w, dtype=np.float64)
    assert np.all(np.isfinite(w)), f"{ctx}: non-finite weight {w}"
    assert np.all(w >= -1e-9), f"{ctx}: negative weight {w}"
    assert abs(w.sum() - 1.0) < 1e-6, f"{ctx}: weights sum to {w.sum()}, not 1"


# --------------------------------------------------------------------------- #
# 1. End-to-end: no NaN / inf collapse
# --------------------------------------------------------------------------- #
def test_t3_contrib_baselines_no_nan_collapse():
    """All three baselines must run and stay finite end-to-end.

    Same spirit as ``test_all_methods_no_nan_collapse``: finite global params,
    finite accuracies, finite weights, at every recorded checkpoint.
    """
    split = _tiny_split(seed=0)
    cfg = FedConfig(rounds=4, gamma=0.5, n_rare=1, seed=0, round_zero_zero=False)

    for name, trainer in _make_trainers(split, cfg).items():
        hist = trainer.train(eval_every=2)

        assert getattr(trainer, "final_params", None) is not None, (
            f"{name}: final_params is None"
        )
        for p in trainer.final_params:
            assert np.all(np.isfinite(p)), (
                f"{name}: final params contain NaN/inf "
                f"(shape={p.shape}, bad={np.sum(~np.isfinite(p))})"
            )
        assert len(hist.all_acc) >= 2, f"{name}: nothing recorded"
        for a in hist.all_acc + hist.last_n_acc:
            assert np.isfinite(a), f"{name}: non-finite accuracy {a}"
            assert 0.0 <= a <= 1.0, f"{name}: accuracy out of range: {a}"
        for w in hist.weights:
            assert np.all(np.isfinite(w)), f"{name}: non-finite weights {w}"


# --------------------------------------------------------------------------- #
# 2. Every recorded weight vector lives on the probability simplex
# --------------------------------------------------------------------------- #
def test_t3_contrib_weights_are_on_the_simplex():
    """Non-negative and summing to 1, every round each baseline records one.

    CELM eq. 12 and SpectralFed Algorithm 1 line 14 both state the constraint
    explicitly (``c_i >= 0``, ``sum_i c_i = 1``); the CoAst weight map is this
    repo's disclosed adaptation and must satisfy it too.
    """
    split = _tiny_split(seed=1)
    cfg = FedConfig(rounds=4, gamma=0.5, n_rare=1, seed=1, round_zero_zero=False)

    for name, trainer in _make_trainers(split, cfg).items():
        hist = trainer.train(eval_every=1)
        assert len(hist.weights) == len(hist.rounds)
        for rnd, w in zip(hist.rounds, hist.weights):
            _assert_simplex(w, f"{name} round {rnd}")


def test_coast_score_weights_are_on_the_simplex_every_round():
    """CoAst records a score-derived weight vector on every single round."""
    split = _tiny_split(seed=2, n_attacks=3)
    cfg = FedConfig(rounds=6, gamma=1.0, n_rare=1, seed=2, round_zero_zero=False)
    trainer = CoAstTrainer(split, cfg, k=2)
    trainer.train(eval_every=6)
    assert len(trainer.score_weight_history) == cfg.rounds
    for i, w in enumerate(trainer.score_weight_history, start=1):
        _assert_simplex(w, f"coast score round {i}")


# --------------------------------------------------------------------------- #
# 3a. CELM -- the published equations, checked numerically
# --------------------------------------------------------------------------- #
def test_celm_contribution_weights_match_paper_equations():
    """Hand-computed pass through CELM eqs. 9-13 (arXiv:2605.18892v1).

    Two clients, two classes, raw evidence ``q~ = [[4, 2], [2, 0]]`` and
    ``b^(t-1) = 1``:

        eq. 9   q     = [[3, 1], [1, 0]]
        eq. 10  cols  = [4, 1]           -> r = [[0.75, 1.0], [0.25, 0.0]]
        eq. 11  c^    = [0.875, 0.125]
        eq. 12  c-    = [0.875, 0.125]   (already sums to 1)
        eq. 13  c     = 0.5*[0.5, 0.5] + 0.5*[0.875, 0.125]
                      = [0.6875, 0.3125]
    """
    evidence = np.array([[4.0, 2.0], [2.0, 0.0]])
    prev = np.array([0.5, 0.5])
    w = celm_contribution_weights(evidence, bias=1.0, prev_weights=prev, ema_beta=0.5)
    np.testing.assert_allclose(w, [0.6875, 0.3125], atol=1e-9)
    _assert_simplex(w, "celm hand case")


def test_celm_eq9_clip_zeroes_evidence_below_the_global_bias():
    """Eq. 9's ``max(0, q~ - b)`` must actually cut a below-baseline client.

    A client whose logits never beat the previous global model's own probe
    baseline contributes zero evidence in every class and is driven towards
    weight 0 by the EMA -- this is the whole point of the debiasing step.
    """
    evidence = np.array([[10.0, 10.0], [0.5, 0.4]])
    w1 = celm_contribution_weights(evidence, bias=5.0, prev_weights=np.array([0.5, 0.5]))
    assert w1[1] == pytest.approx(0.25)   # 0.5*0.5 + 0.5*0.0
    # Iterating the frozen evidence drives it monotonically toward 0.
    w2 = celm_contribution_weights(evidence, bias=5.0, prev_weights=w1)
    assert w2[1] < w1[1]
    _assert_simplex(w1, "celm clip")
    _assert_simplex(w2, "celm clip iterated")


def test_celm_evidence_is_class_normalized_not_magnitude_dominated():
    """Eq. 10 normalizes *per class column*, so raw logit scale cannot buy weight.

    Scaling every client's evidence for a class by the same positive constant
    leaves that class's column of ``r`` unchanged, hence the weights unchanged.
    A magnitude-based (non-column-normalized) stand-in would fail this.
    """
    base = np.array([[3.0, 1.0], [1.0, 3.0]])
    scaled = base * np.array([100.0, 1.0])[None, :]   # blow up class 0 only
    prev = np.array([0.5, 0.5])
    w_base = celm_contribution_weights(base, bias=0.0, prev_weights=prev)
    w_scaled = celm_contribution_weights(scaled, bias=0.0, prev_weights=prev)
    np.testing.assert_allclose(w_base, w_scaled, atol=1e-9)


def test_logit_maximization_probe_raises_the_target_class_logit():
    """Eq. 6 is genuine gradient *ascent* on the pre-softmax logit ``s_c(x)``.

    Verifies the probe direction (a sign error here would silently invert the
    whole contribution ranking) and that the L2 penalty keeps ``x`` finite.
    """
    layer_sizes = (6, 8, 3)
    model = MLP(layer_sizes, seed=0)
    params = init_params(layer_sizes, seed=3)
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal((3, 6)).astype(np.float32)

    model.set_params(params)
    logits0 = model._model(x0, training=False).numpy()
    before = np.diag(logits0)

    after, x_star = logit_maximization_probe(
        model, params, n_features=6, n_classes=3, x_init=x0,
        steps=60, lr=0.05, l2=1e-3,
    )
    assert np.all(np.isfinite(after)) and np.all(np.isfinite(x_star))
    assert np.all(after > before - 1e-6), (
        f"logit maximization must not decrease s_c: {before} -> {after}"
    )
    assert np.any(after > before + 1e-4), "probe made no progress at all"


def test_celm_freezes_weights_after_the_warmup_period():
    """Eq. 14: ``c^(t) = c^(T_w)`` for every ``t > T_w`` -- the paper's
    defining behavioural claim for CELM, and the reason it is cheap.

    With ``rounds=6`` and ``warmup_frac=1/3``, ``T_w = 2``: exactly two rounds
    of probing may happen, and every weight vector recorded from round 3
    onwards must be bit-identical to the round-2 value.
    """
    split = _tiny_split(seed=3)
    cfg = FedConfig(rounds=6, gamma=1.0, n_rare=1, seed=3, round_zero_zero=False)
    trainer = CELMTrainer(split, cfg, warmup_frac=1.0 / 3.0, probe_steps=4)
    assert trainer.warmup_rounds == 2

    hist = trainer.train(eval_every=1)
    # Probing happened during warm-up only.
    assert len(trainer.evidence_history) == 2
    assert trainer.evidence_history[0].shape == (
        split.n_clients, split.spec.n_classes,
    )

    frozen = trainer.frozen_weights
    assert frozen is not None
    by_round = dict(zip(hist.rounds, hist.weights))
    for rnd in range(trainer.warmup_rounds, cfg.rounds + 1):
        np.testing.assert_array_equal(
            by_round[rnd], frozen,
            err_msg=f"CELM weights changed after warm-up at round {rnd}",
        )


# --------------------------------------------------------------------------- #
# 3b. SpectralFed -- the published equations, checked numerically
# --------------------------------------------------------------------------- #
def test_von_neumann_entropy_rank_one_update_scores_zero():
    """A collapsed (rank-1) final-layer update carries no spectral diversity.

    ``A = M M^T`` is then rank 1, so under either normalization the single
    non-zero eigenvalue is 1 and ``S = -1*log(1) = 0``. This is the paper's
    stated intuition ("updates that spread more evenly across class
    directions" score higher) at its extreme point.
    """
    M = np.outer(np.array([1.0, 2.0, -3.0]), np.array([0.5, -1.0, 2.0, 0.25]))
    assert von_neumann_entropy(M, "frobenius") == pytest.approx(0.0, abs=1e-9)
    assert von_neumann_entropy(M, "trace") == pytest.approx(0.0, abs=1e-9)
    assert von_neumann_entropy(np.zeros((3, 4))) == 0.0


def test_von_neumann_entropy_matches_hand_computed_eigenvalues():
    """Both normalizations, on ``A = diag(4, 1)`` (i.e. ``M = diag(2, 1)``).

    eq. 12 (Frobenius): ``||A||_F = sqrt(17)``, lambda = (4, 1)/sqrt(17)
    -> S = -(0.970143*log 0.970143 + 0.242536*log 0.242536) = 0.372985
    eq. 11 (trace):     ``Tr(A) = 5``, lambda = (0.8, 0.2)
    -> S = -(0.8*log 0.8 + 0.2*log 0.2) = 0.500402

    The two differ, which is exactly why the module documents Eq. 11 vs. Eq. 12
    as a real ambiguity in the source rather than the "scaling factor" the
    paper calls it.
    """
    M = np.diag([2.0, 1.0])
    lam_f = np.array([4.0, 1.0]) / np.sqrt(17.0)
    lam_t = np.array([0.8, 0.2])
    assert von_neumann_entropy(M, "frobenius") == pytest.approx(
        float(-np.sum(lam_f * np.log(lam_f))), rel=1e-10
    )
    assert von_neumann_entropy(M, "frobenius") == pytest.approx(0.3729848, abs=1e-6)
    assert von_neumann_entropy(M, "trace") == pytest.approx(
        float(-np.sum(lam_t * np.log(lam_t))), rel=1e-10
    )
    assert von_neumann_entropy(M, "trace") == pytest.approx(0.5004024, abs=1e-6)


def test_von_neumann_entropy_is_maximized_by_isotropic_updates():
    """Isotropic (whitened) updates beat increasingly anisotropic ones."""
    iso = von_neumann_entropy(np.eye(4), "trace")
    mild = von_neumann_entropy(np.diag([1.0, 1.0, 1.0, 0.5]), "trace")
    sharp = von_neumann_entropy(np.diag([1.0, 0.1, 0.1, 0.1]), "trace")
    assert iso > mild > sharp > 0.0
    assert iso == pytest.approx(np.log(4.0), rel=1e-10)


def test_spectralfed_weights_follow_algorithm1_lines_13_14():
    """``S~^(1) = S^(1)``; then ``S~ = mu S~ + (1-mu) S``; then normalize."""
    S1 = np.array([3.0, 1.0])
    w1, smooth1 = spectralfed_weights(S1, None, momentum=0.9)
    np.testing.assert_allclose(smooth1, S1)
    np.testing.assert_allclose(w1, [0.75, 0.25], atol=1e-12)

    S2 = np.array([1.0, 3.0])
    w2, smooth2 = spectralfed_weights(S2, smooth1, momentum=0.9)
    np.testing.assert_allclose(smooth2, [0.9 * 3 + 0.1 * 1, 0.9 * 1 + 0.1 * 3])
    np.testing.assert_allclose(w2, np.array([2.8, 1.2]) / 4.0, atol=1e-12)
    _assert_simplex(w1, "spectralfed w1")
    _assert_simplex(w2, "spectralfed w2")
    # Higher entropy -> higher weight (the paper's stated direction).
    assert w1[0] > w1[1]


def test_spectralfed_weights_degenerate_round_falls_back_to_uniform():
    w, _ = spectralfed_weights(np.zeros(3), None)
    np.testing.assert_allclose(w, np.full(3, 1 / 3), atol=1e-12)


def test_spectralfed_downweights_a_spectrally_collapsed_client():
    """Trainer-level: the entropy is read off the *final layer*, class-major.

    Client 0 gets a rank-1 final-layer update (all class rows proportional --
    the "collapsed" case), client 1 an isotropic one. SpectralFed must weight
    client 1 strictly higher. This also pins the orientation: the Gram matrix
    is ``C x C``, not ``d x d``.
    """
    split = _tiny_split(seed=4)
    cfg = FedConfig(rounds=2, gamma=1.0, n_rare=1, seed=4)
    trainer = SpectralFedTrainer(split, cfg)

    layer_sizes = trainer._model.layer_sizes
    g = init_params(layer_sizes, seed=0)
    d, C = g[-2].shape

    collapsed = [p.copy() for p in g]
    collapsed[-2] = g[-2] + np.outer(
        np.linspace(1.0, 2.0, d), np.ones(C)
    ).astype(np.float32)

    rng = np.random.default_rng(0)
    diverse = [p.copy() for p in g]
    diverse[-2] = (g[-2] + rng.standard_normal((d, C))).astype(np.float32)

    S = trainer._client_entropies([collapsed, diverse], g)
    assert S[0] == pytest.approx(0.0, abs=1e-6), (
        "a rank-1 final-layer update must have zero von Neumann entropy"
    )
    assert S[1] > 0.1
    w, _ = spectralfed_weights(S, None, trainer.momentum)
    assert w[1] > w[0]
    _assert_simplex(w, "spectralfed collapsed vs diverse")


# --------------------------------------------------------------------------- #
# 3c. CoAst -- the published equations, checked numerically
# --------------------------------------------------------------------------- #
def test_ternary_quantize_keeps_top_r_percent_as_signs():
    """CoAst Algorithm 1 / eq. 7 on a hand-computable layer.

    ``Delta = [0.5, -0.2, 0.05, 0.01]`` with ``r = 0.5`` keeps the two largest
    magnitudes (0.5 and -0.2) as their signs and zeroes the rest.
    """
    delta = [np.array([0.5, -0.2, 0.05, 0.01])]
    q = ternary_quantize_update(delta, prune_ratio=0.5)
    np.testing.assert_allclose(q[0], [1.0, -1.0, 0.0, 0.0])

    # r = 0.25 keeps exactly one entry; values are ternary, never magnitudes.
    q1 = ternary_quantize_update(delta, prune_ratio=0.25)
    np.testing.assert_allclose(q1[0], [1.0, 0.0, 0.0, 0.0])
    assert set(np.unique(q1[0])) <= {-1.0, 0.0, 1.0}

    # Per-layer, not global: a tiny second layer still keeps at least one entry.
    q2 = ternary_quantize_update(
        [np.array([100.0, 90.0]), np.array([-1e-6, 1e-9])], prune_ratio=0.5
    )
    assert np.count_nonzero(q2[1]) >= 1


def test_coast_sign_agreement_matches_eq13():
    """Eq. 13: ``sum_h sgn(q[h]) sgn(U[h])``; zeroed entries drop out.

    ``q = [1, -1, 0, 0]`` against ``U = [+, +, +, +]`` gives ``1 - 1 = 0``;
    against ``U = [+, -, -, -]`` gives ``1 + 1 = 2``.
    """
    q = [np.array([1.0, -1.0, 0.0, 0.0])]
    assert coast_sign_agreement(q, [np.ones(4)]) == pytest.approx(0.0)
    assert coast_sign_agreement(q, [np.array([1.0, -1.0, -1.0, -1.0])]) == pytest.approx(2.0)
    # Perfect anti-alignment is maximally negative.
    assert coast_sign_agreement(q, [np.array([-1.0, 1.0, 1.0, 1.0])]) == pytest.approx(-2.0)
    # Multi-layer sums across arrays.
    assert coast_sign_agreement(
        [np.array([1.0]), np.array([-1.0])], [np.array([1.0]), np.array([-1.0])]
    ) == pytest.approx(2.0)


def test_coast_fedavg_mode_leaves_the_global_model_untouched():
    """The paper-faithful mode must not change aggregation at all.

    CoAst states it "neither affects the local training procedure nor changes
    the global models"; ``weight_mode="fedavg"`` is the reading that honours
    that, and it must aggregate with plain FedAvg weights every round while
    still producing scores.
    """
    split = _tiny_split(seed=5, n_attacks=3)
    cfg = FedConfig(rounds=5, gamma=1.0, n_rare=1, seed=5, round_zero_zero=False)
    trainer = CoAstTrainer(split, cfg, k=2, weight_mode="fedavg")
    hist = trainer.train(eval_every=1)

    expected = fedavg_weights(trainer.client_sizes)
    for rnd, w in zip(hist.rounds, hist.weights):
        np.testing.assert_allclose(w, expected, atol=1e-9, err_msg=f"round {rnd}")
    # Scores are still produced -- the measurement runs regardless.
    assert np.any(trainer.contribution_scores != 0.0)


def test_coast_downweights_a_label_corrupted_client():
    """The property the method exists for: a bad contributor scores lowest.

    Three IID clients; client 1's labels are cyclically shifted, so its
    top-``r%`` update directions systematically disagree with where the global
    model actually goes over the next ``k`` rounds (eq. 13). Run in the
    paper-faithful ``weight_mode="fedavg"``, so the score is a pure
    measurement and cannot be self-fulfilling through the aggregation.
    """
    split = _iid_split_with_one_corrupt_client(seed=6, n_attacks=3, corrupt_idx=1)
    cfg = FedConfig(rounds=12, gamma=1.0, n_rare=1, seed=6, round_zero_zero=False)
    trainer = CoAstTrainer(split, cfg, k=2, weight_mode="fedavg")
    trainer.train(eval_every=12)

    scores = trainer.contribution_scores
    assert np.all(np.isfinite(scores))
    clean = [scores[0], scores[2]]
    assert scores[1] < min(clean), (
        "CoAst must rank the label-corrupted client below both clean clients; "
        f"got {scores}"
    )

    # And the score->weight map propagates that: the corrupted client is
    # down-weighted relative to both clean clients.
    w = trainer.score_weight_history[-1]
    _assert_simplex(w, "coast corrupt-client weights")
    assert w[1] < min(w[0], w[2])


def test_coast_rejects_bad_arguments():
    split = _tiny_split(seed=7)
    cfg = FedConfig(rounds=2, gamma=1.0, n_rare=1, seed=7)
    with pytest.raises(ValueError, match="weight_mode"):
        CoAstTrainer(split, cfg, weight_mode="nope")
    with pytest.raises(ValueError, match="k must be"):
        CoAstTrainer(split, cfg, k=0)
    with pytest.raises(ValueError, match="prune_ratio"):
        ternary_quantize_update([np.ones(4)], prune_ratio=0.0)
    with pytest.raises(ValueError, match="normalization"):
        von_neumann_entropy(np.eye(2), normalization="l2")
    with pytest.raises(ValueError, match="warmup_frac"):
        CELMTrainer(split, cfg, warmup_frac=0.0)
    with pytest.raises(ValueError, match="momentum"):
        SpectralFedTrainer(split, cfg, momentum=1.0)
