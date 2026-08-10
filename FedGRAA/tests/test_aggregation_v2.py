"""Tests for FedGRAA-v2's aggregation-layer mechanisms (W2.3/W2.3a, F1/F12).

Covers FEDGRAA_V2_MASTER_PLAN.md §9.2's keystone gates:

* **T-4a** -- count-weighted normalization reproduces the pooled mean
  gradient exactly; the naive (unweighted) sum does not (obstruction A).
* **T-4b** -- the arm must be selected from the *pooled* L_h/L_m, never a
  per-client local choice (obstruction B).
* **T-4**  -- the combined two-phase protocol matches the pooled reference
  to float32 tolerance (the keystone test; no v2 number is trustworthy
  before this is green).
* **T-10** -- v2 weights stay on the simplex at every round, for every ``m``.
* **T-19** -- the strict-reproduction default is unchanged by the W0.2
  promotion of ``ProjectedFedGRAA``/``FedGRAAv2`` into ``aggregation.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fedgraa.aggregation import (
    FedGRAA,
    FedGRAAv2,
    ProjectedFedGRAA,
    _composite_min_arm,
    _loss_and_grad_P,
    project_to_simplex,
    static_reparam_weights,
    two_phase_composite_grad,
)
from fedgraa.config import CIC_IDS_2017, UNSW_NB15, FedConfig
from fedgraa.data import make_synthetic_dataset
from fedgraa.model import MLP, aggregate, init_params
from fedgraa.partition import apply_tags, partition_non_iid, stratum_tags


def _rel_err(a, b) -> float:
    fa = np.concatenate([x.ravel() for x in a]).astype(np.float64)
    fb = np.concatenate([x.ravel() for x in b]).astype(np.float64)
    den = np.linalg.norm(fb)
    return float(np.linalg.norm(fa - fb) / den) if den else float(np.linalg.norm(fa - fb))


def _build_scenario(spec, gamma, seed=0):
    """One reusable (model, params, pooled val, per-client vals) scenario,
    identical in shape to scripts/verify_v2_exactness.py's ``run()``."""
    cfg = FedConfig(gamma=gamma, seed=seed)
    X, y, spec2 = make_synthetic_dataset(spec, seed=seed)
    split = partition_non_iid(
        X, y, spec2, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction, seed=cfg.seed
    )
    layer_sizes = spec2.model_config().layer_sizes
    model = MLP(layer_sizes, seed=0)
    base = init_params(layer_sizes, seed=0)
    rng = np.random.default_rng(0)
    client_params = [
        [p + 0.01 * rng.normal(size=p.shape).astype(np.float32) for p in base]
        for _ in split.clients
    ]
    w0 = static_reparam_weights([c.n_train for c in split.clients])
    params = aggregate(client_params, w0)

    rare_ids = split.val_split.rare_class_ids
    counts = np.bincount(split.y_val, minlength=spec2.n_classes)
    tags = stratum_tags(counts, rare_ids)
    client_vals = [apply_tags(c.X_val, c.y_val, tags, rare_ids) for c in split.clients]
    return model, params, split, client_params, client_vals, w0, layer_sizes


# --------------------------------------------------------------------------- #
# T-4a -- count-weighted sum equals the pooled mean gradient; naive sum does not
# --------------------------------------------------------------------------- #
def test_count_weighted_sum_equals_pooled_gradient(rng):
    """Obstruction (A) in isolation: tf.reduce_mean means a per-client mean
    gradient must be weighted by its own row count before summing, or the
    result is a materially different quantity, not a small perturbation."""
    layer_sizes = (5, 6, 4)
    model = MLP(layer_sizes, seed=0)
    params = init_params(layer_sizes, seed=0)
    X = rng.normal(size=(60, 5)).astype(np.float32)
    y = rng.integers(0, 4, size=60)

    # Uneven shards spanning nearly two orders of magnitude, mirroring the
    # real client-size skew under one-attack-per-client partitioning.
    boundaries = [0, 3, 10, 60]
    shards = [
        (X[boundaries[i]:boundaries[i + 1]], y[boundaries[i]:boundaries[i + 1]])
        for i in range(len(boundaries) - 1)
    ]

    _, pooled_grad = model.loss_and_grad(params, X, y)

    weighted_total = [np.zeros_like(p) for p in params]
    naive_total = [np.zeros_like(p) for p in params]
    n_total = 0
    for Xs, ys in shards:
        _, g = model.loss_and_grad(params, Xs, ys)
        n = len(ys)
        weighted_total = [t + n * gi for t, gi in zip(weighted_total, g)]
        naive_total = [t + gi for t, gi in zip(naive_total, g)]
        n_total += n
    weighted = [t / n_total for t in weighted_total]

    for w_g, p_g in zip(weighted, pooled_grad):
        np.testing.assert_allclose(w_g, p_g, atol=1e-5)

    diverges = any(
        not np.allclose(n_g, p_g, atol=1e-2)
        for n_g, p_g in zip(naive_total, pooled_grad)
    )
    assert diverges, "naive (unweighted) sum unexpectedly matched the pooled gradient"


# --------------------------------------------------------------------------- #
# T-4b -- the arm must be selected from the pooled L_h/L_m, never per-client
# --------------------------------------------------------------------------- #
def test_arm_selected_globally_not_per_client():
    """Obstruction (B) in isolation, on real client-shard data (CIC-IDS-2017,
    seed 0): at least one client's local argmin(L_h, L_m) disagrees with the
    pooled one (measured: 9/14 clients disagree at this seed) -- v2's global
    arm selection must match the pooled reference; a naive per-client-arm,
    count-weighted-otherwise variant must measurably diverge."""
    model, params, split, client_params, client_vals, w0, layer_sizes = (
        _build_scenario(CIC_IDS_2017, gamma=0.5, seed=0)
    )
    fg = FedGRAA(gamma=0.5, layer_sizes=layer_sizes)

    L_h_pooled, _ = _loss_and_grad_P(
        model, params, split.val_split.X_h, split.val_split.y_h
    )
    L_m_pooled, _ = _loss_and_grad_P(
        model, params, split.val_split.X_m, split.val_split.y_m
    )
    _, global_arm = _composite_min_arm(L_h_pooled, L_m_pooled)

    n_disagree = 0
    for v in client_vals:
        L_h, _ = _loss_and_grad_P(model, params, v.X_h, v.y_h)
        L_m, _ = _loss_and_grad_P(model, params, v.X_m, v.y_m)
        _, local_arm = _composite_min_arm(L_h, L_m)
        if local_arm is not None and local_arm != global_arm:
            n_disagree += 1
    assert n_disagree >= 1, "test fixture no longer produces a local/global arm disagreement"

    V0 = fg._composite_grad_P(params, split.val_split)[1]
    V_global, _stats = two_phase_composite_grad(model, params, client_vals, gamma=0.5)
    assert _rel_err(V_global, V0) < 1e-5

    # Count-weighted (A fixed) but each client selects its OWN local arm (B
    # not fixed) -- a clean isolation of obstruction B alone.
    zero_like = [np.zeros_like(p) for p in params]
    g_num, n_den = list(zero_like), 0
    g_u_num, n_u_den = list(zero_like), 0
    for v in client_vals:
        L_h, _ = _loss_and_grad_P(model, params, v.X_h, v.y_h)
        L_m, _ = _loss_and_grad_P(model, params, v.X_m, v.y_m)
        _, local_arm = _composite_min_arm(L_h, L_m)
        Xs, ys = (v.X_h, v.y_h) if local_arm == "h" else (
            (v.X_m, v.y_m) if local_arm == "m" else (None, None)
        )
        if Xs is not None and len(ys):
            _, g = _loss_and_grad_P(model, params, Xs, ys)
            g_num = [a + len(ys) * gi for a, gi in zip(g_num, g)]
            n_den += len(ys)
        if len(v.y_n):
            _, g = _loss_and_grad_P(model, params, v.X_n, v.y_n)
            g_u_num = [a + len(v.y_n) * gi for a, gi in zip(g_u_num, g)]
            n_u_den += len(v.y_n)
    g_a = [t / n_den for t in g_num] if n_den else zero_like
    g_u = [t / n_u_den for t in g_u_num] if n_u_den else zero_like
    V_local_arm = [0.5 * a + 0.5 * u for a, u in zip(g_a, g_u)]

    assert _rel_err(V_local_arm, V0) > 1e-2, (
        "per-client local-arm variant unexpectedly matched the pooled gradient "
        "-- obstruction B should be measurable given a real arm disagreement"
    )


# --------------------------------------------------------------------------- #
# T-4 -- the combined protocol matches the pooled reference (the keystone test)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [CIC_IDS_2017, UNSW_NB15])
@pytest.mark.parametrize("gamma", [1.0, 0.5, 0.0])
def test_v2_two_phase_protocol_matches_pooled_composite_grad(spec, gamma):
    model, params, split, client_params, client_vals, w0, layer_sizes = (
        _build_scenario(spec, gamma=gamma, seed=0)
    )
    fg = FedGRAA(gamma=gamma, layer_sizes=layer_sizes)
    V0 = fg._composite_grad_P(params, split.val_split)[1]
    V3, stats = two_phase_composite_grad(model, params, client_vals, gamma)
    assert _rel_err(V3, V0) < 1e-6
    assert stats["rounds"] == 2


def test_v2_degenerate_empty_stratum_still_matches_pooled():
    """The n_s=0 -> L_s=inf corner (F3 measured this reachable: 2/10 CIC
    seeds have an empty D_V^n) must not evaluate 0/0."""
    from fedgraa.partition import ValidationSplit

    def drop_stratum(v, which):
        e_X, e_y = v.X_h[:0], v.y_h[:0]
        parts = {"h": (v.X_h, v.y_h), "m": (v.X_m, v.y_m), "n": (v.X_n, v.y_n)}
        parts[which] = (e_X, e_y)
        return ValidationSplit(*parts["h"], *parts["m"], *parts["n"], v.rare_class_ids)

    model, params, split, client_params, client_vals, w0, layer_sizes = (
        _build_scenario(CIC_IDS_2017, gamma=0.5, seed=0)
    )
    fg = FedGRAA(gamma=0.5, layer_sizes=layer_sizes)

    for which in ("n", "h", "m"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pooled = drop_stratum(split.val_split, which)
            per_client = [drop_stratum(v, which) for v in client_vals]
            V0 = fg._composite_grad_P(params, pooled)[1]
            V3, _ = two_phase_composite_grad(model, params, per_client, gamma=0.5)
        assert _rel_err(V3, V0) < 1e-6, which


def test_two_phase_composite_grad_rejects_empty_client_list():
    with pytest.raises(ValueError):
        two_phase_composite_grad(
            MLP((3, 4, 2), seed=0), init_params((3, 4, 2), seed=0), [], gamma=0.5
        )


# --------------------------------------------------------------------------- #
# FedGRAAv2 -- refresh-interval bookkeeping (F5) and simplex projection
# --------------------------------------------------------------------------- #
def _make_val_split(rng, n_features, n_classes):
    from fedgraa.partition import ValidationSplit

    def block(n, lo, hi):
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        y = rng.integers(lo, hi, size=n)
        return X, y

    Xh, yh = block(15, 0, n_classes)
    Xm, ym = block(12, 0, n_classes)
    Xn, yn = block(9, 0, n_classes)
    return ValidationSplit(Xh, yh, Xm, ym, Xn, yn, rare_class_ids=[0])


@pytest.mark.parametrize("weight_epochs,m,expected_refreshes", [
    (5, 1, 5), (5, 2, 3), (5, 3, 2), (5, 5, 1), (1, 1, 1),
])
def test_fedgraav2_refresh_count_matches_ceil_formula(
    rng, weight_epochs, m, expected_refreshes
):
    """The number of grad_provider calls per round must be exactly
    ceil(weight_epochs / m) -- the quantity T-12 bills bytes against."""
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    w0 = static_reparam_weights([100, 50, 10])

    val = _make_val_split(rng, n_features=6, n_classes=4)
    oracle_fg = FedGRAA(gamma=0.5, layer_sizes=layer_sizes)
    call_count = {"n": 0}

    def grad_provider(P):
        call_count["n"] += 1
        return oracle_fg._composite_grad_P(P, val)[1]

    fg2 = FedGRAAv2(
        gamma=0.5, eta_prime=0.1, weight_epochs=weight_epochs,
        layer_sizes=layer_sizes, grad_refresh_every=m,
    )
    fg2.update(client_params, w0, grad_provider)
    assert call_count["n"] == expected_refreshes
    assert fg2.last_n_refreshes == expected_refreshes


def test_fedgraav2_rejects_invalid_refresh_interval():
    with pytest.raises(ValueError):
        FedGRAAv2(gamma=0.5, layer_sizes=(4, 3), grad_refresh_every=0)


def test_fedgraav2_output_is_on_simplex(rng):
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    w0 = static_reparam_weights([100, 50, 10])
    val = _make_val_split(rng, n_features=6, n_classes=4)
    oracle_fg = FedGRAA(gamma=1.0, layer_sizes=layer_sizes)

    def grad_provider(P):
        return oracle_fg._composite_grad_P(P, val)[1]

    fg2 = FedGRAAv2(
        gamma=1.0, eta_prime=0.1, weight_epochs=5,
        layer_sizes=layer_sizes, grad_refresh_every=1,
    )
    w1 = fg2.update(client_params, w0, grad_provider)
    assert abs(float(w1.sum()) - 1.0) < 1e-6
    assert w1.min() >= -1e-7


def test_fedgraav2_at_m1_matches_projected_fedgraa_given_identical_gradient():
    """T-4's downstream claim: when grad_refresh_every=1 (no staleness) and
    the injected grad_provider always returns the SAME pooled composite
    gradient a strict ProjectedFedGRAA.update() would compute internally,
    FedGRAAv2 must produce the identical weight trajectory."""
    rng = np.random.default_rng(3)
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    w0 = static_reparam_weights([100, 50, 10])

    ref = ProjectedFedGRAA(gamma=0.5, eta_prime=0.1, weight_epochs=5, layer_sizes=layer_sizes)
    w_ref = ref.update(client_params, w0.copy(), val)

    oracle_fg = FedGRAA(gamma=0.5, layer_sizes=layer_sizes)

    def grad_provider(P):
        return oracle_fg._composite_grad_P(P, val)[1]

    v2 = FedGRAAv2(
        gamma=0.5, eta_prime=0.1, weight_epochs=5,
        layer_sizes=layer_sizes, grad_refresh_every=1,
    )
    w_v2 = v2.update(client_params, w0.copy(), grad_provider)
    np.testing.assert_allclose(w_v2, w_ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# T-19 -- the strict-reproduction default is unchanged by the W0.2 promotion
# --------------------------------------------------------------------------- #
def test_strict_reproduction_default_unchanged():
    """Guards the no-silent-patching mandate: FederatedTrainer(method="fedgraa")
    must still construct the unprojected, strict-reproduction FedGRAA -- never
    ProjectedFedGRAA/FedGRAAv2 -- despite both now living in the same module."""
    from fedgraa.trainer import FederatedTrainer

    X, y, spec = make_synthetic_dataset(CIC_IDS_2017, seed=0)
    cfg = FedConfig(rounds=1, gamma=0.5, n_rare=3, seed=0)
    split = partition_non_iid(X, y, spec, n_rare=cfg.n_rare, val_fraction=cfg.val_fraction, seed=0)
    trainer = FederatedTrainer(split, cfg, method="fedgraa")
    assert type(trainer.fedgraa) is FedGRAA
    assert not isinstance(trainer.fedgraa, ProjectedFedGRAA)


def test_project_to_simplex_is_idempotent_on_the_simplex():
    w = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    projected = project_to_simplex(w)
    np.testing.assert_allclose(projected, w, atol=1e-6)


def test_project_to_simplex_handles_negative_and_unnormalized_input():
    w = np.array([-1.0, 2.0, 0.5], dtype=np.float32)
    projected = project_to_simplex(w)
    assert abs(float(projected.sum()) - 1.0) < 1e-5
    assert projected.min() >= -1e-7
