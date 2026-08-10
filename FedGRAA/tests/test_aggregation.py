"""Tests for aggregation weights and the FedGRAA weight gradient."""

import numpy as np
import pytest

from fedgraa.aggregation import (
    FedGRAA,
    _grad_wrt_weights,
    fedavg_weights,
    static_reparam_weights,
)
from fedgraa.model import MLP, aggregate, init_params
from fedgraa.partition import ValidationSplit


def test_fedavg_weights_proportional_to_size():
    w = fedavg_weights([10, 30, 60])
    np.testing.assert_allclose(w, [0.1, 0.3, 0.6])
    assert abs(w.sum() - 1.0) < 1e-12


def test_static_weights_inverse_and_normalized():
    sizes = [1, 2, 4]
    w = static_reparam_weights(sizes)
    assert abs(w.sum() - 1.0) < 1e-12
    # smallest dataset must get the largest weight
    assert w[0] > w[1] > w[2]
    # exact inverse-proportional values
    inv = np.array([1.0, 0.5, 0.25])
    np.testing.assert_allclose(w, inv / inv.sum())


def _make_val_split(rng, n_features, n_classes):
    def block(n, lo, hi):
        X = rng.normal(size=(n, n_features)).astype(np.float32)
        y = rng.integers(lo, hi, size=n)
        return X, y

    Xh, yh = block(15, 0, n_classes)
    Xm, ym = block(12, 0, n_classes)
    Xn, yn = block(9, 0, n_classes)
    return ValidationSplit(Xh, yh, Xm, ym, Xn, yn, rare_class_ids=[0])


def test_fedgraa_weight_gradient_matches_finite_difference(rng):
    layer_sizes = (6, 7, 5, 4)
    K = 3
    client_params = [init_params(layer_sizes, seed=s) for s in range(K)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    fg = FedGRAA(
        gamma=0.5, eta_prime=0.1, weight_epochs=1, layer_sizes=layer_sizes
    )

    w = np.array([0.2, 0.3, 0.5])

    # Analytic gradient w.r.t. weights at this w
    P = aggregate(client_params, w)
    _, grad_P = fg._composite_grad_P(P, val)
    grad_w = _grad_wrt_weights(grad_P, client_params)

    # Central finite differences of the composite loss (float32, eps=1e-5 is
    # the sweet spot between truncation and round-off error).
    eps = 1e-5
    num = np.zeros(K)
    for k in range(K):
        wp = w.copy(); wp[k] += eps
        wm = w.copy(); wm[k] -= eps
        lp = fg.composite_loss(aggregate(client_params, wp), val)
        lm = fg.composite_loss(aggregate(client_params, wm), val)
        num[k] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(grad_w, num, atol=5e-3)


def test_fedgraa_gamma_extremes_select_objective(rng):
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    P = aggregate(client_params, np.array([0.2, 0.3, 0.5]))

    # gamma = 1 -> composite loss equals the rare-class loss L_u alone
    fg1 = FedGRAA(gamma=1.0, layer_sizes=layer_sizes)
    model = MLP(layer_sizes, seed=0)
    from fedgraa.aggregation import _loss_and_grad_P
    L_u, _ = _loss_and_grad_P(model, P, val.X_n, val.y_n)
    assert abs(fg1.composite_loss(P, val) - L_u) < 1e-5

    # gamma = 0 -> composite loss equals min(L_h, L_m)
    fg0 = FedGRAA(gamma=0.0, layer_sizes=layer_sizes)
    L_h, _ = _loss_and_grad_P(model, P, val.X_h, val.y_h)
    L_m, _ = _loss_and_grad_P(model, P, val.X_m, val.y_m)
    assert abs(fg0.composite_loss(P, val) - min(L_h, L_m)) < 1e-5


def test_fedgraa_update_runs_and_changes_weights(rng):
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    fg = FedGRAA(
        gamma=1.0, eta_prime=0.1, weight_epochs=5, layer_sizes=layer_sizes
    )
    w0 = static_reparam_weights([100, 50, 10])
    w1 = fg.update(client_params, w0, val)
    assert w1.shape == w0.shape
    assert not np.allclose(w0, w1)  # weights actually move


# --------------------------------------------------------------------------- #
# Gap 2.1: Adam reset each round
# --------------------------------------------------------------------------- #

def test_fedgraa_default_hyperparameters_drift_off_simplex():
    """Regression test locking in the root cause documented in
    ``docs/fedgraa_divergence_diagnosis.md``: with the paper's own default
    hyperparameters (eta'=0.1, E'=5) and no simplex projection (deliberate,
    see strict_reproduction_report.md Sec 2.2), repeated FedGRAA weight
    updates against a validation split with a small rare-class pool (mirroring
    real CIC-IDS-2017's 11-36-sample rare classes) drive `w` progressively
    further from the simplex (`sum(w) != 1`, `min(w) < 0`) within a handful of
    rounds. This is the measured mechanism behind FedGRAA's divergence on real
    data; if a future change to FedGRAA/Adam/aggregate silently made this
    stable, that would be a meaningful behavioural change worth noticing.

    Uses a locally-seeded RNG (not the shared ``rng`` fixture) because the
    escalating drift is a property of this specific data realization, verified
    empirically; it is not claimed to hold for arbitrary seeds within 8 rounds.
    """
    local_rng = np.random.default_rng(0)
    layer_sizes = (6, 7, 5, 4)
    K = 5
    client_params = [init_params(layer_sizes, seed=s) for s in range(K)]

    Xh = local_rng.normal(size=(50, 6)).astype(np.float32)
    yh = local_rng.integers(0, 4, size=50)
    Xm = local_rng.normal(size=(40, 6)).astype(np.float32)
    ym = local_rng.integers(0, 4, size=40)
    Xn = local_rng.normal(size=(6, 6)).astype(np.float32)
    yn = local_rng.integers(0, 4, size=6)
    val = ValidationSplit(Xh, yh, Xm, ym, Xn, yn, rare_class_ids=[0])

    fg = FedGRAA(gamma=1.0, eta_prime=0.1, weight_epochs=5, layer_sizes=layer_sizes)
    w = static_reparam_weights([100, 50, 30, 20, 11])
    assert abs(w.sum() - 1.0) < 1e-6  # starts on the simplex (Static Reparam init)

    sum_deviation = []
    for _ in range(8):
        w = fg.update(client_params, w, val)
        sum_deviation.append(abs(float(w.sum()) - 1.0))

    # The deviation from sum(w)=1 must grow, not stay bounded near 0 --
    # this is what "no simplex projection" means in practice under the
    # paper's own default learning rate.
    assert sum_deviation[-1] > 1.0, (
        "expected sum(w) to have drifted well off the simplex after 8 rounds "
        f"of default-hyperparameter FedGRAA updates; got deviations {sum_deviation}"
    )
    assert sum_deviation[-1] > sum_deviation[0], (
        "expected the simplex deviation to grow round-over-round, not shrink"
    )


def test_fedgraa_reduced_eta_prime_stays_near_simplex():
    """Mirror of ``test_fedgraa_default_hyperparameters_drift_off_simplex``:
    the same scenario, same seed, same client params and validation split,
    but with ``eta'=0.001`` (vs. the paper's default 0.1) stays close to the
    simplex instead of drifting away. This is the mechanism behind the
    sensitivity-sweep finding in ``docs/fedgraa_divergence_diagnosis.md`` §6
    (smaller eta' avoids the blowup) reproduced at unit-test scale/speed.
    """
    local_rng = np.random.default_rng(0)
    layer_sizes = (6, 7, 5, 4)
    K = 5
    client_params = [init_params(layer_sizes, seed=s) for s in range(K)]

    Xh = local_rng.normal(size=(50, 6)).astype(np.float32)
    yh = local_rng.integers(0, 4, size=50)
    Xm = local_rng.normal(size=(40, 6)).astype(np.float32)
    ym = local_rng.integers(0, 4, size=40)
    Xn = local_rng.normal(size=(6, 6)).astype(np.float32)
    yn = local_rng.integers(0, 4, size=6)
    val = ValidationSplit(Xh, yh, Xm, ym, Xn, yn, rare_class_ids=[0])

    fg = FedGRAA(gamma=1.0, eta_prime=0.001, weight_epochs=1, layer_sizes=layer_sizes)
    w = static_reparam_weights([100, 50, 30, 20, 11])

    sum_deviation = []
    for _ in range(8):
        w = fg.update(client_params, w, val)
        sum_deviation.append(abs(float(w.sum()) - 1.0))
        assert w.min() > -1e-6, "expected all weights to stay non-negative"

    assert sum_deviation[-1] < 0.1, (
        "expected sum(w) to stay close to the simplex with a small eta'; "
        f"got deviations {sum_deviation}"
    )


def test_fedgraa_update_is_deterministic_across_calls(rng):
    """Gap 2.1: two calls with the same inputs must return the same weights."""
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    fg = FedGRAA(
        gamma=0.5, eta_prime=0.1, weight_epochs=3, layer_sizes=layer_sizes
    )
    w0 = static_reparam_weights([100, 50, 10])

    w_first  = fg.update(client_params, w0.copy(), val)
    w_second = fg.update(client_params, w0.copy(), val)  # same starting weights
    np.testing.assert_allclose(
        w_first, w_second, atol=1e-12,
        err_msg="FedGRAA.update() is not deterministic — Adam state not reset",
    )
