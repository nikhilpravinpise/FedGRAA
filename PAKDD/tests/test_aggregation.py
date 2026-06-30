"""Tests for aggregation weights and the FedGRAA weight gradient."""

import numpy as np
import pytest

from fedgraa.aggregation import (
    FedGRAA,
    _grad_wrt_weights,
    _project_simplex,
    fedavg_weights,
    static_reparam_weights,
)
from fedgraa.model import aggregate, init_params
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
        X = rng.normal(size=(n, n_features))
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
    fg = FedGRAA(gamma=0.5, eta_prime=0.1, weight_epochs=1)

    w = np.array([0.2, 0.3, 0.5])

    # Analytic gradient w.r.t. weights at this w
    P = aggregate(client_params, w)
    _, grad_P = fg._composite_grad_P(P, val)
    grad_w = _grad_wrt_weights(grad_P, client_params)

    # Central finite differences of the composite loss
    eps = 1e-6
    num = np.zeros(K)
    for k in range(K):
        wp = w.copy(); wp[k] += eps
        wm = w.copy(); wm[k] -= eps
        lp = fg.composite_loss(aggregate(client_params, wp), val)
        lm = fg.composite_loss(aggregate(client_params, wm), val)
        num[k] = (lp - lm) / (2 * eps)

    np.testing.assert_allclose(grad_w, num, atol=1e-4)


def test_fedgraa_gamma_extremes_select_objective(rng):
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    P = aggregate(client_params, np.array([0.2, 0.3, 0.5]))

    # gamma = 1 -> composite loss equals the rare-class loss L_u alone
    fg1 = FedGRAA(gamma=1.0)
    from fedgraa.aggregation import _loss_and_grad_P
    L_u, _ = _loss_and_grad_P(P, val.X_n, val.y_n)
    assert abs(fg1.composite_loss(P, val) - L_u) < 1e-9

    # gamma = 0 -> composite loss equals min(L_h, L_m)
    fg0 = FedGRAA(gamma=0.0)
    L_h, _ = _loss_and_grad_P(P, val.X_h, val.y_h)
    L_m, _ = _loss_and_grad_P(P, val.X_m, val.y_m)
    assert abs(fg0.composite_loss(P, val) - min(L_h, L_m)) < 1e-9


def test_fedgraa_update_runs_and_changes_weights(rng):
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    fg = FedGRAA(gamma=1.0, eta_prime=0.1, weight_epochs=5)
    w0 = static_reparam_weights([100, 50, 10])
    w1 = fg.update(client_params, w0, val)
    assert w1.shape == w0.shape
    assert not np.allclose(w0, w1)  # weights actually move


# --------------------------------------------------------------------------- #
# Gap 1.1: simplex projection
# --------------------------------------------------------------------------- #

def test_project_simplex_on_valid_input():
    """A already-valid distribution should be returned unchanged."""
    w = np.array([0.2, 0.5, 0.3])
    p = _project_simplex(w)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)
    assert np.all(p >= 0)
    np.testing.assert_allclose(p, w, atol=1e-10)


def test_project_simplex_clips_negatives():
    """Negative weights must be zeroed and remainder redistributed."""
    w = np.array([-0.5, 1.2, 0.3])
    p = _project_simplex(w)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)
    assert np.all(p >= 0)


def test_project_simplex_uniform_when_all_equal():
    w = np.array([0.0, 0.0, 0.0])
    p = _project_simplex(w)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)
    assert np.all(p >= -1e-12)


def test_fedgraa_update_respects_simplex(rng):
    """Gap 1.1: updated weights must be non-negative and sum to 1."""
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    # Use a large learning rate to stress-test the projection.
    fg = FedGRAA(gamma=1.0, eta_prime=2.0, weight_epochs=20)
    w0 = static_reparam_weights([100, 50, 10])
    w1 = fg.update(client_params, w0, val)
    assert np.all(w1 >= -1e-12), f"Weights contain negatives: {w1}"
    np.testing.assert_allclose(w1.sum(), 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Gap 2.1: Adam reset each round
# --------------------------------------------------------------------------- #

def test_fedgraa_update_is_deterministic_across_calls(rng):
    """Gap 2.1: two calls with the same inputs must return the same weights."""
    layer_sizes = (6, 7, 5, 4)
    client_params = [init_params(layer_sizes, seed=s) for s in range(3)]
    val = _make_val_split(rng, n_features=6, n_classes=4)
    fg = FedGRAA(gamma=0.5, eta_prime=0.1, weight_epochs=3)
    w0 = static_reparam_weights([100, 50, 10])

    w_first  = fg.update(client_params, w0.copy(), val)
    w_second = fg.update(client_params, w0.copy(), val)  # same starting weights
    np.testing.assert_allclose(
        w_first, w_second, atol=1e-12,
        err_msg="FedGRAA.update() is not deterministic — Adam state not reset",
    )
