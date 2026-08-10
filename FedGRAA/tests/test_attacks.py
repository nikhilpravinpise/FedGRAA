"""Tests for the T7 attack layer (fedgraa/attacks.py), per the master plan's
§2.7 test spec (docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md).
Every attack is a pure function -- tested in isolation, no trainer/aggregator
needed, mirroring this project's own "pure math function" testing convention
(fedgraa/baselines_classic.py's scaffold_option2_control_variate etc.).
"""

from __future__ import annotations

import numpy as np
import pytest

from fedgraa.aggregation import _grad_wrt_weights
from fedgraa.attacks import (
    adaptive_attack,
    camouflage_sim_adjust,
    label_flip_attack,
    model_poison_attack,
    pill_blueprint_search,
    rare_class_signature_attack,
)
from fedgraa.model import Adam


# --------------------------------------------------------------------------- #
# D1a: label_flip_attack
# --------------------------------------------------------------------------- #
def test_label_flip_changes_exactly_the_targeted_fraction():
    rng = np.random.default_rng(0)
    y = np.array([1] * 100 + [0] * 100)
    y2 = label_flip_attack(y, {1: 0}, fraction=0.3, rng=rng)
    n_flipped = int(np.sum((y == 1) & (y2 == 0)))
    assert n_flipped == 30


def test_label_flip_untargeted_labels_unaffected():
    rng = np.random.default_rng(1)
    y = np.array([2] * 50 + [1] * 50)
    y2 = label_flip_attack(y, {1: 0}, fraction=1.0, rng=rng)
    # class 2 (not in flip_map) must be completely untouched
    assert np.all(y2[y == 2] == 2)
    # class 1 fully flipped to 0 at fraction=1.0
    assert np.all(y2[y == 1] == 0)


def test_label_flip_fraction_out_of_range_raises():
    rng = np.random.default_rng(0)
    y = np.array([1, 1, 1])
    with pytest.raises(ValueError):
        label_flip_attack(y, {1: 0}, fraction=1.5, rng=rng)
    with pytest.raises(ValueError):
        label_flip_attack(y, {1: 0}, fraction=-0.1, rng=rng)


def test_label_flip_deterministic_given_same_seed():
    y = np.array([1] * 40 + [0] * 40)
    y_a = label_flip_attack(y, {1: 0}, 0.5, np.random.default_rng(42))
    y_b = label_flip_attack(y, {1: 0}, 0.5, np.random.default_rng(42))
    assert np.array_equal(y_a, y_b)


def test_label_flip_zero_fraction_is_no_op():
    rng = np.random.default_rng(0)
    y = np.array([1, 1, 1, 0, 0])
    y2 = label_flip_attack(y, {1: 0}, fraction=0.0, rng=rng)
    assert np.array_equal(y, y2)


# --------------------------------------------------------------------------- #
# D1b: pill_blueprint_search / model_poison_attack
# --------------------------------------------------------------------------- #
def _toy_params(rng, sizes=(4, 5, 3)):
    params = []
    for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
        params.append(rng.normal(size=(fan_in, fan_out)).astype(np.float32))
        params.append(rng.normal(size=fan_out).astype(np.float32))
    return params


def test_pill_blueprint_selects_n_classes_in_last_two_layers():
    rng = np.random.default_rng(0)
    params = _toy_params(rng, sizes=(4, 6, 5))  # 2 layers -> both are "last two"
    masks = pill_blueprint_search(params, n_classes=2, rng=rng)
    # layer 0: fan_out=6, both layers count as "last two" here (n_layers=2)
    assert masks[1].sum() == 2  # bias mask -> 2 selected neurons
    assert masks[3].sum() == 2


def test_pill_blueprint_selects_one_neuron_in_non_last_layers():
    rng = np.random.default_rng(0)
    params = _toy_params(rng, sizes=(4, 8, 8, 5))  # 3 layers: layer 0 is NOT last-two
    masks = pill_blueprint_search(params, n_classes=2, rng=rng)
    assert masks[1].sum() == 1  # layer 0's bias mask: exactly 1 neuron


def test_model_poison_scales_only_masked_entries_without_camouflage():
    data_rng = np.random.default_rng(0)
    params = _toy_params(data_rng)
    delta = [p * 0.1 for p in params]
    # A dedicated, untouched attack rng -- model_poison_attack's first draw
    # from it is pill_blueprint_search's own selection, so a fresh rng with
    # the same seed reproduces the identical mask.
    poisoned = model_poison_attack(delta, params, n_classes=2, scale=10.0,
                                    rng=np.random.default_rng(1))
    masks = pill_blueprint_search(params, n_classes=2, rng=np.random.default_rng(1))
    for d, p, m in zip(delta, poisoned, masks):
        np.testing.assert_allclose(p[~m], d[~m], atol=1e-6)
        # Masked entries actually changed (scale != 1)
        assert not np.allclose(p[m], d[m])


def test_model_poison_produces_finite_output():
    rng = np.random.default_rng(3)
    params = _toy_params(rng)
    delta = [p * 0.1 for p in params]
    poisoned = model_poison_attack(delta, params, n_classes=2, scale=5.0, rng=rng)
    for p in poisoned:
        assert np.all(np.isfinite(p))


def _cos(a, b):
    av = np.concatenate([np.ravel(x) for x in a])
    bv = np.concatenate([np.ravel(x) for x in b])
    return float(np.dot(av, bv) / (np.linalg.norm(av) * np.linalg.norm(bv)))


def test_camouflage_sim_adjust_is_noop_once_at_or_above_threshold():
    """A poisoned update bit-identical to the (sole) benign reference has,
    by construction, self-cosine-similarity == S_max exactly (both sides
    compute the identical `cos(x, x)` expression, so this holds robustly
    under floating point, unlike comparing two merely mathematically-equal
    but differently-rounded vectors) -- the alternating-scale loop must
    never fire."""
    rng = np.random.default_rng(0)
    params = _toy_params(rng, sizes=(3, 3))
    masks = pill_blueprint_search(params, n_classes=2, rng=rng)
    benign = [[p.copy() for p in params]]  # a single benign reference
    poisoned = [p.copy() for p in benign[0]]  # bit-identical to it
    adjusted = camouflage_sim_adjust(poisoned, masks, benign, c_up=1.5, c_down=0.5, c_iter=20)
    for a, b in zip(adjusted, poisoned):
        np.testing.assert_allclose(a, b)


def test_camouflage_sim_adjust_modifies_output_when_below_threshold():
    """A poisoned update genuinely dissimilar from the benign set is below
    S_max, so the loop must actually run and change the output (finite,
    not a no-op) -- the algorithm's own guarantee is "iterate toward the
    threshold," not a specific closed-form convergence rate, so this checks
    that the mechanism does something rather than asserting a monotonic
    similarity improvement the paper itself doesn't promise."""
    rng = np.random.default_rng(2)
    params = _toy_params(rng, sizes=(3, 3))
    masks = pill_blueprint_search(params, n_classes=2, rng=rng)
    benign = [[p * 1.0 for p in params], [p * 1.1 for p in params]]
    poisoned = [rng.normal(size=p.shape).astype(np.float32) * 50.0 for p in params]

    adjusted = camouflage_sim_adjust(poisoned, masks, benign, c_up=1.5, c_down=0.5, c_iter=20)
    for a in adjusted:
        assert np.all(np.isfinite(a))
    changed = any(not np.allclose(a, p) for a, p in zip(adjusted, poisoned))
    assert changed


def test_camouflage_sim_adjust_no_op_with_no_benign_reference():
    rng = np.random.default_rng(0)
    params = _toy_params(rng, sizes=(3, 3))
    masks = pill_blueprint_search(params, n_classes=2, rng=rng)
    result = camouflage_sim_adjust(params, masks, benign_deltas=[])
    for a, b in zip(result, params):
        np.testing.assert_allclose(a, b)


# --------------------------------------------------------------------------- #
# D1c: rare_class_signature_attack -- the mechanism-specific attack
# --------------------------------------------------------------------------- #
def test_rare_class_signature_attack_constructed_delta_exploits_grad_p():
    """Regression test per §2.7 item 3: the constructed delta's dot product
    with grad_P is more negative than an honest client's, on a synthetic
    fixture -- exactly the quantity that drives the weight update."""
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    grad_P = [rng.normal(size=p.shape).astype(np.float32) for p in params]

    fake = rare_class_signature_attack(params, grad_P, eta_fake=0.5)
    dot_fake = sum(float(np.sum(g * f)) for g, f in zip(grad_P, fake))
    dot_honest = sum(float(np.sum(g * p)) for g, p in zip(grad_P, params))
    assert dot_fake < dot_honest


def test_rare_class_signature_attack_increases_aggregation_weight():
    """Regression test per §2.7 item 4: applied through the real weight-
    gradient formula (`_grad_wrt_weights`, FedGRAA-v2's actual mechanism)
    and one Adam descent step, the attacker's weight increases more than an
    honest client's would, relative to the same starting weight."""
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    grad_P = [rng.normal(size=p.shape).astype(np.float32) for p in params]
    honest_other = [p + rng.normal(scale=0.01, size=p.shape).astype(np.float32) for p in params]

    fake = rare_class_signature_attack(params, grad_P, eta_fake=0.5)

    # dL'/dw_k = <grad_P, P_k> for each of two "clients": the honest one and
    # the attacker, both starting from weight 0.5.
    grad_w_honest = _grad_wrt_weights(grad_P, [honest_other])[0]
    grad_w_attacker = _grad_wrt_weights(grad_P, [fake])[0]

    w_honest = np.array([0.5], dtype=np.float32)
    w_attacker = np.array([0.5], dtype=np.float32)
    Adam(lr=0.1).step([w_honest], [np.array([grad_w_honest])])
    Adam(lr=0.1).step([w_attacker], [np.array([grad_w_attacker])])

    assert w_attacker[0] > w_honest[0]


def test_rare_class_signature_attack_round_one_degeneracy_is_honest():
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    fake = rare_class_signature_attack(params, grad_P=None)
    for a, b in zip(fake, params):
        np.testing.assert_allclose(a, b)


def test_rare_class_signature_attack_deterministic_given_same_seed():
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    grad_P = [rng.normal(size=p.shape).astype(np.float32) for p in params]
    a = rare_class_signature_attack(params, grad_P, eta_fake=0.5, noise_scale=0.1,
                                     rng=np.random.default_rng(7))
    b = rare_class_signature_attack(params, grad_P, eta_fake=0.5, noise_scale=0.1,
                                     rng=np.random.default_rng(7))
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)


def test_rare_class_signature_attack_eta_fake_bounds_effect_size():
    """Larger eta_fake -> the fake delta moves further from the global
    params in the -grad_P direction (sanity check on the formula's own
    monotonicity, not asserted-then-forgotten)."""
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    grad_P = [rng.normal(size=p.shape).astype(np.float32) for p in params]
    small = rare_class_signature_attack(params, grad_P, eta_fake=0.1)
    large = rare_class_signature_attack(params, grad_P, eta_fake=1.0)

    def dist(a, b):
        return sum(float(np.sum((x - y) ** 2)) for x, y in zip(a, b)) ** 0.5

    assert dist(large, params) > dist(small, params)


# --------------------------------------------------------------------------- #
# D1d: adaptive_attack
# --------------------------------------------------------------------------- #
def test_adaptive_attack_converges_to_lower_detection_score():
    """Regression test per §2.7 item 8: optimizes against a mocked defense
    score function, converges to a lower score than a non-adaptive
    (fixed-parameter) attack of the same base type."""
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    delta = [p * 0.1 for p in params]

    def defense_score(candidate):
        # Mocked "anomaly score": L2 norm of the candidate delta -- a
        # detector that flags large perturbations. Lower scale -> lower score.
        return sum(float(np.sum(x ** 2)) for x in candidate)

    def base_attack(scale):
        return model_poison_attack(delta, params, n_classes=2, scale=scale,
                                    rng=np.random.default_rng(0))

    non_adaptive_score = defense_score(base_attack(scale=50.0))
    best_params, best_score = adaptive_attack(
        base_attack, base_kwargs={},
        param_grid=[("scale", [1.0, 2.0, 5.0, 10.0, 50.0])],
        defense_fn=defense_score, n_restarts=5,
        rng=np.random.default_rng(1),
    )
    assert best_score <= non_adaptive_score


def test_adaptive_attack_deterministic_given_same_seed():
    rng = np.random.default_rng(0)
    params = _toy_params(rng)
    delta = [p * 0.1 for p in params]

    def defense_score(candidate):
        return sum(float(np.sum(x ** 2)) for x in candidate)

    def base_attack(scale):
        return model_poison_attack(delta, params, n_classes=2, scale=scale,
                                    rng=np.random.default_rng(0))

    _, score_a = adaptive_attack(
        base_attack, {}, [("scale", [1.0, 5.0, 10.0])], defense_score,
        n_restarts=4, rng=np.random.default_rng(9),
    )
    _, score_b = adaptive_attack(
        base_attack, {}, [("scale", [1.0, 5.0, 10.0])], defense_score,
        n_restarts=4, rng=np.random.default_rng(9),
    )
    assert score_a == score_b
