"""T7 attack layer (docs/research_c/STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md
§2; judgment calls in
docs/research_c/STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md, IDs JC-T7-1..15).

Four attacks (D1a-d), in ascending order of novelty, each an attacker-client
wrapper that intercepts a client's honestly-trained update *before* it
reaches the aggregator -- orthogonal to the trainer/aggregator split, so any
trainer's per-client output can be poisoned without touching the trainer
itself:

* :func:`label_flip_attack` -- D1a, the field-standard baseline attack.
* :func:`model_poison_attack` -- D1b, "pill"-style parameter-space
  perturbation (Guo et al., AAAI 2026, arXiv:2407.15389). Core mechanism
  only (JC-T7-8): blueprint neuron selection (rule 1/2) via greedy
  max-weight search, amplification of the client's own real delta at those
  neurons, and SimAdjust camouflage (Algorithm 2). The six named "dynamic
  search pattern" combinations and the disconnection-update refinement are
  not implemented (JC-T7-8).
* :func:`rare_class_signature_attack` -- D1c, **the mechanism-specific
  attack, novel to this project**: exploits FedGRAA-v2's own printed
  weight-update gradient (`fedgraa.aggregation._grad_wrt_weights`:
  `dL'/dw_k = <grad_P, P_k>`) directly, per JC-T7-13/15.
* :func:`adaptive_attack` -- D1d, an AutoAdapt-style wrapper (Krauß et al.,
  NDSS 2024, not read this pass -- JC-T7-14) that optimizes a base attack's
  parameters against a *given* defense score function via random-restart
  local search.

Every attack is a pure function of its inputs (no hidden state, no RNG
default without an explicit seed argument), so each is independently
testable without touching the trainer or aggregator.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .model import Params

__all__ = [
    "label_flip_attack",
    "pill_blueprint_search",
    "camouflage_sim_adjust",
    "model_poison_attack",
    "rare_class_signature_attack",
    "adaptive_attack",
    "wrap_model_poison_attack",
]


# --------------------------------------------------------------------------- #
# D1a -- label flipping (field-standard baseline)
# --------------------------------------------------------------------------- #
def label_flip_attack(
    y: np.ndarray,
    flip_map: Dict[int, int],
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Flip a ``fraction`` of samples whose label is a key of ``flip_map`` to
    that key's mapped value. Labels not in ``flip_map`` and the
    ``(1-fraction)`` majority of matching samples are left untouched.

    ``fraction`` must be in ``[0, 1]``. A deterministic, seeded selection
    (via ``rng``) of *which* matching rows flip, not the first ``fraction``
    by index -- avoids a systematic bias toward whichever rows happen to
    sort first.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    y_out = np.asarray(y).copy()
    for source, target in flip_map.items():
        idx = np.flatnonzero(y_out == source)
        if idx.size == 0:
            continue
        n_flip = int(round(fraction * idx.size))
        if n_flip == 0:
            continue
        chosen = rng.choice(idx, size=n_flip, replace=False)
        y_out[chosen] = target
    return y_out


# --------------------------------------------------------------------------- #
# D1b -- "pill"-style model poisoning (Guo et al., AAAI 2026)
# --------------------------------------------------------------------------- #
def pill_blueprint_search(
    params_ref: Params, n_classes: int, rng: np.random.Generator,
) -> Params:
    """Rule-1/rule-2 blueprint neuron selection via greedy max-weight search
    (Guo et al. §4.2, JC-T7-8): one neuron per linear layer except the last
    two, which each select ``n_classes`` neurons. Layer 0's selection is a
    random start point; every subsequent layer's selection is the top
    ``N^p_i`` neurons by summed incoming weight from the previous layer's
    selected neurons (the paper's own "approximate max pill search").

    Returns a boolean-mask ``Params`` (same shapes as ``params_ref``, one
    mask array per ``[W, b]`` pair) marking the selected ("pill") entries.
    """
    n_layers = len(params_ref) // 2
    if n_layers == 0:
        return []
    masks: Params = []
    prev_idx: Optional[np.ndarray] = None
    for i in range(n_layers):
        W = params_ref[2 * i]
        fan_out = W.shape[1]
        is_last_two = i >= n_layers - 2
        n_select = min(n_classes if is_last_two else 1, fan_out)
        if prev_idx is None:
            idx = rng.choice(fan_out, size=n_select, replace=False)
        else:
            scores = np.sum(np.abs(W[prev_idx, :]), axis=0)
            idx = np.argsort(-scores)[:n_select]
        w_mask = np.zeros_like(W, dtype=bool)
        w_mask[:, idx] = True
        b_mask = np.zeros(fan_out, dtype=bool)
        b_mask[idx] = True
        masks.append(w_mask)
        masks.append(b_mask)
        prev_idx = idx
    return masks


def _flatten(p: Params) -> np.ndarray:
    return np.concatenate([np.ravel(np.asarray(a, dtype=np.float64)) for a in p])


def _cosine_sim(a: Params, b: Params) -> float:
    av, bv = _flatten(a), _flatten(b)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    return float(np.dot(av, bv) / denom) if denom > 0 else 0.0


def camouflage_sim_adjust(
    poisoned: Params,
    masks: Params,
    benign_deltas: Sequence[Params],
    c_up: float = 1.05,
    c_down: float = 0.95,
    c_iter: int = 10,
) -> Params:
    """Algorithm 2's SimAdjust (Guo et al., JC-T7-7): iteratively scale the
    non-pill parameters up by ``c_up`` (even iterations) and the pill
    parameters down by ``c_down`` (odd iterations) until the poisoned
    update's cosine similarity to the benign-mean update meets or exceeds
    ``S_max = max(0, max_i Sim(benign_mean, benign_deltas[i]))``, or
    ``c_iter`` iterations elapse. A no-op if ``benign_deltas`` is empty (no
    reference to camouflage against, e.g. a single-client synthetic
    fixture).

    Only SimAdjust is implemented, not the paper's DistAdjust variant
    (JC-T7-7's scope note): SimAdjust already supplies the
    detection-evasion axis the T7 harness needs.
    """
    if not benign_deltas:
        return [p.copy() for p in poisoned]
    n_arrays = len(poisoned)
    benign_mean: Params = [
        np.mean([bd[j] for bd in benign_deltas], axis=0) for j in range(n_arrays)
    ]
    s_max = max(0.0, max(_cosine_sim(benign_mean, bd) for bd in benign_deltas))

    current = [p.copy() for p in poisoned]
    for it in range(c_iter):
        if _cosine_sim(benign_mean, current) >= s_max:
            break
        if it % 2 == 0:
            current = [
                (m.astype(np.float32) * 1.0 + (~m).astype(np.float32) * c_up) * d
                for d, m in zip(current, masks)
            ]
        else:
            current = [
                (m.astype(np.float32) * c_down + (~m).astype(np.float32) * 1.0) * d
                for d, m in zip(current, masks)
            ]
    return current


def model_poison_attack(
    delta: Params,
    params_ref: Params,
    n_classes: int,
    scale: float = 10.0,
    rng: Optional[np.random.Generator] = None,
    benign_deltas: Sequence[Params] = (),
) -> Params:
    """"Pill"-style model-poisoning attack (D1b): amplify a client's own
    real trained delta at the blueprint-selected ("pill") neurons by
    ``scale``, then camouflage the result against ``benign_deltas`` (if
    any) via :func:`camouflage_sim_adjust`.

    ``delta`` is the attacker's own honestly-computed parameter delta
    (post-training minus pre-round global params) -- the attack perturbs it,
    it does not fabricate an update from nothing.
    """
    rng = rng if rng is not None else np.random.default_rng()
    masks = pill_blueprint_search(params_ref, n_classes, rng)
    poisoned = [d.copy() for d in delta]
    for j, m in enumerate(masks):
        poisoned[j][m] = poisoned[j][m] * scale
    return camouflage_sim_adjust(poisoned, masks, benign_deltas)


# --------------------------------------------------------------------------- #
# D1c -- the mechanism-specific attack (novel to this project, JC-T7-13/15)
# --------------------------------------------------------------------------- #
def rare_class_signature_attack(
    params_global: Params,
    grad_P: Optional[Params],
    eta_fake: float = 0.5,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Params:
    """FedGRAA-v2's own attack surface: fabricates client params that exploit
    ``dL'/dw_k = <grad_P, P_k>`` (``fedgraa.aggregation._grad_wrt_weights``)
    directly, per JC-T7-13.

    Returns ``P_global - eta_fake * grad_P + noise`` (params, not a delta --
    matches ``client_train_round``'s own return convention). By
    construction, ``<grad_P, P_fake> = <grad_P, P_global> -
    eta_fake*||grad_P||^2``, strictly more negative than an honest client's
    contribution in expectation, which Adam's descent on the aggregation
    weight ``w`` turns directly into a weight increase -- the closed-form
    property ``test_rare_class_signature_attack_increases_aggregation_weight``
    pins as a regression test, not merely asserted.

    ``grad_P=None`` (round-1 degeneracy, or any round before the attacker
    has observed a broadcast ``theta'_g``, JC-T7-15) falls back to an
    **honest, unmodified contribution** (``params_global`` unchanged) --
    there is nothing to exploit yet.
    """
    if grad_P is None:
        return [p.copy() for p in params_global]
    rng = rng if rng is not None else np.random.default_rng()
    out: Params = []
    for p, g in zip(params_global, grad_P):
        fake = p - eta_fake * g
        if noise_scale > 0:
            fake = fake + rng.normal(scale=noise_scale, size=fake.shape).astype(fake.dtype)
        out.append(fake.astype(np.float32))
    return out


# --------------------------------------------------------------------------- #
# D1d -- adaptive attacker (AutoAdapt-style, JC-T7-14)
# --------------------------------------------------------------------------- #
def adaptive_attack(
    base_attack: Callable[..., Params],
    base_kwargs: dict,
    param_grid: Sequence[Tuple[str, Sequence[float]]],
    defense_fn: Callable[[Params], float],
    n_restarts: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Params, float]:
    """Adaptive attacker (D1d): optimizes ``base_attack``'s numeric
    parameters (named in ``param_grid`` as ``(kwarg_name, candidate_values)``
    pairs) against a given ``defense_fn`` score (lower = more evasive, e.g.
    an anomaly score from a T7 defense reproduction) via random-restart
    local search over the grid -- not AutoAdapt's own specific optimizer
    (JC-T7-14, not read this pass), but a genuine adaptive-search mechanic:
    it converges to whichever grid point ``defense_fn`` scores lowest, which
    is what ``test_adaptive_attack_converges_to_lower_detection_score``
    (§2.7 item 8) pins.

    Returns ``(best_params, best_score)``.
    """
    rng = rng if rng is not None else np.random.default_rng()
    best_params: Optional[Params] = None
    best_score = float("inf")
    keys = [k for k, _ in param_grid]
    for _ in range(max(1, n_restarts)):
        trial_kwargs = dict(base_kwargs)
        for key, values in param_grid:
            trial_kwargs[key] = values[int(rng.integers(len(values)))]
        candidate = base_attack(**trial_kwargs)
        score = defense_fn(candidate)
        if score < best_score:
            best_score = score
            best_params = candidate
    assert best_params is not None
    return best_params, best_score


# --------------------------------------------------------------------------- #
# Uniform (P_current, P_global) -> P_poisoned adapter, used by the T7
# defense-reproduction trainers (fedgraa/baselines_t7_adversarial.py) so
# they never special-case an individual attack's own argument shape.
# --------------------------------------------------------------------------- #
def wrap_model_poison_attack(
    rng: np.random.Generator,
    n_classes: int,
    scale: float = 10.0,
    benign_deltas_provider: Optional[Callable[[], Sequence[Params]]] = None,
) -> Callable[[Params, Params], Params]:
    """Adapts :func:`model_poison_attack` (which operates on a delta) to the
    T7 harness's uniform ``(P_current, P_global) -> P_poisoned`` attacker-
    injection contract (§2.6 step 5). ``benign_deltas_provider``, if given,
    is called fresh each invocation to supply this round's other clients'
    deltas for camouflage (JC-T7-7) -- a callable rather than a fixed list
    since the benign set changes every round.
    """
    def _fn(P_current: Params, P_global: Params) -> Params:
        delta = [c - g for c, g in zip(P_current, P_global)]
        benign = benign_deltas_provider() if benign_deltas_provider is not None else ()
        poisoned_delta = model_poison_attack(delta, P_global, n_classes, scale, rng, benign)
        return [d + g for d, g in zip(poisoned_delta, P_global)]
    return _fn
