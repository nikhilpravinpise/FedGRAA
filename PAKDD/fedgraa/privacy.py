"""Differential privacy for the rarity-histogram channel (W2.2, F3/§5.2).

Two pieces:

* :func:`laplace_mechanism` -- the Laplace mechanism itself, ``b = sensitivity
  / epsilon``, applied to the (SecAgg-summed) label-count histogram. The
  sensitivity of a single client's histogram under add/remove-one-row
  neighbouring is ``Delta_1 = 2`` (one row moves from one class's count to
  another's, or appears/disappears -- the redesign doc's Sec. 10 arithmetic,
  unchanged here).
* Two composition accountants over ``T`` rounds -- **advanced composition**
  (Dwork, Rothblum & Vadhan; see Dwork & Roth, *The Algorithmic Foundations of
  Differential Privacy*, 2014, Thm. 3.20) and **Renyi DP** (Mironov, "Renyi
  Differential Privacy," CSF 2017, arXiv:1702.07476) -- because §5.3 requires
  reporting *both*, and because the RDP accountant gives the "tighter number"
  the plan asks for. Neither is a hand-rolled approximation: both are the
  textbook closed-form results for the Laplace mechanism, not a numerical
  search over an arbitrary composition bound. No third-party DP-accounting
  package (e.g. ``dp-accounting``, ``opacus``) is pinned in
  ``requirements.txt``; rather than add an unvetted new dependency mid-project,
  both accountants are implemented directly from their published closed forms
  and cross-checked against the plan's own worked example (§5.3: eps_total ~=
  8.89 at eps_round=0.1, delta=1e-5, T=200) in ``test_privacy.py``.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "SENSITIVITY",
    "laplace_mechanism",
    "aggregate_noise_std",
    "advanced_composition_epsilon",
    "laplace_rdp",
    "rdp_to_dp",
    "composed_rdp_epsilon",
    "composed_epsilon",
]

# L1 sensitivity of one client's label-count histogram under add/remove-one-row
# neighbouring datasets (privacy_compliant_fedgraa_redesign.md §10; F3/§5.2).
SENSITIVITY = 2.0


def laplace_mechanism(
    counts: np.ndarray,
    epsilon: float,
    sensitivity: float = SENSITIVITY,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add ``Laplace(0, sensitivity/epsilon)`` noise to a count vector.

    Returns a float array (counts are no longer exactly integral after
    noising); callers that need integers back round and clip non-negative
    downstream, matching how a real histogram-release mechanism is consumed.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    rng = rng or np.random.default_rng()
    b = sensitivity / epsilon
    noise = rng.laplace(0.0, b, size=np.asarray(counts).shape)
    return np.asarray(counts, dtype=np.float64) + noise


def aggregate_noise_std(
    epsilon: float, n_clients: int, sensitivity: float = SENSITIVITY
) -> float:
    """Std of the SecAgg-*summed* noise over ``n_clients`` independent
    per-client Laplace releases: ``sigma = (sensitivity/epsilon) * sqrt(2*K)``
    (plan §5.2's table) -- each ``Laplace(0, b)`` term has variance ``2*b**2``,
    and summing ``K`` independent copies scales variance by ``K``.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon must be > 0, got {epsilon}")
    if n_clients < 1:
        raise ValueError(f"n_clients must be >= 1, got {n_clients}")
    b = sensitivity / epsilon
    return b * math.sqrt(2.0) * math.sqrt(n_clients)


# --------------------------------------------------------------------------- #
# Advanced composition (Dwork & Roth 2014, Thm 3.20)
# --------------------------------------------------------------------------- #
def advanced_composition_epsilon(
    epsilon_round: float, rounds: int, delta_prime: float, delta_round: float = 0.0
) -> tuple[float, float]:
    """Advanced-composition ``(epsilon_total, delta_total)`` over ``rounds``
    adaptive ``(epsilon_round, delta_round)``-DP mechanisms.

    ``epsilon_total = sqrt(2 * rounds * ln(1/delta_prime)) * epsilon_round
    + rounds * epsilon_round * (exp(epsilon_round) - 1)``,
    ``delta_total = rounds * delta_round + delta_prime``.

    The Laplace mechanism used here is pure (``delta_round = 0``)-DP, so the
    default reproduces the plan's §5.3 worked example exactly: at
    ``epsilon_round=0.1``, ``rounds=200``, ``delta_prime=1e-5`` this returns
    ``epsilon_total ~= 8.89``.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    if not 0.0 < delta_prime < 1.0:
        raise ValueError(f"delta_prime must be in (0, 1), got {delta_prime}")
    term1 = math.sqrt(2.0 * rounds * math.log(1.0 / delta_prime)) * epsilon_round
    term2 = rounds * epsilon_round * (math.exp(epsilon_round) - 1.0)
    eps_total = term1 + term2
    delta_total = rounds * delta_round + delta_prime
    return eps_total, delta_total


# --------------------------------------------------------------------------- #
# Renyi DP (Mironov, CSF 2017)
# --------------------------------------------------------------------------- #
def laplace_rdp(alpha: float, epsilon0: float) -> float:
    """Renyi-DP epsilon at order ``alpha`` for a single pure-``epsilon0``-DP
    Laplace mechanism release (closed form for the Laplace mechanism; see
    Mironov 2017 and the equivalent derivation in Zhu & Wang, "Improving
    Sparse Vector Technique with Renyi Differential Privacy," or any RDP
    reference table for the Laplace mechanism):

    ``eps(alpha) = 1/(alpha-1) * log(
        alpha/(2*alpha-1) * exp((alpha-1)*epsilon0)
        + (alpha-1)/(2*alpha-1) * exp(-alpha*epsilon0)
    )``
    """
    if alpha <= 1.0:
        raise ValueError(f"alpha must be > 1, got {alpha}")
    if epsilon0 <= 0:
        raise ValueError(f"epsilon0 must be > 0, got {epsilon0}")
    term1 = (alpha / (2.0 * alpha - 1.0)) * math.exp((alpha - 1.0) * epsilon0)
    term2 = ((alpha - 1.0) / (2.0 * alpha - 1.0)) * math.exp(-alpha * epsilon0)
    return (1.0 / (alpha - 1.0)) * math.log(term1 + term2)


def rdp_to_dp(rdp_at_alpha: Dict[float, float], delta: float) -> float:
    """Standard RDP -> ``(epsilon, delta)``-DP conversion (Mironov 2017,
    Prop. 3): ``epsilon = rdp(alpha) + log(1/delta)/(alpha-1)``, minimized
    over the supplied orders ``alpha``.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    candidates = [
        eps_a + math.log(1.0 / delta) / (alpha - 1.0)
        for alpha, eps_a in rdp_at_alpha.items()
    ]
    return min(candidates)


def _default_alpha_grid() -> np.ndarray:
    return np.concatenate(
        [np.arange(1.01, 2.0, 0.01), np.arange(2.0, 64.0, 0.5)]
    )


def composed_rdp_epsilon(
    epsilon_round: float,
    rounds: int,
    delta: float,
    alphas: Optional[Sequence[float]] = None,
) -> float:
    """Composed ``epsilon`` over ``rounds`` rounds via the RDP accountant.

    RDP composes additively across rounds (``rdp_total(alpha) = rounds *
    rdp_round(alpha)``); the result is converted back to ``(epsilon, delta)``-DP
    via :func:`rdp_to_dp`, minimized over a grid of Renyi orders. This is
    the "tighter number" the plan's §5.3 asks to report alongside advanced
    composition.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    alphas = alphas if alphas is not None else _default_alpha_grid()
    composed = {a: rounds * laplace_rdp(a, epsilon_round) for a in alphas}
    return rdp_to_dp(composed, delta)


def composed_epsilon(
    epsilon_round: float, rounds: int, delta: float
) -> Dict[str, float]:
    """Report both accountants side by side, per §5.3's instruction to
    "report the composed (epsilon, delta) ... never a per-round epsilon" and
    to give both the advanced-composition and the RDP number."""
    adv_eps, adv_delta = advanced_composition_epsilon(epsilon_round, rounds, delta)
    rdp_eps = composed_rdp_epsilon(epsilon_round, rounds, delta)
    return {
        "advanced_composition_epsilon": adv_eps,
        "advanced_composition_delta": adv_delta,
        "rdp_epsilon": rdp_eps,
        "delta": delta,
    }
