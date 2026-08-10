"""Significance-testing primitives for the Stage B ablation (stage_b_plan.md §6.1).

No statistics code existed anywhere in the repo before this module — both
seed-sweep tools (``scripts/run_seed_sweep.py``,
``scripts/seed_sweep_combined_fix.py``) compute only ``np.mean``/``np.std``.
This module adds the bootstrap CI, paired significance tests, effect size,
and multiple-comparison correction the Stage B ablation table needs, built
against SciPy references rather than hand-rolled formulas wherever SciPy
already implements the primitive correctly (Wilcoxon, paired t).

**Degenerate-cell safety (stage_b_plan.md §4).** The rare-class accuracy
metric is a bounded mean of per-class recalls, and several ablation cells
have exactly zero variance across seeds (e.g. the combined-fix arm landed on
identical accuracy at every one of its first 3 seeds). A naive paired
t-test or Cohen's d on such data divides by a zero standard deviation. Every
function here detects that condition explicitly and returns a well-defined,
finite result instead of ``NaN`` or a bare Python exception — see each
docstring for its exact degenerate-case contract.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import numpy as np
from scipy import stats as _scipy_stats

__all__ = [
    "bootstrap_ci",
    "paired_ttest",
    "wilcoxon_signed_rank",
    "cohens_d_paired",
    "holm_bonferroni",
]


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for ``statistic(values)``.

    Resamples ``values`` with replacement ``n_boot`` times using a fixed,
    seeded RNG (``np.random.default_rng(seed)``) so the CI is exactly
    reproducible across runs, and returns the
    ``(alpha/2, 1 - alpha/2)`` percentiles of the resampled statistic.

    **Degenerate case:** if every value is identical (``np.ptp(values) ==
    0``), the bootstrap distribution is a point mass at that value by
    construction — resampling is skipped and the point CI ``(v, v)`` is
    returned directly rather than run through 10,000 no-op resamples.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    if n == 0:
        raise ValueError("bootstrap_ci requires at least one value")

    point = float(np.asarray(statistic(arr)))
    if np.ptp(arr) == 0.0:
        return (point, point)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    resamples = arr[idx]
    try:
        boot_stats = np.asarray(statistic(resamples, axis=1), dtype=np.float64)
    except TypeError:
        boot_stats = np.array([float(np.asarray(statistic(row))) for row in resamples])

    lo, hi = np.percentile(boot_stats, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return (float(lo), float(hi))


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Paired (dependent-samples) t-test, ``scipy.stats.ttest_rel`` under the hood.

    **Degenerate case:** if every paired difference ``a_i - b_i`` is
    identical, the sample standard error of the differences is (near) zero,
    so the t-statistic is a literal ``0/0`` (differences all zero) or a
    division by ~zero (differences constant but nonzero) — both undefined
    or numerically unstable in a way that would silently produce ``NaN``
    or an unreproducible platform-dependent huge number. Rather than
    report a spurious near-infinite significance from a zero-variance
    sample, this returns ``(0.0, 1.0)`` — "not significant" —
    and defers the sign call to :func:`wilcoxon_signed_rank`, which handles
    the same degeneracy by construction (see its docstring).

    **The degeneracy test is relative, not exact** (corrected 2026-07-28).
    An exact ``ptp == 0.0`` check misses the case that actually arises:
    ``[0.7, 0.8] - [0.6, 0.7]`` is constant in exact arithmetic but its two
    float64 entries differ by ~1.1e-16, which was enough to slip past the
    guard and yield ``t = 1.8e15, p = 3.5e-16``. Fewer than two pairs, and
    any non-finite result from ``scipy``, are also mapped to
    ``(0.0, 1.0)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = a - b

    # A paired t-test needs at least two pairs; with one, the sample standard
    # deviation is undefined.
    if diffs.size < 2:
        return (0.0, 1.0)

    # ⚠️ ``np.ptp(diffs) == 0.0`` -- the original guard -- is an EXACT float
    # comparison, and the degenerate case that matters in practice does not
    # hit it exactly. ``[0.7, 0.8] - [0.6, 0.7]`` is a constant difference
    # mathematically, but in float64 the two entries differ by ~1.1e-16, so
    # ``ptp`` is nonzero, the guard does not fire, and ``ttest_rel`` divides
    # by a ~1e-17 standard error to produce **t = 1.8e15, p = 3.5e-16** --
    # exactly the "spurious near-infinite significance" this function's
    # docstring says it prevents. Found by
    # ``tests/test_pipeline_invariants.py`` (T-15) and recorded in
    # ``docs/audit/PRE_COMPUTE_AUDIT_2026-07-28.md``.
    #
    # The fix is a *relative* degeneracy test: treat the differences as
    # constant when their spread is negligible against their own scale. The
    # tolerance is loose enough to catch accumulated round-off and far tighter
    # than any real seed-to-seed variation in an accuracy metric (Stage B's
    # own paired differences have std ~1e-2, twelve orders of magnitude above
    # this threshold).
    scale = max(float(np.max(np.abs(diffs))), 1.0)
    if float(np.std(diffs, ddof=1)) <= 1e-12 * scale:
        return (0.0, 1.0)

    t, p = _scipy_stats.ttest_rel(a, b)
    if not (np.isfinite(t) and np.isfinite(p)):
        return (0.0, 1.0)
    return (float(t), float(p))


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Wilcoxon signed-rank test, ``scipy.stats.wilcoxon`` under the hood.

    **Zero-diff safety:** ``scipy.stats.wilcoxon`` raises ``ValueError``
    when every paired difference is zero (nothing survives the default
    ``zero_method="wilcox"`` tie-dropping) or when too few non-zero
    differences remain to compute a rank statistic. Both cases mean "no
    detectable difference" — caught and reported as ``(0.0, 1.0)`` instead
    of propagating the exception.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = a - b
    if np.allclose(diffs, 0.0):
        return (0.0, 1.0)
    try:
        stat, p = _scipy_stats.wilcoxon(a, b)
    except ValueError:
        return (0.0, 1.0)
    return (float(stat), float(p))


def cohens_d_paired(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's *d* for paired samples: ``mean(a - b) / std(a - b, ddof=1)``.

    **Degenerate case:** if the paired differences have zero sample
    variance, the ratio is a literal ``0/0`` (differences all zero — no
    effect at all, returns ``0.0``) or a division by zero (differences
    constant but nonzero — a perfectly consistent effect within the
    sample). The latter is a genuine unbounded standardized effect size
    (zero within-sample noise around a nonzero mean difference), reported
    as signed infinity (``+inf``/``-inf``) rather than clipped to an
    arbitrary finite number — that is the mathematically honest value, and
    unlike ``NaN`` it is unambiguous and orders correctly against any
    finite ``d``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = a - b
    # Check degeneracy on the raw (bit-identical) differences via ptp, not on
    # a computed std -- summing identical floats and dividing by n can pick
    # up ~1e-16 rounding noise that makes an exact-zero std test spuriously
    # false (the exact failure mode this function exists to avoid).
    if np.ptp(diffs) == 0.0:
        constant = float(diffs[0])
        if constant == 0.0:
            return 0.0
        return float(np.sign(constant) * np.inf)
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    return mean_diff / std_diff


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """Holm–Bonferroni step-down adjusted p-values, order-preserving.

    Standard Holm (1979) procedure: sort ascending, multiply the *i*-th
    smallest (0-indexed) by ``(m - i)``, enforce the running-maximum
    monotonicity constraint (an adjusted p-value can never be smaller than
    one earlier in the sorted order — that would let a less significant
    result "borrow" significance from a more significant one), clip to
    ``1.0``, then restore the caller's original ordering.
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.shape[0]
    if m == 0:
        return []

    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    multipliers = m - np.arange(m)
    adjusted_sorted = np.minimum(sorted_p * multipliers, 1.0)
    # Running maximum enforces monotonicity in the sorted order.
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)

    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]
