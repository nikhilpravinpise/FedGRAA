"""Unit tests for the FedGRAA divergence-diagnosis instrumentation.

These test the pure analysis/ablation helpers in ``scripts/diagnose_fedgraa.py``
directly (no training, no TF), so they run in well under a second and can act
as a permanent regression guard on the diagnostic logic itself: if
``analyze_history`` or ``project_to_simplex`` silently break, the science
verdict in ``docs/fedgraa_divergence_diagnosis.md`` would be re-derived from
broken instrumentation without anyone noticing.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.diagnose_fedgraa import analyze_history, project_to_simplex  # noqa: E402
from fedgraa.trainer import TrainHistory  # noqa: E402


# --------------------------------------------------------------------------- #
# project_to_simplex
# --------------------------------------------------------------------------- #
def test_project_to_simplex_sums_to_one_and_nonnegative():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.normal(size=7).astype(np.float32) * 5.0
        proj = project_to_simplex(w)
        assert proj.shape == w.shape
        assert np.all(proj >= -1e-6)
        assert abs(proj.sum() - 1.0) < 1e-4


def test_project_to_simplex_identity_on_simplex_point():
    w = np.array([0.2, 0.3, 0.5], dtype=np.float32)
    proj = project_to_simplex(w)
    np.testing.assert_allclose(proj, w, atol=1e-5)


def test_project_to_simplex_handles_all_negative():
    w = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    proj = project_to_simplex(w)
    assert abs(proj.sum() - 1.0) < 1e-4
    assert np.all(proj >= 0.0)


def test_project_to_simplex_clips_negative_weight():
    """A weight far outside the simplex (as FedGRAA's unconstrained Adam step
    can produce) must land back inside it."""
    w = np.array([5.0, -3.0, 0.5, 0.2], dtype=np.float32)
    proj = project_to_simplex(w)
    assert np.all(proj >= -1e-6)
    assert abs(proj.sum() - 1.0) < 1e-4


# --------------------------------------------------------------------------- #
# analyze_history
# --------------------------------------------------------------------------- #
def _history_with(composite_loss, weights):
    h = TrainHistory(method="fedgraa", gamma=1.0, n_rare=3)
    for i, (loss, w) in enumerate(zip(composite_loss, weights)):
        h.record(i, all_a=0.5, last_a=0.5, w=np.asarray(w, dtype=np.float32), comp_loss=loss)
    return h


def test_analyze_history_detects_monotone_case():
    # Strictly non-increasing composite loss + weights that stay on the simplex.
    losses = [10.0, 8.0, 6.0, 4.0, 2.0]
    weights = [[0.5, 0.5]] * 5
    h = _history_with(losses, weights)
    diag = analyze_history(h)
    assert diag["monotone_lemma1"] is True
    assert diag["first_monotonicity_violation_round"] is None
    assert diag["first_simplex_violation_round"] is None
    assert diag["final_weight_sum"] == pytest.approx(1.0)


def test_analyze_history_detects_violation_round_and_blowup():
    # Loss decreases then blows up at round 3.
    losses = [5.0, 4.0, 3.0, 30.0, 60.0]
    weights = [[0.5, 0.5]] * 5
    h = _history_with(losses, weights)
    diag = analyze_history(h)
    assert diag["monotone_lemma1"] is False
    assert diag["first_monotonicity_violation_round"] == 3
    assert diag["composite_loss_blowup_ratio"] == pytest.approx(60.0 / 5.0)


def test_analyze_history_detects_simplex_drift():
    losses = [5.0, 4.0, 3.0]
    # Round 0-1 stay on the simplex; round 2 drifts (sum != 1, one weight < 0).
    weights = [[0.5, 0.5], [0.6, 0.4], [1.8, -0.8]]
    h = _history_with(losses, weights)
    diag = analyze_history(h)
    assert diag["first_simplex_violation_round"] == 2
    assert diag["weight_sum_trajectory"] == pytest.approx([1.0, 1.0, 1.0])
    assert diag["weight_min_trajectory"][-1] == pytest.approx(-0.8)


def test_analyze_history_empty_composite_loss_is_vacuously_monotone():
    h = TrainHistory(method="fedavg", gamma=None, n_rare=3)
    h.record(0, all_a=0.4, last_a=0.1, w=np.array([1.0]), comp_loss=float("nan"))
    diag = analyze_history(h)
    assert diag["monotone_lemma1"] is True
    assert diag["composite_loss_blowup_ratio"] is None
