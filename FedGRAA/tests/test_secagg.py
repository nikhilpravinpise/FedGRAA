"""Tests for the secure-aggregation channel (W2.1, F4).

T-6: individual client contributions must be structurally unreachable, not
merely unused. T-7: dropout tolerance -- a valid renormalized result when
enough clients survive, an explicit error (never a silently wrong answer)
when too few do.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from fedgraa.secagg import SecAggChannel, SecAggDropoutError


def test_sum_over_multiple_clients():
    ch = SecAggChannel(n_clients=3, threshold=3)
    ch.submit([np.array([1.0, 2.0]), np.array(5.0)])
    ch.submit([np.array([3.0, 4.0]), np.array(1.0)])
    ch.submit([np.array([0.0, 0.0]), np.array(-1.0)])
    total = ch.aggregate()
    np.testing.assert_allclose(total[0], [4.0, 6.0])
    np.testing.assert_allclose(total[1], 5.0)


def test_aggregate_mean_divides_by_survivors_not_registered():
    ch = SecAggChannel(n_clients=5, threshold=2)
    ch.submit([np.array(10.0)])
    ch.submit([np.array(20.0)])
    mean = ch.aggregate_mean()
    np.testing.assert_allclose(mean[0], 15.0)  # /2 survivors, not /5 registered


def test_submit_shape_mismatch_raises():
    ch = SecAggChannel(n_clients=2, threshold=2)
    ch.submit([np.array(1.0), np.array(2.0)])
    with pytest.raises(ValueError):
        ch.submit([np.array(1.0)])  # wrong number of arrays


def test_submit_beyond_n_clients_raises():
    ch = SecAggChannel(n_clients=1, threshold=1)
    ch.submit([np.array(1.0)])
    with pytest.raises(RuntimeError):
        ch.submit([np.array(2.0)])


def test_aggregate_is_single_use():
    ch = SecAggChannel(n_clients=1, threshold=1)
    ch.submit([np.array(1.0)])
    ch.aggregate()
    with pytest.raises(RuntimeError):
        ch.submit([np.array(2.0)])


def test_calling_aggregate_twice_raises():
    """Regression: a second aggregate() call on an already-closed channel
    must raise, not silently re-return the same total -- audit-found bug,
    the original code only guarded submit() post-close, not aggregate()
    itself."""
    ch = SecAggChannel(n_clients=1, threshold=1)
    ch.submit([np.array(1.0)])
    ch.aggregate()
    with pytest.raises(RuntimeError):
        ch.aggregate()


# --------------------------------------------------------------------------- #
# T-7 -- dropout tolerance
# --------------------------------------------------------------------------- #
def test_dropout_below_threshold_raises_explicit_error():
    """Never a silently wrong answer: fewer survivors than the Shamir
    reconstruction threshold must raise, not return a partial sum."""
    ch = SecAggChannel(n_clients=10, threshold=7)
    for _ in range(6):
        ch.submit([np.array(1.0)])
    with pytest.raises(SecAggDropoutError):
        ch.aggregate()


def test_dropout_at_or_above_threshold_succeeds():
    ch = SecAggChannel(n_clients=10, threshold=7)
    for _ in range(7):
        ch.submit([np.array(2.0)])
    total = ch.aggregate()
    np.testing.assert_allclose(total[0], 14.0)


def test_dropout_exactly_at_threshold_is_the_boundary():
    """threshold-1 survivors fail, threshold survivors succeed -- the
    boundary itself, not just "well above" / "well below" it."""
    for n_submitted, should_succeed in [(6, False), (7, True)]:
        ch = SecAggChannel(n_clients=10, threshold=7)
        for _ in range(n_submitted):
            ch.submit([np.array(1.0)])
        if should_succeed:
            ch.aggregate()
        else:
            with pytest.raises(SecAggDropoutError):
                ch.aggregate()


def test_invalid_threshold_construction_raises():
    with pytest.raises(ValueError):
        SecAggChannel(n_clients=5, threshold=0)
    with pytest.raises(ValueError):
        SecAggChannel(n_clients=5, threshold=6)
    with pytest.raises(ValueError):
        SecAggChannel(n_clients=0, threshold=1)


# --------------------------------------------------------------------------- #
# T-6 -- individual contributions are structurally unreachable, not merely
# unused. Checked by introspection, not by "the algorithm happens not to
# call such a method."
# --------------------------------------------------------------------------- #
def test_no_public_or_private_method_returns_an_individual_contribution():
    """No method on SecAggChannel takes or implies a client index/identity and
    returns a value -- the only per-round information surfaced is a count
    (`n_submitted`) and the aggregate sum/mean."""
    allowed_public_api = {
        "submit", "aggregate", "aggregate_mean",
        "n_clients", "threshold", "n_submitted",
    }
    members = {
        name for name, _ in inspect.getmembers(SecAggChannel)
        if not name.startswith("_")
    }
    assert members == allowed_public_api, (
        f"unexpected public surface on SecAggChannel: {members - allowed_public_api}"
    )
    # None of the exposed methods accept any kind of client-index/id parameter.
    for name in ("submit", "aggregate", "aggregate_mean"):
        sig = inspect.signature(getattr(SecAggChannel, name))
        params = set(sig.parameters) - {"self"}
        assert not any("client" in p or "index" in p or "id" in p for p in params), (
            f"{name}{sig} exposes a client-identifying parameter"
        )


def test_individual_contributions_are_not_retained_after_submit():
    """After several submissions, no attribute on the instance holds a
    per-client value -- only the running elementwise total and a plain
    integer counter (the "unreachable by construction" property, not just
    "no getter exists")."""
    ch = SecAggChannel(n_clients=3, threshold=3)
    contributions = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    for c in contributions:
        ch.submit([c])

    state = vars(ch)
    assert set(state.keys()) == {
        "_n_clients", "_threshold", "_total", "_n_submitted", "_closed",
    }
    # The running total must be the SUM, never equal to (or contain as a
    # sub-array) any individual contribution alone.
    total = state["_total"][0]
    np.testing.assert_allclose(total, np.array([9.0, 12.0]))
    for c in contributions:
        assert not np.allclose(total, c)
