"""CLI flag tests for run_experiments.py (stage_a_plan.md §6.6: --workers, --ns).

main() parses args and immediately runs training, so this test doesn't invoke
it end-to-end; instead it imports the real ``build_parser()`` from
run_experiments.py directly (no hand-duplicated shadow parser -- a prior
audit found that duplicating the parser here let a real default-mismatch bug
in --round-zero-zero go undetected, since the shadow copy had the identical
bug rather than catching it).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_experiments import build_parser  # noqa: E402


def test_workers_and_ns_defaults():
    args = build_parser().parse_args([])
    assert args.workers is None
    assert args.ns == [3, 5, 7]


def test_workers_and_ns_can_be_overridden():
    args = build_parser().parse_args(["--workers", "3", "--ns", "3", "5"])
    assert args.workers == 3
    assert args.ns == [3, 5]


def test_round_zero_zero_defaults_true_and_matches_fedconfig():
    """Regression guard: --round-zero-zero used to silently default to False
    via bare store_true, contradicting FedConfig's own default of True."""
    from fedgraa.config import FedConfig

    args = build_parser().parse_args([])
    assert args.round_zero_zero is True
    assert args.round_zero_zero == FedConfig().round_zero_zero


def test_round_zero_zero_can_be_disabled():
    args = build_parser().parse_args(["--no-round-zero-zero"])
    assert args.round_zero_zero is False
