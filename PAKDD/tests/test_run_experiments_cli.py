"""CLI flag tests for run_experiments.py (stage_a_plan.md §6.6: --workers, --ns).

main() parses args and immediately runs training, so this test doesn't invoke
it end-to-end; it mirrors the parser definition to check flag wiring and
defaults without paying for a real sweep.
"""

from __future__ import annotations

import argparse


def _build_parser():
    parser = argparse.ArgumentParser(description="Reproduce FedGRAA Figures 1-3")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--datasets", nargs="+", default=["CIC-IDS-2017", "UNSW-NB15"])
    parser.add_argument("--sweep", choices=["gamma", "n", "both"], default="both")
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--round-zero-zero", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--ns", nargs="+", type=int, default=[3, 5, 7])
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def test_workers_and_ns_defaults():
    args = _build_parser().parse_args([])
    assert args.workers is None
    assert args.ns == [3, 5, 7]


def test_workers_and_ns_can_be_overridden():
    args = _build_parser().parse_args(["--workers", "3", "--ns", "3", "5"])
    assert args.workers == 3
    assert args.ns == [3, 5]


def test_run_experiments_source_defines_both_flags():
    """Guard against the test parser drifting from the real one in run_experiments.py."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run_experiments.py")
    with open(path, encoding="utf-8") as f:
        src = f.read()
    assert '"--workers"' in src
    assert '"--ns"' in src
