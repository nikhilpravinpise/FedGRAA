"""Resumable, worker-capped driver for the Stage B ablation campaign (stage_b_plan.md T3/§6.2).

Enumerates the ``(arm, seed)`` grid for one dataset across the four
factorial arms -- **literal**, **projection-alone**, **eta-alone
(canonical)**, **both** -- skips any ``(arm, seed)`` whose diagnostic JSON
already exists on disk with all three gamma keys populated, and runs the
rest as independent ``scripts/diagnose_fedgraa.py`` subprocesses (one
process per run, worker-capped, since each CIC-IDS-2017 run loads a 1.46 GB
array and the campaign must stay under available RAM -- stage_b_plan.md
§3 "Parallelism & RAM").

This script does not itself train anything beyond invoking the existing,
already-tested ``diagnose_fedgraa.py`` entry point once per missing run --
it is pure orchestration, matching stage_b_plan.md's filename convention
(§3/§6.2/Appendix) exactly so seed-0 files already committed under
``results/diagnostics/`` are recognized and reused rather than re-run.

Usage::

    # Enumerate + run every missing (arm, seed) for CIC-IDS-2017, seeds 0-9,
    # capped at 3 concurrent worker processes.
    python scripts/run_ablation_seed_sweep.py --dataset CIC-IDS-2017 --seeds 0-9 --workers 3

    # Dry run: show what would be launched without running anything.
    python scripts/run_ablation_seed_sweep.py --dataset UNSW-NB15 --seeds 0-9 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DIAGNOSE_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagnose_fedgraa.py")
DEFAULT_OUT_DIR = "results/diagnostics"
DEFAULT_GAMMAS = [0.0, 0.5, 1.0]
GAMMA_KEYS = {"g0.0", "g0.5", "g1.0"}


# --------------------------------------------------------------------------- #
# Arm -> CLI-flag / tag mapping (stage_b_plan.md §3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ArmSpec:
    name: str
    cli_flags: List[str] = field(default_factory=list)
    base_tag: Optional[str] = None  # None => no --tag at seed 0

    def project(self) -> bool:
        return "--project" in self.cli_flags


ARMS = {
    "literal": ArmSpec("literal", cli_flags=[], base_tag=None),
    "projection_alone": ArmSpec("projection_alone", cli_flags=["--project"], base_tag=None),
    "eta_alone": ArmSpec(
        "eta_alone",
        cli_flags=["--eta-prime", "0.01", "--weight-epochs", "5"],
        base_tag="etaonly_eta0.01_E5",
    ),
    "both": ArmSpec(
        "both",
        cli_flags=["--project", "--eta-prime", "0.01", "--weight-epochs", "5"],
        base_tag="combined_eta0.01_E5",
    ),
}
ALL_ARM_NAMES = ["literal", "projection_alone", "eta_alone", "both"]


def _tag_for(arm: ArmSpec, seed: int) -> Optional[str]:
    """The ``--tag`` value diagnose_fedgraa.py should receive for this
    (arm, seed) -- seed 0 uses the arm's base tag (possibly None/absent);
    seed>0 appends ``_seed{s}`` (or is exactly ``seed{s}`` when the arm has
    no base tag), matching the existing committed files' naming."""
    if seed == 0:
        return arm.base_tag
    if arm.base_tag:
        return f"{arm.base_tag}_seed{seed}"
    return f"seed{seed}"


def output_path(dataset: str, arm_name: str, seed: int, rounds: int, out_dir: str = DEFAULT_OUT_DIR) -> str:
    """Reproduce diagnose_fedgraa.py's ``main()`` output-filename convention
    exactly: ``{DS}_{variant}{_tag}_r{rounds}.json``."""
    arm = ARMS[arm_name]
    dataset_tag = dataset.replace("-", "_")
    variant = "projected" if arm.project() else "literal"
    tag = _tag_for(arm, seed)
    suffix = f"_{tag}" if tag else ""
    return os.path.join(out_dir, f"{dataset_tag}_{variant}{suffix}_r{rounds}.json")


# --------------------------------------------------------------------------- #
# Resumability: skip any (arm, seed) whose output already has all 3 gammas
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RunSpec:
    dataset: str
    arm: str
    seed: int
    rounds: int
    eval_every: int
    gammas: List[float]
    data_dir: str
    cache_dir: str
    out_dir: str

    @property
    def path(self) -> str:
        return output_path(self.dataset, self.arm, self.seed, self.rounds, self.out_dir)

    def cli_args(self) -> List[str]:
        arm = ARMS[self.arm]
        tag = _tag_for(arm, self.seed)
        args = [
            "--dataset", self.dataset,
            "--gammas", *[str(g) for g in self.gammas],
            "--rounds", str(self.rounds),
            "--eval-every", str(self.eval_every),
            "--seed", str(self.seed),
            "--data-dir", self.data_dir,
            "--cache-dir", self.cache_dir,
            "--out-dir", self.out_dir,
            *arm.cli_flags,
        ]
        if tag:
            args += ["--tag", tag]
        return args


def _has_all_gammas(path: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return GAMMA_KEYS.issubset(data.keys())


def enumerate_runs(
    dataset: str,
    seeds: Sequence[int],
    arms: Sequence[str] = ALL_ARM_NAMES,
    rounds: int = 60,
    eval_every: int = 1,
    gammas: Sequence[float] = DEFAULT_GAMMAS,
    data_dir: str = "data",
    cache_dir: str = ".cache",
    out_dir: str = DEFAULT_OUT_DIR,
) -> List[RunSpec]:
    """Every ``(arm, seed)`` for this dataset, in a stable order."""
    runs = []
    for arm in arms:
        for seed in seeds:
            runs.append(RunSpec(
                dataset=dataset, arm=arm, seed=seed, rounds=rounds, eval_every=eval_every,
                gammas=list(gammas), data_dir=data_dir, cache_dir=cache_dir, out_dir=out_dir,
            ))
    return runs


def missing_runs(runs: Sequence[RunSpec]) -> List[RunSpec]:
    """Filter to only the runs whose output JSON is absent or incomplete
    (missing one or more of the 3 gamma keys) -- the resumability contract:
    re-running the driver after an interruption re-does at most the runs
    that never finished, never a run whose output is already complete."""
    return [r for r in runs if not _has_all_gammas(r.path)]


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _run_one(run: RunSpec) -> "subprocess.CompletedProcess":
    cmd = [sys.executable, DIAGNOSE_SCRIPT] + run.cli_args()
    return subprocess.run(cmd, check=False)


def execute(runs: Sequence[RunSpec], workers: int = 1) -> int:
    """Run every ``RunSpec`` (assumed already filtered to the missing set),
    up to ``workers`` concurrent subprocesses. Returns the count of runs
    that exited non-zero."""
    if not runs:
        return 0
    failures = 0
    if workers <= 1:
        for run in runs:
            print(f">>> {run.dataset} arm={run.arm} seed={run.seed} -> {run.path}")
            result = _run_one(run)
            if result.returncode != 0:
                failures += 1
                print(f"  [FAILED] exit code {result.returncode}")
        return failures

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, run): run for run in runs}
        for future in as_completed(futures):
            run = futures[future]
            result = future.result()
            status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
            print(f"  [{status}] {run.dataset} arm={run.arm} seed={run.seed} -> {run.path}")
            if result.returncode != 0:
                failures += 1
    return failures


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_seed_token(token: str) -> List[int]:
    """Parse one seed token: a bare int (``"3"``) or an inclusive range
    (``"0-9"``). §6.7's optional native range-shorthand for this driver
    (``run_seed_sweep.py`` keeps its existing space-separated-list CLI)."""
    token = token.strip()
    if "-" in token and not token.startswith("-"):
        lo_s, _, hi_s = token.partition("-")
        lo, hi = int(lo_s), int(hi_s)
        if hi < lo:
            raise argparse.ArgumentTypeError(f"invalid seed range '{token}': end < start")
        return list(range(lo, hi + 1))
    return [int(token)]


def parse_seeds(tokens: Sequence[str]) -> List[int]:
    seeds: List[int] = []
    for tok in tokens:
        seeds.extend(parse_seed_token(tok))
    # de-duplicate, preserve ascending order
    return sorted(set(seeds))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="CIC-IDS-2017")
    ap.add_argument("--seeds", nargs="+", default=["0-9"],
                     help="Seed tokens: bare ints and/or ranges, e.g. --seeds 0-9  or  --seeds 0 1 2")
    ap.add_argument("--arms", nargs="+", default=ALL_ARM_NAMES, choices=ALL_ARM_NAMES)
    ap.add_argument("--gammas", nargs="+", type=float, default=DEFAULT_GAMMAS)
    ap.add_argument("--rounds", type=int, default=60)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--cache-dir", default=".cache")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--workers", type=int, default=1,
                     help="Concurrent subprocess cap (stage_b_plan.md §3: keep CIC-IDS-2017 runs "
                          "under ~11 GB RAM -- 3 is the plan's suggested cap for that dataset).")
    ap.add_argument("--dry-run", action="store_true",
                     help="Print the missing (arm, seed) runs and their commands without executing them.")
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.out_dir, exist_ok=True)

    all_runs = enumerate_runs(
        args.dataset, seeds, arms=args.arms, rounds=args.rounds, eval_every=args.eval_every,
        gammas=args.gammas, data_dir=args.data_dir, cache_dir=args.cache_dir, out_dir=args.out_dir,
    )
    to_run = missing_runs(all_runs)
    already_done = len(all_runs) - len(to_run)

    print("=" * 78)
    print(f"  Ablation seed sweep -- {args.dataset}")
    print(f"  arms={args.arms} seeds={seeds} rounds={args.rounds}")
    print(f"  {len(all_runs)} total (arm, seed) runs | {already_done} already complete on disk "
          f"| {len(to_run)} to run")
    print("=" * 78)

    if args.dry_run:
        for run in to_run:
            print(f"  WOULD RUN: {' '.join([sys.executable, DIAGNOSE_SCRIPT] + run.cli_args())}")
        return 0

    if not to_run:
        print("Nothing to do -- every (arm, seed) already has a complete diagnostic JSON.")
        return 0

    failures = execute(to_run, workers=args.workers)
    print("=" * 78)
    if failures:
        print(f"  {failures}/{len(to_run)} runs FAILED. Re-run this driver to retry only those "
              f"(completed runs are skipped automatically).")
        return 1
    print(f"  All {len(to_run)} runs completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
