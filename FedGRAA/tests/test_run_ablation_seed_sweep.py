"""Unit tests for the ablation seed-sweep driver (stage_b_plan.md T3/§6.2).

These test the pure enumeration/skip/CLI-building logic with no training and
no subprocess launched -- ``execute()`` is exercised with the actual runner
monkeypatched out, so a full test run costs milliseconds and provably
demonstrates the resumability contract: re-running the driver against a
directory that already has complete diagnostic JSONs re-does nothing.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_ablation_seed_sweep import (  # noqa: E402
    ALL_ARM_NAMES,
    ARMS,
    RunSpec,
    enumerate_runs,
    execute,
    missing_runs,
    output_path,
    parse_seed_token,
    parse_seeds,
)


# --------------------------------------------------------------------------- #
# Arm -> filename convention (regression guard against the committed files)
# --------------------------------------------------------------------------- #
def test_output_path_matches_committed_literal_seed0():
    assert output_path("CIC-IDS-2017", "literal", 0, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_literal_r60.json")


def test_output_path_matches_committed_projection_alone_seed0():
    assert output_path("CIC-IDS-2017", "projection_alone", 0, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_projected_r60.json")


def test_output_path_matches_committed_both_arm_seeds():
    # These three files are already committed under results/diagnostics/.
    assert output_path("CIC-IDS-2017", "both", 0, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_projected_combined_eta0.01_E5_r60.json")
    assert output_path("CIC-IDS-2017", "both", 1, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_projected_combined_eta0.01_E5_seed1_r60.json")
    assert output_path("CIC-IDS-2017", "both", 2, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_projected_combined_eta0.01_E5_seed2_r60.json")


def test_output_path_eta_alone_canonical_is_new_naming():
    assert output_path("CIC-IDS-2017", "eta_alone", 0, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_literal_etaonly_eta0.01_E5_r60.json")
    assert output_path("UNSW-NB15", "eta_alone", 3, 60) == \
        os.path.join("results/diagnostics", "UNSW_NB15_literal_etaonly_eta0.01_E5_seed3_r60.json")


def test_output_path_literal_and_projection_alone_seed_gt_0_uses_bare_seed_tag():
    # literal/projection-alone have no base tag, so seed>0 uses "seed{s}" verbatim.
    assert output_path("CIC-IDS-2017", "literal", 4, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_literal_seed4_r60.json")
    assert output_path("CIC-IDS-2017", "projection_alone", 4, 60) == \
        os.path.join("results/diagnostics", "CIC_IDS_2017_projected_seed4_r60.json")


def test_all_four_arms_are_distinct_cli_flag_combinations():
    flag_sets = {tuple(ARMS[name].cli_flags) for name in ALL_ARM_NAMES}
    assert len(flag_sets) == 4  # literal / project / eta / project+eta all distinct


# --------------------------------------------------------------------------- #
# CLI args
# --------------------------------------------------------------------------- #
def test_cli_args_both_arm_includes_project_and_eta_flags():
    run = RunSpec(
        dataset="CIC-IDS-2017", arm="both", seed=1, rounds=60, eval_every=1,
        gammas=[0.0, 0.5, 1.0], data_dir="data", cache_dir=".cache",
        out_dir="results/diagnostics",
    )
    args = run.cli_args()
    assert "--project" in args
    assert "--eta-prime" in args and args[args.index("--eta-prime") + 1] == "0.01"
    assert "--weight-epochs" in args and args[args.index("--weight-epochs") + 1] == "5"
    assert "--tag" in args and args[args.index("--tag") + 1] == "combined_eta0.01_E5_seed1"
    assert "--seed" in args and args[args.index("--seed") + 1] == "1"


def test_cli_args_literal_seed0_has_no_tag_flag():
    run = RunSpec(
        dataset="CIC-IDS-2017", arm="literal", seed=0, rounds=60, eval_every=1,
        gammas=[0.0, 0.5, 1.0], data_dir="data", cache_dir=".cache",
        out_dir="results/diagnostics",
    )
    assert "--tag" not in run.cli_args()
    assert "--project" not in run.cli_args()


# --------------------------------------------------------------------------- #
# Enumeration + skip logic (resumability)
# --------------------------------------------------------------------------- #
def test_enumerate_runs_covers_full_grid():
    runs = enumerate_runs("CIC-IDS-2017", seeds=range(3), arms=ALL_ARM_NAMES)
    assert len(runs) == 4 * 3  # 4 arms x 3 seeds


def test_missing_runs_skips_complete_files_and_keeps_incomplete_or_absent(tmp_path):
    out_dir = str(tmp_path)
    dataset = "CIC-IDS-2017"

    # A complete file (all 3 gamma keys) for (literal, seed 0) -- should be skipped.
    complete_path = output_path(dataset, "literal", 0, 60, out_dir)
    with open(complete_path, "w", encoding="utf-8") as f:
        json.dump({"g0.0": {}, "g0.5": {}, "g1.0": {}}, f)

    # An incomplete file (missing g1.0) for (projection_alone, seed 0) -- must NOT be skipped.
    incomplete_path = output_path(dataset, "projection_alone", 0, 60, out_dir)
    os.makedirs(os.path.dirname(incomplete_path), exist_ok=True)
    with open(incomplete_path, "w", encoding="utf-8") as f:
        json.dump({"g0.0": {}, "g0.5": {}}, f)

    # (eta_alone, seed 0) has no file at all -- must NOT be skipped.

    runs = enumerate_runs(dataset, seeds=[0], arms=["literal", "projection_alone", "eta_alone"], out_dir=out_dir)
    to_run = missing_runs(runs)
    to_run_arms = {r.arm for r in to_run}

    assert "literal" not in to_run_arms
    assert "projection_alone" in to_run_arms
    assert "eta_alone" in to_run_arms


def test_missing_runs_ignores_corrupt_json_and_reruns_it(tmp_path):
    out_dir = str(tmp_path)
    path = output_path("CIC-IDS-2017", "literal", 0, 60, out_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")

    runs = enumerate_runs("CIC-IDS-2017", seeds=[0], arms=["literal"], out_dir=out_dir)
    to_run = missing_runs(runs)
    assert len(to_run) == 1


def test_rerun_against_fully_complete_directory_does_nothing(tmp_path, monkeypatch):
    """The resumability contract, proved end-to-end: pre-populate every
    (arm, seed) output as complete, then confirm missing_runs() -> [] and
    execute() never invokes the (monkeypatched) runner."""
    out_dir = str(tmp_path)
    dataset = "CIC-IDS-2017"
    seeds = [0, 1]

    all_runs = enumerate_runs(dataset, seeds=seeds, arms=ALL_ARM_NAMES, out_dir=out_dir)
    for run in all_runs:
        os.makedirs(os.path.dirname(run.path), exist_ok=True)
        with open(run.path, "w", encoding="utf-8") as f:
            json.dump({"g0.0": {}, "g0.5": {}, "g1.0": {}}, f)

    to_run = missing_runs(all_runs)
    assert to_run == []

    calls = []
    monkeypatch.setattr(
        "scripts.run_ablation_seed_sweep._run_one",
        lambda run: calls.append(run) or type("R", (), {"returncode": 0})(),
    )
    failures = execute(to_run, workers=1)
    assert failures == 0
    assert calls == []  # the runner was never invoked


def test_execute_calls_runner_once_per_missing_run(monkeypatch):
    runs = enumerate_runs("CIC-IDS-2017", seeds=[0, 1], arms=["literal"])
    calls = []

    def fake_run_one(run):
        calls.append(run)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("scripts.run_ablation_seed_sweep._run_one", fake_run_one)
    failures = execute(runs, workers=1)
    assert failures == 0
    assert len(calls) == 2


def test_execute_counts_failures(monkeypatch):
    runs = enumerate_runs("CIC-IDS-2017", seeds=[0, 1], arms=["literal"])

    def fake_run_one(run):
        return type("R", (), {"returncode": 0 if run.seed == 0 else 1})()

    monkeypatch.setattr("scripts.run_ablation_seed_sweep._run_one", fake_run_one)
    failures = execute(runs, workers=1)
    assert failures == 1


# --------------------------------------------------------------------------- #
# Seed range parsing (§6.7 native range shorthand)
# --------------------------------------------------------------------------- #
def test_parse_seed_token_range():
    assert parse_seed_token("0-9") == list(range(10))


def test_parse_seed_token_single():
    assert parse_seed_token("3") == [3]


def test_parse_seeds_dedupes_and_sorts():
    assert parse_seeds(["0-2", "1", "5"]) == [0, 1, 2, 5]
