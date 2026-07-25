"""Stage B completeness check (stage_b_plan.md §7).

Asserts the full Stage-B artifact set exists, is loadable, and is
internally consistent, exiting non-zero on any gap. Standalone (not part of
``pytest tests``) so the unit suite stays decoupled from large committed
binaries and from the (currently un-run) 10-seed data campaign -- same
idiom as ``scripts/check_stage_a_complete.py``: ``fail()``/``ok()``,
aggregate every failure before exiting, ``sys.exit(main())``.

Usage::

    python scripts/check_stage_b_complete.py
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.check_doc_links import find_broken_links  # noqa: E402
from scripts.run_ablation_seed_sweep import ALL_ARM_NAMES, output_path  # noqa: E402

DIAG_DIR = "results/diagnostics"
ABLATION_DIR = "results/ablation"
THEORY_DOC = "docs/current/fedgraa_convergence_defect.md"
MANIFEST_PATH = "results/ablation/README.md"

DATASETS = ["CIC-IDS-2017", "UNSW-NB15"]
GAMMA_KEYS = {"g0.0", "g0.5", "g1.0"}
REQUIRED_DIAGNOSIS_FIELDS = {"final_all_acc", "final_last_n_acc", "composite_loss_blowup_ratio"}
N_SEEDS_REQUIRED = 10
ROUNDS = 60

REQUIRED_ARM_CELL_KEYS = {"rare_acc", "all_acc", "blowup"}
REQUIRED_CELL_FIELDS = {"seeds", "per_seed", "mean", "ci_95"}
REQUIRED_COMPARISON_ARMS = {"literal", "projection_alone", "eta_alone", "fedavg", "static"}
REQUIRED_COMPARISON_FIELDS = {"p_ttest", "p_wilcoxon", "cohens_d", "p_holm"}

REQUIRED_THEORY_HEADINGS = [
    # (substring to search for, human label) -- deliberately substrings, not
    # exact heading text, so minor wording edits don't break the guard. Each
    # substring is chosen specific enough that it can't be satisfied by
    # incidental prose elsewhere in the doc (a bare "why" or "defect" would
    # trivially appear in ordinary text and defeat the point of the guard).
    ("feasibility-set defect", "the feasibility-set defect"),
    ("does not cover this", "why Appendix A.1's proof doesn't cover it"),
    ("corrected convergence proposition", "the corrected convergence proposition"),
    ("consolidated ablation table", "the consolidated ablation table section"),
    ("honest limitations", "the honest-limitations section"),
]

FAILURES: list = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


# --------------------------------------------------------------------------- #
# 1. Per-seed diagnostic JSONs: 4 arms x 10 seeds x 2 datasets
# --------------------------------------------------------------------------- #
def check_diagnostic_json(path: str, label: str) -> bool:
    if not os.path.exists(path):
        fail(f"{label}: missing {path}")
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        fail(f"{label}: {path} failed to load ({exc})")
        return False
    if not GAMMA_KEYS.issubset(data.keys()):
        missing = GAMMA_KEYS - set(data.keys())
        fail(f"{label}: {path} missing gamma keys {sorted(missing)}")
        return False
    for gkey in GAMMA_KEYS:
        diag = data[gkey].get("diagnosis", {})
        missing_fields = REQUIRED_DIAGNOSIS_FIELDS - set(diag.keys())
        if missing_fields:
            fail(f"{label}: {path}[{gkey}] missing diagnosis fields {sorted(missing_fields)}")
            return False
    return True


def check_arm_seed_coverage() -> None:
    for dataset in DATASETS:
        for arm in ALL_ARM_NAMES:
            n_complete = 0
            for seed in range(N_SEEDS_REQUIRED):
                path = output_path(dataset, arm, seed, ROUNDS, DIAG_DIR)
                if check_diagnostic_json(path, f"{dataset}/{arm}/seed{seed}"):
                    n_complete += 1
            if n_complete == N_SEEDS_REQUIRED:
                ok(f"{dataset}/{arm}: all {N_SEEDS_REQUIRED} seeds present with complete diagnosis blocks")
            else:
                fail(f"{dataset}/{arm}: only {n_complete}/{N_SEEDS_REQUIRED} seeds complete")


# --------------------------------------------------------------------------- #
# 2. Consolidated ablation table JSON
# --------------------------------------------------------------------------- #
def check_ablation_table_json(dataset: str) -> None:
    dataset_tag = dataset.replace("-", "_")
    path = os.path.join(ABLATION_DIR, f"{dataset_tag}_ablation_table.json")
    label = f"{dataset} ablation table"
    if not os.path.exists(path):
        fail(f"{label}: missing {path}")
        return
    try:
        with open(path, encoding="utf-8") as f:
            table = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        fail(f"{label}: {path} failed to load ({exc})")
        return

    n_failures_before = len(FAILURES)

    for arm in ALL_ARM_NAMES:
        arm_table = table.get("arms", {}).get(arm)
        if arm_table is None:
            fail(f"{label}: missing arm '{arm}'")
            continue
        for gkey in GAMMA_KEYS:
            cell_group = arm_table.get(gkey)
            if cell_group is None:
                fail(f"{label}: arm '{arm}' missing {gkey}")
                continue
            missing_cells = REQUIRED_ARM_CELL_KEYS - set(cell_group.keys())
            if missing_cells:
                fail(f"{label}: arm '{arm}'[{gkey}] missing cells {sorted(missing_cells)}")
                continue
            for cell_name in REQUIRED_ARM_CELL_KEYS:
                cell = cell_group[cell_name]
                missing_fields = REQUIRED_CELL_FIELDS - set(cell.keys())
                if missing_fields:
                    fail(f"{label}: arm '{arm}'[{gkey}][{cell_name}] missing fields {sorted(missing_fields)}")
                elif len(cell.get("per_seed", [])) != N_SEEDS_REQUIRED:
                    fail(f"{label}: arm '{arm}'[{gkey}][{cell_name}] has "
                         f"{len(cell.get('per_seed', []))} seeds, expected {N_SEEDS_REQUIRED}")

    for method in ("fedavg", "static"):
        baseline = table.get("baselines", {}).get(method)
        if baseline is None:
            fail(f"{label}: missing baseline '{method}'")
            continue
        if len(baseline.get("rare_acc", {}).get("per_seed", [])) != N_SEEDS_REQUIRED:
            fail(f"{label}: baseline '{method}' rare_acc has "
                 f"{len(baseline.get('rare_acc', {}).get('per_seed', []))} seeds, expected {N_SEEDS_REQUIRED}")

    for gkey in GAMMA_KEYS:
        gamma_comps = table.get("comparisons", {}).get(gkey)
        if gamma_comps is None:
            fail(f"{label}: missing comparisons[{gkey}]")
            continue
        missing_arms = REQUIRED_COMPARISON_ARMS - set(gamma_comps.keys())
        if missing_arms:
            fail(f"{label}: comparisons[{gkey}] missing comparison arms {sorted(missing_arms)}")
            continue
        for ref_arm, comp in gamma_comps.items():
            if comp is None:
                fail(f"{label}: comparisons[{gkey}][{ref_arm}] is null (data not yet generated)")
                continue
            missing_fields = REQUIRED_COMPARISON_FIELDS - set(comp.keys())
            if missing_fields:
                fail(f"{label}: comparisons[{gkey}][{ref_arm}] missing fields {sorted(missing_fields)}")

        # comparisons_all_acc mirrors comparisons exactly (F2/A4 -- reporting
        # only rare-class accuracy let a starvation pathology read as parity
        # with the fix; without this check a builder regression that dropped
        # the all-acc family would leave the completeness gate green).
        gamma_comps_all_acc = table.get("comparisons_all_acc", {}).get(gkey)
        if gamma_comps_all_acc is None:
            fail(f"{label}: missing comparisons_all_acc[{gkey}]")
        else:
            missing_arms_all_acc = REQUIRED_COMPARISON_ARMS - set(gamma_comps_all_acc.keys())
            if missing_arms_all_acc:
                fail(f"{label}: comparisons_all_acc[{gkey}] missing comparison arms {sorted(missing_arms_all_acc)}")
            for ref_arm, comp in gamma_comps_all_acc.items():
                if comp is None:
                    fail(f"{label}: comparisons_all_acc[{gkey}][{ref_arm}] is null (data not yet generated)")
                    continue
                missing_fields = REQUIRED_COMPARISON_FIELDS - set(comp.keys())
                if missing_fields:
                    fail(f"{label}: comparisons_all_acc[{gkey}][{ref_arm}] missing fields {sorted(missing_fields)}")

        fx = table.get("factorial", {}).get(gkey)
        if fx is None:
            fail(f"{label}: missing factorial[{gkey}]")
        else:
            missing_fx = {"projection_main_effect", "eta_main_effect", "interaction"} - set(fx.keys())
            if missing_fx:
                fail(f"{label}: factorial[{gkey}] missing {sorted(missing_fx)}")

    if len(FAILURES) == n_failures_before:
        ok(f"{label}: {path} loads with every arm x gamma cell, all 5 comparisons "
           f"(rare-acc and all-acc families), and the factorial read")


# --------------------------------------------------------------------------- #
# 3. Rendered table + figure
# --------------------------------------------------------------------------- #
def check_figure(path: str, label: str) -> None:
    if not os.path.exists(path):
        fail(f"{label}: missing {path}")
        return
    if os.path.getsize(path) == 0:
        fail(f"{label}: {path} is empty (0 bytes)")
        return
    ok(f"{label}: {path} exists and is non-empty")


# --------------------------------------------------------------------------- #
# 4. Theory doc section-heading guard
# --------------------------------------------------------------------------- #
def check_theory_doc() -> None:
    if not os.path.exists(THEORY_DOC):
        fail(f"theory doc: missing {THEORY_DOC}")
        return
    with open(THEORY_DOC, encoding="utf-8") as f:
        text = f.read().lower()
    missing = [label for substr, label in REQUIRED_THEORY_HEADINGS if substr not in text]
    if missing:
        fail(f"theory doc: {THEORY_DOC} missing required section content: {missing}")
    else:
        ok(f"theory doc: {THEORY_DOC} contains all required section content")


# --------------------------------------------------------------------------- #
# 5. Manifest path check (same git ls-files pattern as check_stage_a_complete.py)
# --------------------------------------------------------------------------- #
def check_manifest_paths() -> None:
    if not os.path.exists(MANIFEST_PATH):
        fail(f"manifest: missing {MANIFEST_PATH}")
        return
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        text = f.read()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        tracked = set(
            subprocess.run(
                ["git", "ls-files"], cwd=repo_root, capture_output=True, text=True, check=True,
            ).stdout.splitlines()
        )
    except Exception as exc:
        fail(f"manifest: could not run `git ls-files` ({exc})")
        return

    referenced_paths = sorted(set(re.findall(r"`(results/[^`\s]+)`", text)))
    if not referenced_paths:
        fail(f"manifest: no `results/...` paths found in {MANIFEST_PATH}")
        return

    missing_on_disk = []
    missing_from_git = []
    for p in referenced_paths:
        abs_p = os.path.join(repo_root, p)
        if not os.path.exists(abs_p):
            missing_on_disk.append(p)
            continue
        if p.endswith("/"):
            if not any(t.startswith(p) for t in tracked):
                missing_from_git.append(p)
            continue
        if p not in tracked:
            missing_from_git.append(p)

    if missing_on_disk:
        fail(f"manifest: paths listed but not on disk: {missing_on_disk}")
    if missing_from_git:
        fail(f"manifest: paths listed but not git-tracked: {missing_from_git}")
    if not missing_on_disk and not missing_from_git:
        ok(f"manifest: all {len(referenced_paths)} referenced results/ paths exist on disk and are git-tracked")


def check_doc_links() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    broken = find_broken_links(repo_root)
    if broken:
        for md_file, target in broken:
            fail(f"doc link: {md_file} -> {target} does not resolve")
    else:
        ok("doc links: all relative markdown links resolve")


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 80)
    print("  Stage B completeness check")
    print("=" * 80)

    print("\n-- Per-seed diagnostic JSONs (4 arms x 10 seeds x 2 datasets) --")
    check_arm_seed_coverage()

    print("\n-- Consolidated ablation table JSONs --")
    for dataset in DATASETS:
        check_ablation_table_json(dataset)

    print("\n-- Rendered tables + figures --")
    for dataset in DATASETS:
        dataset_tag = dataset.replace("-", "_")
        check_figure(f"{ABLATION_DIR}/{dataset_tag}_ablation_table.md", f"{dataset} rendered table")
        check_figure(f"{ABLATION_DIR}/{dataset_tag}_ablation.png", f"{dataset} rendered figure")

    print("\n-- Theory doc --")
    check_theory_doc()

    print("\n-- Manifest --")
    check_manifest_paths()

    print("\n-- Documentation links --")
    check_doc_links()

    print("\n" + "=" * 80)
    if FAILURES:
        print(f"  STAGE B INCOMPLETE — {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 80)
        return 1
    print("  STAGE B COMPLETE — all checks passed.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
