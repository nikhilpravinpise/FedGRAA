"""Stage A completeness check (stage_a_plan.md §7).

Asserts the full Stage-A artifact set exists, is loadable, and is internally
consistent, exiting non-zero on any gap. Standalone (not part of ``pytest
tests``) so the unit suite stays decoupled from large committed binaries.

Usage::

    python scripts/check_stage_a_complete.py
"""

from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fedgraa.experiment import load_n_sweep_results, load_results  # noqa: E402
from scripts.check_doc_links import find_broken_links  # noqa: E402

RESULTS_DIR = "results"
FIGURES_DIR = "results/figures"
DIAG_DIR = "results/diagnostics"
MANIFEST_PATH = "results/README.md"

GAMMA_KEYS = {"fedavg", "static", "fedgraa_g0.0", "fedgraa_g0.5", "fedgraa_g1.0"}
N_KEYS = {3, 5, 7}

FAILURES: list = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def check_gamma_sweep_json(path: str, label: str) -> None:
    if not os.path.exists(path):
        fail(f"{label}: missing {path}")
        return
    try:
        results = load_results(path)
    except Exception as exc:
        fail(f"{label}: {path} failed to load ({exc})")
        return
    missing = GAMMA_KEYS - set(results.keys())
    if missing:
        fail(f"{label}: {path} missing method keys {sorted(missing)}")
    else:
        ok(f"{label}: {path} loads with all 5 method keys")


def check_n_sweep_json(path: str, label: str) -> dict | None:
    if not os.path.exists(path):
        fail(f"{label}: missing {path}")
        return None
    try:
        results = load_n_sweep_results(path)
    except Exception as exc:
        fail(f"{label}: {path} failed to load ({exc})")
        return None
    missing_ns = N_KEYS - set(results.keys())
    if missing_ns:
        fail(f"{label}: {path} missing n keys {sorted(missing_ns)}")
        return results
    n_failures_before = len(FAILURES)
    for n in N_KEYS:
        missing_methods = GAMMA_KEYS - set(results[n].keys())
        if missing_methods:
            fail(f"{label}: {path} n={n} missing method keys {sorted(missing_methods)}")
    if len(FAILURES) == n_failures_before:
        ok(f"{label}: {path} loads with n={{3,5,7}} and all 5 method keys per n")
    return results


def check_figure(path: str, label: str) -> None:
    if not os.path.exists(path):
        fail(f"{label}: missing figure {path}")
        return
    if os.path.getsize(path) == 0:
        fail(f"{label}: figure {path} is empty (0 bytes)")
        return
    ok(f"{label}: figure {path} exists and is non-empty")


def check_n_sweep_figure_lines(path: str, label: str) -> None:
    """Rendered guard against the §4 plot_n_sweep regression: each panel must
    show >=5 Line2D objects (fedavg, static, fedgraa_g0.0/0.5/1.0)."""
    if not os.path.exists(path):
        fail(f"{label}: missing figure {path} (cannot check line count)")
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        from fedgraa.plot import plot_n_sweep
        json_path = path.replace("figures/", "").replace("_figure2_3.png", "_n_sweep.json")
        json_path = os.path.join(RESULTS_DIR, os.path.basename(json_path))
        results = load_n_sweep_results(json_path)
        fig = plot_n_sweep(results, metric="last_n")
        for ax in fig.axes:
            n_lines = len(ax.get_lines())
            if n_lines < 5:
                fail(f"{label}: panel has only {n_lines} lines (<5) — FedGRAA curves may be missing")
                return
        ok(f"{label}: every n-sweep panel renders >=5 lines")
    except Exception as exc:
        fail(f"{label}: could not verify rendered line count ({exc})")


def check_combined_fix_band(path: str, label: str, reference_path: str | None = None) -> None:
    if not os.path.exists(path):
        fail(f"{label}: missing {path}")
        return
    import json
    with open(path, encoding="utf-8") as f:
        band = json.load(f)
    expected_keys = {"g0.0", "g0.5", "g1.0"}
    if set(band.keys()) != expected_keys:
        fail(f"{label}: {path} has keys {sorted(band.keys())}, expected {sorted(expected_keys)}")
        return
    required_fields = {
        "gamma", "seeds", "all_acc_per_seed", "last_n_acc_per_seed",
        "blowup_ratio_per_seed", "all_acc_mean", "all_acc_std",
        "last_n_acc_mean", "last_n_acc_std",
    }
    for key, entry in band.items():
        missing = required_fields - set(entry.keys())
        if missing:
            fail(f"{label}: {path} entry {key} missing fields {sorted(missing)}")
            return
    if reference_path and os.path.exists(reference_path):
        with open(reference_path, encoding="utf-8") as f:
            ref = json.load(f)
        if set(ref.keys()) != set(band.keys()):
            fail(f"{label}: {path} schema keys diverge from reference {reference_path}")
            return
        for key in ref:
            ref_fields = set(ref[key].keys())
            band_fields = set(band[key].keys())
            if ref_fields != band_fields:
                fail(f"{label}: {path}[{key}] fields {sorted(band_fields)} != "
                     f"reference fields {sorted(ref_fields)}")
                return
    ok(f"{label}: {path} matches expected combined-fix band schema")


def check_manifest_paths() -> None:
    if not os.path.exists(MANIFEST_PATH):
        fail(f"manifest: missing {MANIFEST_PATH}")
        return
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        text = f.read()

    try:
        tracked = set(
            subprocess.run(
                ["git", "ls-files"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                capture_output=True, text=True, check=True,
            ).stdout.splitlines()
        )
    except Exception as exc:
        fail(f"manifest: could not run `git ls-files` ({exc})")
        return

    import re
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    referenced_paths = sorted(set(re.findall(r"`(results/[^`\s]+)`", text)))
    if not referenced_paths:
        fail("manifest: no `results/...` paths found in README.md")
        return

    missing_on_disk = []
    missing_from_git = []
    for p in referenced_paths:
        abs_p = os.path.join(repo_root, p)
        if not os.path.exists(abs_p):
            missing_on_disk.append(p)
            continue
        if p.endswith("/"):
            # Directory reference: satisfied if any tracked file lives under it.
            if not any(t.startswith(p) for t in tracked):
                missing_from_git.append(p)
            continue
        rel_to_repo_root = os.path.relpath(abs_p, os.path.dirname(repo_root)).replace("\\", "/")
        if rel_to_repo_root not in tracked and p not in tracked:
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


def main() -> int:
    print("=" * 80)
    print("  Stage A completeness check")
    print("=" * 80)

    print("\n-- Gamma-sweep JSONs (Figure 1) --")
    check_gamma_sweep_json(f"{RESULTS_DIR}/CIC-IDS-2017_gamma_sweep.json", "CIC gamma sweep")
    check_gamma_sweep_json(f"{RESULTS_DIR}/UNSW-NB15_gamma_sweep.json", "UNSW gamma sweep")

    print("\n-- N-sweep JSONs (Figures 2-3) --")
    check_n_sweep_json(f"{RESULTS_DIR}/CIC-IDS-2017_n_sweep.json", "CIC n-sweep")
    check_n_sweep_json(f"{RESULTS_DIR}/UNSW-NB15_n_sweep.json", "UNSW n-sweep")

    print("\n-- Figures --")
    check_figure(f"{FIGURES_DIR}/CIC-IDS-2017_figure1.png", "CIC Figure 1")
    check_figure(f"{FIGURES_DIR}/UNSW-NB15_figure1.png", "UNSW Figure 1")
    check_figure(f"{FIGURES_DIR}/CIC-IDS-2017_figure2_3.png", "CIC Figures 2-3")
    check_figure(f"{FIGURES_DIR}/UNSW-NB15_figure2_3.png", "UNSW Figures 2-3")
    for suffix in (
        "composite_loss.png", "summary.md",
        "weights_g0p0.png", "weights_g0p5.png", "weights_g1p0.png",
    ):
        check_figure(f"{FIGURES_DIR}/diagnosis_unsw_combined_fix_{suffix}", "UNSW combined-fix diagnostics")

    print("\n-- N-sweep rendered line count (regression guard for §4 plot bug) --")
    check_n_sweep_figure_lines(f"{FIGURES_DIR}/CIC-IDS-2017_figure2_3.png", "CIC Figures 2-3")
    check_n_sweep_figure_lines(f"{FIGURES_DIR}/UNSW-NB15_figure2_3.png", "UNSW Figures 2-3")

    print("\n-- Combined-fix seed-variance bands --")
    check_combined_fix_band(
        f"{DIAG_DIR}/seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json", "CIC combined-fix band",
    )
    check_combined_fix_band(
        f"{DIAG_DIR}/seed_sweep_UNSW_NB15_fedgraa_combined_fix.json", "UNSW combined-fix band",
        reference_path=f"{DIAG_DIR}/seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json",
    )

    print("\n-- Manifest --")
    check_manifest_paths()

    print("\n-- Documentation links --")
    check_doc_links()

    print("\n" + "=" * 80)
    if FAILURES:
        print(f"  STAGE A INCOMPLETE — {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 80)
        return 1
    print("  STAGE A COMPLETE — all checks passed.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
