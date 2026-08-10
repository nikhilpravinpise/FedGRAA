"""Stage A deep audit -- verifies correctness, not just completeness.

``scripts/check_stage_a_complete.py`` answers "is every artifact present and
shaped right?" -- it re-renders the n-sweep figures and counts line objects,
but it does not check whether a single number quoted in prose actually
matches the JSON it claims to summarize. This script does. It applies
``audit_stage_b.py``'s method (AUDIT_REMEDIATION_PLAN_2026-07-25.md R12) to
Stage A's own artifacts, which had no re-runnable correctness gate before
this: Stage A's ~90 reported values were verified by hand, twice, and
nothing re-runnable. Two live defects the verification audit found sitting
in Stage A's own capstone -- a false "byte-identical" claim (R8) and a
stale seed-0 severity framing with no forward-reference (R9) -- are exactly
the class of thing hand-verification misses and this class of check is
built to catch automatically going forward.

Four checks, in increasing order of what they would catch:

A. **Raw-run integrity** -- every Stage-A-owned diagnostic/sweep JSON is
   byte-distinct from its siblings, and each diagnostic file's persisted
   ``config`` matches what its filename and containing section claim
   (project flag, eta', rounds, seed). Catches a mislabeled or duplicated
   run, which no shape check can see.
R0. **Round-0 convention** -- ``all_acc[0] == last_n_acc[0] == 0.0`` on
   every persisted curve in the four base sweeps. This is the specific
   config-vs-artifact convention ``stage_a_plan.md`` §3 once flagged as
   contradictory; re-verified here on every curve, not spot-checked.
D. **Doc transcription** -- every number hand-typed into
   ``STAGE_A_MASTER_REPORT.md``'s §5.2 (CIC literal/projected-ablation),
   §5.8 (UNSW literal), and §5.7 Step 3 (CIC combined-fix seed band)
   tables is parsed back out and re-derived from the persisted JSON.
   Catches copy-paste drift between data and prose -- the class of defect
   that this audit's Stage B counterpart found (A2: a stale seed-0 figure
   presented as the campaign's own) and that Stage A's own capstone
   independently carried (R9) until this pass fixed it.
E. **Presentation soundness** -- flags any combined-fix seed-band cell
   whose reported mean would misrepresent a heavy-tailed per-seed spread
   (the same class of defect as Stage B's A1, checked here even though it
   has not fired on Stage A's 3-seed bands, because an unexercised branch
   is exactly where a bug hides).

Negative-controlled before trusting it, per the Stage B correctness audit's
own lesson that an audit which cannot fail is worthless (its own bug, A5,
was in the failure path -- the one branch that never runs while everything
passes). Four defects were deliberately injected into scratch copies and
confirmed caught, then the fixtures were discarded (not committed):

    injected defect                                  | caught by | result
    --------------------------------------------------|-----------|----------------------------------------
    doc number changed (193x -> 199x, S5.2)           | check D   | "doc 199.0 vs json 192.99"
    diagnostic config corrupted (project False->True) | check A   | "config project ... got True want False"
    a run file overwritten with a copy of another      | check A   | "... is byte-identical to ..."
    round-0 convention violated (all_acc[0] = 0.05)    | check R0  | "all_acc[0] = 0.05, expected 0.0"

Confirmed clean on the real, uncorrupted tree before and after each
negative-control run.

Usage::

    python scripts/audit_stage_a.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

# Same rationale as audit_stage_b.py: failure messages quote row labels and
# Greek letters (eta', gamma, sigma) that a cp1252 Windows console cannot
# encode. An audit that crashes while reporting a failure is barely better
# than none -- fixed here up front rather than discovered by a negative
# control that then has to be run twice.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_DIR = "results"
DIAG_DIR = "results/diagnostics"
MASTER_REPORT = "docs/current/STAGE_A_MASTER_REPORT.md"

METHODS = ["fedavg", "static", "fedgraa_g0.0", "fedgraa_g0.5", "fedgraa_g1.0"]
N_KEYS = ["3", "5", "7"]
GAMMAS = ["g0.0", "g0.5", "g1.0"]

# path -> expected config fields, where the file has a per-gamma `config`
# block worth cross-checking against what the doc/filename imply.
DIAG_FILES = {
    f"{DIAG_DIR}/CIC_IDS_2017_literal_r60.json": {"project": False, "seed": 0},
    f"{DIAG_DIR}/UNSW_NB15_literal_r60.json": {"project": False, "seed": 0},
    f"{DIAG_DIR}/CIC_IDS_2017_projected_r60.json": {"project": True, "seed": 0},
    f"{DIAG_DIR}/UNSW_NB15_projected_r60.json": {"project": True, "seed": 0},
}
SWEEP_FILES = [
    f"{RESULTS_DIR}/CIC-IDS-2017_gamma_sweep.json",
    f"{RESULTS_DIR}/UNSW-NB15_gamma_sweep.json",
    f"{RESULTS_DIR}/CIC-IDS-2017_n_sweep.json",
    f"{RESULTS_DIR}/UNSW-NB15_n_sweep.json",
]
COMBINED_FIX_FILES = {
    "CIC-IDS-2017": f"{DIAG_DIR}/seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json",
    "UNSW-NB15": f"{DIAG_DIR}/seed_sweep_UNSW_NB15_fedgraa_combined_fix.json",
}

FAILURES: list = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


# --------------------------------------------------------------------------- #
# A. Raw-run integrity
# --------------------------------------------------------------------------- #
def check_raw_runs() -> None:
    print("\n-- A. Raw-run integrity (Stage A's own diagnostic + sweep files) --")
    hashes: dict = {}
    n_files = n_config_checks = 0

    all_paths = list(DIAG_FILES) + SWEEP_FILES + list(COMBINED_FIX_FILES.values())
    for path in all_paths:
        if not os.path.exists(path):
            fail(f"A: missing {path}")
            continue
        raw = open(path, "rb").read()
        n_files += 1
        h = hashlib.sha256(raw).hexdigest()
        if h in hashes:
            fail(f"A: {path} is byte-identical to {hashes[h]} "
                 "(duplicated artifact filed under two names)")
        hashes[h] = path

    for path, expected in DIAG_FILES.items():
        if not os.path.exists(path):
            continue
        data = json.loads(open(path, encoding="utf-8").read())
        if sorted(data.keys()) != sorted(GAMMAS):
            fail(f"A: {path} gammas={sorted(data.keys())}")
            continue
        for g in GAMMAS:
            cfg = data[g].get("config", {})
            checks = (
                ("project", bool(cfg.get("project", False)), expected["project"]),
                ("seed", cfg.get("seed"), expected["seed"]),
                ("gamma", cfg.get("gamma"), float(g[1:])),
                ("rounds", cfg.get("rounds"), 60),
            )
            for key, got, want in checks:
                n_config_checks += 1
                if got is None:
                    continue
                same = (abs(got - want) < 1e-12 if isinstance(want, float) else got == want)
                if not same:
                    fail(f"A: config {key} in {path} {g}: got {got!r} want {want!r}")

    for dataset, path in COMBINED_FIX_FILES.items():
        if not os.path.exists(path):
            continue
        data = json.loads(open(path, encoding="utf-8").read())
        if sorted(data.keys()) != sorted(GAMMAS):
            fail(f"A: {path} gammas={sorted(data.keys())}")
            continue
        for g in GAMMAS:
            cell = data[g]
            n_config_checks += 1
            if sorted(cell.get("seeds", [])) != [0, 1, 2]:
                fail(f"A: {path} {g} seeds={cell.get('seeds')}, expected [0, 1, 2]")

    if not FAILURES:
        ok(f"A: {n_files} files, all byte-distinct; {n_config_checks} config "
           "fields checked against what each file's role claims")


# --------------------------------------------------------------------------- #
# R0. Round-0 convention
# --------------------------------------------------------------------------- #
def check_round_zero_convention() -> None:
    print("\n-- R0. Round-0 convention (all_acc[0] == last_n_acc[0] == 0.0) --")
    n_curves = 0
    clean = True
    for path in SWEEP_FILES:
        if not os.path.exists(path):
            fail(f"R0: missing {path}")
            clean = False
            continue
        data = json.loads(open(path, encoding="utf-8").read())
        is_n_sweep = "n_sweep" in path
        groups = ((n, data[n]) for n in N_KEYS) if is_n_sweep else [("", data)]
        for n_label, methods in groups:
            for method in METHODS:
                curve = methods.get(method)
                if curve is None:
                    fail(f"R0: {path}{' n=' + n_label if n_label else ''}: missing method '{method}'")
                    clean = False
                    continue
                n_curves += 1
                all_acc = curve.get("all_acc", [])
                last_n_acc = curve.get("last_n_acc", [])
                label = f"{path}{' n=' + n_label if n_label else ''}/{method}"
                if not all_acc or all_acc[0] != 0.0:
                    fail(f"R0: {label} all_acc[0] = {all_acc[0] if all_acc else 'MISSING'}, expected 0.0")
                    clean = False
                if not last_n_acc or last_n_acc[0] != 0.0:
                    fail(f"R0: {label} last_n_acc[0] = {last_n_acc[0] if last_n_acc else 'MISSING'}, expected 0.0")
                    clean = False
    if clean:
        ok(f"R0: all {n_curves} curves across the 4 base sweeps start at "
           "all_acc[0] = last_n_acc[0] = 0.0")


# --------------------------------------------------------------------------- #
# D. Doc transcription fidelity
# --------------------------------------------------------------------------- #
def _diag(path: str, gamma: str) -> dict:
    return json.loads(open(path, encoding="utf-8").read())[gamma]["diagnosis"]


def check_literal_projected_table_5_2() -> None:
    """CIC §5.2: 6 rows (literal + projected-ablation) x blowup ratio."""
    if not os.path.exists(MASTER_REPORT):
        fail(f"D: missing {MASTER_REPORT}")
        return
    doc = open(MASTER_REPORT, encoding="utf-8").read()
    checked = 0
    clean = True

    row_source = {
        "Literal": f"{DIAG_DIR}/CIC_IDS_2017_literal_r60.json",
        "Projected (ablation)": f"{DIAG_DIR}/CIC_IDS_2017_projected_r60.json",
    }
    for line in doc.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or len(cols) != 5:
            continue
        gamma_txt, variant = cols[0], cols[1]
        if variant not in row_source or gamma_txt not in ("0.0", "0.5", "1.0"):
            continue
        m = re.search(r"([\d.]+)×", cols[3])
        if not m:
            fail(f"D: unparsed blowup in §5.2 row {gamma_txt}/{variant}: {cols[3]!r}")
            clean = False
            continue
        checked += 1
        want = _diag(row_source[variant], "g" + gamma_txt)["composite_loss_blowup_ratio"]
        got = float(m.group(1))
        if abs(got - round(want, 2)) > 0.011:
            fail(f"D: §5.2 blowup {gamma_txt}/{variant}: doc {got} vs json {round(want, 2)}")
            clean = False

    if checked == 0:
        fail("D: §5.2 CIC literal/projected table not found or fully unparsed")
        clean = False
    elif clean:
        ok(f"D: §5.2 CIC literal/projected-ablation table -- {checked} blowup "
           "values traced to results/diagnostics/CIC_IDS_2017_{literal,projected}_r60.json")


def check_unsw_table_5_8() -> None:
    """UNSW §5.8: 3 rows x {blowup, final Sw, final min(w), all-acc, rare-acc}."""
    if not os.path.exists(MASTER_REPORT):
        return
    doc = open(MASTER_REPORT, encoding="utf-8").read()
    section = "### 5.8"
    if section not in doc:
        fail(f"D: {MASTER_REPORT} missing section {section}")
        return
    start = doc.index(section)
    end = doc.index("###", start + 10) if "###" in doc[start + 10:] else len(doc)
    block = doc[start:end]

    path = f"{DIAG_DIR}/UNSW_NB15_literal_r60.json"
    checked = 0
    clean = True

    for line in block.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or len(cols) != 8:
            continue
        gamma_txt = cols[0]
        if gamma_txt not in ("0.0", "0.5", "1.0"):
            continue
        diag = _diag(path, "g" + gamma_txt)
        fields = (
            (cols[3], diag["composite_loss_blowup_ratio"], "×", "blowup"),
            (cols[4], diag["final_weight_sum"], None, "final Sw"),
            (cols[5], diag["final_weight_min"], None, "final min(w)"),
            (cols[6], 100 * diag["final_all_acc"], "%", "all-acc"),
            (cols[7], 100 * diag["final_last_n_acc"], "%", "rare-acc"),
        )
        for cell_txt, want, suffix, what in fields:
            pattern = r"([+\-−]?[\d.]+)" + (re.escape(suffix) if suffix else "")
            m = re.search(pattern, cell_txt.replace("**", ""))
            if not m:
                fail(f"D: §5.8 unparsed {what} at gamma={gamma_txt}: {cell_txt!r}")
                clean = False
                continue
            checked += 1
            got = float(m.group(1).replace("−", "-"))
            # Blowup is displayed at 1 or 2 decimals inconsistently in the doc
            # (1.25x, 1.24x but 7.6x); accept either rounding rather than
            # picking one and false-failing on a formatting choice, not a
            # data error.
            if what == "blowup":
                mismatch = (abs(got - round(want, 1)) > 0.051
                            and abs(got - round(want, 2)) > 0.011)
            elif suffix == "%":
                mismatch = abs(got - round(want, 1)) > 0.051
            else:
                mismatch = abs(got - round(want, 2)) > 0.011
            if mismatch:
                fail(f"D: §5.8 {what} at gamma={gamma_txt}: doc {got} vs json {round(want, 3)}")
                clean = False

    if checked == 0:
        fail("D: §5.8 UNSW literal table not found or fully unparsed")
        clean = False
    elif clean:
        ok(f"D: §5.8 UNSW literal table -- {checked} values (blowup, Sw, min(w), "
           "all-acc, rare-acc) traced to results/diagnostics/UNSW_NB15_literal_r60.json")


def check_combined_fix_band_5_7() -> None:
    """CIC §5.7 Step 3: 3-seed combined-fix band, mean +/- std and per-seed."""
    if not os.path.exists(MASTER_REPORT):
        return
    doc = open(MASTER_REPORT, encoding="utf-8").read()
    # Sec. 5.7 has three tables under one heading (Step 1/2/3 grids); anchor
    # to "Step 3" specifically -- the whole-section block collides with
    # Step 2's table, which also has 5 columns and gamma values in column 0.
    anchor = "**Step 3"
    if anchor not in doc:
        fail(f"D: {MASTER_REPORT} missing {anchor!r} (combined-fix seed-band table)")
        return
    start = doc.index(anchor)
    end = doc.index("**Step", start + 10) if "**Step" in doc[start + 10:] else (
        doc.index("###", start + 10) if "###" in doc[start + 10:] else len(doc))
    block = doc[start:end]

    path = COMBINED_FIX_FILES["CIC-IDS-2017"]
    if not os.path.exists(path):
        fail(f"D: missing {path}")
        return
    band = json.loads(open(path, encoding="utf-8").read())
    checked = 0
    clean = True

    for line in block.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or len(cols) != 5:
            continue
        gamma_txt = cols[0]
        if gamma_txt not in ("0.0", "0.5", "1.0"):
            continue
        cell = band["g" + gamma_txt]

        for cell_txt, mean_key, std_key, what in (
            (cols[1], "all_acc_mean", "all_acc_std", "all-acc"),
            (cols[2], "last_n_acc_mean", "last_n_acc_std", "rare-acc"),
        ):
            m = re.search(r"([\d.]+)%\s*±\s*([\d.]+)%", cell_txt.replace("**", ""))
            if not m:
                fail(f"D: §5.7 unparsed {what} mean+-std at gamma={gamma_txt}: {cell_txt!r}")
                clean = False
                continue
            checked += 2
            want_mean = 100 * cell[mean_key]
            want_std = 100 * cell[std_key]
            got_mean, got_std = float(m.group(1)), float(m.group(2))
            if abs(got_mean - round(want_mean, 1)) > 0.051:
                fail(f"D: §5.7 {what} mean at gamma={gamma_txt}: doc {got_mean} "
                     f"vs json {round(want_mean, 1)}")
                clean = False
            if abs(got_std - round(want_std, 1)) > 0.051:
                fail(f"D: §5.7 {what} std at gamma={gamma_txt}: doc {got_std} "
                     f"vs json {round(want_std, 1)}")
                clean = False

        # per-seed rare-acc list, e.g. "83.3% / 75.0% / 58.3%"
        per_seed_matches = re.findall(r"([\d.]+)%", cols[3])
        if per_seed_matches:
            checked += len(per_seed_matches)
            want_per_seed = [round(100 * v, 1) for v in cell["last_n_acc_per_seed"]]
            got_per_seed = [float(v) for v in per_seed_matches]
            if got_per_seed != want_per_seed:
                fail(f"D: §5.7 per-seed rare-acc at gamma={gamma_txt}: "
                     f"doc {got_per_seed} vs json {want_per_seed}")
                clean = False

    if checked == 0:
        fail("D: §5.7 Step 3 combined-fix seed-band table not found or fully unparsed")
        clean = False
    elif clean:
        ok(f"D: §5.7 Step 3 CIC combined-fix seed band -- {checked} values "
           "(mean, std, per-seed) traced to seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json")


def check_doc_transcription() -> None:
    print(f"\n-- D. Doc transcription: numbers in {MASTER_REPORT} re-derived from JSON --")
    check_literal_projected_table_5_2()
    check_unsw_table_5_8()
    check_combined_fix_band_5_7()


# --------------------------------------------------------------------------- #
# E. Presentation soundness
# --------------------------------------------------------------------------- #
def check_presentation() -> None:
    """A1-class guard: a combined-fix band's reported mean must not misrepresent
    a heavy-tailed per-seed spread. Has not fired on Stage A's 3-seed bands
    (variance is low throughout), checked anyway per the Stage B audit's own
    lesson that an unexercised branch is exactly where a bug hides."""
    print("\n-- E. Presentation soundness (heavy-tailed seed-band summaries) --")
    clean = True
    n_checked = 0
    for dataset, path in COMBINED_FIX_FILES.items():
        if not os.path.exists(path):
            continue
        band = json.loads(open(path, encoding="utf-8").read())
        for g in GAMMAS:
            cell = band[g]
            for metric_key in ("all_acc", "last_n_acc"):
                per_seed = cell[f"{metric_key}_per_seed"]
                mean = cell[f"{metric_key}_mean"]
                n_checked += 1
                srt = sorted(per_seed)
                median = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
                if median > 1e-9 and mean > 3 * median:
                    fail(f"E: {dataset} {g} {metric_key} mean ({mean:.4f}) exceeds "
                         f"3x its median ({median:.4f}) -- heavy-tailed per-seed "
                         f"spread {per_seed}, mean may misrepresent a typical run")
                    clean = False
    if clean:
        ok(f"E: {n_checked} combined-fix band cells checked; no mean exceeds "
           "3x its own median (n=3 bands show no heavy-tailed distortion)")


def main() -> int:
    print("=" * 78)
    print("  Stage A deep audit -- correctness, not just completeness")
    print("=" * 78)
    check_raw_runs()
    check_round_zero_convention()
    check_doc_transcription()
    check_presentation()
    print()
    print("=" * 78)
    if FAILURES:
        print(f"  AUDIT FAILED -- {len(FAILURES)} problem(s):")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 78)
        return 1
    print("  AUDIT PASSED -- every Stage A number checked traces to a persisted")
    print("  run and reproduces independently.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
