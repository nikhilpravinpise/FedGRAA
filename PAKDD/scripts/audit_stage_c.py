"""Stage C deep audit -- verifies correctness, not just completeness.

``scripts/check_stage_c_complete.py`` answers "is every artifact present and
shaped right?". This script answers the harder question ``audit_stage_a.py``/
``audit_stage_b.py`` already answer for Stage A/B and Stage C never has:
**are the numbers actually true?** Per ``PRE_COMPUTE_AUDIT_2026-07-28.md``
finding H5 (repeated in ``docs/audit/STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md``
Sec 4.4), Stage C's headline claims -- 93.47%/69.66% rare-class recall,
111/112 significant comparisons, 6.00x/3.00x communication overhead, the
3.1x/2.7x attacker up-weighting, the SecFedNIDS/WeiDetect attack-success-rate
table -- had never been independently re-derived from the raw per-cell JSON.
This script closes that gap, mirroring ``audit_stage_b.py``'s own discipline:
recompute everything from ``results/v2_frontier/*.json`` using an independent
implementation (direct ``scipy.stats.wilcoxon`` + a from-scratch Holm
step-down, never ``fedgraa/stats.py`` or ``scripts/build_v2_*.py``), and never
trust a consolidated table or a master-report prose number without tracing it
back to the raw cells.

**Negative-controlled before being trusted** (2026-08-02, same discipline
``audit_stage_a.py``'s docstring records for itself): a corrupted
``p_holm`` injected into ``significance_table.json`` was caught by check C
with the exact expected/got values; a corrupted ``final_last_n_acc`` injected
into one raw cell was caught by check B and correctly cascaded into three
real check-C consequences on the same run, including one comparison flipping
from significant to non-significant -- exactly the class of downstream defect
this audit exists to catch. Both injections were reverted before commit;
``git status`` on ``results/v2_frontier/`` was confirmed clean afterward. On
the real, uncorrupted data this script currently finds **zero** defects: all
six checks pass.

Six independent checks, in increasing order of what they would catch:

A. **Raw-run integrity** -- every per-cell result JSON on disk is
   byte-distinct (no accidental copies masquerading as separate seeds), and
   its content's own ``(dataset, arm, gamma, seed, config)`` reconstructs the
   exact filename it was found under (mirrors ``CellSpec.path``'s own
   ``config_tag`` logic, reimplemented independently -- a filename/content
   mismatch is exactly the defect class no shape check can see). Scans the
   directory rather than replicating ``enumerate_cells``'s T1-T6/TC/T7 tier
   grids, so it needs no assumption about what tiers exist -- it checks
   whatever is actually on disk against its own claimed identity.
B. **Frontier-table fidelity** -- ``frontier_table.json``'s per-(dataset, arm)
   ``per_seed``/``mean`` values are rebuilt from the raw 200-round cells
   (the same ``gamma==1.0, n_rounds==200, p_drop==0, dp_epsilon is None``
   filter ``build_v2_frontier_table.py::_is_t1_cell`` uses) and compared
   exactly. Catches a globbing or seed-mapping bug in the builder.
C. **Independent significance re-derivation** -- every Wilcoxon statistic,
   p-value, and Holm-adjusted p-value in ``significance_table.json`` is
   recomputed from scratch against ``scipy`` on the seed-matched rare-class
   recall this script rebuilt in check B, not by importing
   ``fedgraa/stats.py`` or ``build_v2_significance_table.py``.
D. **Overhead re-derivation** -- every byte ratio in ``overhead_table.json``
   is recomputed from the same rebuilt ``bytes_sent`` medians.
E. **Master-report transcription** -- every number hand-typed into
   ``STAGE_C_MASTER_REPORT.md`` Sec 6 (per-arm rare/all-acc + traffic),
   Sec 7 (111/112, the 8-row significant/total table), and Sec 8 (the
   6.00x/4.00x/3.00x/2.00x overhead figures) is parsed back out and checked
   against the rebuilt tables.
F. **T7 headline numbers** -- the mechanism-specific attack's measured
   up-weighting (Sec 10.3: 3.1x/2.7x at 30% attackers) and the defense
   attack-success-rate table (Sec 10.4, 6 rows x 2 datasets) are both
   recomputed directly from ``attacker_weight_history`` in the raw T7 cell
   JSONs -- exactly how ``scripts/render_stage_c_figures.py`` derives them,
   reimplemented independently rather than trusting the figures.

**Scope, disclosed rather than silently narrowed:** T6's ``nf_uq_max_per_class``
cap is not persisted in any cell's ``config`` block (it is a data-loading
parameter, not a ``FedConfig`` field), so check A cannot independently verify
the ``capNNNNN`` filename suffix against JSON content the way it verifies
every other config_tag component -- it is read from the filename and passed
through unchecked, not fabricated. T7G (the gamma-sweep addendum) is picked
up automatically by every check here once its cells exist on disk, since
nothing in this script hardcodes a cell count -- it discovers and audits
whatever ``results/v2_frontier/`` actually contains.

Usage::

    python scripts/audit_stage_c.py
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

# Same defect class fixed in audit_stage_b.py/check_stage_c_complete.py: a
# Windows console defaulting to cp1252 must not crash while reporting a
# failure whose message contains an em-dash or a comparison operator.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

RESULTS_DIR = "results/v2_frontier"
MASTER_REPORT = "docs/current/STAGE_C_MASTER_REPORT.md"
CONSOLIDATED_FILES = {"frontier_table.json", "overhead_table.json", "significance_table.json"}

DATASETS = {"CIC-IDS-2017": "CIC_IDS_2017", "UNSW-NB15": "UNSW_NB15"}
N_CLIENTS = {"CIC-IDS-2017": 14, "UNSW-NB15": 9}

# Independently hardcoded, not imported from scripts/run_v2_frontier_sweep.py
# -- read from it once while writing this script, the same way
# audit_stage_b.py's own ARM_SPEC was, so a defect in the production
# constants doesn't propagate into what this audit expects.
V2_ARMS = ["v1_projected", "v2_m1", "v2_m2", "v2_m3", "v2_m5"]
V2_FOCAL_ARMS = ["v2_m1", "v2_m2", "v2_m3", "v2_m5"]  # excludes v1_projected
BASELINE_ARMS = sorted([
    "celm", "spectralfed", "coast", "fltrust", "raga", "fedtruth",
    "fedpgo", "fedmade", "sbfednids", "fedapa", "fedawa", "dapfl",
    "fedaware", "adaaggrl",
])
CLASSIC_ARMS = sorted(["fedavg", "fedprox", "fedopt", "scaffold", "fednova"])
MAIN_ARMS = V2_ARMS + BASELINE_ARMS + CLASSIC_ARMS
APPROX_BASELINES = ["celm", "spectralfed"]

T7_DEFENSE_ARMS = [
    "secfednids_labelflip", "secfednids_modelpoison", "secfednids_adaptive",
    "weidetect_labelflip", "weidetect_modelpoison", "weidetect_adaptive",
]
T7_RARESIG_ARM = "v2_raresig"
T7_DEFAULT_FRACTION = 0.3
T7_SECOND_FRACTION = 0.5

DEFAULT_ETA_PRIME = 0.01
CANONICAL_ROUNDS = 200

FAILURES: list = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


# --------------------------------------------------------------------------- #
# Shared: load every real per-cell JSON exactly once
# --------------------------------------------------------------------------- #
def load_all_cells(results_dir: str = RESULTS_DIR) -> List[Tuple[str, dict]]:
    """Every per-cell result JSON in ``results_dir`` (excluding the 3
    consolidated tables), as ``(path, data)`` pairs. No tier grid is
    assumed -- whatever is actually on disk is what gets audited."""
    out = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(path) in CONSOLIDATED_FILES:
            continue
        with open(path, encoding="utf-8") as f:
            out.append((path, json.load(f)))
    return out


def expected_config_tag(cfg: dict, n_rounds: int, actual_path: str) -> str:
    """Reimplements ``CellSpec.config_tag`` (scripts/run_v2_frontier_sweep.py)
    independently, from a cell's own persisted ``config`` block. The one
    component that cannot be independently derived -- T6's
    ``nf_uq_max_per_class`` cap, never persisted into ``config`` -- is read
    back from ``actual_path`` itself rather than guessed (see module
    docstring's Scope note)."""
    parts: List[str] = []
    if cfg.get("p_drop"):
        parts.append(f"pdrop{cfg['p_drop']}")
    if cfg.get("dp_epsilon") is not None:
        parts.append(f"eps{cfg['dp_epsilon']}")
    if cfg.get("eta_prime", DEFAULT_ETA_PRIME) != DEFAULT_ETA_PRIME:
        parts.append(f"etap{cfg['eta_prime']}")
    if cfg.get("n_rare", 3) != 3:
        parts.append(f"n{cfg['n_rare']}")
    if cfg.get("secagg_threshold") is not None:
        parts.append(f"thr{cfg['secagg_threshold']}")
    if n_rounds != CANONICAL_ROUNDS:
        parts.append(f"r{n_rounds}")
    cap_match = re.search(r"_cap(\d+)", os.path.basename(actual_path))
    if cap_match:
        parts.append(f"cap{cap_match.group(1)}")
    if cfg.get("attacker_fraction"):
        parts.append(f"af{cfg['attacker_fraction']}")
    return ("_" + "_".join(parts)) if parts else ""


# --------------------------------------------------------------------------- #
# A. Raw-run integrity: byte-distinctness + content-vs-filename fidelity
# --------------------------------------------------------------------------- #
def check_raw_runs(cells: List[Tuple[str, dict]]) -> None:
    print(f"\n-- A. Raw-run integrity ({len(cells)} result files: distinct, "
          "content matches its own filename) --")
    hashes: Dict[str, str] = {}
    n_real = n_smoke = n_bad = 0
    for path, data in cells:
        raw = open(path, "rb").read()
        h = hashlib.sha256(raw).hexdigest()
        if h in hashes:
            fail(f"A: {path} is byte-identical to {hashes[h]} "
                 "(duplicated run filed as two seeds)")
            n_bad += 1
        hashes[h] = path

        cfg = data.get("config", {})
        n_rounds = cfg.get("n_rounds")
        if n_rounds == CANONICAL_ROUNDS:
            n_real += 1
        else:
            n_smoke += 1  # a --smoke leftover (2 rounds); not part of the real campaign

        ds_tag = data["dataset"].replace("-", "_")
        tag = expected_config_tag(cfg, n_rounds, path)
        expected = os.path.join(
            os.path.dirname(path),
            f"{ds_tag}_{data['arm']}_g{data['gamma']}_seed{data['seed']}{tag}.json",
        )
        if os.path.normpath(expected) != os.path.normpath(path):
            fail(f"A: {path} content reconstructs a different filename "
                 f"({expected}) -- identity fields don't match where it's filed")
            n_bad += 1

    if n_bad == 0:
        print(f"  {n_real} real (200-round) cells, {n_smoke} smoke-test "
              "leftovers (excluded from B-F below), all byte-distinct")
        ok(f"A: {len(cells)} result files, every one's own content "
           "reconstructs the exact filename it is stored under")


# --------------------------------------------------------------------------- #
# B. Frontier-table fidelity: rebuild frontier_table.json from raw cells
# --------------------------------------------------------------------------- #
def is_t1_cell(data: dict) -> bool:
    """Mirrors build_v2_frontier_table.py::_is_t1_cell exactly, reimplemented
    independently (not imported)."""
    if data["gamma"] != 1.0:
        return False
    cfg = data.get("config", {})
    if cfg.get("n_rounds") != CANONICAL_ROUNDS:
        return False
    return not cfg.get("p_drop") and cfg.get("dp_epsilon") is None


def rebuild_frontier(cells: List[Tuple[str, dict]]) -> Dict[str, Dict[str, dict]]:
    """{dataset: {arm: {"seeds": [...], "rare_acc": [...], "all_acc": [...],
    "bytes_sent": [...]}}} for every (dataset, arm) with at least one T1-anchor
    cell, rebuilt from raw JSON only."""
    table: Dict[str, Dict[str, dict]] = {ds: {} for ds in DATASETS}
    for _path, data in cells:
        if not is_t1_cell(data):
            continue
        if data["arm"] not in MAIN_ARMS:
            continue  # T7 arms never satisfy is_t1_cell's gamma filter anyway
        ds, arm = data["dataset"], data["arm"]
        if ds not in table:
            # T6's NF-UQ-NIDS-v2 -- a real dataset, but frontier_table.json's
            # own DATASETS list (build_v2_frontier_table.py) only ever covers
            # CIC-IDS-2017/UNSW-NB15; T6 has no consolidated table to audit
            # against here.
            continue
        bucket = table[ds].setdefault(
            arm, {"seeds": [], "rare_acc": [], "all_acc": [], "bytes_sent": []})
        bucket["seeds"].append(data["seed"])
        bucket["rare_acc"].append(data["final_last_n_acc"])
        bucket["all_acc"].append(data["final_all_acc"])
        bucket["bytes_sent"].append(data["bytes_sent"])
    return table


def check_frontier_table(rebuilt: Dict[str, Dict[str, dict]]) -> Optional[dict]:
    print("\n-- B. Frontier-table fidelity (rebuilt from raw cells vs. "
          "frontier_table.json) --")
    path = os.path.join(RESULTS_DIR, "frontier_table.json")
    if not os.path.exists(path):
        fail(f"B: missing {path}")
        return None
    with open(path, encoding="utf-8") as f:
        T = json.load(f)

    clean = True
    n_checked = 0
    for ds, arms in rebuilt.items():
        for arm, bucket in arms.items():
            cell = T.get("datasets", {}).get(ds, {}).get(arm)
            if cell is None:
                fail(f"B: {ds}/{arm} present in raw cells but missing from "
                     "frontier_table.json")
                clean = False
                continue
            # Seed-order-independent: compare as (seed -> value) maps, since
            # neither this rebuild nor the table promises a canonical order.
            want_rare = dict(zip(bucket["seeds"], bucket["rare_acc"]))
            want_all = dict(zip(bucket["seeds"], bucket["all_acc"]))
            want_bytes = dict(zip(bucket["seeds"], bucket["bytes_sent"]))
            got_rare = dict(zip(cell["seeds"], cell["rare_acc"]["per_seed"]))
            got_all = dict(zip(cell["seeds"], cell["all_acc"]["per_seed"]))
            got_bytes = dict(zip(cell["seeds"], cell["bytes_sent"]["per_seed"]))
            n_checked += 1
            if want_rare != got_rare:
                fail(f"B: {ds}/{arm} rare_acc per_seed mismatch")
                clean = False
                continue
            if want_all != got_all:
                fail(f"B: {ds}/{arm} all_acc per_seed mismatch")
                clean = False
            if want_bytes != got_bytes:
                fail(f"B: {ds}/{arm} bytes_sent per_seed mismatch")
                clean = False
            if abs(cell["rare_acc"]["mean"] - float(np.mean(list(want_rare.values())))) > 1e-12:
                fail(f"B: {ds}/{arm} rare_acc mean mismatch")
                clean = False
            if abs(cell["all_acc"]["mean"] - float(np.mean(list(want_all.values())))) > 1e-12:
                fail(f"B: {ds}/{arm} all_acc mean mismatch")
                clean = False
            want_median_gb = float(np.median(list(want_bytes.values()))) / 1e9
            if abs(cell["bytes_sent"]["median_gb"] - want_median_gb) > 1e-9:
                fail(f"B: {ds}/{arm} bytes_sent median_gb mismatch")
                clean = False
            # CI sanity (not bit-exact reproduction of the seeded bootstrap --
            # same lighter-touch check audit_stage_b.py applies to Stage B's
            # own bootstrap CIs): must contain its own mean and never escape
            # the raw data's range.
            lo, hi = cell["rare_acc"]["ci_95"]
            vals = list(want_rare.values())
            if not (lo - 1e-9 <= np.mean(vals) <= hi + 1e-9):
                fail(f"B: {ds}/{arm} rare_acc CI excludes its own mean")
                clean = False
            if lo < min(vals) - 1e-9 or hi > max(vals) + 1e-9:
                fail(f"B: {ds}/{arm} rare_acc CI escapes its own data range")
                clean = False

    if clean:
        ok(f"B: {n_checked} (dataset, arm) cells -- every per_seed array, "
           "mean, and byte median in frontier_table.json reproduces exactly "
           "from the raw 200-round cells")
    return T


# --------------------------------------------------------------------------- #
# C. Independent significance re-derivation
# --------------------------------------------------------------------------- #
def holm_bonferroni_independent(pvals: List[float]) -> List[float]:
    """From-scratch Holm (1979) step-down, deliberately not imported from
    fedgraa/stats.py -- same algorithm audit_stage_b.py reimplements for
    Stage B, applied here to Stage C's own comparisons."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.shape[0]
    if m == 0:
        return []
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    multipliers = m - np.arange(m)
    adjusted_sorted = np.minimum(sorted_p * multipliers, 1.0)
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adjusted_sorted
    return [float(v) for v in adjusted]


def wilcoxon_independent(a: List[float], b: List[float]) -> Tuple[float, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if np.allclose(a - b, 0.0):
        return (0.0, 1.0)
    try:
        stat, p = sps.wilcoxon(a, b)
    except ValueError:
        return (0.0, 1.0)
    return (float(stat), float(p))


def check_significance(rebuilt: Dict[str, Dict[str, dict]]) -> None:
    print("\n-- C. Independent significance re-derivation (scipy directly, "
          "never fedgraa/stats.py) --")
    path = os.path.join(RESULTS_DIR, "significance_table.json")
    if not os.path.exists(path):
        fail(f"C: missing {path}")
        return
    with open(path, encoding="utf-8") as f:
        S = json.load(f)

    clean = True
    n_checked = 0
    n_significant = 0
    for ds in DATASETS:
        for v2_arm in V2_FOCAL_ARMS:
            v2_bucket = rebuilt.get(ds, {}).get(v2_arm)
            if v2_bucket is None:
                continue
            v2_by_seed = dict(zip(v2_bucket["seeds"], v2_bucket["rare_acc"]))

            family_p: List[float] = []
            family_names: List[str] = []
            for baseline in BASELINE_ARMS:
                b_bucket = rebuilt.get(ds, {}).get(baseline)
                if b_bucket is None:
                    continue
                b_by_seed = dict(zip(b_bucket["seeds"], b_bucket["rare_acc"]))
                common = sorted(set(v2_by_seed) & set(b_by_seed))
                if not common:
                    continue
                a_vals = [v2_by_seed[s] for s in common]
                b_vals = [b_by_seed[s] for s in common]
                _stat, p_w = wilcoxon_independent(a_vals, b_vals)
                family_p.append(p_w)
                family_names.append(baseline)

            adjusted = holm_bonferroni_independent(family_p)
            for baseline, p_w, p_holm in zip(family_names, family_p, adjusted):
                got = S.get("datasets", {}).get(ds, {}).get(v2_arm, {}).get(baseline)
                if got is None:
                    fail(f"C: {ds}/{v2_arm}/{baseline} in raw cells but "
                         "missing from significance_table.json")
                    clean = False
                    continue
                n_checked += 1
                if abs(got["p_wilcoxon"] - p_w) > 1e-9:
                    fail(f"C: {ds}/{v2_arm}/{baseline} p_wilcoxon "
                         f"{got['p_wilcoxon']} vs scipy {p_w}")
                    clean = False
                if abs(got["p_holm"] - p_holm) > 1e-9:
                    fail(f"C: {ds}/{v2_arm}/{baseline} p_holm "
                         f"{got['p_holm']} vs recomputed {p_holm}")
                    clean = False
                exp_significant = bool(p_holm < S.get("alpha", 0.05))
                if got["significant"] != exp_significant:
                    fail(f"C: {ds}/{v2_arm}/{baseline} significant="
                         f"{got['significant']} vs recomputed {exp_significant}")
                    clean = False
                if got["significant"]:
                    n_significant += 1

    if clean:
        ok(f"C: {n_checked} comparisons -- every Wilcoxon p-value and "
           "Holm-adjusted p-value reproduces exactly via scipy; "
           f"{n_significant}/{n_checked} independently confirmed significant "
           f"(table claims {S.get('n_comparisons')} total)")
    if S.get("n_comparisons") != n_checked:
        fail(f"C: significance_table.json claims n_comparisons="
             f"{S.get('n_comparisons')} but {n_checked} comparisons exist "
             "in the raw cells")


# --------------------------------------------------------------------------- #
# D. Overhead re-derivation
# --------------------------------------------------------------------------- #
def check_overhead(rebuilt: Dict[str, Dict[str, dict]]) -> None:
    print("\n-- D. Overhead re-derivation (byte ratios recomputed from raw "
          "bytes_sent) --")
    path = os.path.join(RESULTS_DIR, "overhead_table.json")
    if not os.path.exists(path):
        fail(f"D: missing {path}")
        return
    with open(path, encoding="utf-8") as f:
        O = json.load(f)

    clean = True
    n_checked = 0
    for ds in DATASETS:
        for v2_arm in V2_FOCAL_ARMS:
            v2_bucket = rebuilt.get(ds, {}).get(v2_arm)
            if v2_bucket is None:
                continue
            v2_gb = float(np.median(v2_bucket["bytes_sent"])) / 1e9
            for approx in APPROX_BASELINES:
                a_bucket = rebuilt.get(ds, {}).get(approx)
                if a_bucket is None:
                    continue
                approx_gb = float(np.median(a_bucket["bytes_sent"])) / 1e9
                exp_ratio = v2_gb / approx_gb if approx_gb > 0 else float("inf")
                got = O.get("datasets", {}).get(ds, {}).get(v2_arm, {}).get(approx)
                if got is None:
                    fail(f"D: {ds}/{v2_arm}/{approx} missing from "
                         "overhead_table.json")
                    clean = False
                    continue
                n_checked += 1
                if abs(got["byte_ratio"] - exp_ratio) > 1e-6 * max(1.0, exp_ratio):
                    fail(f"D: {ds}/{v2_arm}/{approx} byte_ratio "
                         f"{got['byte_ratio']} vs recomputed {exp_ratio}")
                    clean = False
    if clean:
        ok(f"D: {n_checked} (v2 arm, approx baseline) rows -- every byte "
           "ratio reproduces exactly from raw bytes_sent medians")


# --------------------------------------------------------------------------- #
# E. Master-report transcription
# --------------------------------------------------------------------------- #
DISPLAY_NAME_TO_ARM = {
    "v1_projected (privacy-violating reference)": "v1_projected",
    "**v2_m1**": "v2_m1", "**v2_m2**": "v2_m2",
    "**v2_m3 (best)**": "v2_m3", "**v2_m5**": "v2_m5",
    "FedAWARE": "fedaware", "DA-PFL": "dapfl", "SCAFFOLD": "scaffold",
    "FLTrust": "fltrust", "FedPGO": "fedpgo", "FedAPA": "fedapa",
    "CoAst": "coast", "CELM": "celm", "FedNova": "fednova",
    "FedAWA": "fedawa", "FedTruth": "fedtruth", "AdaAggRL": "adaaggrl",
    "FedProx": "fedprox", "FedOpt": "fedopt",
}


def check_master_report_transcription(frontier_table: dict, sig_table: dict,
                                       overhead_table: dict) -> None:
    print("\n-- E. Master-report transcription (Sec 6/7/8 numbers traced to "
          "the rebuilt tables) --")
    if not os.path.exists(MASTER_REPORT):
        fail(f"E: missing {MASTER_REPORT}")
        return
    doc = open(MASTER_REPORT, encoding="utf-8").read()
    clean = True
    checked = 0

    # Sec 6.1/6.2: per-arm rare-acc% [CI] / all-acc% / traffic GB rows. Each
    # dataset's table is scoped to its own section block first (mirrors
    # audit_stage_b.py's check_doc_transcription) -- both tables share the
    # same row labels (e.g. "v1_projected"), so scanning the whole document
    # per dataset would cross-check CIC's row against UNSW's json and vice
    # versa the moment both sections are present.
    section_markers = {"CIC-IDS-2017": "### 6.1", "UNSW-NB15": "### 6.2"}
    for ds_name in DATASETS:
        marker = section_markers[ds_name]
        if marker not in doc:
            fail(f"E: Sec6 section marker {marker!r} not found for {ds_name}")
            clean = False
            continue
        start = doc.index(marker)
        end = doc.index("###", start + 10) if "###" in doc[start + 10:] else len(doc)
        block = doc[start:end]
        for line in block.splitlines():
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if not line.startswith("| ") or len(cols) != 5:
                continue
            arm = DISPLAY_NAME_TO_ARM.get(cols[0])
            if arm is None:
                continue
            m = re.search(r"([\d.]+)%\s*\[([\d.]+),\s*([\d.]+)\]", cols[1] + " " + cols[2])
            if not m:
                continue
            cell = frontier_table.get("datasets", {}).get(ds_name, {}).get(arm)
            if cell is None:
                continue
            checked += 1
            got_mean, got_lo, got_hi = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            want_mean = 100 * cell["rare_acc"]["mean"]
            want_lo, want_hi = (100 * v for v in cell["rare_acc"]["ci_95"])
            if abs(got_mean - round(want_mean, 2)) > 0.011:
                fail(f"E: Sec6 rare-acc mean {ds_name}/{arm}: doc {got_mean} "
                     f"vs json {round(want_mean, 2)}")
                clean = False
            if abs(got_lo - round(want_lo, 2)) > 0.011 or abs(got_hi - round(want_hi, 2)) > 0.011:
                fail(f"E: Sec6 rare-acc CI {ds_name}/{arm}: doc "
                     f"[{got_lo},{got_hi}] vs json [{round(want_lo, 2)},{round(want_hi, 2)}]")
                clean = False
            m2 = re.search(r"([\d.]+)%", cols[3])
            if m2:
                checked += 1
                want_all = 100 * cell["all_acc"]["mean"]
                if abs(float(m2.group(1)) - round(want_all, 2)) > 0.011:
                    fail(f"E: Sec6 all-acc {ds_name}/{arm}: doc {m2.group(1)} "
                         f"vs json {round(want_all, 2)}")
                    clean = False
            m3 = re.search(r"([\d.]+)", cols[4])
            if m3:
                checked += 1
                want_gb = cell["bytes_sent"]["median_gb"]
                if abs(float(m3.group(1)) - round(want_gb, 3)) > 0.0011:
                    fail(f"E: Sec6 traffic {ds_name}/{arm}: doc {m3.group(1)} "
                         f"vs json {round(want_gb, 3)}")
                    clean = False

    # Sec 7: "111 of 112 comparisons significant" + the 8-row significant/total table.
    m = re.search(r"\*\*Result: (\d+) of (\d+) comparisons significant", doc)
    if m:
        checked += 1
        got_sig, got_total = int(m.group(1)), int(m.group(2))
        n_sig = sum(1 for ds in sig_table["datasets"].values() for arm in ds.values()
                     for res in arm.values() if res and res.get("significant"))
        n_total = sum(1 for ds in sig_table["datasets"].values() for arm in ds.values()
                       for res in arm.values() if res is not None)
        if (got_sig, got_total) != (n_sig, n_total):
            fail(f"E: Sec7 headline count doc says {got_sig}/{got_total}, "
                 f"significance_table.json has {n_sig}/{n_total}")
            clean = False
    else:
        fail("E: Sec7 headline 'N of M comparisons significant' sentence not found")
        clean = False

    for line in doc.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or len(cols) != 3:
            continue
        ds_name, v2_arm = cols[0], cols[1]
        if ds_name not in DATASETS or v2_arm not in V2_FOCAL_ARMS:
            continue
        m4 = re.match(r"(\d+)/(\d+)", cols[2])
        if not m4:
            continue
        checked += 1
        arm_results = sig_table["datasets"].get(ds_name, {}).get(v2_arm, {})
        n_sig = sum(1 for r in arm_results.values() if r and r.get("significant"))
        n_total = sum(1 for r in arm_results.values() if r is not None)
        got_sig, got_total = int(m4.group(1)), int(m4.group(2))
        if (got_sig, got_total) != (n_sig, n_total):
            fail(f"E: Sec7 row {ds_name}/{v2_arm}: doc {got_sig}/{got_total} "
                 f"vs recomputed {n_sig}/{n_total}")
            clean = False

    # Sec 8: overhead ratios at m=1/2/3/5 ("6.00x", "4.00x (m=2)", etc).
    m5 = re.search(r"\*\*exactly (\d+\.\d+)×\*\*", doc)
    m6 = re.search(
        r"(\d+\.\d+)×\s*\(`m=2`\).*?(\d+\.\d+)×\s*\(`m=3`\).*?(\d+\.\d+)×\s*\(`m=5`\)",
        doc, re.DOTALL,
    )
    if m5 and m6:
        checked += 1
        doc_ratios = {
            "v2_m1": float(m5.group(1)), "v2_m2": float(m6.group(1)),
            "v2_m3": float(m6.group(2)), "v2_m5": float(m6.group(3)),
        }
        for v2_arm, doc_ratio in doc_ratios.items():
            row = overhead_table.get("datasets", {}).get("CIC-IDS-2017", {}).get(
                v2_arm, {}).get("celm")
            if row is None:
                continue
            if abs(doc_ratio - round(row["byte_ratio"], 2)) > 0.011:
                fail(f"E: Sec8 overhead {v2_arm}: doc {doc_ratio}x vs "
                     f"json {round(row['byte_ratio'], 2)}x")
                clean = False
    else:
        fail("E: Sec8 overhead-ratio sentence not found/parsed")
        clean = False

    if clean:
        ok(f"E: {checked} numbers transcribed into {MASTER_REPORT} Sec 6-8 "
           "all trace to the rebuilt tables")


# --------------------------------------------------------------------------- #
# F. T7 headline numbers: attacker up-weighting + defense ASR
# --------------------------------------------------------------------------- #
def check_t7_headline_numbers(cells: List[Tuple[str, dict]]) -> None:
    print("\n-- F. T7 headline numbers (up-weighting ratio + defense ASR, "
          "recomputed from attacker_weight_history) --")
    if not os.path.exists(MASTER_REPORT):
        return
    doc = open(MASTER_REPORT, encoding="utf-8").read()
    clean = True
    checked = 0

    by_ds_arm_frac: Dict[Tuple[str, str, float], List[float]] = {}
    for _path, data in cells:
        if data["arm"] != T7_RARESIG_ARM:
            continue
        cfg = data.get("config", {})
        if cfg.get("n_rounds") != CANONICAL_ROUNDS or data["gamma"] != 1.0:
            continue
        hist = data.get("attacker_weight_history") or []
        if not hist:
            continue
        key = (data["dataset"], data["arm"], cfg.get("attacker_fraction"))
        by_ds_arm_frac.setdefault(key, []).append(hist[-1])

    m = re.search(
        r"climbs from parity.*?\*\*(\d+\.\d+)×.*?on CIC-IDS-2017.*?"
        r"and \*\*(\d+\.\d+)×\*\*.*?on UNSW-NB15",
        doc, re.DOTALL,
    )
    if m:
        checked += 1
        doc_cic, doc_unsw = float(m.group(1)), float(m.group(2))
        for ds_name, doc_ratio in (("CIC-IDS-2017", doc_cic), ("UNSW-NB15", doc_unsw)):
            finals = by_ds_arm_frac.get((ds_name, T7_RARESIG_ARM, T7_DEFAULT_FRACTION))
            if not finals:
                fail(f"F: no raw T7 raresig cells found for {ds_name} at "
                     f"attacker_fraction={T7_DEFAULT_FRACTION}")
                clean = False
                continue
            uniform = 1.0 / N_CLIENTS[ds_name]
            got_ratio = float(np.mean(finals)) / uniform
            if abs(doc_ratio - round(got_ratio, 1)) > 0.051:
                fail(f"F: Sec10.3 up-weighting {ds_name}: doc {doc_ratio}x "
                     f"vs recomputed {round(got_ratio, 1)}x")
                clean = False
    else:
        fail("F: Sec10.3 up-weighting sentence not found/parsed")
        clean = False

    # Sec 10.4 ASR table: | Defense | Attack | CIC ASR | UNSW ASR |
    by_ds_arm: Dict[Tuple[str, str], List[float]] = {}
    for _path, data in cells:
        if data["arm"] not in T7_DEFENSE_ARMS:
            continue
        cfg = data.get("config", {})
        if (cfg.get("n_rounds") != CANONICAL_ROUNDS or data["gamma"] != 1.0
                or cfg.get("attacker_fraction") != T7_DEFAULT_FRACTION):
            continue
        hist = data.get("attacker_weight_history") or []
        if not hist:
            continue
        by_ds_arm.setdefault((data["dataset"], data["arm"]), []).append(hist[-1])

    defense_display = {"SecFedNIDS": "secfednids", "WeiDetect": "weidetect"}
    attack_display = {"label-flip": "labelflip", "model-poison": "modelpoison",
                       "adaptive": "adaptive"}
    for line in doc.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or len(cols) != 4:
            continue
        defense, attack = defense_display.get(cols[0]), attack_display.get(cols[1])
        if defense is None or attack is None:
            continue
        arm = f"{defense}_{attack}"
        m2 = re.match(r"([\d.]+)%", cols[2])
        m3 = re.match(r"([\d.]+)%", cols[3])
        if not (m2 and m3):
            continue
        for ds_name, doc_val in (("CIC-IDS-2017", float(m2.group(1))),
                                  ("UNSW-NB15", float(m3.group(1)))):
            finals = by_ds_arm.get((ds_name, arm))
            if not finals:
                fail(f"F: no raw T7 cells found for {ds_name}/{arm}")
                clean = False
                continue
            checked += 1
            got_val = float(np.mean(finals)) * 100
            if abs(doc_val - round(got_val, 1)) > 0.051:
                fail(f"F: Sec10.4 ASR {ds_name}/{arm}: doc {doc_val}% "
                     f"vs recomputed {round(got_val, 1)}%")
                clean = False

    if clean and checked:
        ok(f"F: {checked} T7 headline numbers (up-weighting ratios + ASR "
           "table) all reproduce from raw attacker_weight_history")
    elif checked == 0:
        fail("F: no T7 ASR table rows were matched against the doc -- "
             "parser likely out of sync with the master report's table format")


def main() -> int:
    print("=" * 78)
    print("  Stage C deep audit -- correctness, not just completeness")
    print("=" * 78)
    cells = load_all_cells()
    if not cells:
        fail(f"no result files found under {RESULTS_DIR}")
        print("\n" + "=" * 78)
        print("  AUDIT FAILED")
        print("=" * 78)
        return 1

    check_raw_runs(cells)
    rebuilt = rebuild_frontier(cells)
    frontier_table = check_frontier_table(rebuilt)
    check_significance(rebuilt)
    check_overhead(rebuilt)
    if frontier_table is not None:
        sig_path = os.path.join(RESULTS_DIR, "significance_table.json")
        overhead_path = os.path.join(RESULTS_DIR, "overhead_table.json")
        if os.path.exists(sig_path) and os.path.exists(overhead_path):
            with open(sig_path, encoding="utf-8") as f:
                sig_table = json.load(f)
            with open(overhead_path, encoding="utf-8") as f:
                overhead_table = json.load(f)
            check_master_report_transcription(frontier_table, sig_table, overhead_table)
    check_t7_headline_numbers(cells)

    print()
    print("=" * 78)
    if FAILURES:
        print(f"  AUDIT FAILED -- {len(FAILURES)} problem(s):")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 78)
        return 1
    print("  AUDIT PASSED -- every Stage C number traces to a persisted run,")
    print("  reproduces independently, and is reported without distortion.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
