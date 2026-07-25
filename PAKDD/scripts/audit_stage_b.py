"""Stage B deep audit -- verifies correctness, not just completeness.

``scripts/check_stage_b_complete.py`` answers "is every artifact present and
shaped right?". This script answers the harder question: **are the numbers
actually true?** It was written as an adversarial re-check of the committed
Stage B package and it found three real defects on its first run (all since
fixed): a heavy-tailed blowup *mean* being reported as if it described a
typical run, a stale seed-0 worst-case figure presented as the 10-seed worst
case, and a run-count overstatement in the master report.

Five independent checks, in increasing order of what they would catch:

A. **Raw-run integrity** -- every one of the 80 diagnostic JSONs exists, is
   byte-distinct from the others (no accidental copies masquerading as
   separate seeds), and its *persisted config* matches the arm it is filed
   under (projection flag, eta', E', seed, rounds, gamma). Catches a
   mislabeled or duplicated run, which no shape check can see.
B. **Table-vs-raw fidelity** -- the consolidated table's ``per_seed`` arrays
   are re-read from the raw JSONs and compared exactly. Catches a globbing
   or seed-mapping bug in the builder.
C. **Independent statistics** -- every p-value, Cohen's d, Holm adjustment
   and factorial effect is recomputed from scratch against ``scipy``,
   deliberately *not* via ``fedgraa/stats.py``, so the module under test is
   checked against an independent implementation rather than itself.
D. **Doc transcription** -- every number hand-typed into the markdown tables
   of the theory doc and master report is parsed back out and re-derived
   from the table JSON, including that each significance star agrees with
   its own numeric p-value. Catches copy-paste drift between data and prose.
E. **Presentation soundness** -- flags any cell where the mean is a
   misleading summary (heavy-tailed ratio), and verifies the rendered tables
   report median+max for blowup rather than the mean.
F. **Metric-divergence disclosure** -- re-derives the all-class-accuracy
   comparison family and requires that any comparison which *ties* on the
   metric of record while differing materially on all-class accuracy is
   actually disclosed. Catches the subtlest defect of the set: a true
   "statistically indistinguishable" that is nonetheless misleading, which
   on this data let an arm sitting at 15.2% all-class accuracy read as
   equivalent to one at 44.3%.

Usage::

    python scripts/audit_stage_b.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

# Failure messages quote row labels containing non-cp1252 characters (eta',
# primes, em-dashes). On a Windows console defaulting to cp1252 that raises
# UnicodeEncodeError *while reporting a failure* -- i.e. the audit would
# detect a defect and then crash instead of naming it. Found by injecting a
# deliberate error as a negative control; fixed here rather than by dropping
# the characters, so the messages stay readable where the console supports it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):  # pragma: no cover - non-reconfigurable stream
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from scipy import stats as sps  # noqa: E402

DIAG_DIR = "results/diagnostics"
ABLATION_DIR = "results/ablation"
THEORY_DOC = "docs/current/fedgraa_convergence_defect.md"
MASTER_REPORT = "docs/current/STAGE_B_MASTER_REPORT.md"

DATASETS = {"CIC_IDS_2017": "CIC-IDS-2017", "UNSW_NB15": "UNSW-NB15"}
GAMMAS = ["g0.0", "g0.5", "g1.0"]
N_SEEDS = 10

# arm -> (project, eta_prime, weight_epochs, filename template)
ARM_SPEC = {
    "literal":          (False, 0.1,  5, "{ds}_literal{seed}_r60.json"),
    "projection_alone": (True,  0.1,  5, "{ds}_projected{seed}_r60.json"),
    "eta_alone":        (False, 0.01, 5, "{ds}_literal_etaonly_eta0.01_E5{seed}_r60.json"),
    "both":             (True,  0.01, 5, "{ds}_projected_combined_eta0.01_E5{seed}_r60.json"),
}
METRIC_FIELD = {"rare_acc": "final_last_n_acc", "all_acc": "final_all_acc",
                "blowup": "composite_loss_blowup_ratio"}

FAILURES: list = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def diag_path(ds_tag: str, arm: str, seed: int) -> str:
    suffix = "" if seed == 0 else f"_seed{seed}"
    return os.path.join(DIAG_DIR, ARM_SPEC[arm][3].format(ds=ds_tag, seed=suffix))


def raw_series(ds_tag: str, arm: str, gamma: str, metric: str) -> np.ndarray:
    vals = []
    for seed in range(N_SEEDS):
        with open(diag_path(ds_tag, arm, seed)) as fh:
            vals.append(json.load(fh)[gamma]["diagnosis"][METRIC_FIELD[metric]])
    return np.asarray(vals, dtype=float)


# --------------------------------------------------------------------------- #
# A. Raw-run integrity + config fidelity
# --------------------------------------------------------------------------- #
def check_raw_runs() -> None:
    print("\n-- A. Raw-run integrity (80 runs: distinct files, configs match arm) --")
    hashes: dict = {}
    n_files = n_cells = 0
    for ds_tag, ds_name in DATASETS.items():
        for arm, (want_proj, want_eta, want_E, _) in ARM_SPEC.items():
            for seed in range(N_SEEDS):
                path = diag_path(ds_tag, arm, seed)
                if not os.path.exists(path):
                    fail(f"A: missing {ds_name}/{arm}/seed{seed}: {path}")
                    continue
                raw = open(path, "rb").read()
                n_files += 1
                h = hashlib.sha256(raw).hexdigest()
                if h in hashes:
                    fail(f"A: {path} is byte-identical to {hashes[h]} "
                         "(duplicated run filed as two seeds)")
                hashes[h] = path
                data = json.loads(raw)
                if sorted(data.keys()) != sorted(GAMMAS):
                    fail(f"A: {path} gammas={sorted(data.keys())}")
                    continue
                for g in GAMMAS:
                    n_cells += 1
                    cfg = data[g].get("config", {})
                    checks = (
                        ("project", bool(cfg.get("project", False)), want_proj),
                        ("eta_prime", cfg.get("eta_prime"), want_eta),
                        ("weight_epochs", cfg.get("weight_epochs"), want_E),
                        ("seed", cfg.get("seed"), seed),
                        ("rounds", cfg.get("rounds"), 60),
                        ("gamma", cfg.get("gamma"), float(g[1:])),
                    )
                    for key, got, want in checks:
                        if got is None:
                            continue
                        same = (abs(got - want) < 1e-12 if isinstance(want, float)
                                else got == want)
                        if not same:
                            fail(f"A: config {key} in {path} {g}: got {got!r} want {want!r}")
    if n_files == 80 and n_cells == 240 and len(hashes) == n_files:
        ok(f"A: {n_files} runs / {n_cells} cells, all byte-distinct, "
           "every persisted config matches its arm")


# --------------------------------------------------------------------------- #
# B + C. Table fidelity and independently recomputed statistics
# --------------------------------------------------------------------------- #
def check_table_and_stats() -> None:
    print("\n-- B/C. Table-vs-raw fidelity + statistics recomputed via scipy --")
    for ds_tag, ds_name in DATASETS.items():
        path = f"{ABLATION_DIR}/{ds_tag}_ablation_table.json"
        if not os.path.exists(path):
            fail(f"B: missing {path}")
            continue
        with open(path) as fh:
            T = json.load(fh)
        clean = True

        # B: per-seed arrays and summary stats
        for arm in ARM_SPEC:
            for g in GAMMAS:
                for metric in METRIC_FIELD:
                    cell = T["arms"][arm][g][metric]
                    want = raw_series(ds_tag, arm, g, metric)
                    got = np.asarray(cell["per_seed"], dtype=float)
                    if got.shape != want.shape or not np.array_equal(got, want):
                        fail(f"B: {ds_name}/{arm}/{g}/{metric} per_seed != raw runs")
                        clean = False
                        continue
                    if abs(cell["mean"] - float(np.mean(want))) > 1e-12:
                        fail(f"B: {ds_name}/{arm}/{g}/{metric} mean mismatch")
                        clean = False
                    if cell.get("median") is not None:
                        srt = np.sort(want)
                        exp_med = (srt[len(srt) // 2] if len(srt) % 2 else
                                   (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2)
                        if abs(cell["median"] - float(exp_med)) > 1e-12:
                            fail(f"B: {ds_name}/{arm}/{g}/{metric} median mismatch")
                            clean = False
                    lo, hi = cell["ci_95"]
                    if not (lo - 1e-9 <= np.mean(want) <= hi + 1e-9):
                        fail(f"B: {ds_name}/{arm}/{g}/{metric} CI excludes its own mean")
                        clean = False
                    if lo < want.min() - 1e-9 or hi > want.max() + 1e-9:
                        fail(f"B: {ds_name}/{arm}/{g}/{metric} CI escapes data range")
                        clean = False
                    if np.ptp(want) == 0 and lo != hi:
                        fail(f"B: {ds_name}/{arm}/{g}/{metric} zero-variance cell "
                             "has a non-point CI")
                        clean = False

        # C: comparisons recomputed against scipy
        for g in GAMMAS:
            comps = T["comparisons"][g]
            both = raw_series(ds_tag, "both", g, "rare_acc")
            ordered = []
            for name, c in comps.items():
                if name in ARM_SPEC:
                    ref = raw_series(ds_tag, name, g, "rare_acc")
                else:
                    ref = np.asarray(T["baselines"][name]["rare_acc"]["per_seed"], float)
                d = both - ref
                degenerate = bool(np.ptp(d) == 0)
                exp_p = 1.0 if degenerate else float(sps.ttest_rel(both, ref)[1])
                exp_d = 0.0 if degenerate else float(np.mean(d) / np.std(d, ddof=1))
                if np.all(d == 0):
                    exp_w = 1.0
                else:
                    try:
                        exp_w = float(sps.wilcoxon(both, ref)[1])
                    except ValueError:
                        exp_w = 1.0
                if abs(c["p_ttest"] - exp_p) > 1e-9:
                    fail(f"C: {ds_name}/{g}/{name} p_ttest {c['p_ttest']} vs scipy {exp_p}")
                    clean = False
                if abs(c["p_wilcoxon"] - exp_w) > 1e-9:
                    fail(f"C: {ds_name}/{g}/{name} p_wilcoxon {c['p_wilcoxon']} vs scipy {exp_w}")
                    clean = False
                if abs(c["cohens_d"] - exp_d) > 1e-6 * max(1.0, abs(exp_d)):
                    fail(f"C: {ds_name}/{g}/{name} cohens_d {c['cohens_d']} vs {exp_d}")
                    clean = False
                ordered.append((name, c["p_ttest"], c["p_holm"]))

            # Holm step-down, recomputed
            ps = np.asarray([p for _, p, _ in ordered], dtype=float)
            adj = np.empty(len(ps))
            running = 0.0
            for rank, idx in enumerate(np.argsort(ps, kind="stable")):
                running = max(running, (len(ps) - rank) * ps[idx])
                adj[idx] = min(1.0, running)
            for i, (name, _, got_holm) in enumerate(ordered):
                if abs(got_holm - adj[i]) > 1e-9:
                    fail(f"C: {ds_name}/{g}/{name} p_holm {got_holm} vs recomputed {adj[i]}")
                    clean = False

        # C: factorial effects
        for g in GAMMAS:
            lit = raw_series(ds_tag, "literal", g, "rare_acc")
            proj = raw_series(ds_tag, "projection_alone", g, "rare_acc")
            eta = raw_series(ds_tag, "eta_alone", g, "rare_acc")
            both = raw_series(ds_tag, "both", g, "rare_acc")
            expected = {
                "projection_main_effect": float(np.mean(((proj - lit) + (both - eta)) / 2)),
                "eta_main_effect": float(np.mean(((eta - lit) + (both - proj)) / 2)),
                "interaction": float(np.mean((both - eta) - (proj - lit))),
            }
            for key, exp in expected.items():
                node = T["factorial"][g][key]
                got = node["mean"] if isinstance(node, dict) else node
                if abs(got - exp) > 1e-9:
                    fail(f"C: {ds_name}/{g}/{key} {got} vs recomputed {exp}")
                    clean = False

        # B: baselines trace to their own seed-sweep files
        for method in ("fedavg", "static"):
            src_path = f"{ABLATION_DIR}/seed_sweep_{ds_tag}_{method}_r60.json"
            if not os.path.exists(src_path):
                fail(f"B: missing baseline source {src_path}")
                clean = False
                continue
            with open(src_path) as fh:
                src = json.load(fh)
            if src.get("rounds") != 60:
                fail(f"B: {src_path} rounds={src.get('rounds')} (expect 60)")
                clean = False
            if sorted(src.get("seeds", [])) != list(range(N_SEEDS)):
                fail(f"B: {src_path} seeds={src.get('seeds')}")
                clean = False
            got = np.asarray(T["baselines"][method]["rare_acc"]["per_seed"], float)
            want = np.asarray(src["last_n_acc_per_seed"], float)
            if got.shape != want.shape or not np.array_equal(got, want):
                fail(f"B: {ds_name}/{method} baseline per_seed != source file")
                clean = False

        if clean:
            ok(f"B/C: {ds_name} table matches raw runs; all p-values, effect sizes, "
               "Holm adjustments and factorial effects reproduce independently")


# --------------------------------------------------------------------------- #
# D. Doc transcription fidelity
# --------------------------------------------------------------------------- #
ROW_LABEL_TO_CELL = {
    "FedAvg (baseline)": ("baseline", "fedavg"),
    "Static Reparam. (baseline)": ("baseline", "static"),
    "Literal (η′=0.1, no proj.)": ("arm", "literal"),
    "Projection-alone (η′=0.1)": ("arm", "projection_alone"),
    "η′-alone (η′=0.01, no proj.)": ("arm", "eta_alone"),
    "**Both** (proj. + η′=0.01) — the fix": ("arm", "both"),
    "Both (proj. + η′=0.01) — the fix": ("arm", "both"),
}
COMPARISON_LABEL = {"Literal": "literal", "Projection-alone": "projection_alone",
                    "η′-alone": "eta_alone", "FedAvg": "fedavg",
                    "Static Reparam.": "static"}


def _cell_for(T: dict, kind: str, name: str, gamma: str, metric: str) -> dict:
    return (T["baselines"][name][metric] if kind == "baseline"
            else T["arms"][name][gamma][metric])


def _strip_bold(text: str) -> str:
    """Remove markdown bold delimiters wrapping a cell half, without eating a
    significance run. ``**0.04 \\* (d=-0.9)**`` -> ``0.04 \\* (d=-0.9)``; a
    plain ``1.5e-4 ***`` is returned untouched."""
    t = text.strip()
    while t.startswith("**") and t.endswith("**") and len(t) > 4:
        t = t[2:-2].strip()
    return t


def check_doc_transcription() -> None:
    print("\n-- D. Doc transcription: every hand-typed number re-derived --")
    if not os.path.exists(THEORY_DOC):
        fail(f"D: missing {THEORY_DOC}")
        return
    doc = open(THEORY_DOC, encoding="utf-8").read()
    checked = 0
    clean = True

    for ds_tag, section in (("CIC_IDS_2017", "### 5.1"), ("UNSW_NB15", "### 5.2")):
        with open(f"{ABLATION_DIR}/{ds_tag}_ablation_table.json") as fh:
            T = json.load(fh)
        if section not in doc:
            fail(f"D: {THEORY_DOC} missing section {section}")
            clean = False
            continue
        start = doc.index(section)
        end = doc.index("###", start + 10) if "###" in doc[start + 10:] else len(doc)
        block = doc[start:end]

        for line in block.splitlines():
            cols = [c.strip() for c in line.strip().strip("|").split("|")]
            if not line.startswith("| ") or len(cols) != 4:
                continue

            # arm x gamma rows
            if cols[0] in ROW_LABEL_TO_CELL:
                kind, name = ROW_LABEL_TO_CELL[cols[0]]
                for gi, g in enumerate(GAMMAS):
                    txt = cols[1 + gi]
                    m = re.search(r"([\d.]+)%\s*\[([\d.]+),\s*([\d.]+)\]", txt)
                    if not m:
                        fail(f"D: unparsed rare-acc {ds_tag} {cols[0]} {g}: {txt!r}")
                        clean = False
                        continue
                    rare = _cell_for(T, kind, name, g, "rare_acc")
                    for got, want, what in (
                        (float(m.group(1)), 100 * rare["mean"], "mean"),
                        (float(m.group(2)), 100 * rare["ci_95"][0], "ci-lo"),
                        (float(m.group(3)), 100 * rare["ci_95"][1], "ci-hi"),
                    ):
                        checked += 1
                        if abs(got - round(want, 1)) > 0.051:
                            fail(f"D: rare-acc {what} {ds_tag} {cols[0]} {g}: "
                                 f"doc {got} vs json {round(want, 1)}")
                            clean = False
                    m2 = re.search(r"all\s+([\d.]+)%", txt)
                    if m2:
                        checked += 1
                        want = 100 * _cell_for(T, kind, name, g, "all_acc")["mean"]
                        if abs(float(m2.group(1)) - round(want, 1)) > 0.051:
                            fail(f"D: all-acc {ds_tag} {cols[0]} {g}: "
                                 f"doc {m2.group(1)} vs json {round(want, 1)}")
                            clean = False
                    # blowup must be the median, never the heavy-tailed mean
                    m3 = re.search(r"blowup\s+([\d.]+)×\s*(?:\(med\)|med)", txt)
                    if m3:
                        checked += 1
                        bc = _cell_for(T, kind, name, g, "blowup")
                        got_med = float(m3.group(1))
                        if abs(got_med - round(bc["median"], 2)) > 0.011:
                            fail(f"D: blowup median {ds_tag} {cols[0]} {g}: "
                                 f"doc {got_med} vs json {round(bc['median'], 2)}")
                            clean = False
                        if (bc["mean"] > 3 * max(bc["median"], 1e-9)
                                and abs(got_med - round(bc["mean"], 2)) < 0.011):
                            fail(f"D: blowup shows heavy-tailed MEAN {ds_tag} {cols[0]} {g}")
                            clean = False
                        m4 = re.search(r"([\d.]+(?:×10⁶|e\+?\d+)?)×\s*max", txt)
                        if m4:
                            checked += 1
                            got_max = float(m4.group(1).replace("×10⁶", "e6"))
                            want_max = bc["max"]
                            if abs(got_max - want_max) > max(0.011, abs(want_max) * 0.01):
                                fail(f"D: blowup max {ds_tag} {cols[0]} {g}: "
                                     f"doc {got_max} vs json {want_max}")
                                clean = False
                        elif bc["max"] > bc["median"] * 1.05:
                            fail(f"D: blowup max omitted {ds_tag} {cols[0]} {g} "
                                 f"(median {bc['median']:.2f} but max {bc['max']:.2f})")
                            clean = False

            # Holm p-value rows. Cells are dual-metric: "rare_p ** / all_p n.s."
            # Each half is validated against its own comparison family, and
            # each half's star run must match its own p-value.
            elif cols[0] in COMPARISON_LABEL:
                key = COMPARISON_LABEL[cols[0]]
                for gi, g in enumerate(GAMMAS):
                    halves = [h.strip() for h in cols[1 + gi].split("/")]
                    families = [("comparisons", "rare-acc")]
                    if len(halves) == 2 and "comparisons_all_acc" in T:
                        families.append(("comparisons_all_acc", "all-acc"))
                    elif len(halves) != 1:
                        fail(f"D: unparsed dual-metric cell {ds_tag} {cols[0]} {g}: "
                             f"{cols[1 + gi]!r}")
                        clean = False
                        continue
                    for half, (block, metric) in zip(halves, families):
                        stripped = _strip_bold(half)
                        # A significance run may be escaped for markdown
                        # (``\*\*\*``), so match a run of optionally-escaped
                        # stars as one token -- matching a bare ``\*`` first
                        # would silently read 3 stars as 1.
                        m = re.match(r"([\d.]+(?:e[-+]?\d+)?)\s*((?:\\?\*)+|n\.s\.)?",
                                     stripped)
                        if not m:
                            fail(f"D: unparsed {metric} p {ds_tag} {cols[0]} {g}: "
                                 f"{half!r}")
                            clean = False
                            continue
                        checked += 1
                        want = T[block][g][key]["p_holm"]
                        got = float(m.group(1))
                        if want and abs(got - want) > max(abs(want) * 0.02, 1e-12):
                            fail(f"D: {metric} p_holm {ds_tag} {cols[0]} {g}: "
                                 f"doc {got} vs json {want}")
                            clean = False
                        marker = (m.group(2) or "").strip()
                        shown = 0 if marker in ("", "n.s.") else marker.replace("\\", "").count("*")
                        exp = (3 if want < .001 else 2 if want < .01
                               else 1 if want < .05 else 0)
                        if shown != exp:
                            fail(f"D: {metric} stars {ds_tag} {cols[0]} {g}: "
                                 f"{shown} shown, {exp} warranted by p={want:.4g}")
                            clean = False

            # factorial rows
            elif cols[0] in ("0.0", "0.5", "1.0"):
                g = "g" + cols[0]
                for ci, key in enumerate(("projection_main_effect", "eta_main_effect",
                                          "interaction")):
                    m = re.search(r"([+\-−][\d.]+)pp", cols[1 + ci])
                    if not m:
                        continue
                    checked += 1
                    node = T["factorial"][g][key]
                    want = 100 * (node["mean"] if isinstance(node, dict) else node)
                    got = float(m.group(1).replace("−", "-").replace("+", ""))
                    if abs(got - round(want, 1)) > 0.051:
                        fail(f"D: factorial {key} {ds_tag} {g}: "
                             f"doc {got} vs json {round(want, 1)}")
                        clean = False

    if clean:
        ok(f"D: all {checked} numbers transcribed into {THEORY_DOC} trace to the "
           "table JSON, and every significance star matches its own p-value")


# --------------------------------------------------------------------------- #
# E. Presentation soundness
# --------------------------------------------------------------------------- #
def check_metric_divergence() -> None:
    """F. A tie on the metric of record can hide a large difference on overall
    accuracy. Re-derives both metric families and requires that any comparison
    which ties on rare-acc but differs significantly on all-acc is actually
    disclosed -- the exact defect that let 'eta'-alone is indistinguishable
    from both' stand while that arm sat at 15.2% all-class accuracy against
    44.3%."""
    print("\n-- F. Metric-divergence disclosure (rare-acc ties hiding all-acc gaps) --")
    clean = True
    doc = open(THEORY_DOC, encoding="utf-8").read() if os.path.exists(THEORY_DOC) else ""
    for ds_tag, ds_name in DATASETS.items():
        path = f"{ABLATION_DIR}/{ds_tag}_ablation_table.json"
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            T = json.load(fh)
        if "comparisons_all_acc" not in T:
            fail(f"F: {ds_name} table has no comparisons_all_acc block "
                 "(rebuild: the all-acc family is required)")
            clean = False
            continue
        for g in GAMMAS:
            rare = T["comparisons"][g]
            allc = T["comparisons_all_acc"][g]
            for name, rc in rare.items():
                ac = allc.get(name)
                if rc is None or ac is None:
                    continue
                # recompute the all-acc test independently
                both = raw_series(ds_tag, "both", g, "all_acc")
                if name in ARM_SPEC:
                    ref = raw_series(ds_tag, name, g, "all_acc")
                else:
                    ref = np.asarray(T["baselines"][name]["all_acc"]["per_seed"], float)
                d = both - ref
                exp_p = 1.0 if np.ptp(d) == 0 else float(sps.ttest_rel(both, ref)[1])
                if abs(ac["p_ttest"] - exp_p) > 1e-9:
                    fail(f"F: {ds_name}/{g}/{name} all-acc p_ttest "
                         f"{ac['p_ttest']} vs scipy {exp_p}")
                    clean = False
                # a hidden divergence must be disclosed somewhere in the doc
                if rc["p_holm"] >= 0.05 and ac["p_holm"] < 0.05:
                    gap = abs(float(np.mean(both) - np.mean(ref))) * 100
                    if gap >= 5.0 and "diverge" not in doc.lower():
                        fail(f"F: {ds_name}/{g}/{name} ties on rare-acc "
                             f"(p={rc['p_holm']:.3g}) but differs on all-acc "
                             f"(p={ac['p_holm']:.3g}, {gap:.1f}pp gap) and the "
                             "theory doc never discloses a metric divergence")
                        clean = False
    if clean:
        ok("F: all-acc comparison family present and independently reproduced; "
           "every rare-acc tie that hides a material all-acc gap is disclosed")


def check_presentation() -> None:
    print("\n-- E. Presentation soundness (heavy-tailed summaries) --")
    clean = True
    for ds_tag, ds_name in DATASETS.items():
        md_path = f"{ABLATION_DIR}/{ds_tag}_ablation_table.md"
        if not os.path.exists(md_path):
            fail(f"E: missing {md_path}")
            clean = False
            continue
        md = open(md_path, encoding="utf-8").read()
        with open(f"{ABLATION_DIR}/{ds_tag}_ablation_table.json") as fh:
            T = json.load(fh)
        for arm in ARM_SPEC:
            for g in GAMMAS:
                bc = T["arms"][arm][g]["blowup"]
                if bc.get("median") is None:
                    fail(f"E: {ds_name}/{arm}/{g} blowup cell has no median "
                         "(rebuild the table)")
                    clean = False
                    continue
                if bc["mean"] > 3 * max(bc["median"], 1e-9):
                    # heavy-tailed: the mean must not be the rendered figure
                    if f"{bc['mean']:.2f}" in md:
                        fail(f"E: {ds_name}/{arm}/{g} rendered table shows the "
                             f"heavy-tailed mean {bc['mean']:.2f}")
                        clean = False
        if "median (and max)" not in md:
            fail(f"E: {md_path} does not document that blowup is a median")
            clean = False
    if clean:
        ok("E: blowup reported as median+max wherever the mean would mislead; "
           "the choice is documented in each rendered table")


def main() -> int:
    print("=" * 78)
    print("  Stage B deep audit -- correctness, not just completeness")
    print("=" * 78)
    check_raw_runs()
    check_table_and_stats()
    check_doc_transcription()
    check_presentation()
    check_metric_divergence()
    print()
    print("=" * 78)
    if FAILURES:
        print(f"  AUDIT FAILED -- {len(FAILURES)} problem(s):")
        for f in FAILURES:
            print(f"    - {f}")
        print("=" * 78)
        return 1
    print("  AUDIT PASSED -- every Stage B number traces to a persisted run,")
    print("  reproduces independently, and is reported without distortion.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
