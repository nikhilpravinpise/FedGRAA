# How badly did the PDF↔MD fidelity bug affect Stages A/B/C? — Impact Assessment

**Date:** 2026-08-13
**Scope:** Traces the blast radius of the `PAKDD_Final_Version_Submit_v3.pdf` → `.md` conversion bug (documented in [`PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md`](PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md) and [`../paper-files/FedGRAA_PDF_vs_MD_Fidelity_Report_v3_FINAL.md`](../paper-files/FedGRAA_PDF_vs_MD_Fidelity_Report_v3_FINAL.md)) across the whole repo: does it touch Stage A, B, or C's actual research — the reproduction, the theory, the FedGRAA-v2 experiments — or is it contained to documentation? Verified independently against `git log`, repo-wide grep, and direct reads of every implicated file; not a summary of the prior two audits.

**Bottom line up front: the research substance of Stages A, B, and C is untouched.** No code, config, dataset, or experiment result was ever generated from, or checked against, the fabricated data. The damage is confined to **one fabricated finding, copy-pasted into two documents**, plus one low-stakes miscitation — and it is currently still live in the tracked repo.

---

## 1. What actually happened, one paragraph

`PAKDD_Final_Version_Submit_v3.md` is an LLM-based (llamaparse) conversion of the submitted paper's PDF. The conversion hallucinated three giant HTML tables (~870–936 numeric cells) purporting to give exact per-round accuracy values for Figures 1–3. The PDF contains no such tables anywhere — those figures are plotted raster line charts with only axis tick labels, no per-point numbers. Every one of those ~900 cell values was invented by the conversion tool, not read off the source paper. Everything else in the conversion (prose, math, all 10 equations, Algorithm 1, Tables 1–2's real sample-distribution data, the one genuine small results table, all 14 references) was independently re-verified against page screenshots and found faithful.

## 2. Did it touch the actual research? No — verified, not assumed

| Question | Check performed | Result |
|---|---|---|
| Does any training/data/pipeline script read the fabricated `.md` or its predecessor `draft1.md`? | `grep -rl "PAKDD_Final_Version_Submit_v3\|draft1" **/*.py` (excl. `.venv`), plus a filename grep across every non-`.md` file in the repo | Only hits are the fidelity-verification scripts themselves (`docs/paper-files/verify_*.py`) and one lineage mention in the LaTeX submission's AI-disclosure paragraph. **Zero pipeline consumption.** |
| Does any result, config, or dataset file contain the fabricated numbers? | `grep -rl` for the fabricated table's column headers and value fingerprints across `results/`, `src/`, `scripts/`, `data/`, `*.json`, `*.yaml`, `*.csv` | No hits (the only numeric coincidences were unrelated columns in the raw CIC-IDS-2017/UNSW-NB15 CSVs). |
| Are the CIC-IDS-2017 / UNSW-NB15 datasets themselves affected? | Separate, unrelated forensic audit already exists: `docs/audit/data_provenance_audit.md` (2026-07-23) | "Both datasets are genuine, complete, and correctly labeled — bit-faithful to canonical published distributions and to the paper's own Tables 1–2." No connection to this bug. |
| Does Stage B's theoretical argument (simplex-projection defect, convergence proof) rest on the fabricated finding? | Read `STAGE_B_MASTER_REPORT.md:120,173,1191` | Stage B's §4.2 argument is explicitly grounded in `PAPER_AMBIGUITY_REPORT.md` items **#6** (the `min()` non-smoothness) and **#13** (convexity vs. the tanh-MLP) — not item #9 (the contaminated one). #9 is load-bearing for nothing in Stage B. |
| Do Stage C's (FedGRAA-v2) numbers derive from the fabricated table? | Read `STAGE_C_MASTER_REPORT.md:550`, the report's own AI-disclosure | "Every number in this report is read from `results/v2_frontier/*.json`... none is hand-transcribed or asserted ahead of the data producing it." Independently generated, unrelated data source. |
| Do any other audit documents (`STAGE_C_IMPACT_ON_STAGE_AB`, `INDEPENDENT_STAGE_C_CODE_VS_PAPER_AUDIT`, etc.) cite the fabricated finding? | Grepped `docs/audit/` for the fabrication's fingerprints and for "SUSPICIOUS DATA" / "implausibly static" | No matches outside the three sites in §3 below. |

**Conclusion:** Stage A's reproduction, Stage B's theory and ablation campaign, and Stage C's FedGRAA-v2 frontier experiments are all built from this project's own re-implementation, its own training runs, and its own measured results — never from the source paper's (real or hallucinated) reported numbers. The fabrication had no path into any of that.

## 3. What it did affect: three citation sites, one of them load-bearing

The fabricated tables' column headers (`"Adaptive/FedAvg/Static (Last n classes)"`) were quoted as if genuine paper evidence in:

| Document | Status | What's wrong | How bad |
|---|---|---|---|
| `docs/current/PAPER_AMBIGUITY_REPORT.md`, item **#9** | **RETRACTED 2026-08-13** (fixed, per [`PDF_MD_FIDELITY_REMEDIATION_PLAN_2026-08-13.md`](PDF_MD_FIDELITY_REMEDIATION_PLAN_2026-08-13.md) §1.1) | Claimed the source paper's per-round accuracy values are suspiciously static/frozen (e.g. "CIC 'Static (All classes)' stays exactly `0.44` from round 60–200"). The table those numbers come from doesn't exist in the PDF. | This was the **origin point** — first committed 2026-07-10 (`6c33f3d`), 15 days earlier than the file's own internal "2026-07-25" dateline suggests. |
| `docs/current/STAGE_A_MASTER_REPORT.md`, finding **#9** | **RETRACTED 2026-08-13** (fixed, per remediation plan §1.2) | Same claim, copied into the project's most-cited Stage A capstone. First appears 2026-07-25 (`155828c`); survived the 2026-08-10 PAKDD→FedGRAA rename (`15ce51e`) unchanged. | Was the highest exposure — this is the document Stage B explicitly points readers to for "upstream context," and it made a direct data-fabrication accusation against the source paper's authors on the strength of a hallucination. |
| `understanding/02_the_problem.md`, line 52 | **Re-pointed 2026-08-13** (fixed, per remediation plan §1.3) | Cited the fabricated `"FedAvg (Last n classes)"` column as evidence FedAvg gets ~0% on rare classes. | Low-stakes: the underlying claim is independently true (visible in the real chart images and consistent with FedAvg's known behavior); only the citation was wrong, not the conclusion — now re-pointed at the real chart images. |
| `docs/paper/draft1.md` | Superseded, out of citable lineage | Same signature present | Not re-audited in depth; flagged only in case anything still reads from it. Nothing in the repo does (see §2). |

**What survives, explicitly:** `PAPER_AMBIGUITY_REPORT.md` item **#8** and `STAGE_A_MASTER_REPORT.md` finding **#8** — the *real*, small 5-row comparison table's suspiciously round repeating-decimal values (66.66 / 88.88 / 77.77 / 33.33) — are a separate, legitimate observation about genuine paper content, confirmed present in the PDF. **Losing confidence in #9 does not mean losing #8.** The "this paper's headline numbers look statistically implausible" thread still stands on its own evidence; only the "and also the per-round tables freeze" reinforcement was fake.

## 4. Severity, stage by stage

- **Stage A (reproduction):** No effect on the reproduction itself (code, data, results). One narrative finding in its capstone report (`STAGE_A_MASTER_REPORT.md` #9) is fabricated-evidence-based and currently live. **Fix scope: one paragraph.**
- **Stage B (theory + ablation):** No effect. The formal argument is grounded in unrelated, unaffected ambiguity items (#6, #13). The capstone report only inherits exposure by linking to Stage A's document, not by relying on its content.
- **Stage C (FedGRAA-v2):** No effect. Explicitly disclosed as sourced entirely from its own `results/v2_frontier/*.json`.
- **Project-wide research narrative:** One overstated accusation against the source paper's authors, currently live in the two most load-bearing audit documents in the repo. This is a rigor/integrity problem in the project's own audit trail, not in its experiments.

## 4.1 The other two non-trivial defects (#4, #6): also checked, also zero effect

§2–4 above only traced defect #1 (the fabricated tables) — the one with a plausible path into the project's narrative. The fidelity audit's other two above-cosmetic defects (#3, #5, #7 are pure formatting/punctuation, not checked further) were verified separately:

- **Defect #4 — Algorithm 2 line 1 signature, `θ_k^t` in the flawed `.md` where the PDF has `θ_c^t`** (the vector of *all* clients' models, not one client's). Checked whether the project's actual FedGRAA implementation or any theory document derived anything from this mis-transcribed subscript. It doesn't: `fedgraa/aggregation.py:248` implements the weight update from Eq. 10 (`θ'_g = Σ_k w_k θ_k`) directly — that equation was transcribed correctly and was never one of the flagged defects — and takes `client_params` (the full list) as its actual argument, matching the PDF's real semantics regardless of what line 159's label said. Grepped every doc in the repo for any discussion of Algorithm 2's input signature specifically; the only place this defect is mentioned anywhere in the repo is the fidelity audit that found it. **No downstream dependency, dormant defect** — still uncorrected at `docs/paper/PAKDD_Final_Version_Submit_v3.md:159`, but nothing reads that line for meaning.
- **Defect #6 — Appendix, Lemma 1 proof, dropped trailing "3" marker** (an informal inline cross-reference to Assumption 3, mirroring "β-smooth 1" = Assumption 1 two words earlier in the same sentence). Checked the project's own convergence-proof work — the actual Stage B theoretical deliverable — for whether it inherited the wrong assumption dependency. It didn't: `docs/current/fedgraa_convergence_defect.md:35` independently states Lemma 1's guarantee holds "under the paper's Assumptions 1–3," and `understanding/05_convergence_proof.md:35-38` independently enumerates and numbers all four assumptions correctly, tying Lemma 1 to 1–3 and leaving Assumption 4 (client participation) separate. Both derived this straight from the paper's substance, not from the dropped marker. **No downstream dependency** — the marker was a stylistic aid, and its absence caused no misattribution anywhere in the repo.

Both defects are exactly what the original fidelity audit already called them (Medium and Low severity, respectively, on transcription fidelity grounds) — the addition here is confirming, by direct check rather than inference, that neither one left the one file they live in.

## 5. `.gitignore` risk — resolved 2026-08-13, reverted by user

**Status update:** the `.gitignore` change described below has been undone. Reverified directly: `git diff .gitignore` against `HEAD` is now empty (matches the committed version, no exclusion lines), and `git check-ignore` on the audit doc, `docs/paper-files/`, and `docs/meeting_prep/` returns no match (exit 1 — not ignored). `git status` now shows all four of these as ordinary untracked (`??`) entries:

```
?? docs/audit/PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md
?? docs/audit/PDF_MD_FIDELITY_IMPACT_ON_STAGES_ABC_2026-08-13.md
?? docs/meeting_prep/
?? docs/paper-files/
```

This changes the finding materially, not just cosmetically: before the revert, these files would have been *silently* excluded from every future `git add -A` / `git status` view — invisible unless someone thought to check `.gitignore` specifically. Now they behave like any other new, uncommitted work: visible in plain `git status`, and one `git add` away from being tracked. **The "accusation ships, correction doesn't" asymmetry from the original finding is closed.** Nothing here is committed yet — that's still a separate, ordinary to-do — but it is no longer at risk of disappearing without anyone noticing.

*(Original finding, for the record: the working tree had contained an uncommitted `.gitignore` diff adding `docs/audit/PDF_MD_FIDELITY_AND_CONTAMINATION_AUDIT_2026-08-13.md`, `docs/meeting_prep/`, and `docs/paper-files/` as ignored paths, while `STAGE_A_MASTER_REPORT.md` and `PAPER_AMBIGUITY_REPORT.md` — carrying the live fabricated-evidence finding — remained tracked. Flagged as a factual asymmetry, not an accusation of intent; likeliest explanation was routine scratch-doc hygiene that hadn't yet been reconciled with what it was excluding.)*

## 6. Fixes — APPLIED 2026-08-13

All four fixes below were applied in full, per [`PDF_MD_FIDELITY_REMEDIATION_PLAN_2026-08-13.md`](PDF_MD_FIDELITY_REMEDIATION_PLAN_2026-08-13.md).

**6.1 — `docs/current/PAPER_AMBIGUITY_REPORT.md`, item #9:** ✅ Applied. Replaced with a dated retraction note (remediation plan §1.1), preserving the finding number and history rather than silently deleting it. Finding #8 explicitly called out as unaffected.

**6.2 — `docs/current/STAGE_A_MASTER_REPORT.md`, finding #9:** ✅ Applied. Same retraction note, adapted (remediation plan §1.2). Confirmed no secondary references to finding #9 elsewhere in the file.

**6.3 — `understanding/02_the_problem.md`, line 52:** ✅ Applied. Citation re-pointed from the fabricated table to the real chart images (Figure 1); underlying claim (FedAvg ≈0% on rare classes) unchanged, since it was already correct (remediation plan §1.3).

**6.4 — `docs/paper-files/` delivery defects:** ✅ Applied. The three PNGs were moved into `docs/paper-files/figures/`, resolving `PAKDD_Final_Version_Submit_v3_CORRECTED.md`'s broken image references. `verify_corrected_md.py`'s dependency on the missing `PAKDD_Final_Version_Submit_v3_pdftotext_layout.txt` was dropped (remediation plan §2.2, Option 2 — inline ground-truth strings instead of an external diff, mirroring `verify_references.py`'s already-standalone pattern) rather than regenerated via `pdftotext`, to avoid re-introducing dependence on the same class of automated-extraction tool this bug originated from. Both `docs/paper-files/verify_corrected_md.py` (48/48 pass, exit 0) and `verify_references.py` (14/14 references match, exit 0) now run standalone from a clean checkout.

## 7. One-line answer to "how bad"

**Zero impact on any Stage A/B/C experiment, result, or code — verified twice, including an independent adversarial re-check and a final re-verification pass. One fabricated accusation against the source paper's authors was live in two tracked documents; both are now retracted, and the one low-stakes miscitation is re-pointed.** The `.gitignore` risk that would have hidden the fix was resolved (§5). **Status: CLOSED 2026-08-13** — all of §6.1–6.4 applied and independently re-verified (§5 of the remediation plan's own verification sweep: zero remaining fabricated-phrase matches in `docs/current/` and `understanding/`, both `docs/paper-files/verify_*.py` scripts pass standalone, all three figure PNGs present under `docs/paper-files/figures/`).

## 8. Final re-verification pass — 2026-08-13

Re-ran the checks below after the `.gitignore` revert, to confirm nothing else drifted:

- `git status` shows no modifications to any file this doc's claims depend on (`STAGE_A_MASTER_REPORT.md`, `PAPER_AMBIGUITY_REPORT.md`, `STAGE_B_MASTER_REPORT.md`, `STAGE_C_MASTER_REPORT.md`, `understanding/02_the_problem.md` are all unchanged since the prior pass) — only the four `docs/audit/`/`docs/paper-files/`/`docs/meeting_prep/` paths are untracked, exactly as expected.
- Finding #9 re-confirmed still live, word-for-word, at `PAPER_AMBIGUITY_REPORT.md:223`, `STAGE_A_MASTER_REPORT.md:319`, and the miscitation at `understanding/02_the_problem.md:52`.
- `docs/paper-files/verify_corrected_md.py` re-run directly: still fails on the missing `pdftotext` layout file (§6.4 defect confirmed still present, not fixed by the revert).
- §2–4's pipeline-independence and Stage B/C-independence findings were already independently verified by a separate adversarial agent pass earlier in this session (all six claims CONFIRMED, no corrections) and are unaffected by the `.gitignore` change, which only touched file-tracking state, not repo content.

No new issues found. The doc's conclusions stand; only §5 (now resolved) and §6.4 (now unblocked) changed.
