# Publication Roadmap — FedGRAA: Reproduction, Critique & Extension

**Last refreshed:** 2026-07-25, against `HEAD` with Stages A and B complete and all four gates green (R4 of [`../audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md`](../audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md)).
**Current state in one line:** Stage A ✅ · Stage B ✅ · **Stage C 0%, blocked on one mentor conversation, not on compute** (§6).

> **Maintainer's note.** This document went two full stages stale once — it described Stage A as unstarted while Stages A and B were both finished and gated. It is the first thing a returning session or the mentor reads, so **refresh §1 and §6 whenever a stage closes.** If the state line above disagrees with the gates, trust the gates.

**Purpose of this document:** a single entry point for "what's left to do to genuinely publish this." It is a router, not a re-audit — it points to the existing docs that already contain the detailed work, and lays out the sequence, decision points, and process knowledge to get from here to a real submission.

**Confirmed strategy:** finish the faithful reproduction → package the critique/diagnosis as a rigorous component → build the full extension for an A/A* venue (AAAI/IJCAI-tier). Paced by readiness, no fixed deadline yet. Working under an IIT Guwahati professor with day-to-day PhD-student mentorship.

---

## 1. Where things stand

**Stages A and B are both complete, gated, and independently audited.** A faithful, Keras-based reimplementation of FedGRAA lives in `PAKDD/fedgraa/`, with **142/142 tests passing** and both datasets (CIC-IDS-2017, UNSW-NB15) independently verified as genuine, canonical data — see `../audit/data_provenance_audit.md`. Every deliberate deviation from the paper is logged in `../archive/strict_reproduction_report.md`. Stage A's capstone is [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md); Stage B's is [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md). Four gates guard them — two completeness (`check_stage_a_complete.py`, `check_stage_b_complete.py`) and two correctness (`audit_stage_a.py`, `audit_stage_b.py`).

**The central finding is a real, defensible result.** The paper's Algorithm 2 has no simplex projection on the aggregation weight vector `w`. On real data, `w` leaves the probability simplex in round 1, at every γ, on both datasets — causing the composite loss to blow up and rare-class accuracy to collapse to 0% (vs. the paper's claimed ~88%). This is fully diagnosed, not just observed, in `fedgraa_divergence_diagnosis.md`, and formalized with a corrected convergence proposition in [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §4. The defect holds on **60/60** literal-arm cells at 10 seeds, and the corrected algorithm's feasibility prediction holds on **60/60**. A minimal fix — simplex projection + η′=0.01 instead of 0.1 — is validated across a 10-seed, 2×2 factorial, dual-metric ablation on both datasets, and is deliberately kept diagnostic-only rather than silently merged into the strict-reproduction baseline.

**The paper itself has been critiqued rigorously.** `PAPER_AMBIGUITY_REPORT.md` catalogs 17 distinct issues in the original paper — contradictions between SGD-notation and Adam-prose, a convexity assumption in the convergence proof that the paper's own non-convex tanh-MLP violates, an undefined "last-n accuracy" metric, and evidence that some reported numbers (33.33/66.66/77.77/88.88) look like repeating-decimal placeholders rather than genuine measurements.

**The literature review is already publication-grade.** [`../research_c/literature_review_and_publication_strategy.md`](../research_c/literature_review_and_publication_strategy.md) is a multi-pass, snowball-searched review across 30+ papers at A*/A venues (CCS, NDSS, AAAI, IJCAI, NeurIPS, ICML), with an explicit scooping-risk verdict (no near-duplicate found) and a recommended novelty angle (few-shot/prototype-guided hybrid aggregation) plus venue ranking (AAAI/IJCAI first).

**A second extension angle exists but is unbuilt.** [`../research_c/privacy_compliant_fedgraa_redesign.md`](../research_c/privacy_compliant_fedgraa_redesign.md) (~730 lines) identifies that the original paper's design assumes server-side oracle knowledge of which clients/classes are rare, and a server-held labelled validation set — both of which arguably violate FL's core privacy premise. It proposes a concrete redesign, grounded in 7 families of prior work, but none of it is implemented.

**There is also a didactic series worth mining for the write-up.** `../../understanding/00_START_HERE.md` … `../../understanding/10_SOTA_adaptive_aggregation.md` is an 11-part plain-language walkthrough of the method, the convergence proof (`../../understanding/05_convergence_proof.md`), and the SOTA baseline landscape (`../../understanding/10_SOTA_adaptive_aggregation.md`). These feed Stage B (the proof-sketch and corrected theory) and Stage C (the baseline comparison and Related Work) directly.

**What is now done** *(all of these were listed as missing in this document's previous revision — recorded explicitly so the change of state is legible)*:

| Previously missing | Now |
|---|---|
| Only the CIC-IDS-2017 γ-sweep (n=3) persisted | **All four base sweeps persisted** — CIC γ/n, UNSW γ/n; `check_stage_a_complete.py` exit 0 |
| Combined fix seed-swept on CIC only | **UNSW band exists** |
| No formal statistical testing (3-seed bands only) | **10-seed bootstrap CIs + paired *t* + Wilcoxon + Cohen's *d* + Holm**, on rare-class **and** all-class accuracy |
| No ablation table | **Full 2×2 factorial**, both datasets, both metrics, with auto-flagged metric divergence |
| No corrected convergence theory | [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §4 — feasibility + stationarity proposition, empirically verified 60/60 |

**What's still missing for a submission:**
- **No manuscript exists anywhere** — everything is standalone analysis docs plus two capstone reports. This is the single largest gap.
- **Stage C is at 0%** and is gated on one mentor conversation (see §6).
- **The provenance question is unresolved.** Whether the target manuscript is published determines whether Stage B can stand as a citable critique at all — see [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) §21 and [`../audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md`](../audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md) D1.
- Baselines beyond FedAvg/Static are implemented (`../../fedgraa/baselines.py` has FedProx, FedOpt, FedSND) but **never run** against FedGRAA — verified still true. A separate SOTA-baseline catalogue, `../../understanding/10_SOTA_adaptive_aggregation.md`, flags AFL, q-FedAvg, and FedAdp as "must-have" comparisons — none run yet.
- The extension method (few-shot/prototype hybrid, or the privacy redesign) is unimplemented.
- **~140 carried-over bibliography entries in the Stage C corpus were never independently re-verified** — see [`../research_c/STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](../research_c/STAGE_C_LITERATURE_REAUDIT_2026-07-25.md). This must be closed before Stage C's Related Work is written on top of it.

---

## 2. The three-stage strategy

### Stage A — Close out the faithful reproduction — ✅ **COMPLETE**

Goal: every table/figure in the original paper has a corresponding persisted, reproduced (or honestly-diverging) counterpart in `results/`.

> **Capstone: [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md)** (976 lines) · Plan: [`stage_a_plan.md`](stage_a_plan.md) · Gates: `check_stage_a_complete.py` (completeness) + `audit_stage_a.py` (correctness) · Manifest: [`../../results/README.md`](../../results/README.md)

Delivered: CIC-IDS-2017 γ- and n-sweeps; UNSW-NB15 γ- and n-sweeps; UNSW combined-fix seed band; all figures regenerated; the `plot_n_sweep` curve-dropping defect fixed and regression-tested; `seed_sweep_combined_fix.py` generalized to any dataset. The n=7 ≡ n=5 result on UNSW is a documented honest divergence, not a bug — UNSW has only 5 genuinely-rare classes, so `rare_client_ids(7)` truncates to the same 5.

### Stage B — Package the critique as a rigorous, citable component — ✅ **COMPLETE**

Goal: turn the diagnosis from "internal audit trail" into something that reads as a paper section on its own.

> **Capstone: [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md)** · Theory: [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) · Plan: [`stage_b_plan.md`](stage_b_plan.md) · Gates: `check_stage_b_complete.py` (completeness) + `audit_stage_b.py` (correctness) · Tables: [`../../results/ablation/`](../../results/ablation/)

Delivered: a formal defect statement and a corrected convergence proposition (feasibility + stationarity, §4); `fedgraa/stats.py` (bootstrap CIs, paired *t* / Wilcoxon, Cohen's *d*, Holm–Bonferroni); a 10-seed, 4-arm × 2-dataset × 3-γ factorial (240 diagnosis blocks); and a `build_ablation_table.py` / `render_ablation.py` pipeline reporting **both** rare-class and all-class accuracy with auto-flagged metric divergence.

**Two results worth carrying into Stage C, because both refuted a prior expectation:**
- The pre-registered n=3 hypothesis — "both levers are always independently necessary" — was **overturned**. Which lever carries the arm is *dataset-dependent* (η′ on CIC, projection on UNSW).
- Reporting only rare-class accuracy made `η′-alone` look equivalent to the full fix. On all-class accuracy it collapses to 15.2% (CIC) and 13.3% (UNSW) at γ=1. **No arm dominates on both metrics at every γ**; `both` is the only arm never catastrophically wrong on either.

> ⚠️ **Framing caveat.** This stage was scoped as a standalone citable critique of a *published* algorithm. That premise is unverified — see §6 action 1. The technical content is unaffected either way; the framing is not.

### Stage C — Build the A/A* extension

Goal: the actual novel contribution that earns a spot at AAAI/IJCAI-tier venue, on top of Stage A+B's foundation.

- **Decision point — bring this to the prof/PhD mentor, don't decide alone:** few-shot/prototype-guided hybrid aggregation (the lit review's recommended angle, competitive with FedAPA/FedAWA-style 2025 papers) vs. the privacy-compliant redesign (already has a design doc, addresses a real flaw in the original paper, arguably a less crowded angle) — or scope for both if time allows.
- Implement the chosen extension in `fedgraa/`.
- Run it against FedAvg, Static, the fixed-FedGRAA baseline, and the currently-unrun baselines (AFL, q-FedAvg, FedAdp, plus the already-implemented FedProx/FedOpt).
- Add the corrected convergence theory the fix currently lacks.
- Per the lit review's own recommendations: add a scalability experiment (50–200 clients, vs. the current 9–14) and an extreme non-IID robustness experiment.
- Re-run the snowball literature/scooping search closer to submission — `literature_review_and_publication_strategy.md` already flags two very recent (weeks-old, as of its last pass) adjacent papers (FedVG, Fed-LAA) worth re-checking for follow-ons.

---

## 3. Working with the IITG prof + PhD mentor

- Bring the completed Stage A reproduction and Stage B critique package to the mentor *before* starting Stage C — this is the natural, concrete checkpoint to jointly pick the Stage C novelty angle and target venue together, since that choice shapes months of subsequent work.
- Raise authorship expectations explicitly and early (you, the PhD student, the prof) — this is a normal conversation to have up front, not an awkward one to defer.
- Ask about lab compute resources once Stage C's scope is set — current work is deliberately CPU-only (a GPU attempt was tried and measured 25% *slower* for this small model), but a 50–200 client scalability study may change that calculus and be worth lab GPU time.
- Use the PhD student as first-pass reviewer for manuscript drafts before they go to the prof — standard practice, and useful for calibrating how much detail/formality the write-up needs.
- When a venue is chosen, work backward from its actual deadline — AAAI/IJCAI run on fixed annual cycles (see §5), so "pace by readiness" works for Stage A/B but Stage C will eventually need a real date.

---

## 4. Manuscript assembly (once Stage B is packaged)

- Use LaTeX via Overleaf — standard for CS venues, and AAAI/IJCAI both publish official author-kit templates (search "AAAI author kit" / "IJCAI author kit" for the current year's template once a target cycle is chosen).
- Map existing docs to paper sections rather than writing from scratch:
  - `literature_review_and_publication_strategy.md` → first draft of Related Work.
  - `fedgraa_divergence_diagnosis.md` + `PAPER_AMBIGUITY_REPORT.md` → the critique/correction section from Stage B.
  - `privacy_compliant_fedgraa_redesign.md` or the few-shot design (whichever is chosen in Stage C) → Method section.
   - `../audit/data_provenance_audit.md` + `../archive/strict_reproduction_report.md` → Experimental Setup / reproducibility appendix.
   - `../../understanding/05_convergence_proof.md` → scaffolding for the corrected convergence analysis; `../../understanding/10_SOTA_adaptive_aggregation.md` → the baseline-comparison framing and part of Related Work.
- Anonymize for double-blind review (no author names, no identifying institution references, no self-citations that reveal identity) — the original FedGRAA paper itself follows this convention ("No Author Given"), so the norm is already visible in the source material.

---

## 5. Publishing-process primer (you're new to this — start here)

**On writing and reading papers:**
- Simon Peyton Jones, *"How to Write a Great Research Paper"* (talk + slides) — the standard first reference for structuring a CS paper.
- S. Keshav, *"How to Read a Paper"* — useful both for reading related work efficiently and for understanding what reviewers look for.
- The NeurIPS reproducibility checklist (search "NeurIPS reproducibility checklist") is a good model of the rigor bar — this project's audit practice already meets or exceeds it in most places.

**Tools:**
- **Overleaf** — collaborative LaTeX writing (needed for §4).
- **Zotero** — reference management; import as citations accumulate rather than at the end.
- **arXiv** — for posting a preprint once a draft is stable (common and generally encouraged in ML/security before or alongside submission, but check the target venue's specific preprint policy first).
- **Connected Papers** and **Semantic Scholar** — citation-graph tools that complement the manual snowball search already done in the lit review; good for a final saturation check before submission.
- **OpenReview** — where AAAI/IJCAI-adjacent venues increasingly host reviews; worth being familiar with the interface before a submission is due.

**Norms worth internalizing now:**
- Authorship order and contribution — worth an explicit conversation (see §3), not left implicit.
- Double-blind review means no de-anonymizing details anywhere in the submission, including filenames, code repo links, or acknowledgments.
- **Honest negative-result reporting is the project's biggest asset, not a liability.** The central finding — "a published FL-IDS paper's algorithm doesn't converge as described" — must be reported plainly, with the mechanism and the fix, not softened or hidden. This is exactly the kind of rigor reviewers reward, and it's already this project's strongest habit (see the audit-of-audits practice, the disclosed-not-hidden regression in `../../latest_baseline.md`, and the explicit "no silent patching" stance on the validated fix).

**Venue mechanics:**
- AAAI and IJCAI run on fixed annual submission cycles with hard deadlines (typically several months before the conference) — once Stage C's scope is firm, look up the next open cycle and work backward from it rather than continuing indefinitely on "pace by readiness."
- PAKDD itself is worth keeping in mind as a fallback/complementary venue — the lit review notes PAKDD hasn't previously published in this specific niche (gradient-adaptive aggregation for FL-IDS), which cuts both ways: a fit argument for novelty, a risk argument for reviewer unfamiliarity. Worth a explicit discussion with the mentor once Stage C's angle is chosen.

---

## 6. Immediate next actions

**Stages A and B are complete.** All six actions this section previously enumerated have been executed, gated, and audited. The frontier is now **Stage C, which is at 0% and blocked on a conversation, not on compute.**

**The one action that unblocks everything — talk to the mentor.** Four questions, one conversation:

1. **Provenance.** Is FedGRAA the lab's own submission, or a third party's unpublished manuscript? Two zero-result web searches plus a DBLP and Semantic Scholar sweep found no published trace (see [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §7 and [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) §21). **This branches everything downstream:**
   - *If it's the lab's own* — Stage B was never a critique; it is internal QA, and its defect finding, fix, and corrected theory become §4–§5 of a corrected resubmission, co-authored.
   - *If it's a third party's* — the critique-as-paper plan is not viable, and Stage B becomes internal validation plus a Method-section ingredient for Stage C.
2. **The Stage C angle.** Few-shot/prototype-guided hybrid aggregation, or the privacy-compliant redesign? Both are scoped; neither is implemented. This has been "the next decision" for three consecutive passes.
3. **Compute.** Everything so far is CPU-only. Stage C's baseline comparison (FedProx/FedOpt/FedSND + AFL/q-FedAvg/FedAdp across both datasets) is materially larger than anything run to date. What is actually available?
4. **Authorship** on whatever Stage C becomes.

**Once those are answered, in this order:**

1. Close the **Stage C bibliography exposure** — ~140 carried-over entries were never independently re-verified ([`../research_c/STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](../research_c/STAGE_C_LITERATURE_REAUDIT_2026-07-25.md)). Do this *before* writing Related Work on top of it.
2. **Run the SOTA baselines** against FedGRAA — the one experimental gap that no amount of writing closes.
3. **Implement the chosen extension**, then design its evaluation against the Stage B factorial as the internal control.
4. **Draft the manuscript.** Stage A + Stage B already supply the reproduction, the defect, the corrected theory, and the ablation — the Methods and Experiments sections are substantially pre-written.

> Share [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md) and [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) with the mentor as the package. They are written to be read together and are the strongest evidence the project has.
