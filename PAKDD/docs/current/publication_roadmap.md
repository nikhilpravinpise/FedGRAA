# Publication Roadmap — FedGRAA: Reproduction, Critique & Extension

**Purpose of this document:** a single entry point for "what's left to do to genuinely publish this." It is a router, not a re-audit — it points to the existing docs that already contain the detailed work, and lays out the sequence, decision points, and process knowledge to get from here to a real submission.

**Confirmed strategy:** finish the faithful reproduction → package the critique/diagnosis as a rigorous component → build the full extension for an A/A* venue (AAAI/IJCAI-tier). Paced by readiness, no fixed deadline yet. Working under an IIT Guwahati professor with day-to-day PhD-student mentorship.

---

## 1. Where things stand

**The reproduction is strong and well-audited.** A faithful, Keras-based reimplementation of FedGRAA lives in `PAKDD/fedgraa/`, with 79/79 tests passing and both datasets (CIC-IDS-2017, UNSW-NB15) independently verified as genuine, canonical data — see `data_provenance_audit.md`. Every deliberate deviation from the paper is logged in `strict_reproduction_report.md`.

**The central finding is a real, defensible result.** The paper's Algorithm 2 has no simplex projection on the aggregation weight vector `w`. On real data, `w` leaves the probability simplex in round 1, at every γ, on both datasets — causing the composite loss to blow up and rare-class accuracy to collapse to 0% (vs. the paper's claimed ~88%). This is fully diagnosed, not just observed, in `fedgraa_divergence_diagnosis.md`. A minimal fix — simplex projection + η′=0.01 instead of 0.1 — is validated across 3 seeds, recovering 72–94% rare-class accuracy, and is deliberately kept diagnostic-only rather than silently merged into the strict-reproduction baseline.

**The paper itself has been critiqued rigorously.** `PAPER_AMBIGUITY_REPORT.md` catalogs 17 distinct issues in the original paper — contradictions between SGD-notation and Adam-prose, a convexity assumption in the convergence proof that the paper's own non-convex tanh-MLP violates, an undefined "last-n accuracy" metric, and evidence that some reported numbers (33.33/66.66/77.77/88.88) look like repeating-decimal placeholders rather than genuine measurements.

**The literature review is already publication-grade.** `literature_review_and_publication_strategy.md` is a multi-pass, snowball-searched review across 30+ papers at A*/A venues (CCS, NDSS, AAAI, IJCAI, NeurIPS, ICML), with an explicit scooping-risk verdict (no near-duplicate found) and a recommended novelty angle (few-shot/prototype-guided hybrid aggregation) plus venue ranking (AAAI/IJCAI first).

**A second extension angle exists but is unbuilt.** `privacy_compliant_fedgraa_redesign.md` (~730 lines) identifies that the original paper's design assumes server-side oracle knowledge of which clients/classes are rare, and a server-held labelled validation set — both of which arguably violate FL's core privacy premise. It proposes a concrete redesign, grounded in 7 families of prior work, but none of it is implemented.

**There is also a didactic series worth mining for the write-up.** `understanding/00_START_HERE.md` … `understanding/10_SOTA_adaptive_aggregation.md` is an 11-part plain-language walkthrough of the method, the convergence proof (`05_convergence_proof.md`), and the SOTA baseline landscape (`10_SOTA_adaptive_aggregation.md`). These feed Stage B (the proof-sketch and corrected theory) and Stage C (the baseline comparison and Related Work) directly.

**What's missing for a submission:**
- No manuscript exists anywhere — everything is ~20 standalone analysis docs plus `PAKDD_STATUS_REPORT.md`.
- Experimental matrix is incomplete: only the CIC-IDS-2017 γ-sweep (n=3) is persisted to `results/`. The n=5/7 sweeps (paper's Fig. 2–3) and the UNSW-NB15 γ-sweep are runnable but not yet run/persisted. The combined fix has only been seed-swept on CIC, not UNSW.
- Baselines beyond FedAvg/Static are implemented (`fedgraa/baselines.py` has FedProx, FedOpt, FedSND) but never run against FedGRAA. A separate SOTA-baseline catalogue, `understanding/10_SOTA_adaptive_aggregation.md`, flags AFL, q-FedAvg, and FedAdp as "must-have" comparisons — none run yet.
- No formal statistical testing (only 3-seed variance bands, no t-tests/bootstrap CIs).
- No ablation table packaging the projection-vs-η′ isolation tests that already exist informally in the diagnosis doc.
- No corrected convergence theory for the fixed algorithm (the diagnosis is empirical, not a new proof).
- The extension method (few-shot/prototype hybrid, or the privacy redesign) is unimplemented.

---

## 2. The three-stage strategy

### Stage A — Close out the faithful reproduction

Goal: every table/figure in the original paper has a corresponding persisted, reproduced (or honestly-diverging) counterpart in `results/`.

- Run and persist the n=5 and n=7 client sweeps on CIC-IDS-2017 (paper's Fig. 2–3 equivalent). `run_experiments.py` / `scripts/real_reproduction.py` already support this — check whether a flag or config change is needed for n beyond the current n=3 default.
- Run and persist the UNSW-NB15 γ-sweep (paper's Table/Fig. equivalent for the second dataset) — currently only the literal divergence diagnosis has been run there, not the base reproduction sweep.
- Re-run the combined fix (projection + η′=0.01) with a seed sweep on UNSW-NB15, mirroring what's already done for CIC.
- Regenerate figures from these persisted results using `scripts/render_diagnostics.py` or `fedgraa/plot.py`.

### Stage B — Package the critique as a rigorous, citable component

Goal: turn the diagnosis from "internal audit trail" into something that reads as a paper section on its own — this is the fallback/anchor contribution, real even if Stage C stalls or runs long.

- Write a precise, formal statement of the defect (no simplex projection → weight vector exits the feasible set → Lemma 1's non-increasing-loss guarantee no longer holds) with a short proof sketch of *why* the paper's convergence proof doesn't cover this case (it assumes convexity the tanh-MLP doesn't have — already identified in `PAPER_AMBIGUITY_REPORT.md`).
- Build a proper ablation table: projection alone vs. η′ reduction alone vs. both, currently scattered across the diagnosis doc's sections — consolidate into one table with more than 3 seeds and real significance testing (bootstrap confidence intervals or paired t-tests) rather than variance bands.
- This section should stand alone as "Why a published FL-IDS aggregation algorithm doesn't converge, and how to fix it" — useful framing to test with the mentor even before Stage C is scoped.

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
  - `data_provenance_audit.md` + `strict_reproduction_report.md` → Experimental Setup / reproducibility appendix.
  - `understanding/05_convergence_proof.md` → scaffolding for the corrected convergence analysis; `understanding/10_SOTA_adaptive_aggregation.md` → the baseline-comparison framing and part of Related Work.
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
- **Honest negative-result reporting is the project's biggest asset, not a liability.** The central finding — "a published FL-IDS paper's algorithm doesn't converge as described" — must be reported plainly, with the mechanism and the fix, not softened or hidden. This is exactly the kind of rigor reviewers reward, and it's already this project's strongest habit (see the audit-of-audits practice, the disclosed-not-hidden regression in `latest_baseline.md`, and the explicit "no silent patching" stance on the validated fix).

**Venue mechanics:**
- AAAI and IJCAI run on fixed annual submission cycles with hard deadlines (typically several months before the conference) — once Stage C's scope is firm, look up the next open cycle and work backward from it rather than continuing indefinitely on "pace by readiness."
- PAKDD itself is worth keeping in mind as a fallback/complementary venue — the lit review notes PAKDD hasn't previously published in this specific niche (gradient-adaptive aggregation for FL-IDS), which cuts both ways: a fit argument for novelty, a risk argument for reviewer unfamiliarity. Worth a explicit discussion with the mentor once Stage C's angle is chosen.

---

## 6. Immediate next actions

1. Check `run_experiments.py` / `scripts/real_reproduction.py` for how to trigger n=5 and n=7 client sweeps on CIC-IDS-2017; run and persist to `results/`.
2. Run and persist the UNSW-NB15 γ-sweep (base reproduction, not just the diagnosis).
3. Re-run the combined fix with a seed sweep on UNSW-NB15.
4. Regenerate figures from the newly persisted results.
5. Once Stage A is done, draft the Stage B critique section (formal defect statement + consolidated ablation table + significance testing) as a standalone markdown doc, then share Stage A+B with the PhD mentor to jointly scope Stage C.
