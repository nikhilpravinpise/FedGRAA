# Research Direction Audit — Are We Going the Right Way?

> ### ⚠️ Partially superseded — 2026-07-25
>
> **Written at 01:04, before the Stage B campaign completed at 03:01 (`4459f53`).** This document's premise that the Stage B campaign is *unrun* was overtaken within ninety minutes of writing. Read it with that correction applied.
>
> **Superseded — do not act on:**
> - Every statement that the Stage B compute campaign is pending or unrun. **It is complete**: 10 seeds × 4 arms × 2 datasets × 3 γ, both metrics, committed and gated (`check_stage_b_complete.py` and `audit_stage_b.py` both exit 0).
> - **§5 action 5** — *"cut to 5 seeds, not 10."* Acting on this today would mean **discarding a finished 10-seed campaign to rebuild it smaller.** The compute is spent; the resulting factorial is more defensible than an n=5 version; and under both branches of D1 the ablation retains its value. See [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §21.
> - §5 action 1's *"do not run compute… until it is answered"* clause — moot for Stage B specifically, since that compute had already completed. It **remains live for Stage C**, which is exactly where it now bites.
>
> **Still live — this is the valuable part of this document:**
> - **D1 (CRITICAL — provenance).** Still the highest-leverage open question in the project. Partially actioned: now fully disclosed in [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §21 and [`../current/fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §7, and the unbacked word "published" has been removed from the theory doc's title. **The question itself is still unanswered** and still needs the mentor.
> - **D2, D3, D4, D5, D6** — all unaffected by the campaign completing. D3 (the Stage C decision is the bottleneck) and D4 (baseline debt) are, if anything, *more* acute now that Stages A and B are both closed.
> - **§5 actions 1–4, 6, 7, 8** and the entire **"Stop doing"** list.
>
> **Tracked by:** [`AUDIT_REMEDIATION_PLAN_2026-07-25.md`](AUDIT_REMEDIATION_PLAN_2026-07-25.md) (R5). Current project state: [`../current/publication_roadmap.md`](../current/publication_roadmap.md).

---


**Compiled:** 2026-07-25 · **Method:** ARS `deep-research` skill, `review` mode (editor-in-chief + devil's-advocate + ethics lenses) applied to the *research programme*, not to a manuscript · **Scope:** the whole FedGRAA reproduction → critique → extension strategy as recorded in `publication_roadmap.md`, `STAGE_A_MASTER_REPORT.md`, `stage_b_plan.md` / `STAGE_B_REMAINING_TASKS.md`, `fedgraa_convergence_defect.md`, `docs/research_c/STAGE_C_LITERATURE_MASTER.md`, and `docs/paper/FedGRAA_Simulated_Peer_Review.md`.

**Bottom line up front.** The *quality* of the work is well above field norm and the *kind* of contribution being aimed at is right. The **sequencing and the framing are wrong**, and one load-bearing assumption underneath Stage B has never been checked and does not survive checking: **FedGRAA is not a published paper.** Stage B is currently scoped as a standalone, citable critique of a published algorithm; that object cannot exist. Everything else in this document follows from that, plus a second problem the first one aggravates — Stage C, the only stage that produces a paper, is at 0% and has been "the next decision" for three consecutive passes.

**Verdict: MAJOR REVISION of the strategy — not reject, not stay-the-course.** Keep Stage A entirely. Re-role Stage B from "anchor contribution" to "Method-section ingredient" and cut its compute campaign accordingly. Move Stage C to the front and unblock it this week with one conversation.

---

## 1. What is genuinely strong (do not change these)

These are stated first because they are real, they are unusual, and the corrections in §2 should not be read as devaluing them.

| # | Strength | Evidence |
|---|---|---|
| S1 | **Reproduction rigor is above field norm.** Line-for-line Keras reimplementation, 139/139 tests, forensic dataset provenance audit against raw CSV SHA-256s and canonical distributions, and every interpretive choice resolved to exactly one *documented, non-configurable* decision rather than a menu of flags. | `STAGE_A_MASTER_REPORT.md` §3, §4.1; `docs/audit/data_provenance_audit.md` |
| S2 | **The central finding is real, mechanistic, and falsifiable — not just an observed gap.** "`w` leaves the simplex in round 1, every γ, both datasets, zero exceptions," with the causal isolation done properly (same gradient, same η′, projection toggled → 193× blowup collapses to 4.3×). | `fedgraa_convergence_defect.md` §2 |
| S3 | **The fix is validated, and the honesty around it is exemplary.** Combined fix recovers 72–94% rare-class accuracy across seeds, and is deliberately kept diagnostic-only rather than merged into the reproduction baseline. The "no silent patching" mandate is held consistently across ~15 documents. | `STAGE_A_MASTER_REPORT.md` §5.7; `fedgraa_convergence_defect.md` §6 |
| S4 | **The corrected theory claims less than it could, on purpose.** Stationarity rather than global optimality, with an explicit "rigor flag" listing the three things a referee-grade proof pass would still need (β_eff, formal sub-differential treatment, explicit rate constant). Most students overclaim here; this underclaims. | `fedgraa_convergence_defect.md` §4 |
| S5 | **Literature verification discipline is better than most published related-work sections.** The web → arXiv → DBLP escalation path, the "gray zone = FAIL" rule, two caught hallucinated citations, and — critically — the **self-audit that caught the review asserting tool-verified claims it had not actually verified**, disclosed rather than buried. | `STAGE_C_LITERATURE_MASTER.md` §3, §12 |
| S6 | **Independent convergent evidence.** A simulated 5-reviewer panel, run against the manuscript, independently reached the same three critical findings as the from-scratch empirical audit. That convergence is meaningful signal that the critique targets reviewer-legible defects, not reproduction idiosyncrasies. | `docs/paper/FedGRAA_Simulated_Peer_Review.md` Appendix |

**Read together:** you have built the evidence base for a good paper. You have not started the paper, and the thing you have been treating as the safe fallback paper is not one.

---

## 2. Findings — where the direction is wrong

Severity uses the ARS convention: **CRITICAL** = invalidates a core plan assumption; **MAJOR** = materially reduces the chance of publication; **MODERATE** = real cost, recoverable.

### D1 — CRITICAL: Stage B's anchor contribution has no venue, because the target paper is not published

`publication_roadmap.md` §2 scopes Stage B as *"the fallback/anchor contribution, real even if Stage C stalls or runs long,"* and `fedgraa_convergence_defect.md` is titled **"Why a *published* FL-IDS aggregation algorithm doesn't converge, and how to fix it."**

FedGRAA is not published.

**Evidence gathered this pass (2026-07-25):**
- The manuscript in `docs/paper/` is `PAKDD_Final_Version_Submit_v3.pdf` — an anonymized submission draft ("No Author Given," "No Institute Given").
- Web search for `"FedGRAA" federated gradient-refined adaptive averaging intrusion detection` — **zero results naming FedGRAA.** Every returned link is a different, real, differently-named paper (NIDS-FGPA, EdgeDetect, FedWGCA, FedDef).
- Web search for the exact title string `"Federated Gradient-Refined Adaptive Averaging" PAKDD` — **zero results.** Same failure signature the project's own literature docs use to declare CT-BRFL and FedSND non-existent.
- No prior document in this repo — across ~9,600 lines — states the paper's publication status, DOI, venue outcome, or provenance. The project's own docs refer to "the original authors" as third parties (`REPRODUCTION_AUDIT.md`, `PAPER_AMBIGUITY_REPORT.md`) without ever recording who they are or where the manuscript came from.

**Why this is fatal to Stage B *as currently framed*:**

1. **Reproducibility venues require a published target.** ReScience C, the ML Reproducibility Challenge, and TMLR's reproducibility track all define their scope as replication of *published* work. A critique of an unpublished manuscript is out of scope by construction.
2. **Reviewers cannot verify your central claim.** The entire Stage B argument is "the paper's Algorithm 2, as printed, omits a projection step." A reviewer who cannot obtain the paper cannot check the algorithm box, cannot check Lemma 1, cannot check the reported 88.88%. The claim becomes unfalsifiable-in-practice, which is the worst possible property for a critique paper.
3. **The technical content, stripped of the critique framing, is not novel.** Euclidean projection onto the probability simplex is Duchi et al. (2008) — your own §4 cites it. Projected subgradient descent on a compact convex set is textbook. Everything publishable in Stage B lives in the sentence "*a specific published algorithm* gets this wrong." Remove "published," and the contribution collapses to a known correction applied to an object no one else can inspect.
4. **If the manuscript is a third party's, there is a confidentiality question**, not just a venue question — publicly critiquing a named, unpublished, anonymized submission obtained through a lab or review channel is a norms problem regardless of how correct the critique is.

**This is the single highest-leverage unknown in the project, and it is answerable in one conversation.** Two branches, both actionable:

| Branch | If true | What Stage B becomes |
|---|---|---|
| **(a) FedGRAA is your lab's own paper** (prof's / PhD student's PAKDD submission) — the most likely reading given the repo contains the final-submission PDF and the project runs under that lab | This was never a critique. It is **internal QA of your own group's work**, and you have done the group a large favour. | The **corrected resubmission**, co-authored. The simulated review's own R1–R7 revision list (~10–14 weeks) *is* the plan, and it is a good one. Your defect finding + fix + corrected theory become §4 and §5 of that paper, with you as an author. Stage B stops being a separate paper and becomes the reason the resubmission is credible. |
| **(b) FedGRAA is a third party's unpublished manuscript** | The critique-as-paper plan is not viable, and naming the target publicly is questionable. | **Internal validation + a Method-section ingredient for Stage C.** The defect analysis becomes the motivation paragraph of your Stage C paper ("unconstrained learned aggregation weights are numerically fragile; we constrain ours to Δ and prove stationarity"), citing the general phenomenon rather than the unpublishable manuscript. |

Note that under **both** branches, the ~6–10 hours of unattended compute plus finalization pass in `STAGE_B_REMAINING_TASKS.md` is buying something smaller than "a standalone citable component." That does not make it worthless — it makes it a sizing decision, addressed in §3.

### D2 — MAJOR: effort is allocated inversely to publication value

Rough accounting of the present state:

| Artifact class | Volume | Publication value |
|---|---|---|
| Analysis / audit / plan documents | ~9,600 lines across ~25 files, including 4 audit docs, an audit-of-audits, and 5 overlapping literature reviews later consolidated into a 6th | Indirect — supports a paper, is not one |
| Reproduction code + tests | `fedgraa/` + 139 tests, Stage B infra fully built and tested | High, but it is infrastructure |
| Persisted experimental results | Stage A complete; Stage B campaign unrun | Necessary, insufficient |
| **Manuscript** | **0 lines** | **The only thing that gets published** |

`publication_roadmap.md` §1 states this itself: *"No manuscript exists anywhere — everything is ~20 standalone analysis docs."* That was written before Stage A closed, and it is still true after Stage B's infrastructure closed.

The auditing habit that produces S1 and S5 has crossed into diminishing returns. Concretely: `STAGE_C_LITERATURE_MASTER.md` was compiled 2026-07-24 to *supersede five overlapping literature documents*, was self-audited the same day, was audited again in a second pass the same day, and was re-audited again on 2026-07-25. Four verification passes over one document in two days. The 2026-07-25 pass's net yield was one citation upgrade (Fed-LAA), one new "do not cite" entry, and one status-tag change — all genuinely correct, none of which changes a single decision.

**The failure mode has a name and it is not laziness — it is that auditing is legible progress and drafting is not.** Every audit produces a clean, satisfying artifact. A first manuscript draft produces something visibly bad. The project has optimized hard for the first feeling.

### D3 — MAJOR: the Stage C decision is the bottleneck and has been deferred three times

Both `publication_roadmap.md` §2 and `STAGE_C_LITERATURE_MASTER.md` §10 correctly say the Stage C angle (few-shot/prototype hybrid vs. privacy-compliant redesign) should be decided *with* the mentor, not unilaterally. That is right in principle. It has also been the stated next action since 2026-07-04, restated 2026-07-24, and restated again in the Stage B punch list — and it has not happened.

Everything downstream is blocked on it: which baselines to implement, which datasets to add, how much compute is needed, what the manuscript's §3 even says. **This is one conversation and it unblocks months of work.** It is the highest-return action available and it costs nothing but scheduling.

`STAGE_C_LITERATURE_MASTER.md` §10 already does the hard part — it lays out the trade-off honestly and recommends Option B (privacy redesign) on implementation-readiness grounds, noting Option A's competitive window is narrowing (FeCoGraph, FedFSA, PROTEAN, FedP3E all landed in 2025 alone). **Concur with that recommendation.** Option B has exact byte counts, a threat model, a test spec, and a function-level codebase mapping already written; Option A has none of that and faces fresher competition. Bring both, recommend B, and let the mentor overrule if they have information you don't.

### D4 — MODERATE: baseline debt is the real distance to A/A*, and current compute cannot pay it

`STAGE_C_LITERATURE_MASTER.md` §9 lists ~15 required head-to-head baselines: FedAvg, FedProx, SCAFFOLD, FedNova, FedOpt, FedAPA, FedAWA, DA-PFL, FedDRL, FedNolowe, FedMADE, SBFedNIDS, DWAFL, FD-IDS, NIDS-FGPA, Jiang et al.

Currently run: **FedAvg and Static Reparameterization.** FedProx/FedOpt/FedSND are implemented in `baselines.py` and have never been executed once.

At the documented ~15–20 min per (arm, seed) on CPU, a 15-baseline × 10-seed × 2-dataset × 3-γ grid is not reachable on the current hardware — the Stage B campaign alone (80 cells, 9 already done) is estimated at 6–10 hours at `--workers 3`.

`publication_roadmap.md` §3 says to *"ask about lab compute resources once Stage C's scope is set."* **That ordering is backwards.** Compute availability determines which Stage C angle is affordable — a 50–200-client scalability study is a different machine than a 14-client tabular MLP. Compute is an *input* to the angle decision, not a consequence of it. Fold it into the same conversation as D3.

### D5 — MODERATE: the "publishable critique" framing has one more soft spot worth knowing about

Even granting D1's branch (a) — that this is your own lab's paper and the resubmission is the plan — note what the simulated review concluded: **Reject, resubmit encouraged**, with ~10–14 weeks of required revisions, explicitly exceeding a Major-Revision window. That panel's finding R2 asks the authors to *"supply raw, seed-level training logs... replace any value that cannot be reproduced from genuine logs"* — because the reported 33.33/66.66/77.77/88.88 pattern is statistically inconsistent with stochastic measurement.

That is a delicate conversation to have with a supervisor, and it is coming. Have it early, framed as "here is what a reviewer will do to this table," not as an accusation. The evidence is strong (`PAPER_AMBIGUITY_REPORT.md` #8, #9), it is going to surface at any venue, and it is far better surfaced internally by you now.

### D6 — MINOR: one literature claim should be tightened before it gets quoted

`STAGE_C_LITERATURE_MASTER.md` §3.2 lists **FedSND** as a "confirmed hallucination" cited as FedGRAA's own reference [12]. The *no-such-paper* part is correct — three tools, zero trace. But the precise state of affairs is checkable from the manuscript file itself and is a stronger claim than the doc currently makes:

- The manuscript's **body text** (§1) says *"Zhang et al. [12] introduced FedSND."*
- The manuscript's **reference [12]** is: *Zhang, X., Li, T., Jin, R., Yang, Q.: "Federated learning on non-iid data via local and global adaptation," Proceedings of the 38th ICML, pp. 8340–8351 (2020), https://arxiv.org/pdf/2306.14443.*
- The name "FedSND" appears nowhere in that reference, and the entry is internally inconsistent on its face: it attributes an **ICML 2020** venue to an arXiv ID whose `2306` prefix denotes a **June 2023** preprint.

So the accurate finding is: **the manuscript attaches a non-existent method name to a real-but-misattributed reference** — not "a fabricated bibliography entry." That version is more precise, independently checkable by any reviewer from the two files alone, and harder to argue with. Worth correcting in the master doc, since it will be quoted.

---

## 3. Ethics review (ARS `ethics_review_agent` lens)

**Verdict: CONDITIONAL** — no blocking issue, three conditions to satisfy before any submission.

| Item | Status | Condition |
|---|---|---|
| **AI-usage disclosure** | ⚠️ Not yet drafted | This programme is substantially AI-assisted (ARS pipelines for the literature review, the simulated peer review, this audit). Every target venue now requires a disclosure statement. Draft it once, early, via `/ars-disclosure`; do not improvise it at submission. |
| **Attribution / verification integrity** | ⚠️ Known open exposure | The 2026-07-24 self-audit (`STAGE_C_LITERATURE_MASTER.md` §12) established that ~140 bibliography entries carried over from the five source documents were **never independently re-verified in that pass** and inherit unaudited confidence — in a document whose own first draft asserted verifications that had not been run. Before any of those entries enter a manuscript, they need a real check. This is the most likely path from "excellent process" to "a retracted-citation embarrassment." |
| **Fair representation of the critiqued work** | ⚠️ Branch-dependent | See D1. Under branch (b), naming an unpublished third-party manuscript in a public critique is a norms problem independent of correctness. Resolve provenance before anything is written for public consumption. |
| **Dual-use screening** | ✅ Cleared | Defensive intrusion detection; no offensive capability produced. |
| **Data provenance** | ✅ Cleared | Both datasets are public research benchmarks, forensically verified, correctly cited. |

---

## 4. Devil's advocate — the strongest case that this project is *not* going anywhere

Stated at full strength, then answered, because a direction audit that only produces polite adjustments is not doing its job.

> **"You have spent months building an exhaustively-audited reproduction of an unpublished, anonymous, 13-reference workshop-grade manuscript with fabricated-looking results and at least one invented citation. The reproduction is excellent. The object is not worth it. Everything of value you have produced — a simplex projection, a stationarity proof, a 10-seed ablation harness — is standard technique applied to a paper nobody will ever read. The audits keep producing artifacts, which keeps feeling like progress, which is why the actual paper has zero lines nine months in. The honest move is to keep the codebase, discard the FedGRAA framing entirely, and start a real paper on the FL-IDS minority-class problem using this as infrastructure."**

**How much of that is right: most of the diagnosis, not the conclusion.**

Right: the object is weaker than the effort invested in it; the critique-as-paper plan does not work (D1); the audit loop is a substitute for drafting (D2); the technical content is standard in isolation.

Wrong: "discard and restart" throws away the two things that are hardest to buy. First, **a working, tested, provenance-verified FL-IDS experimental platform** — 139 tests, both canonical datasets audited to the row, a seed-sweep harness, a stats module with bootstrap CIs and Holm–Bonferroni, resumable multi-seed drivers. That is typically the first three months of any FL paper and you already have it. Second, **a real, measured, non-obvious empirical result**: learned aggregation weights on extreme-skew IDS data leave the simplex immediately and catastrophically, and severity scales with client skew (193× on CIC's 11-sample Heartbleed vs. 7.6× on UNSW's 174-sample Worms). That finding is *interesting independent of FedGRAA* and belongs in your Stage C paper's motivation regardless of what happens to the manuscript that prompted it.

So: the devil's advocate wins on framing and sequencing, loses on scrap-it. The correction is to **stop treating FedGRAA as the subject and start treating it as the motivating instance** — which is exactly what happens naturally once Stage C is the paper and Stage B is its §4.

---

## 5. Decision and revised sequence

**Overall verdict: MAJOR REVISION of the strategy.** Stage A unchanged. Stage B re-roled and cut. Stage C promoted to the front and unblocked immediately.

### Do this week

1. **Settle provenance.** Who wrote FedGRAA, where did this PDF come from, what happened to the PAKDD submission, and are you a co-author on any resubmission? One conversation with the prof / PhD mentor. **Every other decision in this document branches on the answer**, so do not run compute or write prose until it is answered.
2. **Force the Stage C angle decision in the same conversation.** Bring `STAGE_C_LITERATURE_MASTER.md` §10's table. Recommend **Option B (privacy-compliant redesign)** — it is implementation-ready, less crowded, and addresses a flaw the field has independently called out (FedTruth, CoAst). Bring Option A as the alternative and let the mentor decide.
3. **Ask the compute question in that same conversation**, as an input to (2), not a follow-up. "Which of these two angles can we actually afford on available hardware?" is a legitimate and clarifying question.
4. **Raise the authorship conversation** while you are already in it. `publication_roadmap.md` §3 has been right about this for a while; it is a normal conversation, and it gets more awkward the longer it waits.

### Then, in order

5. **Cut the Stage B campaign to what a paper section actually needs.** Do not run 71 fresh (arm, seed) cells to populate a section whose venue is now in question. Decide the ablation table's final shape *first* — most likely 4 arms × 2 datasets × 3 γ at **5 seeds**, not 10 — then run exactly that. The stats module already reports bootstrap CIs and both parametric and non-parametric tests, which is what makes n=5 defensible. The infrastructure is built and resumable, so this is a sizing decision, not a rebuild.
6. **Start the manuscript now, before the results are in.** Overleaf, the target venue's author kit, section headers, and the placeholder tables you intend to fill. A skeleton with `\todo{}` markers surfaces what is genuinely missing in an afternoon — far faster and more honestly than another audit pass. This is the single habit change that matters most.
7. **Freeze the literature documents.** One further pass, at submission time, with exactly two jobs: read Jiang et al. (*Neurocomputing* 604:128317, 2024) in full — it is the closest philosophical precedent and the most likely thing a reviewer discovers unprompted — and verify the ~140 unaudited carried-over bibliography entries that actually end up cited. Nothing else. No new consolidation documents.
8. **Implement and run the baselines the chosen angle requires**, in the §9 priority order, sized to real compute. FedProx and FedOpt are already implemented and have never been run once — start there, since it costs nothing but time and immediately answers "does the fixed FedGRAA beat anything that isn't a strawman?"

### Stop doing

- Audits of audits. The verification discipline is a genuine strength; a fourth pass over the same document in two days is that strength eating the schedule.
- New consolidation documents. There are now six literature documents where the sixth exists to supersede the first five. A seventh will not help.
- Re-verifying the same ~15 core papers. They are verified. Record it and move on.

---

## 6. Honest limitations of this audit

- **The provenance question is unresolved, not answered.** Two web searches returning zero results establishes that FedGRAA has no public trace *findable by general search*; it does not conclusively establish non-publication (a paywalled or very recently indexed venue could evade it, and this pass did not query DBLP or Semantic Scholar directly — the escalation path this project's own literature docs mandate). The negative result is strong enough to act on and to make the question urgent; it is not strong enough to close the question without the mentor's answer. Applying the project's own evidentiary standard: this is a strong negative, not yet a confirmed one.
- **This audit assessed strategy, not correctness.** It did not re-derive the convergence proposition, re-run any experiment, or re-verify any dataset claim. Those were audited independently on 2026-07-23 and this pass takes those results as given.
- **This is a single-perspective assessment** — one reviewer's lens across editorial, skeptic, and ethics roles, not five independent panelists. Where it disagrees with the mentor, the mentor has context this document does not.
- **This audit is itself AI-generated**, and is subject to the same disclosure obligation described in §3.

---

*Compiled 2026-07-25 via the ARS `deep-research` skill, `review` mode. Verification performed this pass: two web searches for the FedGRAA title and acronym (both zero-result, 2026-07-25); direct inspection of `docs/paper/PAKDD_Final_Version_Submit_v3.md` §1 and reference [12]; a document/volume census of `PAKDD/docs/`. All other findings are read from existing project documents and cited to them inline.*
