# Direction Assessment — Are Stage A/B/C On Track?

> ### ⚠️ Partially superseded — 2026-07-25
>
> **Written at 01:00, before the Stage B campaign completed at 03:01 (`4459f53`).** Read it with that correction applied.
>
> **Superseded — do not act on:**
> - **§2 "Stage B — Infrastructure complete; one compute task remaining."** That compute task **has been run.** Stage B is complete, committed, and gated; its capstone is [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md).
> - The test count (139) — now **142**.
> - **§3's finding** that `publication_roadmap.md` is stale, and **§4/§7's** framing of the Stage B campaign as the critical path. The roadmap was refreshed on 2026-07-25 (R4); the campaign is done.
> - §4's "Recommended sequence" items 1–2.
>
> **Still live:**
> - **§4 item 3 / §7's bottom line: the mentor conversation is the real blocker on Stage C**, and nothing in this repository can substitute for it. This is now *the* critical path, and it has only become more so.
> - **§5's Socratic thesis check** — options A/B/C/D are still open and still undecided. Note that option A ("Stage B stands alone") is the one D1 of [`../audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md`](../audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md) puts at risk, pending provenance.
> - **§6's provisional chapter-shape** and §1's ARS-pipeline placement.
>
> **Tracked by:** [`../audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md`](../audit/AUDIT_REMEDIATION_PLAN_2026-07-25.md) (R5). Current project state: [`publication_roadmap.md`](publication_roadmap.md).

---


**Compiled**: 2026-07-25, using the ARS `academic-pipeline` skill (to place this project within the full research→write→review→revise→finalize workflow) and the ARS `academic-paper` `plan` mode / Socratic framework (to surface the open thesis/contribution questions that gate the next real milestone).

**Short answer: yes, correctly positioned — with one stale document to fix and one sequencing risk worth naming explicitly before it compounds.**

---

## 1. Where this project actually sits in the ARS pipeline

`academic-pipeline` defines a 10-stage arc: RESEARCH → WRITE → INTEGRITY → REVIEW → REVISE → RE-REVIEW → RE-REVISE → FINAL INTEGRITY → FINALIZE → PROCESS SUMMARY. The project's own Stage A/B/C labels (from [`publication_roadmap.md`](publication_roadmap.md)) are a *different* numbering — easy to conflate, so mapped explicitly here:

| Project's own stage | What it is | Maps to ARS pipeline stage |
|---|---|---|
| Stage A — faithful reproduction | Reimplement FedGRAA, verify data provenance, run the paper's own experiments, diagnose the divergence | **ARS Stage 1 (RESEARCH)** — evidence-gathering, closer to `deep-research`'s fact-check/systematic-review modes than to any writing |
| Stage B — critique packaging | Formalize the convergence-defect finding as a citable theory section + statistically rigorous ablation | **Still ARS Stage 1 (RESEARCH)** — synthesis of findings, not manuscript drafting |
| Stage C — novel extension | Build and evaluate a new method (few-shot/prototype hybrid or privacy redesign), positioned against the literature | **Still ARS Stage 1 (RESEARCH)**, transitioning into **ARS Stage 2 (WRITE)** once the method is built, evaluated, and the manuscript drafting actually starts |

**The load-bearing fact:** per `publication_roadmap.md` itself, *"No manuscript exists anywhere — everything is ~20 standalone analysis docs."* No ARS Stage 2 (WRITE) has begun. This means the project has **not** skipped ahead — it has not started drafting a paper before the research foundation (reproduction + critique + literature positioning) was solid. That is the correct order, and it's worth stating plainly because it's the single most common failure mode `academic-pipeline` is designed to prevent (Anti-Pattern table: "orchestrator/user doing substantive writing before the research stage's deliverables — RQ Brief, Methodology Blueprint, Bibliography, Synthesis — are in hand").

The earlier work in this session — the simulated 5-reviewer peer review of the *original* FedGRAA paper ([`FedGRAA_Simulated_Peer_Review.md`](../paper/FedGRAA_Simulated_Peer_Review.md)) — is also correctly sequenced: it's an ARS Stage-3-*shaped* exercise applied to someone else's paper as **evidence for this project's own Stage 1 synthesis**, not a premature review of a manuscript that doesn't exist yet. It corroborates the Stage A/B findings from an independent "blind panel" angle. Good use of the tool, not a detour.

---

## 2. Current state of each stage, verified against the actual files (not the roadmap's stale summary — see §3)

### Stage A — ✅ Complete
[`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md) header: *"Status: Stage A ✅ complete. Completion commit: `be4ef03`."* Both datasets' γ-sweeps and n-sweeps are run, persisted, and rendered; 79/79 tests pass; the central finding (missing simplex projection → round-1 divergence) is fully diagnosed with measured blowup ratios on both datasets; 17 paper ambiguities are catalogued. This is further corroborated by `git log` — `42c3e3a Stage A: complete faithful reproduction (CIC n-sweep, UNSW gamma/n/combined-fix)` and a follow-up correction commit `a17a443`.

### Stage B — Infrastructure complete; one compute task remaining
[`STAGE_B_REMAINING_TASKS.md`](STAGE_B_REMAINING_TASKS.md): the theory (defect statement + corrected convergence proposition, both proof-sketched), the statistics module (`fedgraa/stats.py`, 19 tests), the ablation driver/builder/renderer, and the completeness checker are **all built and tested** (139/139 tests green). What remains is **exactly one thing**: the 10-seed × 4-arm × 2-dataset compute campaign (`run_ablation_seed_sweep.py`, ~6–10h unattended parallel or ~18–23h sequential) plus a ~30–60 min mechanical finalization pass. The doc's own bottom line: *"The gap between 'Stage B infrastructure-complete' and 'Stage B done' is compute time, not further design or engineering work."*

### Stage C — Literature review substantially advanced; method undecided
Two new documents dated **today** (2026-07-25) exist under [`../research_c/`](../research_c/): [`STAGE_C_LITERATURE_MASTER.md`](../research_c/STAGE_C_LITERATURE_MASTER.md) (compiled 2026-07-24, ~170 papers surveyed across 5+ passes, consolidating and superseding 5 earlier documents that are *not* deleted but retained as the verification trail) and [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](../research_c/STAGE_C_LITERATURE_REAUDIT_2026-07-25.md) (a same-day fact-check follow-up pass). This is real, rigorous work — hallucinated citations caught and removed (FedSND, FeCWGAN-GP, CT-BRFL, pFedCross), a corrected venue ranking (**PAKDD is CORE B, not A** — a materially important correction, see §4.4), and a precisely identified competitor set (FedAPA, FedAWA, FedMADE, SBFedNIDS, DA-PFL) that any Stage C method must benchmark against. **What is not yet decided**: which Stage C method to actually build. `publication_roadmap.md` §3 is explicit that this choice — few-shot/prototype hybrid vs. the privacy-compliant redesign (FedGRAA-v2) vs. both — is *"a decision point — bring this to the prof/PhD mentor, don't decide alone."* No evidence in the tree that this mentor conversation has happened yet.

---

## 3. Finding worth fixing: `publication_roadmap.md` is now stale

`publication_roadmap.md` — the document explicitly designed to be *"a single entry point for 'what's left to do'"* — currently still describes Stage A as **not yet started** (§6, "Immediate next actions": *"Execute Stage A in this order..."*) and its §1 "Where things stand" summary predates the Stage B infrastructure build and both Stage C literature documents entirely. This is a real inconsistency, not a nitpick: it is the document a returning session (or the PhD mentor, if shown this repo) would read first, and it currently underclaims the project's actual progress by roughly two stages.

**Recommended fix** (mechanical, ~20–30 min): update `publication_roadmap.md` §1 and §6 to point to `STAGE_A_MASTER_REPORT.md` (done), `STAGE_B_REMAINING_TASKS.md` (the real "what's left"), and the two Stage C literature docs — replacing the stale "Execute Stage A" instructions with "Run the Stage B compute campaign, then bring Stage A+B+C-literature to the mentor."

---

## 4. The one sequencing risk worth naming

**Stage C literature work has outpaced Stage B's actual remaining task, and Stage B's remaining task is cheap.** `STAGE_B_REMAINING_TASKS.md` describes what's left as *"one command to run twice, four to run once, then a ~30-minute writing pass"* — a ~6–10 hour unattended compute campaign. Meanwhile, two substantial Stage C literature documents were produced today, and per the roadmap's own sequencing (§3: *"Bring the completed Stage A reproduction and Stage B critique package to the mentor before starting Stage C"*), Stage C's *literature survey* is reasonable to run in parallel (it costs no compute and de-risks the eventual method choice), but Stage C's *method decision and implementation* is explicitly gated on that not-yet-documented mentor conversation, which is itself gated on Stage A+B being presentable as a finished package.

This is not a crisis — literature reconnaissance done early is not wasted work, and the master doc is genuinely high quality (it caught and removed real hallucinated citations, which is exactly the kind of rigor that protects the eventual submission). But it is worth being explicit that **the critical-path item, right now, is the Stage B compute campaign** — not more literature verification — because it is both the cheapest remaining task in the entire roadmap and the one that closes out the "fallback/anchor contribution real even if Stage C stalls" package the roadmap itself identifies as the thing to bring to the mentor first.

**Recommended order for the next work session:**
1. Run the Stage B 10-seed ablation campaign (`STAGE_B_REMAINING_TASKS.md` §2.1–2.3) — this can run unattended; nothing below depends on watching it.
2. While it runs: fix `publication_roadmap.md`'s staleness (§3 above) and do the ~30–60 min Stage B finalization checklist once the campaign lands.
3. Take the completed Stage A + Stage B package (plus the Stage C literature master as supporting context) to the mentor — this is a **human action**, not something further agent work can substitute for, and it's the actual blocker on Stage C's method decision.
4. Only after that conversation: begin Stage C implementation of whichever method (or combination) is chosen.

---

## 5. Socratic thesis check (ars-plan Step 0 / Step 1 / Step 2.5) — open, not answered here

`academic-paper` `plan` mode's first two steps ask, before any chapter planning: *"Is your research question finalized? Can you state it in one sentence?"* and *"What is your paper arguing?"* Per the plan mode protocol's own boundary rule (the Kong L2 verb test: the mentor may only surface and ask, never propose, substitute, rank, expand, or select a contribution claim for the user), this assessment does **not** answer these on the user's behalf — but it can state precisely what the project's own documents already show the candidate answers are, since laying out real options grounded in existing docs is different from choosing among them:

| Candidate thesis / contribution | Grounded in | What it would require to finish |
|---|---|---|
| **A. "Stage B stands alone"** — *A published FL-IDS aggregation algorithm's convergence proof and its algorithm as printed disagree; here is the formal defect, the fix, and the statistically validated correction* | [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md), Stage B ablation (pending compute) | Nothing beyond finishing Stage B — this is the "anchor contribution, real even if Stage C stalls" the roadmap already names |
| **B. "Few-shot/prototype hybrid"** — a new aggregation method combining FedGRAA's gradient-refinement idea with prototype exchange (per PROTEAN, the closest same-domain precedent) or few-shot personalization | `STAGE_C_LITERATURE_MASTER.md` §8, §10 gap-closing checklist | Full Stage C implementation + benchmarking against FedAPA/FedAWA/FedMADE/SBFedNIDS/DA-PFL |
| **C. "Privacy-compliant redesign (FedGRAA-v2)"** — fix the two structural privacy flaws (server-side oracle knowledge of rare classes; server-held labeled `D_V`) via inferred rarity + federated validation backward pass | [`privacy_compliant_fedgraa_redesign.md`](../research_c/privacy_compliant_fedgraa_redesign.md), `STAGE_C_LITERATURE_MASTER.md` §7 | Implementation of the ≈6×-communication-cost federated backward pass; empirical validation that it preserves Eq. 8–10 exactly |
| **D. "B, then C"** — ship Stage B as a standalone contribution now (lower risk, faster), pursue B or C as a follow-on paper | Combination of the above | Sequencing decision only — no new technical work beyond A + (B or C) |

This table is descriptive of what already exists in the project's own documents, not a recommendation — per the plan-mode boundary, that choice belongs to the user and the mentor, and the roadmap already correctly identifies it as a mentor-level decision rather than something to resolve solo.

**One fact relevant to weighing these, surfaced only today by the Stage C literature work:** PAKDD's actual current CORE ranking is **B**, not A — independently confirmed across two sources in the re-audit. This matters for options B/C's venue framing: `publication_roadmap.md` §5 already flags PAKDD as a "fallback/complementary venue" with AAAI/IJCAI as the A*/A target, and the corrected ranking sharpens rather than changes that framing — but it's a fact the mentor conversation should have accurately, since an earlier draft's premise ("PAKDD = CORE A") was wrong.

---

## 6. Provisional chapter-shape (not a commitment — for orientation only)

Independent of which Stage C option is chosen, the existing project material already maps cleanly onto standard IMRaD sections, which is useful to see even before the thesis is finalized (this is *not* a Chapter Plan in the `plan` mode sense — that requires the actual Socratic dialogue with the user once the thesis is fixed):

| Eventual paper section | Already-written source material |
|---|---|
| Introduction / motivation | `publication_roadmap.md`, the original FedGRAA problem framing |
| Related Work | `STAGE_C_LITERATURE_MASTER.md` §4 (full reference catalog) → first draft directly |
| Reproduction / Experimental Setup | `STAGE_A_MASTER_REPORT.md` §2–4, `data_provenance_audit.md`, `strict_reproduction_report.md` |
| The Defect / Correction (Stage B, if option A or D) | `fedgraa_convergence_defect.md` — already written end-to-end as a paper section |
| Method (if option B or C) | `privacy_compliant_fedgraa_redesign.md` or a new few-shot/prototype design doc (not yet written) |
| Results | Stage A/B persisted `results/` artifacts + (for B/C) new Stage C experiments |
| Discussion / Limitations | Already drafted in `fedgraa_convergence_defect.md` §6–7; would extend to cover the chosen Stage C method |

---

## 7. Bottom line

**Direction: correct.** The project has not written ahead of its evidence, has not skipped the integrity/reproduction groundwork, and has produced two independently-derived critique artifacts (the from-scratch Stage A/B audit and this session's simulated blind peer-review panel) that corroborate each other — a genuinely strong position to bring to a mentor. The only concrete corrections are administrative, not technical: refresh `publication_roadmap.md`'s stale status section, and prioritize finishing the (cheap, already-built) Stage B compute campaign over continuing to deepen Stage C literature work that is already sufficient to support a mentor conversation. The actual next bottleneck is not more agent-driven research — it's the human mentor conversation that decides Stage C's angle, which nothing in this repository can substitute for.
