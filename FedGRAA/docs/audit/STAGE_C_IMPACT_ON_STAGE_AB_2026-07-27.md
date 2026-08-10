# Does `Final_Stage_C.md` Change Anything in Stage A/Stage B?

**Compiled:** 2026-07-27. **Scope:** compares [`research_c/Final_Stage_C.md`](../research_c/Final_Stage_C.md) (S10, the final Stage C literature entry point) against [`current/STAGE_A_MASTER_REPORT.md`](../current/STAGE_A_MASTER_REPORT.md), [`current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md), both plan docs, `STAGE_B_REMAINING_TASKS.md`, `STAGE_DIRECTION_ASSESSMENT.md`, `fedgraa_convergence_defect.md`, the two Stage A/B audit docs, the `fedgraa/` codebase, and the manuscript draft (`docs/paper/PAKDD_Final_Version_Submit_v3.md`).

**Method:** direct read of `Final_Stage_C.md` in full (922 lines); a background research pass read all nine listed Stage A/B documents in full and reported verbatim findings; independently spot-checked against `fedgraa/aggregation.py`, `fedgraa/baselines.py`, `fedgraa/stats.py`, and the paper draft to confirm code-level claims rather than trust either document's prose.

> **⚠️ Correction (2026-07-27, same day, post-initial-draft of this report).** The first version of this report treated `PAKDD_Final_Version_Submit_v3.md` as *the* manuscript draft and flagged its uncorrected ref [12] ("FedSND") as an open action item. That was based on an incomplete picture of the project's own workflow: **`PAKDD_Final_Version_Submit_v3.md` is deliberately kept as the untouched original manuscript** — no fixes are applied there. All corrections (from Stage B's audits and now cross-checked against Stage C) are applied to **[`draft1.md`](../paper/draft1.md)**, the actual working/corrected draft, after which the corresponding edits are reverted out of `v3.md` so it stays pristine. §1 below is corrected accordingly.

---

## Verdict

**No, Final_Stage_C.md does not overturn anything Stage A or Stage B did.** Stage A (reproduction) and Stage B (defect diagnosis + fix + statistics) never made a novelty claim, a venue-ranking claim, or a competitor-landscape claim — that was always explicitly out of scope, deferred to "Stage C." Final_Stage_C.md is that deferred literature work arriving. Everywhere it touches something Stage A/B already asserted (PAKDD's venue rank, the convergence defect, the privacy flaws, the datasets used), it **corroborates** Stage A/B rather than contradicting it — in one case (§12, cross-check against `fedgraa_convergence_defect.md`) it says so explicitly: *"no drift found."*

There is exactly **one concrete correctness problem** it surfaces that Stage A/B's own artifacts should act on, and it isn't new — earlier passes already flagged it, but it was never fixed:

> **`FedSND` is a hallucinated method, and it is currently coded, planned, and cited as if it were real.**

Everything else is new *input* for decisions Stage A/B already knew were pending (the Stage C method angle, the baseline grid, the paper's Related Work section) rather than a correction to work already completed.

---

## 1. `FedSND` — already fixed where it matters, still loose in two other places

This needs three-way bookkeeping across the paper's two files and the code, because the project deliberately splits "frozen original" from "working draft":

| Where | Status | Detail |
|---|---|---|
| `docs/paper/PAKDD_Final_Version_Submit_v3.md:24` | **Untouched by design.** Still reads *"Zhang et al. [12] introduced FedSND..."* | This file is intentionally kept as the pristine original manuscript — no corrections are ever applied here. Its uncorrected state is expected, not a gap. |
| `docs/paper/draft1.md` (abstract, §1 line 24, ref [12] at line 1715) | **✅ Already fixed, 2026-07-26 — a day before Final_Stage_C.md.** In-text: *"Wang et al. [12] introduced *FedNova*, which normalizes local client updates..."* Bibliography: ref [12] is now Wang/Liu/Liang/Joshi/Poor, NeurIPS 2020, arXiv:2007.07481, with an inline correction note: *"this entry previously cited 'Zhang et al., FedSND' — confirmed a fabricated citation by three independent bibliographic tools... Replaced with FedNova, the literature master's recommended closest real analogue."* | This is exactly the fix Final_Stage_C.md §9.1 item 12 recommends ("replace with FedAWA or FedNova"). **It was already done, sourced from an earlier Stage C pass (`STAGE_C_LITERATURE_MASTER.md` §3.2) and `INDEPENDENT_FULL_AUDIT_2026-07-26.md`, before the final consolidated document even existed.** Final_Stage_C.md corroborates this fix rather than requesting it. Draft1 also independently: withdrew the unsupported "+55% over FedAvg" claim, added a scope-disclosure note to Appendix A.1 correctly limiting the convergence theorem to a stationarity result, and corrected three other misattributed/mis-dated references (4, 8, 11) — all Stage B/audit work, all consistent with, not contradicted by, Final_Stage_C.md. |
| `fedgraa/baselines.py:278-299` | **Still open.** `FedSNDTrainer` remains an unannotated class, docstring: *"FedSND baseline: distribution-aware static reweighting... addressing gap m-3"* — no note that the method it's named for doesn't exist | Code-level artifact, untouched by the draft1/v3 split. If this is ever run head-to-head and reported as "FedGRAA vs. FedSND," that would cite a method that doesn't exist. Should be relabeled (e.g. "a synthetic ablation inspired by the manuscript's own retracted citation, not a reproduction of a published method") or dropped before it's ever run. |
| `stage_a_plan.md`, `stage_b_plan.md`, `STAGE_B_REMAINING_TASKS.md`, `STAGE_A_MASTER_REPORT.md` §12.3/§14.3 | **Still open.** All list "FedProx/FedOpt/**FedSND** implemented-but-unrun" as a Stage C to-do, with no fabrication flag attached in these particular files | Low urgency (nothing has been run yet) but worth a one-line annotation so a future pass doesn't schedule FedSND as if it were a real baseline to beat. |

**Net effect on the previous version of this report:** the manuscript-side fix is done, and done before Final_Stage_C.md was written — so §9.1 item 12 is corroboration, not a new requirement. What remains open is narrower than originally stated: only the `baselines.py` code annotation and the plan-doc to-do lists, not the manuscript.

Everything else below is corroboration or new forward-looking input, not a defect in existing work.

---

## 2. What Final_Stage_C.md corroborates (no change needed)

| Stage A/B claim | Final_Stage_C.md's independent finding |
|---|---|
| PAKDD is CORE B, not A (`STAGE_DIRECTION_ASSESSMENT.md` §5, corrected 2026-07-25) | §6/§6.1: **re-confirmed directly at `portal.core.edu.au`** this pass, plus new sourced numbers (Computer Networks 2-yr citedness 4.413 vs. LNCS 1.409) that make the "not A*/A" defense *weaker* than previously argued — sharpens, does not reverse, the existing correction |
| Datasets: CIC-IDS-2017 + UNSW-NB15 only, no others (Stage A/B, throughout) | §5: same two datasets remain the "primary/secondary" pair; no Stage A/B claim about them is contradicted |
| The two structural privacy flaws — server-side oracle on rare classes, server-held `D_V` (`STAGE_DIRECTION_ASSESSMENT.md` §5 option C, `privacy_compliant_fedgraa_redesign.md`) | §7.1: same two flaws, same framing, same "FedGRAA-v2" fix concept. §7.3 adds exact byte/DP-budget numbers on top of the existing design, doesn't change it |
| The Algorithm 2 divergence / missing simplex projection (`fedgraa_convergence_defect.md`, Stage B's own theory contribution) | §1.4 / §4.11a: explicitly **cross-checked this pass, "no drift"** — and independently corroborates the diagnosis by noting CELM treats the same projection as a basic design requirement, i.e. the literature agrees Stage B's fix is standard, necessary practice |
| Stage B's statistics infrastructure (bootstrap CI, paired t, Wilcoxon, Holm-Bonferroni, N=10 seeds) is complete | §5 (footnote), §9.1 item 8: cites the *existing* `fedgraa/stats.py` machinery as *already being the right answer* to a reviewer objection about low-support rare classes — treats it as an asset to lead with, not a gap |

None of these require any Stage A/B document to be edited.

---

## 3. What Final_Stage_C.md adds that Stage A/B never had (new input, not a correction)

Stage A and Stage B explicitly scoped novelty claims, competitor identification, and baseline selection as **out of scope, deferred to Stage C** (`stage_b_plan.md` §9, `STAGE_B_MASTER_REPORT.md` §22.3, `STAGE_A_MASTER_REPORT.md` §12.3, verbatim). Final_Stage_C.md is that deferred work. It supplies several things that genuinely did not exist anywhere in the tree before:

1. **The actual competitor landscape.** None of FedPGO, CELM, FedAWARE, NIDS-FGPA, Jiang et al., or AdaAggRL appear anywhere in Stage A or Stage B — those documents only knew the vaguer, older short-list (FedAPA, FedAWA, FedMADE, SBFedNIDS, DA-PFL) from an earlier Stage C pass. Final_Stage_C.md's §4.7 and §9.1 item 2 supply a much longer, priority-ordered baseline list headed by **FedPGO** (same dataset, same design space — "the single most consequential paper in this corpus"), which supersedes the old "AFL/q-FedAvg/FedAdp are must-have" framing in `stage_a_plan.md`/`stage_b_plan.md` with a materially different, more specific priority order.
2. **A concrete dataset upgrade path (§5, "Option D"): NF-UQ-NIDS-v2 / NetFlow-V3.** Stage B's own remaining-work list (`STAGE_B_MASTER_REPORT.md` §22.3) only ever said "a third dataset" as a placeholder with nothing named. Final_Stage_C.md's Option D is a specific, scoped recommendation (real rather than simulated non-IID structure, no algorithmic change) that didn't exist in any prior Stage A/B document.
3. **Adversarial robustness as a required section (§4.11b, §9.1 item 3).** This topic — SecFedNIDS, WeiDetect, AutoAdapt, the "FedGRAA's premise structurally conflicts with poisoning-defense premises" argument — appears **nowhere** in Stage A or Stage B, not even in their own "deferred to Stage C" lists. This is a genuinely new scope item, not a refinement of something already planned.
4. **A citable scaffold for hardening Stage B's convergence proof (§4.11a).** Stage B already has a corrected proposition (feasibility by construction via projection; stationarity, not global-optimum, convergence) but flags three "rigor gaps" as open: exact β, a rigorous treatment of `min(L_h, L_m)`'s sub-differential, and an explicit rate constant. Final_Stage_C.md's FedAAW/Parsa-et-al./FedAWARE citations are new tooling for closing exactly those flagged gaps — they don't identify a new problem, they hand Stage B a way to finish one it already named.
5. **A previously-unremarked fact about the existing codebase (§7.3, last bullet):** `DatasetSpec.rare_client_ids()` "already has a data-driven, non-oracle fallback path that current reproduction runs simply never exercise." This is worth flagging to whoever implements FedGRAA-v2 — part of the privacy fix may already be sitting unused in code Stage A wrote.
6. **The final, submission-ready Related Work paragraph and citation list (§8.1-8.3).** Neither `draft1.md` (the working draft) nor `PAKDD_Final_Version_Submit_v3.md` (the frozen original) cites any of FedPGO/CELM/FedAWARE/NIDS-FGPA/Jiang et al./AdaAggRL/SecFedNIDS (confirmed by direct grep of both files) — expected, since Related Work hasn't been rewritten yet, not a defect. **When it is, the rewrite belongs in `draft1.md`, not `v3.md`**, per the project's established workflow (§1 above).

None of items 1-6 require re-opening Stage A or Stage B's completed work. They are exactly what those stages' own documents said they were waiting on: the still-open "which Stage C angle, which baselines, which datasets" decision that `STAGE_DIRECTION_ASSESSMENT.md` explicitly flagged as blocked on a mentor conversation that, per that document, had not yet happened.

---

## 4. Net effect on the pending mentor decision

`STAGE_DIRECTION_ASSESSMENT.md` framed the open choice as A vs. B vs. C vs. "B then C" (Stage B stands alone / few-shot hybrid / privacy redesign / sequenced). Final_Stage_C.md's §10 sharpens that same trade-off rather than replacing it:

- It reaffirms the privacy redesign (B) as the more implementation-ready angle — consistent with what `STAGE_B_MASTER_REPORT.md` §22.2 already cited from an earlier Stage C pass.
- It changes *why*: not "less crowded" (that rationale is now explicitly retired — CELM/SpectralFed occupy the data-free version of the same idea) but "a comparative experiment with a publishable result either way" (exact-gradient-at-6×-cost vs. approximate-at-near-zero-cost).
- It adds Option D (dataset migration) as a recommended addition that composes with either A or B, not a competing fifth option.

This narrows the decision space the mentor conversation needs to resolve; it doesn't remove the need for that conversation, which — per Stage A/B's own documents — is still the actual blocker on starting Stage C.

---

## 5. Action items, prioritized

1. ~~Fix ref [12] in the manuscript~~ — **already done**, in `draft1.md` (2026-07-26, predates Final_Stage_C.md). Do **not** apply this fix to `PAKDD_Final_Version_Submit_v3.md` — that file is intentionally kept as the untouched original; corrections belong in `draft1.md` only.
2. **Annotate or remove `FedSNDTrainer` in `fedgraa/baselines.py`** before it is ever run head-to-head — it reproduces a fabricated citation, and the paper-side fix (item 1) hasn't propagated to this code file yet.
3. **Update the Stage C baseline-to-do lists** in `stage_a_plan.md`/`stage_b_plan.md`/`STAGE_B_REMAINING_TASKS.md`/`STAGE_A_MASTER_REPORT.md` §12.3/§14.3 — flag FedSND as fabricated where it's listed, and reflect Final_Stage_C.md §9.1 item 2's much more specific priority ordering (FedPGO first) rather than the older, vaguer AFL/q-FedAvg/FedAdp framing.
4. ~~Bring §10's sharpened B+D recommendation, plus §6.1's venue-quality finding, to the still-pending mentor conversation~~ — **✅ resolved, 2026-07-28; corrected twice later the same day.** The author decided directly, without the external mentor meeting this item assumed: implement `Final_Stage_C.md` §10's **Option C — "both, sequenced."** B (privacy redesign) and D (dataset migration) are the active scope now and are substantially implemented (`REMAINING_COMPUTE_WORK.md` §2–§7); A (the few-shot/prototype hybrid) is a committed **follow-on**, sequenced after B+D, not started and not scheduled to start this pass (`REMAINING_COMPUTE_WORK.md` §3a). The T3/T5-vs-external-compute cost trade-off (B+D's own baselines) remains deferred until real frontier numbers are in hand; A's own baseline grid is deferred further still, to A's own later phase. §6.1's venue-quality finding (PAKDD = CORE B) was not separately re-litigated by this decision; nothing in it argued against proceeding.
5. No edits needed to `fedgraa_convergence_defect.md`, the privacy-redesign doc, `draft1.md`, or any completed Stage A/B experimental results — all corroborated as-is.
