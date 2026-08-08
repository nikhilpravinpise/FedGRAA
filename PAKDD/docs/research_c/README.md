# Stage C — literature foundation and FedGRAA-v2 build record

Two things live in this folder: the literature review that grounded Stage
C's novel-contribution claim, and the build/plan documents for FedGRAA-v2
itself (the privacy-compliant redesign + dataset migration + adversarial-
robustness evaluation that Stage C actually delivered).

## Current build & plan documents

| File | What it is |
|---|---|
| [`Final_Stage_C.md`](Final_Stage_C.md) | The final, consolidated literature entry point — novelty claim plus an explicit do-not-write list. Supersedes `STAGE_C_LITERATURE_MASTER.md`. |
| [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) | The FedGRAA-v2 build plan (design, theory, experiments, code, gates) — Option B+D. |
| [`FEDGRAA_V2_THEORY.md`](FEDGRAA_V2_THEORY.md) | The theory workstream (W5): the v2 convergence theorem as the generalization of Stage B's proposition. |
| [`STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md`](STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md) | Execution-ready inventory of everything left after the compute campaign (T7 adversarial robustness, manuscript finishing). |
| [`STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`](STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md) | Judgment calls resolving ambiguities in the T7 primary sources (WeiDetect, FLPoison, etc.) before implementation. |
| [`COMPUTE_INFRASTRUCTURE_2026-07-27.md`](COMPUTE_INFRASTRUCTURE_2026-07-27.md) | Cloud-platform evaluation (Azure / DigitalOcean / Camber rejected, Modal selected) for the compute campaign. |
| [`STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`](STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md) | Decision to reimplement the FedPGO baseline faithfully from its primary source rather than an earlier proxy. |
| [`FedPGO/FedPGO.md`](FedPGO/FedPGO.md) | Extracted primary-source text of the FedPGO paper (Bikila & Čapek, *Computer Networks* 2026), used for the faithful reimplementation above. |
| [`STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md`](STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md) | Dated session record: campaign verification, FedPGO result commit, a primary-source correction. |
| [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md) | Earlier remaining-work map, largely superseded by `STAGE_C_REMAINING_WORK_MASTER_PLAN_2026-08-02.md` above; kept for the Option A (few-shot hybrid) deferral record in its §3a. |

## Literature review trail (chronological)

The novelty claim wasn't taken on faith — this is the actual search/verify
history behind it, oldest first:

1. [`literature_review_and_publication_strategy.md`](literature_review_and_publication_strategy.md) — the original review positioning FedGRAA's novelty at A*/A-tier venues.
2. [`research_followup.md`](research_followup.md) — early notes (2026-06-28) on privacy-compliant alternatives to FedGRAA's oracle/validation-set assumptions.
3. [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) — the blueprint identifying FedGRAA's two privacy-violating assumptions and proposing the v2 redesign; still current for sections the master plan hasn't replaced.
4. [`sagitra_report.md`](sagitra_report.md) — raw, unedited output of an automated literature-survey tool (~400 entries). ⚠️ Not citable standalone; every finding in it was independently re-verified and folded into the redesign document above.
5. [`STAGE_C_DEEP_RESEARCH_2026-07-25.md`](STAGE_C_DEEP_RESEARCH_2026-07-25.md) — found the claimed "unoccupied" novelty intersection was actually occupied by FedPGO/CELM.
6. [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md) — targeted re-verification of four open threads (Fed-LAA, pFedCross, etc.).
7. [`STAGE_C_FASTTRACK_FINAL_PASS_2026-07-26.md`](STAGE_C_FASTTRACK_FINAL_PASS_2026-07-26.md) — final saturation pass cross-checking the (revised) novelty claim and retrieving FedPGO/CELM full text.
8. [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) — consolidated master review superseding the five docs above.
9. [`FedGRAA_Deep_Research_Compendium.md`](FedGRAA_Deep_Research_Compendium.md) — exhaustive, independently-verified expansion of every paper/venue/claim, with corrections and hallucination flags.
10. [`Final_Stage_C.md`](Final_Stage_C.md) — the terminal document; supersedes `STAGE_C_LITERATURE_MASTER.md` (listed again above as the current entry point).

Back to [the doc map](../README.md).
