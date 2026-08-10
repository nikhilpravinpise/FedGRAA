# Paper drafts — the manuscript's lineage

Four `.md` files (plus the `.tex`/`.pdf` pair for the two submission-ready
versions), in a straight line from first draft to corrected resubmission.
Each file carries its own banner making its status explicit — this table
just puts the chain in one place instead of requiring you to open each file
to find it.

| Order | File | What it is | Status |
|---|---|---|---|
| 1 | [`draft1.md`](draft1.md) | Earliest full draft of the FedGRAA manuscript. Its abstract was later hand-edited to withdraw the original "outperforms FedAvg by 55%" claim, but Section 5's result tables were **not** updated to match — abstract and body disagree with each other in this file. | ⚠️ Superseded, do not cite either its abstract or its body numbers |
| 2 | [`PAKDD_Final_Version_Submit_v3.md`](PAKDD_Final_Version_Submit_v3.md) / [`.pdf`](PAKDD_Final_Version_Submit_v3.pdf) | The frozen, anonymized submission draft actually sent out (88%/55%-over-FedAvg claims). | ⚠️ Superseded — rejected, see row 3 |
| 3 | [`FedGRAA_Simulated_Peer_Review.md`](FedGRAA_Simulated_Peer_Review.md) | A 5-agent simulated peer-review panel's report on v3 — the independent critique that reached the same conclusion this project's own audits did, and named the specific revisions required. | Resolved — every finding addressed in row 4 |
| 4 | [`FedGRAA_v2_PAKDD_Submission.md`](FedGRAA_v2_PAKDD_Submission.md) / [`.tex`](FedGRAA_v2_PAKDD_Submission.tex) / [`.pdf`](FedGRAA_v2_PAKDD_Submission.pdf) | The corrected resubmission: simplex projection, PL-based convergence argument, 14-baseline comparison, privacy-compliant redesign, T7 adversarial-robustness results. | ✅ **Current** — the one file where narrative and numbers agree |

**If you only read one of these, read #4.** #1-#3 exist so the correction
trail is auditable, not because any number in them is still true.

One open question this lineage doesn't resolve on its own: the *original*
target manuscript's publication status (never submitted vs. under review vs.
withdrawn) hasn't been established — see
[`../audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md`](../audit/RESEARCH_DIRECTION_AUDIT_2026-07-25.md)
and [`../current/STAGE_B_MASTER_REPORT.md`](../current/STAGE_B_MASTER_REPORT.md) §21.

Back to [the doc map](../README.md).
