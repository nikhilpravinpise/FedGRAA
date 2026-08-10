# Independent audits — the receipts

This folder is not narrative documentation — it's the project's own record
of adversarial self-checking. Each file is an independent pass whose job was
to try to break a claim made elsewhere in the repo (a number, a
completeness claim, a fidelity claim) by re-deriving it from raw data rather
than trusting the prose that reported it. Sixteen passes, 2026-07-23 through
2026-08-03. None found the core science wrong; several found real doc drift,
reporting bugs, or overclaims, all closed the same day they were found — that
loop, not a clean scoreboard, is the point of keeping every pass instead of
just the final one.

## Chronological

| Date | File | What it checked | Verdict |
|---|---|---|---|
| 07-23 | [`full_audit_2026-07-23.md`](full_audit_2026-07-23.md) | Whole-repo: paper→code fidelity, tests, data, divergence diagnosis | Sound and faithful |
| 07-25 | [`AUDIT_REMEDIATION_PLAN_2026-07-25.md`](AUDIT_REMEDIATION_PLAN_2026-07-25.md) | Tracks R1-R12 remediation items from the two 07-25 audits below | Closed 2026-07-26 |
| 07-25 | [`RESEARCH_DIRECTION_AUDIT_2026-07-25.md`](RESEARCH_DIRECTION_AUDIT_2026-07-25.md) | Strategic-direction review (editor / devil's-advocate lens) of Stage B framing and the provenance question | Partially superseded by later Stage C work |
| 07-25 | [`STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md`](STAGE_AB_VERIFICATION_AUDIT_2026-07-25.md) | Independent re-derivation of every Stage A/B numeric claim from raw data | Zero numerical errors; doc/version-control issues only |
| 07-25 | [`STAGE_B_CORRECTNESS_AUDIT_2026-07-25.md`](STAGE_B_CORRECTNESS_AUDIT_2026-07-25.md) | Adversarial, automated re-derivation of Stage B numbers specifically | 4 reporting/summary defects found, no numeric errors |
| 07-26 | [`INDEPENDENT_FULL_AUDIT_2026-07-26.md`](INDEPENDENT_FULL_AUDIT_2026-07-26.md) | Full-repo independent pass | Science sound; flagged the paper draft's abstract numbers as contradicted by the project's own data |
| 07-27 | [`STAGE_C_IMPACT_ON_STAGE_AB_2026-07-27.md`](STAGE_C_IMPACT_ON_STAGE_AB_2026-07-27.md) | Whether Stage C's literature findings overturn any Stage A/B conclusion | No — only corroborates |
| 07-28 | [`PRE_COMPUTE_AUDIT_2026-07-28.md`](PRE_COMPUTE_AUDIT_2026-07-28.md) | Whole-repo audit immediately before launching the paid Modal compute campaign | 6 harness defects found (5 silent), all fixed same day; mechanisms re-verified sound |
| 08-02 | [`INDEPENDENT_REPO_AUDIT_2026-08-02_EXTERNAL.md`](INDEPENDENT_REPO_AUDIT_2026-08-02_EXTERNAL.md) | Fresh, skeptical external-style pass | Interrupted by a usage limit; partial coverage honestly disclosed |
| 08-02 | [`INDEPENDENT_REPO_AUDIT_2026-08-02_COMPLETE.md`](INDEPENDENT_REPO_AUDIT_2026-08-02_COMPLETE.md) | Follow-up closing the coverage gap the EXTERNAL pass left | Code/tests confirmed sound; docs had drift, now fixed |
| 08-02 | [`STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md`](STAGE_ABC_COMPLETENESS_AUDIT_2026-08-02.md) | Every completeness claim across Stages A/B/C, cross-checked against raw artifacts and gates | Docs lagged reality in places; all fixed |
| 08-03 | [`INDEPENDENT_STAGE_C_CODE_VS_PAPER_AUDIT_2026-08-03.md`](INDEPENDENT_STAGE_C_CODE_VS_PAPER_AUDIT_2026-08-03.md) | Six-agent adversarial audit of Stage C code vs. the paper draft | 2 HIGH-severity overclaims found in T7 adversarial-robustness prose, fixed |

## Topic audits (not stage-dated)

| File | What it checked | Verdict |
|---|---|---|
| [`REPRODUCTION_AUDIT.md`](REPRODUCTION_AUDIT.md) | Line-level fidelity of the code to the paper as currently implemented | 2 deviations found, both fixed same day |
| [`SPEEDUP_REPRODUCTION_AUDIT.md`](SPEEDUP_REPRODUCTION_AUDIT.md) | Whether the XLA speedup work changed algorithmic behavior | Not the speedup — a separate numerical bug caused the collapse |
| [`data_provenance_audit.md`](data_provenance_audit.md) | CIC-IDS-2017 / UNSW-NB15 raw CSVs, forensically checked against canonical sources | Genuine, complete, correctly labelled — not the cause of the divergence |
| [`nf_uq_nids_v2_provenance_audit.md`](nf_uq_nids_v2_provenance_audit.md) | NF-UQ-NIDS-v2 CSV genuineness for the Stage C dataset migration | Genuine official release; one code/layout mismatch found and fixed |

Two additional audits exist locally (mentor publication-record and
GitHub-repo provenance analysis, both dated 2026-08-03) but are deliberately
**not tracked in git** — they discuss a named third party's unpublished work
ahead of an actual conversation with them, which doesn't belong in a public
repository regardless of how the underlying provenance question eventually
resolves.

Back to [the doc map](../README.md).
