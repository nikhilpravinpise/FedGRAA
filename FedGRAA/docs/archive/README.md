# Archived docs — superseded or historical, not current state

**Default: do not treat anything in this folder as current.** Each
superseded file below carries its own `⚠️ ARCHIVED` banner explaining what
replaced it. These are kept for historical record (git history alone
doesn't make prior reasoning easy to find), not because they're still
authoritative. For current project status, start at
[`../current/publication_roadmap.md`](../current/publication_roadmap.md) (the
single current entry point) and [`README.md`](../../README.md). (Note:
`PAKDD_STATUS_REPORT.md` at the repo root used to be the pointer here, but it
is itself superseded — see its own banner.)

**Two disclosed exceptions:** `GPU_DECISION_AND_NEXT_STEPS.md` and
`speedup_plan_v2.md` live in this folder for organizational reasons but are
**not** superseded — their content is still current and still cited
elsewhere as live ground truth. Each carries its own `ℹ️` banner saying so
explicitly, rather than the standard `⚠️ ARCHIVED` one.

| File | What it was | Superseded by |
|---|---|---|
| `plan.md` | Original 2026-06-28 build plan (hand-written NumPy MLP) | `strict_reproduction_report.md` (Keras-only rewrite) |
| `reproduction_gaps.md` | 2026-06-30 gap-fix tracker against the NumPy codebase. **Its claimed simplex-projection fix was later deliberately reverted** — current code has no projection | `../audit/REPRODUCTION_AUDIT.md`, `../current/fedgraa_divergence_diagnosis.md` |
| `full_reproduction_gap_analysis.md` | 2026-06-30 independent gap audit (NumPy-era) | `../audit/REPRODUCTION_AUDIT.md` |
| `paper_vs_code_line_by_line_gap_analysis.md` | 2026-07-09 line-by-line audit against a codebase state with since-removed ambiguity toggles | `../audit/REPRODUCTION_AUDIT.md` |
| `strict_reproduction_report.md` | Post-Keras-rewrite architecture snapshot (2026-06-28) — itself later superseded as the codebase kept moving | `../audit/REPRODUCTION_AUDIT.md`, `../current/PAPER_AMBIGUITY_REPORT.md` (current architecture/ambiguity record) |
| `speedup_plan.md` | v1 speedup plan; some of its own measurements were later found wrong | `speedup_plan_v2.md` |
| `GPU_DECISION_AND_NEXT_STEPS.md` | GPU-vs-CPU benchmark and decision record (GPU measured 25% slower at this model size) | Not superseded — still the live, current explanation for "why CPU-only" cited by the root `README.md`; catalogued here because it lives in this folder, not because its content is stale |
| `timing_sweep_analysis.md` | Pre-optimization diagnosis of why the sweep was slow | `speedup_plan_v2.md` (fixed the bottlenecks it names) |
| `v1_regression_root_cause_analysis.md` | Root-cause of a NaN-collapse regression from the XLA speedup (2026-07-21) | Fixed same day (`2bb548a`); current status in `../../PAKDD_STATUS_REPORT.md` |
| `v1_regression_fix_verification.md` | Verification that the above fix worked | Same as above |
| `v2_completion.md` | 2026-07-21 "phases shipped" completion report; its GPU section oversold GPU before the benchmark existed | `speedup_plan_v2.md`, `GPU_DECISION_AND_NEXT_STEPS.md` (GPU); `../../PAKDD_STATUS_REPORT.md` (overall status) |

`speedup_plan_v2.md` (also in this folder) is the terminal document in the
speedup chain above and nothing supersedes it; it is catalogued via the
other rows' "Superseded by" column rather than getting its own row.

**Not catalogued above because it does not physically live in this
folder, despite being referenced from here by some other documents:**
[`../research_c/sagitra_report.md`](../research_c/sagitra_report.md) — raw,
unedited output of an automated literature-survey tool (~400 entries),
explicitly non-citable per its own banner and superseded by
[`../research_c/privacy_compliant_fedgraa_redesign.md`](../research_c/privacy_compliant_fedgraa_redesign.md)
(every paper it surfaced was independently re-verified and folded in). That
redesign document is itself a **live, current** design reference — several
`docs/current/` and `docs/research_c/` documents cite it as still
authoritative for its unreplaced sections — not an archived artifact; only
`sagitra_report.md`'s own raw output is superseded. (A prior pass through
this table pointed at a nonexistent `../current/privacy_compliant_fedgraa_
redesign.md` path and implied `sagitra_report.md` lived in this folder;
both are corrected here.)

If you're looking for the **current** equivalent of any of these, the doc's
own banner names it — follow that link rather than reading the archived
content as fact.
