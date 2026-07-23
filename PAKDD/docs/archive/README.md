# Archived docs — superseded or historical, not current state

**Do not treat anything in this folder as current.** Each file below carries
its own `⚠️ ARCHIVED` banner explaining what superseded it. These are kept
for historical record (git history alone doesn't make prior reasoning easy to
find), not because they're still authoritative. For current project status,
start at [`PAKDD_STATUS_REPORT.md`](../../PAKDD_STATUS_REPORT.md) (repo root)
and [`README.md`](../../README.md).

| File | What it was | Superseded by |
|---|---|---|
| `plan.md` | Original 2026-06-28 build plan (hand-written NumPy MLP) | `strict_reproduction_report.md` (Keras-only rewrite) |
| `reproduction_gaps.md` | 2026-06-30 gap-fix tracker against the NumPy codebase. **Its claimed simplex-projection fix was later deliberately reverted** — current code has no projection | `../audit/REPRODUCTION_AUDIT.md`, `../current/fedgraa_divergence_diagnosis.md` |
| `full_reproduction_gap_analysis.md` | 2026-06-30 independent gap audit (NumPy-era) | `../audit/REPRODUCTION_AUDIT.md` |
| `paper_vs_code_line_by_line_gap_analysis.md` | 2026-07-09 line-by-line audit against a codebase state with since-removed ambiguity toggles | `../audit/REPRODUCTION_AUDIT.md` |
| `sagitra_report.md` | Raw, unedited output of an automated literature-survey tool (~400 entries) | `../current/privacy_compliant_fedgraa_redesign.md` (every paper re-verified and folded in) |
| `speedup_plan.md` | v1 speedup plan; some of its own measurements were later found wrong | `speedup_plan_v2.md` |
| `timing_sweep_analysis.md` | Pre-optimization diagnosis of why the sweep was slow | `speedup_plan_v2.md` (fixed the bottlenecks it names) |
| `v1_regression_root_cause_analysis.md` | Root-cause of a NaN-collapse regression from the XLA speedup (2026-07-21) | Fixed same day (`2bb548a`); current status in `../../PAKDD_STATUS_REPORT.md` |
| `v1_regression_fix_verification.md` | Verification that the above fix worked | Same as above |
| `v2_completion.md` | 2026-07-21 "phases shipped" completion report; its GPU section oversold GPU before the benchmark existed | `speedup_plan_v2.md`, `GPU_DECISION_AND_NEXT_STEPS.md` (GPU); `../../PAKDD_STATUS_REPORT.md` (overall status) |

If you're looking for the **current** equivalent of any of these, the doc's
own banner names it — follow that link rather than reading the archived
content as fact.
