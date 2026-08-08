# Current documents — the live project state

Everything in this folder is the up-to-date account of a stage, unless its
own banner says otherwise (a few are marked superseded-in-place below rather
than moved to [`../archive/`](../archive/), because they're still useful as
runbooks/history for the stage they cover).

**Start here:** [`publication_roadmap.md`](publication_roadmap.md) — the
single router doc for where the project stands and what comes next.

## Stage capstones

| File | What it is |
|---|---|
| [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md) | Exhaustive Stage A capstone: the reproduction methodology, data, every result/figure, and the round-1 simplex-divergence finding. |
| [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) | Exhaustive Stage B capstone: the formal convergence-defect theory, the 10-seed factorial ablation, and every significance test. |
| [`STAGE_C_MASTER_REPORT.md`](STAGE_C_MASTER_REPORT.md) | Exhaustive Stage C capstone: the FedGRAA-v2 privacy-compliant redesign, the 830-cell frontier campaign, the significance layer, and T7 adversarial-robustness results. |

## Supporting analysis (still current)

| File | What it is |
|---|---|
| [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) | A paper-only critique — 17 catalogued ambiguities, contradictions, and undefined terms in the original FedGRAA manuscript text, independent of any code. |
| [`fedgraa_divergence_diagnosis.md`](fedgraa_divergence_diagnosis.md) | The original root-cause analysis: why FedGRAA's rare-class accuracy collapses (missing simplex projection), not a data or implementation bug. |
| [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) | The standalone, paper-section-shaped writeup: states and proves the feasibility defect and the corrected, explicitly weaker convergence proposition. |

## Execution plans and historical runbooks (superseded-in-place)

These aren't wrong, they're just done — kept here rather than in
[`../archive/`](../archive/) because they double as the runbook record for
how each stage was actually executed.

| File | What it was | Status |
|---|---|---|
| [`stage_a_plan.md`](stage_a_plan.md) | Execution plan for closing Stage A reproduction gaps. | Complete — see `STAGE_A_MASTER_REPORT.md` for the outcome |
| [`stage_b_plan.md`](stage_b_plan.md) | Execution plan defining Stage B's scope (theory + ablation + statistics module). | Complete — see `STAGE_B_MASTER_REPORT.md` |
| [`STAGE_B_REMAINING_TASKS.md`](STAGE_B_REMAINING_TASKS.md) | Punch-list for the Stage B 10-seed compute campaign. | Superseded — Stage B is complete; kept as historical runbook |
| [`STAGE_DIRECTION_ASSESSMENT.md`](STAGE_DIRECTION_ASSESSMENT.md) | Tracked whether Stage A/B/C strategy was on track and recorded mentor-decision history. | Largely superseded/historical — Stage C is now complete |

Back to [the doc map](../README.md).
