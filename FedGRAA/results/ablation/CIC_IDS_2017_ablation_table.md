# CIC-IDS-2017 — Stage B ablation table

Rounds = 60. Metric of record = rare-class accuracy (mean per-class recall over the protected classes), reported as mean [95% bootstrap CI] over the per-seed values persisted in this same JSON. All-class accuracy and composite-loss blowup are shown as secondary/diagnostic columns per cell.

**Blowup is reported as median (and max), not mean**, because it is a heavy-tailed ratio rather than a bounded score: a single diverging seed can exceed the other nine by six orders of magnitude, making a mean describe no run that actually occurred. Per-seed blowup arrays and the mean are both retained in the JSON.

Significance markers on a reference-arm cell denote the paired, Holm-Bonferroni-corrected comparison of **both** vs. that arm at the same γ (`* p_holm<.05  ** p_holm<.01  *** p_holm<.001`); exact p-values and effect sizes are in the JSON's `comparisons` block, never only a star here.

| Arm | γ=0.0 | γ=0.5 | γ=1.0 |
|---|---|---|---|
| FedAvg (baseline) | 0.0% [0.0, 0.0] · all 55.7% | 0.0% [0.0, 0.0] · all 55.7% | 0.0% [0.0, 0.0] · all 55.7% |
| Static Reparam. (baseline) | 84.4% [77.2, 91.4] · all 47.7% | 84.4% [77.2, 91.4] · all 47.7% | 84.4% [77.2, 91.4] · all 47.7% |
| Literal (η′=0.1, no proj.) | 0.0% [0.0, 0.0]*** · all 48.9% · blowup 3.70× med, 1.86e+06× max | 0.0% [0.0, 0.0]*** · all 26.7% · blowup 2.71× med, 9.92× max | 10.0% [0.0, 30.0]*** · all 32.9% · blowup 14.19× med, 192.99× max |
| Projection-alone (η′=0.1) | 0.0% [0.0, 0.0]*** · all 59.8% · blowup 1.08× med, 1.18× max | 0.0% [0.0, 0.0]*** · all 54.8% · blowup 2.90× med, 3.91× max | 51.0% [24.2, 77.1] · all 48.4% · blowup 2.21× med, 5.60× max |
| η′-alone (η′=0.01, no proj.) | 27.5% [12.5, 43.3]** · all 71.7% · blowup 1.00× (med) | 78.5% [62.6, 92.2] · all 44.1% · blowup 1.00× (med) | 88.9% [68.9, 100.0] · all 15.2% · blowup 1.00× med, 30.02× max |
| Both (proj. + η′=0.01) — the fix | 69.7% [61.1, 78.9] · all 62.3% · blowup 1.00× (med) | 81.1% [66.4, 92.2] · all 54.2% · blowup 1.00× (med) | 89.3% [83.6, 94.6] · all 44.3% · blowup 1.00× (med) |

## Pre-registered pairwise comparisons (both vs. reference arm)

The **all-acc** column is not decoration. A tie on the metric of record can hide a large difference on overall accuracy, and on this data it does — an arm that matches `both` on rare-class accuracy while collapsing all-class accuracy has not matched `both`. Both metric families are tested on the same seeds and corrected within their own family; rows where the two disagree are flagged **(diverges)**.

**γ=0.0**

| vs. | rare-acc p (Holm) | rare-acc Cohen's d | all-acc p (Holm) | all-acc Cohen's d | agreement |
|---|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 7.968e-07 | 4.56 | 0.1424 | 0.647 | **(diverges)** differs on rare, ties on all-acc |
| Projection-alone (η′=0.1) | 7.968e-07 | 4.56 | 0.1459 | 0.503 | **(diverges)** differs on rare, ties on all-acc |
| η′-alone (η′=0.01, no proj.) | 0.007751 | 1.2 | 0.05211 | -0.92 | **(diverges)** differs on rare, ties on all-acc |
| FedAvg (baseline) | 7.968e-07 | 4.56 | 0.01776 | 1.19 | consistent |
| Static Reparam. (baseline) | 0.007751 | -1.22 | 1.256e-06 | 4.32 | consistent |

*(Exact uncorrected p-values from both the paired t and Wilcoxon tests, plus statistics and paired seed sets, are in the JSON's `comparisons` and `comparisons_all_acc` blocks.)*

**γ=0.5**

| vs. | rare-acc p (Holm) | rare-acc Cohen's d | all-acc p (Holm) | all-acc Cohen's d | agreement |
|---|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 6.174e-06 | 3.59 | 0.003088 | 1.57 | consistent |
| Projection-alone (η′=0.1) | 6.174e-06 | 3.59 | 0.5547 | -0.194 | **(diverges)** differs on rare, ties on all-acc |
| η′-alone (η′=0.01, no proj.) | 1 | 0.076 | 0.285 | 0.59 | consistent |
| FedAvg (baseline) | 6.174e-06 | 3.59 | 0.3784 | -0.449 | **(diverges)** differs on rare, ties on all-acc |
| Static Reparam. (baseline) | 1 | -0.194 | 5.613e-06 | 3.63 | **(diverges)** ties on rare, differs on all-acc |

*(Exact uncorrected p-values from both the paired t and Wilcoxon tests, plus statistics and paired seed sets, are in the JSON's `comparisons` and `comparisons_all_acc` blocks.)*

**γ=1.0**

| vs. | rare-acc p (Holm) | rare-acc Cohen's d | all-acc p (Holm) | all-acc Cohen's d | agreement |
|---|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 0.0001536 | 2.36 | 0.1014 | 0.577 | **(diverges)** differs on rare, ties on all-acc |
| Projection-alone (η′=0.1) | 0.06275 | 0.884 | 0.04243 | -0.881 | **(diverges)** ties on rare, differs on all-acc |
| η′-alone (η′=0.01, no proj.) | 0.9696 | 0.0124 | 0.0005624 | 1.99 | **(diverges)** ties on rare, differs on all-acc |
| FedAvg (baseline) | 1.278e-09 | 9.46 | 0.0001258 | -2.49 | consistent |
| Static Reparam. (baseline) | 0.08901 | 0.738 | 0.01625 | -1.15 | **(diverges)** ties on rare, differs on all-acc |

*(Exact uncorrected p-values from both the paired t and Wilcoxon tests, plus statistics and paired seed sets, are in the JSON's `comparisons` and `comparisons_all_acc` blocks.)*

## 2x2 factorial read (rare-class accuracy)

| γ | Projection main effect | η′ main effect | Interaction |
|---|---|---|---|
| 0.0 | +21.1pp [11.0, 31.7] | +48.6pp [41.1, 56.0] | +42.2pp [21.9, 63.3] |
| 0.5 | +1.3pp [-9.6, 10.9] | +79.8pp [70.1, 89.3] | +2.6pp [-19.2, 21.8] |
| 1.0 | +20.7pp [5.3, 35.7] | +58.6pp [41.5, 75.0] | -40.6pp [-75.8, -0.5] |
