# CIC-IDS-2017 — Stage B ablation table

Rounds = 60. Metric of record = rare-class accuracy (mean per-class recall over the protected classes), reported as mean [95% bootstrap CI] over the per-seed values persisted in this same JSON. All-class accuracy and composite-loss blowup are shown as secondary/diagnostic columns per cell.

Significance markers on a reference-arm cell denote the paired, Holm-Bonferroni-corrected comparison of **both** vs. that arm at the same γ (`* p_holm<.05  ** p_holm<.01  *** p_holm<.001`); exact p-values and effect sizes are in the JSON's `comparisons` block, never only a star here.

| Arm | γ=0.0 | γ=0.5 | γ=1.0 |
|---|---|---|---|
| FedAvg (baseline) | 0.0% [0.0, 0.0] · all 55.7% | 0.0% [0.0, 0.0] · all 55.7% | 0.0% [0.0, 0.0] · all 55.7% |
| Static Reparam. (baseline) | 84.4% [77.2, 91.4] · all 47.7% | 84.4% [77.2, 91.4] · all 47.7% | 84.4% [77.2, 91.4] · all 47.7% |
| Literal (η′=0.1, no proj.) | 0.0% [0.0, 0.0]*** · all 48.9% · blowup 185764.05× | 0.0% [0.0, 0.0]*** · all 26.7% · blowup 3.71× | 10.0% [0.0, 30.0]*** · all 32.9% · blowup 47.21× |
| Projection-alone (η′=0.1) | 0.0% [0.0, 0.0]*** · all 59.8% · blowup 1.10× | 0.0% [0.0, 0.0]*** · all 54.8% · blowup 2.98× | 51.0% [24.2, 77.1] · all 48.4% · blowup 2.75× |
| η′-alone (η′=0.01, no proj.) | 27.5% [12.5, 43.3]** · all 71.7% · blowup 1.00× | 78.5% [62.6, 92.2] · all 44.1% · blowup 1.00× | 88.9% [68.9, 100.0] · all 15.2% · blowup 5.48× |
| Both (proj. + η′=0.01) — the fix | 69.7% [61.1, 78.9] · all 62.3% · blowup 1.00× | 81.1% [66.4, 92.2] · all 54.2% · blowup 1.00× | 89.3% [83.6, 94.6] · all 44.3% · blowup 1.00× |

## Pre-registered pairwise comparisons (both vs. reference arm)

**γ=0.0**

| vs. | p (paired t) | p (Wilcoxon) | Cohen's d | p (Holm) |
|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 1.594e-07 | 0.001953 | 4.56 | 7.968e-07 |
| Projection-alone (η′=0.1) | 1.594e-07 | 0.001953 | 4.56 | 7.968e-07 |
| η′-alone (η′=0.01, no proj.) | 0.004197 | 0.001953 | 1.2 | 0.007751 |
| FedAvg (baseline) | 1.594e-07 | 0.001953 | 4.56 | 7.968e-07 |
| Static Reparam. (baseline) | 0.003875 | 0.01562 | -1.22 | 0.007751 |

**γ=0.5**

| vs. | p (paired t) | p (Wilcoxon) | Cohen's d | p (Holm) |
|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 1.235e-06 | 0.001953 | 3.59 | 6.174e-06 |
| Projection-alone (η′=0.1) | 1.235e-06 | 0.001953 | 3.59 | 6.174e-06 |
| η′-alone (η′=0.01, no proj.) | 0.8155 | 0.8438 | 0.076 | 1 |
| FedAvg (baseline) | 1.235e-06 | 0.001953 | 3.59 | 6.174e-06 |
| Static Reparam. (baseline) | 0.5554 | 1 | -0.194 | 1 |

**γ=1.0**

| vs. | p (paired t) | p (Wilcoxon) | Cohen's d | p (Holm) |
|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 3.84e-05 | 0.003906 | 2.36 | 0.0001536 |
| Projection-alone (η′=0.1) | 0.02092 | 0.03125 | 0.884 | 0.06275 |
| η′-alone (η′=0.01, no proj.) | 0.9696 | 0.4062 | 0.0124 | 0.9696 |
| FedAvg (baseline) | 2.556e-10 | 0.001953 | 9.46 | 1.278e-09 |
| Static Reparam. (baseline) | 0.0445 | 0.125 | 0.738 | 0.08901 |

## 2x2 factorial read (rare-class accuracy)

| γ | Projection main effect | η′ main effect | Interaction |
|---|---|---|---|
| 0.0 | +21.1pp [11.0, 31.7] | +48.6pp [41.1, 56.0] | +42.2pp [21.9, 63.3] |
| 0.5 | +1.3pp [-9.6, 10.9] | +79.8pp [70.1, 89.3] | +2.6pp [-19.2, 21.8] |
| 1.0 | +20.7pp [5.3, 35.7] | +58.6pp [41.5, 75.0] | -40.6pp [-75.8, -0.5] |
