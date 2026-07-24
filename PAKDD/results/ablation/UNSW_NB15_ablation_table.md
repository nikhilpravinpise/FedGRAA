# UNSW-NB15 — Stage B ablation table

Rounds = 60. Metric of record = rare-class accuracy (mean per-class recall over the protected classes), reported as mean [95% bootstrap CI] over the per-seed values persisted in this same JSON. All-class accuracy and composite-loss blowup are shown as secondary/diagnostic columns per cell.

**Blowup is reported as median (and max), not mean**, because it is a heavy-tailed ratio rather than a bounded score: a single diverging seed can exceed the other nine by six orders of magnitude, making a mean describe no run that actually occurred. Per-seed blowup arrays and the mean are both retained in the JSON.

Significance markers on a reference-arm cell denote the paired, Holm-Bonferroni-corrected comparison of **both** vs. that arm at the same γ (`* p_holm<.05  ** p_holm<.01  *** p_holm<.001`); exact p-values and effect sizes are in the JSON's `comparisons` block, never only a star here.

| Arm | γ=0.0 | γ=0.5 | γ=1.0 |
|---|---|---|---|
| FedAvg (baseline) | 0.0% [0.0, 0.0] · all 55.8% | 0.0% [0.0, 0.0] · all 55.8% | 0.0% [0.0, 0.0] · all 55.8% |
| Static Reparam. (baseline) | 48.3% [40.8, 56.0] · all 52.7% | 48.3% [40.8, 56.0] · all 52.7% | 48.3% [40.8, 56.0] · all 52.7% |
| Literal (η′=0.1, no proj.) | 0.1% [0.0, 0.3] · all 54.9% · blowup 5.10× med, 28.16× max | 26.9% [16.8, 36.2] · all 49.6% · blowup 1.50× med, 2.23× max | 27.6% [15.0, 39.9]** · all 29.1% · blowup 8.26× med, 46.85× max |
| Projection-alone (η′=0.1) | 6.4% [0.0, 16.0] · all 59.4% · blowup 1.00× (med) | 47.7% [35.0, 57.6] · all 53.6% · blowup 1.03× med, 1.32× max | 67.2% [59.6, 73.6] · all 51.8% · blowup 1.36× med, 2.06× max |
| η′-alone (η′=0.01, no proj.) | 3.4% [0.4, 7.8] · all 59.7% · blowup 1.00× (med) | 12.1% [5.2, 19.6]* · all 49.2% · blowup 1.00× med, 1.06× max | 36.4% [33.7, 39.8]*** · all 13.3% · blowup 1.19× med, 1.48× max |
| Both (proj. + η′=0.01) — the fix | 1.7% [0.0, 4.6] · all 58.8% · blowup 1.00× (med) | 37.1% [28.4, 45.7] · all 53.2% · blowup 1.03× med, 1.39× max | 69.2% [63.4, 73.8] · all 52.2% · blowup 1.32× med, 1.87× max |

## Pre-registered pairwise comparisons (both vs. reference arm)

**γ=0.0**

| vs. | p (paired t) | p (Wilcoxon) | Cohen's d | p (Holm) |
|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 0.3008 | 0.625 | 0.347 | 0.6061 |
| Projection-alone (η′=0.1) | 0.202 | 0.5 | -0.435 | 0.6061 |
| η′-alone (η′=0.01, no proj.) | 0.09282 | 0.25 | -0.594 | 0.3713 |
| FedAvg (baseline) | 0.2869 | 0.5 | 0.358 | 0.6061 |
| Static Reparam. (baseline) | 6.431e-07 | 0.001953 | -3.88 | 3.215e-06 |

**γ=0.5**

| vs. | p (paired t) | p (Wilcoxon) | Cohen's d | p (Holm) |
|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 0.2197 | 0.1934 | 0.417 | 0.418 |
| Projection-alone (η′=0.1) | 0.209 | 0.1602 | -0.428 | 0.418 |
| η′-alone (η′=0.01, no proj.) | 0.004739 | 0.009766 | 1.18 | 0.01895 |
| FedAvg (baseline) | 2.318e-05 | 0.001953 | 2.52 | 0.0001159 |
| Static Reparam. (baseline) | 0.1037 | 0.1055 | -0.572 | 0.311 |

**γ=1.0**

| vs. | p (paired t) | p (Wilcoxon) | Cohen's d | p (Holm) |
|---|---|---|---|---|
| Literal (η′=0.1, no proj.) | 0.000738 | 0.001953 | 1.58 | 0.002214 |
| Projection-alone (η′=0.1) | 0.5443 | 0.5566 | 0.199 | 0.5443 |
| η′-alone (η′=0.01, no proj.) | 2.647e-05 | 0.003906 | 2.47 | 0.0001059 |
| FedAvg (baseline) | 1.775e-09 | 0.001953 | 7.61 | 8.873e-09 |
| Static Reparam. (baseline) | 0.003149 | 0.001953 | 1.26 | 0.006298 |

## 2x2 factorial read (rare-class accuracy)

| γ | Projection main effect | η′ main effect | Interaction |
|---|---|---|---|
| 0.0 | +2.3pp [-1.0, 6.6] | -0.8pp [-4.5, 1.8] | -8.0pp [-17.3, -0.7] |
| 0.5 | +22.9pp [12.6, 32.5] | -12.7pp [-22.1, -1.1] | +4.3pp [-10.3, 18.1] |
| 1.0 | +36.2pp [25.3, 46.3] | +5.4pp [-1.5, 12.5] | -6.8pp [-18.9, 6.4] |
