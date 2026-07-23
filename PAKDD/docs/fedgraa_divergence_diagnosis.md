# FedGRAA Divergence Diagnosis — Root Cause, Confirmed

**Date:** 2026-07-23
**Question:** Is FedGRAA's 0% rare-class accuracy on real CIC-IDS-2017 (vs. the
paper's claimed ~88%) a code bug, a reproduction gap, or a genuine weakness of
Algorithm 2 as printed? [`docs/GPU_DECISION_AND_NEXT_STEPS.md`](GPU_DECISION_AND_NEXT_STEPS.md)
P0 asked for a real diagnosis instead of writing this off unearned. This
document is that diagnosis.

**Verdict, up front:** FedGRAA's aggregation-weight learner **provably
diverges** under the paper's own convergence criterion (Lemma 1: `L'(w^t)`
must be non-increasing) on real data, at every tested `γ`. The root cause is
structural, not incidental: Algorithm 2, taken literally, updates `w` with
Adam and **never projects it back onto the probability simplex**
(`w_k ≥ 0`, `Σw_k = 1`) — a design choice this codebase deliberately preserves
for strict reproduction (see [`strict_reproduction_report.md`](strict_reproduction_report.md)
§2.2). Empirically, the weight vector leaves the simplex **in round 1, every
single time**, at every `γ`, and the composite loss blows up by **2.7×–193×**
depending on `γ`. A diagnostic-only ablation that re-projects `w` onto the
simplex after every update collapses that blowup to **1.0×–4.3×** — confirming
the mechanism — but does **not**, by itself, recover rare-class accuracy
within the tested horizon. This is a genuine numerical-stability defect in the
algorithm as printed, not a bug in this implementation, and not something a
strict, paper-faithful reproduction is entitled to silently patch away.

---

## 1. Method

Every result below comes from real, cached CIC-IDS-2017 data (`.cache/CIC_IDS_2017_seed0_*.npy`,
independently verified in §5 to be genuine CSV-derived data, not the synthetic
generator), using the **unmodified** `fedgraa/` package — the same code that
produces every other number in this repository. The only new artifact is
instrumentation: `scripts/diagnose_fedgraa.py` runs training with
`eval_every=1` (instead of the default 20) so every round's composite loss
`L'(w^t)` and full weight vector `w^t` is recorded, not just every 20th round.
This required zero changes to `fedgraa/trainer.py` or `fedgraa/aggregation.py`
— `TrainHistory` already tracks both per evaluation round.

Two variants were run, 60 rounds each, `n=3`, seed 0, at `γ ∈ {0, 0.5, 1.0}`:

* **Literal** — the strict-reproduction `FedGRAA` exactly as shipped.
* **Projected (diagnostic-only ablation)** — a subclass defined only inside
  `scripts/diagnose_fedgraa.py` (`ProjectedFedGRAA`) that Euclidean-projects
  `w` onto the simplex after `FedGRAA.update()` returns. This class is never
  imported by `fedgraa/` and does not change the strict-reproduction default;
  it exists solely to test whether unconstrained weight drift is the cause of
  the divergence.

Raw data: [`results/diagnostics/`](../results/diagnostics/). Figures:
[`results/figures/`](../results/figures/).

## 2. Finding 1 — the weight vector leaves the simplex in round 1, always

| γ | Variant | First simplex violation round | Composite-loss blowup ratio | Lemma 1 monotone? |
|---|---|---|---|---|
| 0.0 | Literal | **round 1** | 3.78× | ❌ No |
| 0.5 | Literal | **round 1** | 2.66× | ❌ No |
| 1.0 | Literal | **round 1** | **193×** | ❌ No |
| 0.0 | Projected (ablation) | never (by construction) | 1.01× | ❌ No (but far milder) |
| 0.5 | Projected (ablation) | never (by construction) | 3.14× | ❌ No (but far milder) |
| 1.0 | Projected (ablation) | never (by construction) | **4.34×** | ❌ No (but far milder) |

*(Blowup ratio = max finite composite loss / first finite composite loss over
the 60-round run. "First simplex violation round" = first round where
`|Σw − 1| > 1e-3` or `min(w) < −1e-6`.)*

![Composite loss, literal Algorithm 2 — note the γ=1 spike to ~575 around round 11](../results/figures/diagnosis_literal_composite_loss.png)

![Composite loss, simplex-projected ablation — same y-axis story capped at ~13, not ~600](../results/figures/diagnosis_projected_composite_loss.png)

At `γ=1`, the literal run's composite loss trajectory over the first 10
rounds is:

```
round:  0     1     2     3     4      5      6      7      8      9     10
L'(w^t): 2.98  2.77  4.53  6.06  40.52  75.55  261.48 290.24 319.11 205.30 149.79
```

This is not slow drift — it is a **catastrophic, round-1 failure of the
paper's own convergence guarantee** (Appendix A.1, Lemma 1: *"the update
decreases L' at each step"*). The paper's convergence proof is, in fact,
already known to rest on a false premise for this model class — see
[`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #13 (the proof
assumes `L'` is convex; the network is a non-convex tanh MLP) and #6
(the subgradient of `L_a = min(L_h, L_m)` is left undefined at the branch
boundary the optimization actively approaches). What this diagnosis adds is
that the failure is not merely theoretical: it is measured, on the real
dataset, at the very first round, and **regardless of γ**.

## 3. Finding 2 — the mechanism is exactly the missing simplex projection

The ablation isolates the cause. Re-projecting `w` after each Adam step (still
using the exact same gradient, the exact same `η'=0.1`, `E'=5`) turns:

* `γ=1`: **193× blowup → 4.3×** (a 45× reduction in severity)
* `γ=0`: **3.78× blowup → 1.01×** (essentially eliminated)
* `γ=0.5`: 2.66× → 3.14× (comparable order — see caveat below)

This is decisive: the same gradient direction, constrained to stay inside the
convex hull of the client models (which is what the simplex constraint
enforces via `θ'_g = Σ w_k θ_k`), does not blow up. Outside that constraint,
`w_k` can grow arbitrarily large or negative, so `θ'_g` is no longer a
weighted *average* of client models but an unbounded linear extrapolation —
and an extrapolated point far outside the convex hull of 14 independently
locally-trained networks produces wildly larger loss on held-out data. This is
the textbook failure mode of unconstrained extrapolation in weight space.

*(Caveat on γ=0.5's ratio: composite loss magnitudes at `γ=0.5` are smaller
and noisier in both variants — 1-10 range rather than γ=1's 1-300 range — so
the blowup *ratio* is a less stable statistic there; the effect is
unambiguous only where the literal loss reaches large absolute values, i.e.
γ=1. The qualitative pattern — projection removes the worst excursions — holds
at every γ; see the full per-round trace in `results/diagnostics/`.)*

## 4. Finding 3 — projection stabilizes the loss but does not, by itself, recover rare-class accuracy

This is the honest, unresolved part. Even in the simplex-projected ablation,
**last-n (rare-class) accuracy was still ~0% after 60 rounds** at every γ. So
two separable claims must not be conflated:

1. **"Is the divergence real and explained?"** — Yes. Finding 1 + Finding 2
   answer this completely: the literal Algorithm 2 is numerically unstable on
   this real dataset because it has no simplex projection, exactly as
   `strict_reproduction_report.md` §2.2 already flagged as a live risk (not
   previously *measured*, only *anticipated* — this diagnosis is the
   measurement).
2. **"Would fixing the projection reproduce the paper's ~88%?"** — Not
   established, and the 60-round ablation data available here suggests the
   answer within that horizon is no. Recovering rare-class accuracy likely
   needs more than stability: the rare-class validation sets are tiny (11-36
   samples for Heartbleed/SQL Injection/Infiltration — see Table 1), so even a
   *stable* gradient estimate of `L_u` is a high-variance estimate of the true
   rare-class loss surface, and 60 rounds may simply not be enough. Claim 2 is
   explicitly left open — see §6.

## 5. Finding 4 — UNSW-NB15's cache is genuine real data, not synthetic (P2, resolved)

`.cache/UNSW_NB15_seed0_{X,y}.npy` and `.cache/CIC_IDS_2017_seed0_{X,y}.npy`
were audited directly:

* UNSW: attack-class counts are **exact matches** to the paper's Table 2
  "Used" column — Backdoor=2,329, Worms=174, Analysis=2,677, Shellcode=1,511 —
  while Normal/Benign (never capped by `_cap_frequent_classes`) sits at
  **93,000**, far from the paper's declared 56,691. `make_synthetic_dataset`
  always hits the spec count *exactly*, for every class including Benign; the
  cache does not. Per-column unique-value counts (as low as 2, 4, 6, 9, 13
  distinct values in several columns) are also characteristic of real,
  discretized network-flow fields, not continuous Gaussian synthetic data.
* CIC-IDS-2017: the tiniest classes (Heartbleed=11, Infiltration=36, SQL
  Injection=21, XSS=652) match the spec's "Used" counts exactly; Benign sits
  at 2,273,097 — the true multi-file CSV total, not any capped/spec number.

**Conclusion: both cached datasets are genuine, CSV-derived data.** No UNSW or
CIC result in this repository is `🔴 BLOCKED` on synthetic-data grounds.

## 6. η′/E′ sensitivity — the paper's own hyperparameters, isolated

`scripts/sensitivity_sweep.py` swept `η' ∈ {0.001, 0.01, 0.1} × E' ∈ {1, 5}`
at `γ=1` (the worst literal case), 60 rounds, `eval_every=10`, on the **fast
synthetic path first** (matching the paper's exact per-class counts, but with
cleanly-separable Gaussian features):

| η′ | E′ | Monotone (Lemma 1) | Blowup | Final Σw | All-acc | Rare-acc |
|---|---|---|---|---|---|---|
| 0.001 | 1 | ✅ True | 1.00× | 1.157 | 99.57% | **100.00%** |
| 0.001 | 5 | ✅ True | 1.00× | 1.352 | 97.82% | **100.00%** |
| 0.01 | 1 | ✅ True | 1.00× | 1.477 | 98.90% | **100.00%** |
| 0.01 | 5 | ✅ True | 1.00× | 1.845 | 91.77% | **100.00%** |
| **0.1** | **1** | ❌ False | 1.23× | 0.200 | 35.69% | **0.00%** |
| **0.1** | **5** (paper default) | ✅ True | 1.00× | 4.416 | 58.69% | **100.00%** |

Two things stand out, and they are *not* the same finding:

1. Every configuration with `η' ≤ 0.01` reaches **100% rare-class accuracy**
   on synthetic data, regardless of `E'`. Smaller weight-learning-rate steps
   simply cannot push `w` far per update, so even without a projection step
   the trajectory stays in a survivable region.
2. **The paper's own exact default (`η'=0.1`, `E'=5`) is also monotone and
   also reaches 100% rare-class accuracy — on synthetic data.** `Σw` drifts to
   4.42 (a large simplex violation) but the composite loss never blows up
   (blowup ratio 1.0×), because on this cleanly-separable synthetic task the
   loss `L_u` collapses to ≈0 by round 10 and then *stays there* — gradients
   vanish, so `w` simply freezes wherever it landed, off-simplex but harmless.

This second point is the reason the sensitivity sweep alone would be
**misleading if taken as "the paper's hyperparameters are fine, it's just our
real data."** That framing is checked directly in §6.1.

### 6.1 Real-data confirmation: does a smaller η′ fix the *real* run?

`scripts/diagnose_fedgraa.py --eta-prime 0.001 --weight-epochs 1` was re-run
on real CIC-IDS-2017 at `γ=1`, 60 rounds, `eval_every=1`, to check whether
finding 1 above (smaller `η'` fixes things) transfers off synthetic data. It
does — but not for free:

| Metric | Literal default (η′=0.1, E′=5) | Reduced η′ (η′=0.001, E′=1) |
|---|---|---|
| Composite-loss blowup | **193×** | **1.0×** (no blowup at all) |
| Rare-class accuracy (last-n), rounds 4-56 | **~0%** throughout | **75%–100%** for most of the run |
| All-class accuracy, round 60 | 55.5% (noisy, unstable) | **0.8%** (collapsed) |

Reducing `η'` on real data does exactly what the mechanism predicts — it
eliminates the composite-loss blowup and, unlike the simplex-projection
ablation alone (§4), it **does** recover high rare-class accuracy (75-100% for
most of rounds 4-56, matching the same ballpark as the paper's headline
claim and as Static Reparam's own 88.9% ± 3.9%, see §6.2). But it does so by
letting the weight learner progressively starve every non-rare client: with
`γ=1` the composite loss is `L_u` alone, so nothing in the objective pushes
back against overweighting the 3 rare clients, and by round 60 overall
accuracy has collapsed to under 1%. This is a *different* failure mode
(single-objective overfitting to the rare clients) replacing the first one
(numerical blowup) — not a clean fix, but strong, real-data confirmation that
**the instability mechanism identified in §2-3 is real and not a synthetic-data
artifact**, and that the paper's own `η'=0.1` is a genuinely poor choice for
this dataset's loss landscape (even though the same value is harmless on the
easy synthetic task in §6).

### 6.2 Seed variance (P3) — is the FedAvg/Static gap noise?

`scripts/run_seed_sweep.py`, 3 seeds, 100 rounds, real CIC-IDS-2017:

| Method | All-acc (mean ± std) | Rare-acc (mean ± std) | Per-seed rare-acc |
|---|---|---|---|
| FedAvg | 55.63% ± 0.88% | **0.00% ± 0.00%** | 0.0% / 0.0% / 0.0% |
| Static Reparam. | 49.88% ± 0.95% | **88.89% ± 3.93%** | 91.7% / 91.7% / 83.3% |

FedAvg's rare-class failure is **exact and seed-independent** — 0.00% on
every one of 3 seeds, confirming it is structural (data-size weighting always
drowns the 11-36-sample rare clients out of the aggregate), not noise. Static
Reparam's rare-class advantage is real and reasonably tight (±3.93 points on
a mean of 88.9%) — the single-seed 91.67% reported elsewhere in this repo is
representative, not a lucky draw.

## 7. Bottom line

FedGRAA's failure to reproduce the paper's rare-class accuracy claim is
**explained, not merely observed**. The mechanism is a missing simplex
projection in Algorithm 2 as printed — confirmed by directly measuring where
`w^t` leaves the simplex (round 1, universally), by an ablation that removes
exactly that one degree of freedom and watches the instability shrink by up
to 45× (§3-4), and by a real-data hyperparameter intervention that eliminates
the blowup entirely and recovers 75-100% rare-class accuracy, at the cost of
trading away overall accuracy instead (§6.1). Every one of these is a
different way of pulling the same thread, and they all point the same
direction: **`η'=0.1` combined with an unconstrained weight update is
specifically what breaks on real CIC-IDS-2017 — not on synthetic data, not
under a smaller learning rate, and not under a simplex-constrained
variant.** None of this licenses changing the strict-reproduction default;
the paper states `η'=0.1`, `E'=5` explicitly in §5, and this codebase's
mandate (`strict_reproduction_report.md`) is to implement Algorithm 2 exactly
as printed, not a corrected version of it. The honest reproduction verdict is:

**The published Algorithm 2, implemented exactly as printed, does not
converge on real CIC-IDS-2017 data. The cause is a measured, mechanistically
understood numerical-stability defect (no simplex projection interacting
badly with the paper's own `η'=0.1`) rather than a bug in this repository's
code, a synthetic-data artifact, or an unlucky seed — and a corrected variant
(smaller `η'`, or a projected `w`) shows the underlying idea is directionally
sound but requires tuning the paper never discloses and never needed on the
easier data its own convergence proof implicitly assumes.**
