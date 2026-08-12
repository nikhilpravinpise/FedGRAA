# Stage B, Deeply — The Corrected Theory and the 10-Seed Factorial

**Primary sources:** `docs/current/STAGE_B_MASTER_REPORT.md` (1,204 lines),
`docs/current/fedgraa_convergence_defect.md` (the standalone paper-section
version). Status: ✅ complete. Campaign commit `4459f53`. 142/142 tests at
close.

**What Stage B is, in one sentence:** it turns Stage A's diagnosis from
"we found a bug" into a paper section that stands on its own — formal
theory (why the paper's proof doesn't cover the bug, and what a corrected,
weaker guarantee looks like) plus a statistically rigorous ablation (which
part of the fix does the work, tested at n=10 seeds, not 3).

---

## 1. The theory — three compounding reasons the paper's proof fails

The paper's Appendix A.1 claims (Theorem 1, via Lemma 1/2, under Assumptions
1–3) that `L'(θ_g^t) → inf_θ L'(θ)` — **global** optimality. Three separate
facts, each already catalogued in the paper-ambiguity report, compound into
one argument for *why the proof fails exactly where the empirical defect
appears*:

**(i) Assumption 1 (convex, β-smooth) is false for the model the paper
actually uses.** §5 describes a multi-layer tanh MLP — a standard
non-convex loss surface. Without convexity, "weights converge to `w*`,
the loss-minimizer" loses its foundation before feasibility is even
reached.

**(ii) `L_a = min(L_h, L_m)` is non-smooth exactly where training pushes
it.** `min(f,g)` isn't differentiable where `f=g` — and the entire point of
Eq. 8 is to *balance* `L_h` against `L_m`, i.e. the optimization is driven
*toward* that boundary, not away from it by luck. At the kink, no finite
β-smoothness constant exists, so Assumption 3's `η′≤1/β` is either vacuous
(β=∞) or arbitrarily restrictive right where the algorithm intends to
spend its time.

**(iii) Even granting smoothness by fiat, Lemma 2 tacitly assumes `w∈Δ` —
an assumption nothing supplies.** This is the gap specific to the *measured*
failure. Even a hypothetically well-behaved `L'` — Lemma 1 would still only
bound `L'(w^t)` via `L'(w^{t−1})` and a gradient norm; it says nothing about
*where in ℝ^K* `w^t` lives. Assumption 2 bounds the gradient's *magnitude*,
never the iterate's *location*. A bounded-gradient assumption constrains
how fast `w` can move; it doesn't constrain where it ends up — and
`η′=0.1` × 5 inner epochs is enough displacement to leave `Δ` in round 1.

**One-sentence summary, ready to say out loud:** *"The paper's proof and its
algorithm are about two different mathematical objects — the proof reasons
about optimization over the simplex, implicitly, through its 'weighted
average' framing; the algorithm optimizes over all of ℝ^K."*

## 2. The corrected proposition — what you can actually prove

**Fixed update:** `w^t = Π_Δ(w^{t-1} - η'∂L'(w^{t-1}))`, `Π_Δ` = Euclidean
projection onto the simplex (Duchi et al. 2008).

**Claim 1 — feasibility is structural, not empirical.** `w^t∈Δ` for every
`t≥0` *by construction* — the round-1 blowup is **provably impossible**
under this rule, not merely unobserved in one run.

**Claim 2 — gradient-mapping stationarity, not global optimality.** Because
`L'` isn't convex, you cannot claim global-minimum convergence. What
*does* follow from standard projected-subgradient theory on a compact
convex set: `min_{t≤T} ‖G_{η'}(w^t)‖² = O(1/T)` — the iterates approach a
stationary/KKT point of the *constrained* problem. This is **strictly
weaker** than the paper's Theorem 1, and it's the *correct* guarantee for
the model class the paper's own §5 admits.

**Feasibility is the part of Theorem 1's failure this closes completely.
Global optimality is the part it does not restore** — and given
non-convexity, nothing short of changing the model class could restore it.

**Rigor flags — named, not hidden, per the project's own discipline:**
(a) no exact `β_eff` estimated for this specific architecture; (b) the
`min(L_h,L_m)` sub-differential treatment is a standard-practice sketch,
not a from-scratch measure-theoretic derivation; (c) the `O(1/T)` rate is
asymptotic, no worked constant. None affect the qualitative claim; all
three are exactly what a referee-grade proof pass would tighten.

## 3. The experimental design — the 2×2 factorial

At fixed `E′=5`, four arms are the corners of a factorial over two levers:

| | η′=0.1 (paper default) | η′=0.01 (the cut) |
|---|---|---|
| **projection off** | **literal** | **η′-alone** |
| **projection on** | **projection-alone** | **both** *(the fix)* |

Plus FedAvg and Static Reparam. as γ-independent reference rows.
`60` rounds, `eval_every=1`, `n=3`, **10 seeds (0–9)**, both datasets, all
three γ = **240 measured (arm,seed,γ) cells**. Pre-registered protocol
(decided *before* the 10-seed data existed): bootstrap 95% CI (B=10,000,
fixed seed), 5 pairwise comparisons per (dataset,γ) — paired *t* +
Wilcoxon + Cohen's *d*, Holm–Bonferroni corrected across the 5 — plus a
factorial main-effect/interaction read.

**Wall clock:** CIC campaign 3h51m, UNSW campaign 3h17m (concurrent, CPU
only, `--workers 3`), baselines 1h40m sequential. Total ~5h32m.

## 4. Headline numbers (rare-class accuracy, N=10 seeds, mean [95% CI])

| Arm | CIC-IDS-2017, γ=1 | UNSW-NB15, γ=1 |
|---|---|---|
| FedAvg | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] |
| Static Reparam. | 84.4% [77.2, 91.4] | 48.3% [40.8, 56.0] |
| Literal (paper default) | 10.0% [0.0, 30.0] | 27.6% [15.0, 39.9] |
| Projection-alone | 51.0% [24.2, 77.1] | **67.2%** [59.6, 73.6] |
| η′-alone | **88.9%** [68.9, 100.0] | 36.4% [33.7, 39.8] |
| **Both (the fix)** | **89.3%** [83.6, 94.6] | **69.2%** [63.4, 73.8] |

Composite-loss blowup, worst case over the *whole* campaign: literal
reaches **1.86×10⁶×** on CIC (γ=0, seed 8) and **46.85×** on UNSW (γ=1,
seed 9) — the fix never exceeds **1.02×** on CIC or **1.87×** on UNSW.

## 5. The three findings that survived statistical testing

**Finding 1 — feasibility failure is unconditional; the fix makes it
structurally impossible.** 120/120 unprojected cells: `first_simplex_
violation_round = 1`. 120/120 projected cells: never violated, `Σw` within
`1.2×10⁻⁷` of 1.0 (float32 round-off). Not "usually true" — *identically*
true. This is the pre-registered, falsifiable prediction the campaign was
built to test, and it held at full scale, exactly as the corrected theory
(§2) predicts.

**Finding 2 — which lever does the *accuracy* work is dataset-dependent,
and this overturned the project's own 3-seed-preview hypothesis.** The
n=3 preview said "neither lever alone recovers accuracy; only both do."
At n=10 that's refuted:

- **On CIC-IDS-2017, the η′ cut carries rare-class recovery.** Main effect
  +48.6 to +79.8pp at every γ, vs. projection's +1.3 to +21.1pp.
  `η′-alone` statistically ties `both` at γ≥0.5 (Holm p=1.0, 0.97).
- **On UNSW-NB15, the pattern inverts.** Projection's main effect +22.9 to
  +36.2pp at γ≥0.5; η′'s main effect is *negative or negligible* at every
  γ. `projection-alone` ties `both` at **every** γ (Holm p=0.61/0.42/0.54).
- **One of those ties is hollow, and only checking all-class accuracy
  reveals it.** At CIC γ=1, `η′-alone` matches the fix on rare-acc (88.9%
  vs 89.3%) while sitting at **15.2% all-class accuracy against the fix's
  44.3%** (Holm p=5.6×10⁻⁴, d=+1.99) — it buys rare-class parity by
  starving every frequent class. Reporting only the metric of record would
  have scored that arm as equivalent to the fix. This is the single most
  important nuance to have ready if pressed on "so does the η′ cut alone
  work" — the honest answer is "on rare-class accuracy, sometimes yes; on
  the metric that actually matters for deployment, no."
- **No arm dominates on both metrics at every γ.** At CIC γ=1,
  `projection-alone` is *significantly better* than the fix on all-class
  accuracy (48.4% vs 44.3%, Holm p=0.042). **The defensible claim: `both`
  is the only arm never catastrophically wrong on either metric, on
  either dataset, at any γ** — not a win on any single metric, a
  worst-case-profile argument.

**Finding 3 — on UNSW, the literal algorithm's real failure mode is
unpredictability, not uniform collapse.** On CIC, literal rare-acc is an
exact, seed-independent 0.0% at γ=0/0.5 (10/10 seeds). On UNSW at γ=0.5,
the *same unmodified algorithm* produces per-seed rare-acc of `[29.6, 42.8,
0.4, 43.3, 44.0, 38.5, 3.4, 31.0, 24.2, 11.8]%` — 0.4% to 44.0% on nothing
but a seed change. For a deployed IDS that's arguably the worse failure
mode, and it's only visible once seed count goes from 3 to 10.

## 6. Per-seed head-to-head (the fix vs. literal, run by run)

Stronger than a difference of means — same seed, same γ, direct comparison:

| Dataset | γ=1 blowup: fix strictly lower | γ=1 rare-acc: fix strictly higher |
|---|---|---|
| CIC-IDS-2017 | 9/10 (1 tie) | 9/10 (1 loss) |
| UNSW-NB15 | 10/10 | 10/10 |

**At γ=1 — the regime the paper's own headline claim lives in — the fix
wins 10/10 on both datasets, on both metrics.**

## 7. Why blowup is reported as a median, not a mean

Blowup is a heavy-tailed *ratio*. CIC literal at γ=0, per-seed:
`[3.78, 7.59, 29.76, 2.58, 3.62, 3.59, 2.47, 3.14, 1857580.09, 3.87]` —
**mean = 185,764×, median = 3.70×.** The mean describes no run that
actually occurred; nine of ten runs sit between 2.47× and 29.76×. Every
table in the master reports carries median (and max, when it exceeds
median by >5%) for blowup, mean for accuracy — enforced by a regression
test so it can't silently drift back.

## 8. The n=3-preview vs. n=10-campaign scorecard

Worth having ready to answer "was the full campaign worth running":

| Claim from the 3-seed preview | Status at n=10 |
|---|---|
| `w` leaves the simplex round 1, every γ, both datasets | ✅ Confirmed, strengthened (120/120, and now shown for η′-alone too) |
| The fix gives blowup 1.0× on CIC | ✅ Confirmed (29/30 exactly, 1 at 1.016×) |
| CIC literal blowup worst case ≈193× | ⚠️ Revised **upward four orders of magnitude** to 1.86×10⁶× |
| UNSW literal blowup worst case ≈7.6× | ⚠️ Revised upward to 46.85× |
| "Neither lever alone recovers accuracy; only both do" | ❌ **Refuted** — dataset-dependent, see Finding 2 |

**Two of nine preview claims were materially wrong, and the refuted one was
the headline mechanistic story.** That's the case for having run the
campaign rather than trusting the preview — good material if the mentor
asks "why did this take so much compute."

## 9. What Stage B is explicitly not

No new method, no baselines beyond FedAvg/Static (that's Stage C), no
scalability/non-IID study, no machine-checked proof with exact constants.
One framing risk was open at Stage B's close and is addressed in `04`: the
target manuscript's publication status.

## 10. The correctness audit — `audit_stage_b.py` caught real defects

Worth reading alongside `05`'s AI-integrity section, but the short version:
a *second*, independent script re-derives every number from raw JSON using
SciPy directly (not the project's own `fedgraa/stats.py`), and found six
real reporting defects before this document was written — none were
numerical errors, all were *how correct numbers were summarized*: blowup
reported as a misleading mean (fixed → median, §7 above); seed-0 worst
cases presented as the campaign's worst cases (fixed, §8 above); a run-count
overstatement (80 claimed, 71 actually fresh); a rare-accuracy-only
"statistically indistinguishable" claim that hid a 29pp all-class gap
(fixed → dual-metric reporting, Finding 2 above); and the auditor itself
crashing while trying to report one of these, because a row label
contained "η′" and UTF-8 wasn't configured — fixed same day. Full account:
`05_PAPERS_CODE_AND_AI_INTEGRITY.md` §3.
