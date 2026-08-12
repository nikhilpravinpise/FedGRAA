# Stage C, Deeply — FedGRAA-v2, the Frontier Campaign, and Adversarial Robustness

**Primary sources:** `docs/current/STAGE_C_MASTER_REPORT.md` (550 lines),
`docs/research_c/FEDGRAA_V2_THEORY.md`, `docs/research_c/
FEDGRAA_V2_MASTER_PLAN.md`, `docs/research_c/Final_Stage_C.md`,
`docs/paper/FedGRAA_v2_PAKDD_Submission.md`. Status: ✅ complete except
Option A (deliberately deferred — see §8). 563/563 tests, zero regressions
from A/B.

**What Stage C is, in one sentence:** it takes Stage B's corrected-and-fixed
FedGRAA and asks two questions no prior stage touched — *does the mechanism
require centralizing anything FL exists to avoid?* and *does the
mechanism's own design create an exploitable attack surface?* — while also
closing the baseline-comparison and statistical-rigor gaps a domain-literate
reviewer would ask about first.

---

## 1. The two privacy flaws, and the fix

**Flaw 1 — oracle knowledge of rare classes.** FedGRAA-v1 hard-codes "the
last `n` underrepresented classes" server-side; the partition itself
assumes one attack type per client. A real FL server never observes
client-side label distributions.

**Flaw 2 — a server-held, fully-labelled validation set `D_V`.** Algorithm
2 evaluates its composite loss on a centrally-pooled validation set —
structurally identical to the "root dataset" FLTrust requires, which
FedTruth calls impractical and CoAst calls "difficult to obtain in
practical FL scenarios."

**The fix, concretely:**
- **Rarity ranking** → an online-inferred, SecAgg-summed per-round
  class-frequency histogram built from label counts every client already
  holds locally. `bottom_n_rare_classes` derives the rarest `n` from this
  sum alone — confirmed by direct code reading (`fedgraa/partition.py`'s
  `local_label_histogram`/`bottom_n_rare_classes`).
- **`D_V` replacement** → a **federated backward pass**. The server
  broadcasts the candidate aggregate `θ'_g`; each client returns only
  `∇_P L_k'(θ'_g)` on its own held-out split; the server sums these
  (`two_phase_composite_grad`) to reconstruct `∇_P L' = Σ_k ∇_P L_k'(θ'_g)`
  — **algebraically identical** to what a server-held `D_V` would compute,
  not an approximation.
- **Zero change to Eq. 7–10** — this is a methodology patch to how the
  composite loss's *inputs* are obtained, not a redesign of the objective.
- **Measured exactness: `~2×10⁻⁷` relative error** vs. an oracle-pooled
  reference (`scripts/verify_v2_exactness.py` + a dedicated test).

**Threat model, stated explicitly:** honest-but-curious server, SecAgg's
standard collusion threshold. A fully malicious server and Byzantine
clients beyond the poisoning threat evaluated in §5 are out of scope. The
plaintext client-to-server channel remains exposed to gradient-inversion
attacks exactly as in any plaintext-parameter-exchange FL system — named
as a limitation, not hidden.

## 2. The corrected theorem for the inexact case

Stage B's proposition assumed an *exact* gradient. FedGRAA-v2's gradient is
SecAgg-mediated and therefore inexact in a bounded sense. Extends Stage
B's result by exactly one step:

> **Theorem (ζ-inexact projected subgradient descent on Δ).** Let `L'`
> have bounded subgradients on `Δ` and satisfy the Polyak–Łojasiewicz
> condition. Let the server's gradient estimate satisfy
> `E‖∇̂_P L' − ∇_P L'‖² ≤ ζ²`. Then
> `w^t = Π_Δ(w^{t-1} − η'∇̂_P L'(w^{t-1}))`, `η'≤1/β_eff`, satisfies
> `min_{t≤T} ‖G_{η'}(w^t)‖² = O(1/T) + O(ζ²)`, and `w^t∈Δ` for every `t≥0`
> by construction.

Reduces exactly to Stage B's result at `ζ=0`. The `O(ζ²)` term decomposes
into four named, independently measured sources:

| Source | What it is | Measured |
|---|---|---|
| `ζ_secagg` | Float32 summation-order noise across ~4×10⁴ params, **not** cryptographic noise | `<10⁻⁶` relative error |
| `ζ_dp` | DP noise on the rarity histogram only, never the gradient channel — effect is indirect (a discrete rare-set-flip risk, not a continuous perturbation) | `>10⁻²` isolated, vs. `<10⁻⁵` secagg-only baseline |
| `ζ_stale(m)` | Gap from refreshing the gradient only every `m` inner epochs — the frontier's own knob | Non-decreasing in `m`, no closed form — deferred to the empirical campaign |
| `ζ_dropout` | Gap from a client dropping mid-round, shrinking the effective population | A sampling-variance term |

**Explicitly not claimed** (rigor flags, same discipline as Stage B): an
exact `β_eff`; a formal measure-theoretic sub-differential treatment; a
worked `O(1/T)` rate constant; whether PL actually holds empirically for
`L'` at the trained parameters. There is deliberately **no `ζ_protocol`
term** — licensed by two hard gates (T-4a/T-4b) passing green, not asserted
by omission.

## 3. The campaign — scale and cost

**830 main-frontier cells** (24 arms × 2 datasets × 10 seeds, plus T2's
γ-interaction sweep, T4's robustness-controls sweep, T6's real cross-silo
benchmark) + **90 T7 adversarial-robustness cells** + **40 T7G γ-sweep
addendum** = **960 cells total**, all independently verified against their
own raw JSON. All on **Modal** (serverless cloud Python), real spend.

**Why Modal:** Azure/DigitalOcean both hit card-verification friction;
Camber's execution model is fixed HPC-engine job types with no generic
"run arbitrary pinned-pip Python" path; Modal was purpose-built for exactly
that, with pre-existing credit and no card friction.
(`docs/research_c/COMPUTE_INFRASTRUCTURE_2026-07-27.md`)

**Cost, measured not estimated:** T1–T3 ≈ $13.47 / 111.9 core-hours;
full T1–T7 ≈ $26–29 / ~230 core-hours — under 10.5% of the available
credit. CPU-only throughout (`TF_NUM_INTRAOP_THREADS=1` is hard-coded, so
more cores don't even help — measured: 1 core 681.5s, 2 cores 790.1s
*slower*, 4 cores 617.2s marginal — "no monotonic speedup," use 1 core per
container). A genuinely useful infra finding: `ProcessPoolExecutor` hangs
indefinitely on Linux (20+ min, no crash) because the code assumed
Windows' `spawn` default; Linux defaults to `fork`, which deadlocks after
TensorFlow spins up its own thread pools — fixed by forcing
`multiprocessing.set_start_method("spawn", force=True)`.

**The 19-baseline roster:**
- **14 domain-specific:** CELM, SpectralFed, CoAst (data-free/validation-free
  contribution estimation); FLTrust, RAGA, FedTruth (trust-based robust
  aggregation); FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE,
  AdaAggRL, FedPGO (NIDS-specific adaptive weighting).
- **5 classic FL optimizers:** FedAvg, FedProx, FedOpt, SCAFFOLD, FedNova —
  SCAFFOLD and FedNova built from scratch, faithful to their primary
  sources (Karimireddy et al. ICML 2020; Wang et al. NeurIPS 2020), with a
  regression test proving FedNova's formula collapses to plain FedAvg
  exactly when every client's step count is equal.

## 4. Headline results

| Metric | CIC-IDS-2017 | UNSW-NB15 |
|---|---|---|
| FedGRAA-v2 rare-class recall (best, m=3) | **93.47%** | **69.66%** |
| Best external baseline (FedAWARE) | 71.81% | 32.65% |
| Best classic optimizer (SCAFFOLD) | 31.67% | 30.34% |
| Significant vs. 14 baselines (Holm-corrected) | 55/56 | 56/56 |

**Every classic optimizer except SCAFFOLD scores at or near 0%** on both
datasets (FedAvg 0.00%/0.00%, FedProx 0.00%/1.33%, FedOpt 0.00%/0.44%,
FedNova 0.00%/4.62%). SCAFFOLD's control-variate correction gets it to
31.67%/30.34% — still well below every FedGRAA-v2 refresh interval.
**Conclusion worth saying plainly: size-proportional and drift-corrected
weighting alone, without an explicit rare-class objective, does not solve
this problem.**

**Significance layer:** paired Wilcoxon, Holm-corrected within each
(dataset, v2-arm) family of 14 — **111 of 112 comparisons significant at
α=0.05.** The sole exception: FedAWARE vs. `v2_m1` on CIC (Holm p=0.0625) —
not coincidentally the one baseline closest to FedGRAA-v2's own range, at
the refresh interval where FedGRAA-v2's CI is widest. Reported, not hidden.

## 5. Communication overhead — and a discrepancy worth knowing about

`(1+⌈E'/m⌉)×` a plain-FedAvg-shaped baseline's traffic. At `m=1`: exactly
**6.00×** on both datasets. Drops with `m`: 4.00× (`m=2`), **3.00×
(`m=3`)**, 2.00× (`m=5`). The non-obvious finding: `m=3` — the *middle*
interval at *half* `m=1`'s cost — achieves FedGRAA-v2's own **best**
rare-class recall on both datasets, not merely a decent one.

**⚠️ A discrepancy caught while preparing this briefing, worth having ready
if asked about overhead precisely:** the v2 manuscript itself states this
trade-off two different ways in two different sections. §6 (and the
abstract) say `m=3 → 3.00×`, framed as measured. §4.3 separately says
`m=5 (refreshed once per five inner epochs) → traffic drops to ≈2×`. Both
appear in the same document, unreconciled. The Stage C master report's
own numbers (§8 above, this project's docs) match the `6.00×/3.00×`
framing exactly by the `(1+⌈E'/m⌉)×` formula, so that's very likely the
correct one and §4.3's "≈2×" is either a looser approximate restatement or
a copy-edit artifact — but it's an open, small inconsistency *inside the
resubmission-ready manuscript*, not something this project's own reports
disagree on. Worth a one-line fix before submission; honest to flag if it
comes up rather than assume it's already resolved.

## 6. T7 — adversarial robustness (a first for this project)

**Why this tier exists, in the project's own words:** its literature review
had already written the sentence that makes skipping this indefensible:
*"FedGRAA is therefore not merely undefended against poisoning — it is a
mechanism that structurally amplifies exactly the signal a poisoning
defense suppresses."* No prior stage tested this empirically.

**4 attacks, ascending novelty:**
1. **Label-flipping** — field-standard, relabels a fraction of one class as another.
2. **Model-poisoning ("pill")** — Guo et al., AAAI 2026: blueprint-neuron
   selection + greedy max-weight search + cosine-similarity camouflage,
   core mechanism reproduced from scratch (only the core; not every search
   variant — disclosed, not silently narrowed).
3. **The mechanism-specific attack — novel to this project.** Derived
   directly from FedGRAA-v2's own printed gradient formula,
   `dL'/dw_k = ⟨grad_P, P_k⟩`. The attacker submits `P_global −
   η_fake·grad_P`, which by construction makes its own inner product with
   `grad_P` *more negative* than an honest contribution — and because Adam
   descends on `w` using exactly that inner product, this mechanically
   *increases* the attacker's own aggregation weight. Falls back to an
   honest contribution when no gradient has been observed yet (round 1).
4. **An adaptive wrapper** — random-restart local search over a base
   attack's parameters, scored against a defense's own detectability
   proxy — genuinely does optimization, not a token gesture.

**Measured result:** with 30% adversarial clients, the attacker's mean
assigned weight climbs to a stable **3.1× the uniform per-client share**
on CIC (21.9% vs. 7.1% uniform for 14 clients) and **2.7×** on UNSW (30.1%
vs. 11.1% for 9 clients), holding for ~190 of 200 rounds. At 50% attacker
fraction the per-attacker share drops (more attackers splitting the same
advantage) but still 1.6–2.2× uniform. **This is a real demonstration on a
real 200-round campaign of the literature review's prediction, not a
restated theoretical concern.**

**External defenses — honest results, including a weak one:**

| Defense | Attack | CIC ASR | UNSW ASR |
|---|---|---|---|
| SecFedNIDS | label-flip | 75.0% | 60.0% |
| SecFedNIDS | model-poison | 65.0% | 60.0% |
| SecFedNIDS | adaptive | 55.0% | 53.3% |
| WeiDetect | label-flip | 15.0% | 0.0% |
| WeiDetect | model-poison | 20.0% | 6.7% |
| WeiDetect | adaptive | 60.0% | 33.3% |

(ASR = attack success rate = 1 − detection recall; lower is better.)

**Three findings, stated plainly:**
1. SecFedNIDS's reproduction is a weak defense here — missing 53–75% of
   attackers, notably weaker than its reference implementation's
   reputation might suggest — evaluated on *this project's* datasets, not
   SecFedNIDS's original CIC-Darknet2020/CSE-CIC-IDS2018.
2. WeiDetect is substantially stronger against static attacks (0–20% ASR),
   but the adaptive attacker specifically finds and exploits its weakness
   (60% ASR on CIC) — direct evidence the adaptive wrapper does real
   optimization.
3. **The most consequential finding: every successfully-defended cell's
   rare-class recall collapses to ≈0%.** Both defenses stop the attack
   only by falling back to *unweighted* averaging of retained clients —
   discarding exactly the rare-class-sensitive up-weighting that is
   FedGRAA-v2's entire contribution. **A defense that "wins" by discarding
   the mechanism under evaluation is not a free win.**

AdaAggRL under label-flipping collapses to **0.0% rare-class recall** on
CIC despite being framed in the literature as an RL-learned,
adversarially-aware aggregator — reported as a genuine negative result,
not omitted.

## 7. T7G — the γ-sweep addendum (same-day follow-up, 2026-08-02)

T7's main campaign ran at a single γ=1.0. This addendum measured γ∈{0,0.5}
too, same undefended arm, same seeds, so all three points are directly
comparable.

| Dataset | Attackers | γ=0.0 | γ=0.5 | γ=1.0 |
|---|---|---|---|---|
| CIC-IDS-2017 | 30% | 2.05× (σ=0.122) | **3.34×** (σ=0.007) | 3.07× |
| CIC-IDS-2017 | 50% | 0.93× (σ=0.068) | **1.82×** (σ=0.008) | 1.60× |
| UNSW-NB15 | 30% | 2.33× (σ=0.129) | **2.91×** (σ=0.010) | 2.71× |
| UNSW-NB15 | 50% | 1.24× (σ=0.126) | 2.11× (σ=0.012) | **2.24×** |

**The effect is non-monotonic in γ** — peaks near γ=0.5 in 3 of 4 cells,
not at γ=1.0 (the main campaign's default). **The more consequential
finding is the variance, not the mean:** at γ=0, the across-seed standard
deviation is **10–20× larger** than at γ≥0.5. At γ=0, whether the attack
takes hold at all looks closer to a coin flip across seeds; at γ≥0.5 it's
consistently strong. Hypothesis, named as such and not overclaimed: above a
γ threshold around 0.5, the rare-class-prioritization signal the attack
exploits becomes *reliably present* rather than scaling linearly with γ.

## 8. What Stage C is not — Option A, deliberately deferred

The literature review's §10 offered four angles: **A** — few-shot/
prototype-guided hybrid aggregation (competitive with 2025 FedAPA/FedAWA-
style work, but zero implementation, no design doc, facing fresh 2025–26
competition: FeCoGraph, FedFSA, FedGMKD, PROTEAN, DFML); **B** — the
privacy-compliant redesign (chosen, this stage); **C** — "both, sequenced";
**D** — the NF-UQ-NIDS-v2 dataset migration (also done, this stage). The
author chose **C** directly — *"let's implement it all... and think about
the consequences later on with the actual numbers,"* clarified as *"I meant
option C, doing both sequentially, doing B and D for now"* — **without**
the mentor conversation the literature review itself called for. That's
worth surfacing plainly at the meeting: this specific choice (B+D now, A
later) was made unilaterally, and the roadmap flags it as "worth surfacing
to the mentor explicitly, since it's a deviation from the plan this
document itself laid out." Full context and exact framing: `04` §2 (Q2).

**⚠️ A cross-document lettering trap, worth knowing before it comes up:**
different docs use A/B/C/D for different things. `Final_Stage_C.md` §10
(this stage's own primary source): A=few-shot hybrid, B=privacy redesign,
C=sequencing, D=dataset migration. `STAGE_DIRECTION_ASSESSMENT.md`'s own
table: A="Stage B stands alone," B=few-shot hybrid, C=privacy redesign,
D="B then C" sequencing. **Same underlying decision, different letters** —
documented explicitly in both source files, not silently reconciled. If
asked "which option did you pick," answer in words ("privacy redesign +
dataset migration now, few-shot hybrid later"), not by letter.

## 9. The manuscript — how it answers the peer review

Full peer-review verdict and point-by-point resolution mapping: `05` §2.
The short version: a full v1-only draft already existed in the repo
(missed by earlier Stage C documentation's own filesystem search),
already reviewed by a 5-agent simulated panel and rejected
("Resubmit Encouraged") on missing simplex projection, raw seed-level
logs, a convexity proof mismatched to the model, missing adaptive
baselines, no significance testing, and the Eq.2/Eq.5 FedAvg
contradiction. Every one of those is resolved with real Stage C data —
plus two things neither the original draft nor its review addressed at
all: the privacy-compliant redesign (§1 above) and the adversarial-
robustness evaluation (§6–7 above).
