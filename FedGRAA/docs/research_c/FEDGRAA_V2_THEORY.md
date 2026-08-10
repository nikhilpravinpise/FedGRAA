# FedGRAA-v2 — Theory Workstream (W5)

> **What this document is.** The theory workstream [`FEDGRAA_V2_MASTER_PLAN.md`](FEDGRAA_V2_MASTER_PLAN.md) §6 calls for: build-order steps 1–5 (§6.2), executed against the *implemented* code in `fedgraa/aggregation.py`, `fedgraa/secagg.py`, `fedgraa/privacy.py` and `fedgraa/trainer.py` rather than against a hypothetical design. Steps 6 (deriving `w` from the bound, FedAAW) and 7 (hardening `β_eff`/the `min()` sub-differential/the rate constant) remain out of scope — both are flagged in the master plan as, respectively, "optional, high leverage, requires obtaining the paper" and "mentor-gated" — and are not attempted here.
>
> **What this document is not.** It does not re-derive or re-verify the literature citations the master plan already sourced (F9, F5) — those provenance claims (Schmidt/Le Roux/Bach 2011, Karimi/Nutini/Schmidt 2016, Li et al. ICLR 2020) are carried forward at the master plan's own verification level (§14.3: "Verified by external fetch" for Schmidt et al.; "Catalogued" for the others), not re-fetched here. What *is* new here is connecting each ingredient to the actual implemented mechanism that produces it, since — unlike when §6 was drafted — `SecAggChannel`, the Laplace/RDP accountants, `FedGRAAv2`'s `grad_refresh_every`, and `FederatedTrainerV2`'s dropout model now exist and are tested.
>
> **Compiled:** 2026-07-28, against the implementation landed in this pass (`fedgraa/aggregation.py`'s `FedGRAAv2`/`two_phase_composite_grad`, `fedgraa/secagg.py`, `fedgraa/privacy.py`, `fedgraa/trainer.py`'s `FederatedTrainerV2`/`zeta_probe`) — 234 tests passing, including the T-4/T-4a/T-4b/T-12/T-18 gates this document's claims lean on.

---

## Table of contents

1. [Step 1 — the ζ=0 special case](#1-step-1--the-ζ0-special-case)
2. [Step 2 — Assumption 5 and the target theorem](#2-step-2--assumption-5-and-the-target-theorem)
3. [Step 3 — decomposing ζ into named, code-mapped sources](#3-step-3--decomposing-ζ-into-named-code-mapped-sources)
4. [Step 4 — swapping convexity for the PL condition](#4-step-4--swapping-convexity-for-the-pl-condition)
5. [Step 5 — partial participation](#5-step-5--partial-participation)
6. [Standing discipline and rigor flags carried forward](#6-standing-discipline-and-rigor-flags-carried-forward)
7. [What is asserted vs. verified](#7-what-is-asserted-vs-verified)
8. [AI disclosure](#8-ai-disclosure)

---

## 1. Step 1 — the ζ=0 special case

[`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §4 already proves, for the strict-reproduction (v1) setting where `w`'s gradient `∇_w L'` is computed **exactly** on the server (a pooled, server-held `D_V`):

> **Proposition (Stage B, restated).** Let `L'` have bounded subgradients on the probability simplex `Δ` (Assumption 2) and take `η' ≤ 1/β_eff`. Then the projected update `w^t = Π_Δ(w^{t-1} − η'∂L'(w^{t-1}))` satisfies (1) `w^t ∈ Δ` for every `t ≥ 0` **structurally**, and (2) `min_{t≤T} ‖G_{η'}(w^t)‖² = O(1/T)` — gradient-mapping stationarity, not global optimality, because `L'` is non-convex (§3(i) of that document) and non-smooth at the `{L_h = L_m}` boundary (§3(ii)).

FedGRAA-v2 changes **only** the source of `∂L'(w^{t-1})` — from a pooled server-held computation to the SecAgg-mediated, two-phase reconstruction in [`two_phase_composite_grad`](../../fedgraa/aggregation.py) — while leaving Eq. 7–10 and the projection step untouched (F1, I1 in the master plan's §3 invariant table). The Stage B proposition above is therefore **exactly the `ζ = 0` special case** of the theorem this document states next: when the federated gradient estimate is exact (`ζ = 0`), the `O(ζ²)` term below vanishes and the statement reduces word-for-word to Stage B's. This is not a new proof; it is the base case the v2 theorem must contain, and T-4/T-4a/T-4b (§9.2, all green) are the empirical demonstration that the `ζ = 0` case is in fact reachable by the implemented protocol — [`scripts/verify_v2_exactness.py`](../../scripts/verify_v2_exactness.py) measures it at `2×10⁻⁷` relative error, and `tests/test_aggregation_v2.py::test_v2_two_phase_protocol_matches_pooled_composite_grad` pins it as a regression gate across both datasets and three γ values.

---

## 2. Step 2 — Assumption 5 and the target theorem

**Assumption 5 (bounded gradient-estimation error).** The server's federated composite-loss gradient estimate `∇̂_P L'` satisfies

```
E‖∇̂_P L' − ∇_P L'‖² ≤ ζ²
```

for some `ζ ≥ 0`, where `∇_P L'` is the pooled (oracle) gradient Stage B's proposition assumes exact access to.

**The ingredient that makes this assumption tractable rather than merely stated: Schmidt, Le Roux & Bach, "Convergence Rates of Inexact Proximal-Gradient Methods for Convex Optimization," NeurIPS 2011 (arXiv:1109.2415)** — verified this pass (F5, §14.2 of the master plan) via direct fetch of the arXiv abstract; its result is that inexact (sub)gradient methods retain the error-free convergence rate **provided the per-step error decays appropriately**, with the residual entering the bound additively as an `O(ζ²)` (or, for a fixed non-decaying `ζ`, a floor term) rather than destroying the rate altogether. This is the standard tool for exactly this situation — projected descent with a *noisy* gradient oracle — and its citation is the single piece Stage B's own proposition was missing before this pass.

### Target theorem

> **Theorem (informal, ζ-inexact projected subgradient descent on `Δ`).** Let `L'` have bounded subgradients on `Δ` and satisfy the Polyak–Łojasiewicz (PL) condition on `Δ` (§4, replacing the convexity Stage B's proposition never assumed either — see below). Let the server's composite-loss gradient estimate satisfy `E‖∇̂_P L' − ∇_P L'‖² ≤ ζ²` (Assumption 5). Then projected inexact (sub)gradient descent
>
> ```
> w^t = Π_Δ(w^{t-1} − η'∇̂_P L'(w^{t-1})),   η' ≤ 1/β_eff
> ```
>
> satisfies
>
> ```
> min_{t≤T} ‖G_{η'}(w^t)‖² = O(1/T) + O(ζ²)
> ```
>
> and `w^t ∈ Δ` for every `t ≥ 0` **by construction**, exactly as in the `ζ = 0` case (§1) — the projection step is unaffected by where the gradient came from.

**Proof sketch (extending [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §4's four-step sketch by one step).**

1–4. Identical to Stage B's proof sketch (projection non-expansiveness; the standard projected-(sub)gradient rate; the `min(L_h, L_m)` kink handled via the Clarke sub-differential; non-convexity of the client-model space not invoked). None of these steps reference where `∂L'(w^{t-1})` comes from — they hold for *any* subgradient oracle satisfying the boundedness assumption.

5. **The inexact-oracle step (new).** Decompose the per-step error as `∇̂_P L'(w^{t-1}) = ∂L'(w^{t-1}) + e^t` with `E‖e^t‖² ≤ ζ²`. Standard inexact-(sub)gradient analysis (Schmidt/Le Roux/Bach 2011's Prop. 1/2 machinery, specialized from proximal-gradient to the simpler projected-subgradient case since `L'`'s only "proximal" structure here is the simplex indicator, whose exact prox is `Π_Δ` — already used, not approximated) carries the noise term through the standard descent-lemma argument as an additive `O(ζ²)` contribution to the stationarity bound, alongside the unchanged `O(1/T)` optimization term. Because the two obstructions F12 identified are eliminated by construction in [`two_phase_composite_grad`](../../fedgraa/aggregation.py) (T-4a/T-4b assert this, not merely hope for it), the *protocol* itself contributes no bias to `e^t` — every source of `ζ` in the implemented system is a genuine noise/staleness term, decomposed by source in §3 below, not a hidden arithmetic error the theorem would otherwise need to absorb.

**Three properties this theorem has, inherited from the master plan's own framing (F9):**

- It **reduces to Stage B's proposition exactly** at `ζ = 0` (§1).
- Its `O(ζ²)` term is **exactly the quantity the frontier experiment measures** ([`FederatedTrainerV2`](../../fedgraa/trainer.py)'s `measure_zeta=True` path / `zeta_probe`) — theory and experiment test the same object, which is what makes this theory section load-bearing rather than decorative.
- It covers **every point on the frontier with one statement**: `m=1` (v2-exact, `ζ_stale ≈ 0`) through `m=5` (v2-lazy-5, larger `ζ_stale`) are all instances of the same theorem at different `ζ`, not separate results requiring separate proofs.

---

## 3. Step 3 — decomposing ζ into named, code-mapped sources

> ⚠️ **`ζ_protocol = 0` is a claim, not an omission (F12).** The decomposition below has no `ζ_protocol` term because the two-phase count-weighted protocol (§3 of the master plan; implemented as `two_phase_composite_grad`) is designed so the federated computation is *algebraically identical* to the pooled one — `ζ_protocol = 0` exactly, not approximately. T-4a and T-4b are what license omitting this term: until they are green, it must be carried explicitly, because the naive (both-design-docs-specified) protocol has a large, `m`-independent `ζ_protocol` that would sit as a constant floor under the entire frontier and be indistinguishable from "exactness does not help" (F12's Adam-masking observation makes this concrete: a wrong protocol can look fine in early rounds and diverge only later). **Both T-4a and T-4b are green** (`tests/test_aggregation_v2.py`), so this decomposition proceeds on the assumption that holds.

`ζ² ≲ ζ²_secagg + ζ²_dp + ζ²_stale(m) + ζ²_dropout`, with each term now a specific, code-identifiable quantity rather than a placeholder:

| Term | Source (code) | What it actually is | Measured/measurable |
|---|---|---|---|
| `ζ_secagg` | `SecAggChannel` (`fedgraa/secagg.py`) | **Not cryptographic noise** — this implementation models SecAgg's ideal functionality (exact sum), so `ζ_secagg` here is purely float32 summation-order noise across `~4×10⁴` parameters. A real Bonawitz-et-al.-style deployment adds no *additional* noise beyond this either (SecAgg reconstructs the exact sum when enough shares survive) — the only place a real deployment differs is `ζ_dropout` (below), not `ζ_secagg` itself. | `test_v2_two_phase_protocol_matches_pooled_composite_grad`: `< 10⁻⁶` relative error, both datasets, three γ. |
| `ζ_dp` | `fedgraa/privacy.py`'s `laplace_mechanism`, consumed in `FederatedTrainerV2._histograms_and_tags` | DP noise is added **only to the rarity-ranking histogram** (`h_k^rank`, F3), never to the gradient channel. Its effect on `ζ` is entirely *indirect*: if noise flips which classes `BottomN` selects as rare, the resulting `D_V^n`/`D_V^h`/`D_V^m` stratification differs from the oracle's, producing a genuinely different (not approximately different) `grad_P L'`. At the DP budgets F3/§5.2 measured as viable (`ε ≥ 1`, ≥99.85% `BottomN` recovery), this term is small with high probability but not identically zero — it is a *discrete* failure mode (the rare set either matches or it doesn't), not a small continuous perturbation. | `test_zeta_decomposition_sources_are_separable` (T-18, `tests/test_trainer_v2.py`): forcing a rare-set flip directly measures `ζ_dp`'s isolated contribution (`> 10⁻²`, i.e. not small) against a `ζ_secagg`-only baseline (`< 10⁻⁵`). |
| `ζ_stale(m)` | `FedGRAAv2.grad_refresh_every` (`fedgraa/aggregation.py`) | The gap between the gradient used at inner epoch `e` (refreshed only every `m` epochs) and the gradient a fresh two-phase call would return at the epoch's *actual* (already-Adam-updated) `θ'_g`. This is the frontier's own knob: `m ∈ {1,2,3,5}` traces `ζ_stale` from ≈0 (`m=1`) to a larger, `m`-dependent value. **This is the term the compute campaign (T1–T3) measures empirically as rare-class recall vs. `m`** — no closed-form estimate of `ζ_stale(m)` is offered here beyond "it is non-decreasing in `m`" (more epochs between refreshes ⟹ `θ'_g` has moved further from where the last-computed gradient was evaluated), because the actual magnitude depends on the loss landscape's local curvature, which is an empirical, not analytical, question for this non-convex model. | `test_zeta_decomposition_sources_are_separable` isolates it directly: evaluating the two-phase protocol at a deliberately stale `θ'_g` and comparing against the *current* oracle gradient measures a real, `> 10⁻²` gap, distinct from `ζ_secagg`/`ζ_dp`. The *trajectory* of `ζ_stale(m)` across a full run is what `FederatedTrainerV2(measure_zeta=True)` logs per eval round — see the scope disclosure below. |
| `ζ_dropout` | `FederatedTrainerV2`'s `p_drop`/`secagg_threshold` (`fedgraa/trainer.py`) | When a client drops, the two-phase protocol's `n_arm`, `n_n` (and hence `grad_P L'`) are computed over **fewer rows** than the true population — this is not noise added *to* an otherwise-correct quantity, it is a **smaller but still exactly-computed** quantity (T-4a's count-weighting guarantee holds regardless of how many clients participate; `two_phase_composite_grad` is exact *given* whichever clients submitted). `ζ_dropout` is therefore the gap between "the gradient over the surviving population" and "the gradient over the full population" — a sampling-variance term, not a protocol-correctness term. | `test_dropout_robustness_end_to_end` (T-13) confirms the *trainer* stays numerically stable and produces finite, in-range accuracy across `p_drop ∈ {0, 0.1, 0.3}`; it does not currently isolate `ζ_dropout`'s magnitude the way T-18 isolates `ζ_dp`/`ζ_stale` (see the gap noted below). |

**A disclosed scope gap, stated rather than silently left implicit.** [`zeta_probe`](../../fedgraa/trainer.py) and `FederatedTrainerV2`'s `measure_zeta=True` path compute **both** `∇̂_P L'` and `∇_P L'` freshly at the *same*, current `global_params` (post-round), specifically so the measurement is not confounded by "which inner epoch the last refresh happened at." This means the per-round `zeta` logged in `TrainHistory.zeta` captures `ζ_secagg` and any `ζ_dp`-induced stratification difference, but **not** `ζ_stale(m)`, which by definition requires comparing a *stale* gradient (evaluated at an earlier `θ'_g`) against the *current* oracle — exactly what T-18's dedicated unit test does at a single point, but what the full-trainer per-round log does not do continuously. Measuring `ζ_stale(m)`'s trajectory *inside* `FedGRAAv2.update()`'s inner-epoch loop (not just at the two-phase-protocol level) is real, additional instrumentation work, correctly scoped as part of the compute campaign's own instrumentation pass (§7.2 T1's "annotate each v2 point with its measured median `ζ_t`") rather than claimed as done here. This is recorded in [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md).

---

## 4. Step 4 — swapping convexity for the PL condition

Neither Stage B's proposition nor this document's theorem ever invoked convexity of `L'` (§1's proof sketch step 4 says so explicitly) — the projected-gradient-mapping stationarity argument is precisely the tool available for the non-convex, constrained case. What the **rate constant** in `O(1/T)` depends on, however, is a curvature-type condition; the master plan's F9 catalogues **Karimi, Nutini & Schmidt, "Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak–Łojasiewicz Condition," ECML-PKDD 2016** as the standard replacement for convexity in exactly this role.

**Why PL is the right substitute here, not merely an available one.** The PL condition (`½‖∇f(x)‖² ≥ μ(f(x) − f*)` for some `μ > 0`) is satisfied by a strictly larger function class than convex functions — including many over-parameterized neural-network objectives, which is the relevant regime here (`L'` is a composite loss over a ~4×10⁴-parameter tanh-MLP's *aggregation weights* `w` — an 14- or 9-dimensional simplex, not the network's own parameter space — evaluated through the network). Under PL, gradient descent (and its projected variant) achieves a **linear** rate rather than the sublinear `O(1/T)` this document's theorem states, which would be a genuine strengthening if `L'(w)` (as a function of `w` alone, network parameters held fixed at each round's `θ_k^t`) satisfies PL on `Δ`.

**What is and is not claimed here.** This document does **not** assert that `L'(w)` satisfies PL — that is an empirical/analytical question about the specific composite loss surface at the model's actual trained parameters, not something derivable from the citation alone, and asserting it without checking would repeat exactly the failure mode ([`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §3(i)) that this project's own audit discipline exists to catch (the original paper's Assumption 1 asserted convexity of a demonstrably non-convex object). What *is* claimed: **PL, not convexity, is the correct condition to check** for a rate stronger than `O(1/T) + O(ζ²)`, and the theorem above holds with only the weaker bounded-subgradient assumption regardless of whether PL holds — PL is a possible strengthening, not a load-bearing requirement of the stated result. Verifying PL for this specific model (e.g., empirically, via the standard PL-inequality residual check across training) is future work, appropriately gated behind having T1–T3's actual trained trajectories to check it against — another point where theory and the (as yet unrun) compute campaign meet.

---

## 5. Step 5 — partial participation

The master plan's F9 catalogues **Li et al., ICLR 2020** (the standard reference for FedAvg-style convergence under **partial client participation** — not every client trains every round) as the ingredient for this step. Two things are worth stating precisely about how this interacts with what is actually implemented:

1. **`FederatedTrainerV2` does not currently sample a subset of clients for local training** — every client in `split.clients` trains locally every round (`client_train_round` is called for `self.split.clients`, unconditionally; see [`fedgraa/trainer.py`](../../fedgraa/trainer.py)), matching the base paper's and `FederatedTrainer`'s own convention of full participation in Algorithm 1. What v2 adds is **SecAgg-channel dropout** — some clients that *did* train locally may not survive to contribute to the rarity histograms or the composite-gradient channel that round (`cfg.p_drop`, T-7/T-13). This is a narrower notion of partial participation than Li et al.'s (full local-training subsampling), and the theorem above does not currently need the general result: `ζ_dropout` (§3) already absorbs SecAgg-channel non-participation as a bounded-error term under the existing Assumption-5 framework, without requiring a separate participation-rate parameter in the rate itself.
2. **If/when true client subsampling is added** (e.g. to model a larger real deployment where not every client is reachable every round — relevant for the NF-UQ-NIDS-v2 multi-silo setting, W6), Li et al.'s result is the correct citation to extend the theorem with an explicit participation-rate term, and the `SecAggChannel`/`FedGRAAv2` machinery already built is participation-count-agnostic (the two-phase protocol's normalizers are always the *actual* SecAgg-summed counts, whatever they are) — no structural change to `two_phase_composite_grad` would be needed, only a change to how `FederatedTrainerV2.train()` selects `self.split.clients` each round.

This step is therefore **scoped but not required** by the current implementation: it is correctly catalogued as the reference to reach for if client subsampling for local training (as opposed to SecAgg-channel dropout, already implemented) is added later.

---

## 6. Standing discipline and rigor flags carried forward

[`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §4's own rigor flag lists three items left for a from-scratch proof pass: **(a)** an exact estimated `β_eff` for the paper's tanh-MLP; **(b)** a fully formal (measure-theoretic) treatment of the `min(L_h, L_m)` sub-differential; **(c)** an explicit `O(1/T)` rate constant. **None of these are resolved by this document**, and this document adds a fourth, specific to v2: **(d)** an empirical PL-condition check for `L'(w)` on `Δ` at the model's actual trained parameters (§4 above). All four remain exactly what [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) called them: mentor-gated hardening work, not claims made here. Per this project's standing no-silent-patching mandate, this document states the correct qualitative shape of the result (stationarity, not global optimality; `O(1/T) + O(ζ²)`, not a claim the `ζ` terms vanish) and does not upgrade any of these hedges without doing the underlying work.

**One property worth stating plainly because it is easy to lose in the decomposition table:** the whole reason `ζ_protocol = 0` is assertable (rather than "small") is that T-4a and T-4b are *unit tests on the arithmetic itself*, not statistical measurements with a confidence interval — a future change to `two_phase_composite_grad` that reintroduced either obstruction would fail these tests deterministically, not probabilistically. This is the same "test the mechanism, not just the symptom" discipline [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §3 already established for the simplex-projection fix.

---

## 7. What is asserted vs. verified

- **Verified by the project's own test suite (this pass):** `ζ_protocol = 0` (T-4/T-4a/T-4b, `tests/test_aggregation_v2.py`); the `O(ζ²)` term's decomposition sources are separable and attribute correctly (T-18, `tests/test_trainer_v2.py`); feasibility (`w^t ∈ Δ` structurally) holds for `FedGRAAv2` exactly as it did for `ProjectedFedGRAA` (`test_fedgraav2_output_is_on_simplex`); the ζ=0 case reproduces `ProjectedFedGRAA`'s own trajectory to float32 tolerance given an identical injected gradient (`test_fedgraav2_at_m1_matches_projected_fedgraa_given_identical_gradient`).
- **Verified by the master plan's earlier passes, carried forward here at that verification level:** the Schmidt/Le Roux/Bach 2011 citation (external fetch, F5); Stage B's own proposition and its empirical corroboration (60/60 feasibility cells, `fedgraa_convergence_defect.md` §4).
- **Not verified, and explicitly flagged as such:** whether `L'(w)` actually satisfies the PL condition on `Δ` for this model (§4); the exact functional form/magnitude of `ζ_stale(m)` as a function of `m` (§3 — this is precisely what T1–T3's compute campaign exists to measure, not something derived here); an explicit `O(1/T)` rate constant or `β_eff` value (carried forward from Stage B, unresolved).
- **Judgement, offered as a documented design choice rather than a proven fact:** that SecAgg-channel dropout (as currently implemented) is adequately modeled within the existing `ζ_dropout` term without invoking Li et al.'s full participation-rate machinery (§5) — reasonable given what is actually implemented, but not something this document proves is the *only* correct modeling choice.

---

## 8. AI disclosure

This document was compiled by an AI assistant (Claude Sonnet 5) at the repository author's direction, as part of implementing FEDGRAA_V2_MASTER_PLAN.md's W0–W6 workstreams end to end. It restates [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) §4's already-authored proposition rather than re-deriving it, extends its proof sketch by exactly one step (the inexact-oracle term, §2.5) using the citation the master plan's F5/F9 already sourced and verified by external fetch, and maps the `ζ` decomposition (§3) onto the specific, tested code this pass implemented (`SecAggChannel`, `fedgraa/privacy.py`, `FedGRAAv2.grad_refresh_every`, `FederatedTrainerV2`'s dropout model) rather than describing hypothetical mechanisms. No new literature search was performed for this document — every citation used here is carried forward from the master plan's own §14.2 tooling-disclosure table at that table's stated verification level, and is marked as such throughout rather than re-asserted as newly confirmed. The one genuinely new content in this document is §3's honest scope disclosure that the implemented `zeta_probe`/`measure_zeta` path measures `ζ_secagg` and `ζ_dp`'s downstream effect but not `ζ_stale(m)` continuously — discovered while implementing and testing the mechanism (an initial version of the eval-time zeta computation conflated "last-refreshed gradient" with "current gradient," producing a misleading large `zeta` that turned out to be a measurement-timing artifact, not a real protocol defect; caught by comparing against T-4's known-exact baseline, in the same spirit as this project's F3/F12 findings).
