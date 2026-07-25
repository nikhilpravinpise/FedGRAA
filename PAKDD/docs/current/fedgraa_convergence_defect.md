# Why an FL-IDS Aggregation Algorithm Doesn't Converge, and How to Fix It

**Status:** Complete. §1–4 (theory), §5 (consolidated 10-seed ablation table), and §6–7 (discussion, limitations) are all closed against the full Stage B data campaign (T3): 4 arms × 2 datasets × 10 seeds (80 diagnostic runs) plus 4 baseline seed-sweeps (40 seeds), consolidated via [`../../fedgraa/stats.py`](../../fedgraa/stats.py) with bootstrap CIs, paired significance tests, and the 2×2 factorial read. `python scripts/check_stage_b_complete.py` exits `0` on this tree (completeness), and so does `python scripts/audit_stage_b.py` (correctness: every number below re-derived from the raw runs and independently recomputed against SciPy — see [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) §7.1 for the three defects that audit found and fixed).

**Parent:** [`stage_b_plan.md`](stage_b_plan.md) Tasks T1 (§2–3 below), T2 (§4), T8 (this document, assembled). Builds on [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md) §5 and [`fedgraa_divergence_diagnosis.md`](fedgraa_divergence_diagnosis.md), which established the empirical finding this document formalizes; on [`../../understanding/05_convergence_proof.md`](../../understanding/05_convergence_proof.md), which explains the paper's original proof; and on [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) items #6 and #13, whose consequences §3 works out in full.

**One-paragraph summary.** The FedGRAA paper's Algorithm 2 updates its aggregation-weight vector `w` by gradient descent and never projects it back onto the probability simplex `Δ = {w : w_k ≥ 0, Σw_k = 1}`. On real data, at every tested `γ ∈ {0, 0.5, 1}`, on both CIC-IDS-2017 and UNSW-NB15, `w` leaves `Δ` in round 1 — every single time. Off the simplex, the aggregated model `θ_g = Σ_k w_k θ_k` is no longer a convex combination of client models but an unbounded linear extrapolation, and the paper's own Lemma 1 (composite loss `L'` must be non-increasing) fails immediately. §2 states this as a feasibility-set defect; §3 shows why the paper's Appendix A.1 proof does not — indeed structurally cannot — rule it out; §4 states and sketches a corrected convergence guarantee for the algorithm one would get by adding the missing projection step; §5 is the consolidated, statistically-tested ablation table quantifying the fix across both the projection lever and a companion learning-rate correction; §6 discusses severity and the paper's own hyperparameters; §7 is an honest accounting of what this section does and does not establish.

---

## 1. Setup

**The composite objective (paper Eq. 8–9).** Let `θ_g' = Σ_{k=1}^K w_k θ_k` be the candidate aggregated model formed from the `K` client models `{θ_k}` under weight vector `w`. Two validation losses are computed on disjoint held-out splits — a "high-frequency" split `D_V^h` and a "moderate-frequency" split `D_V^m` (Eq. 8's `L_a = min(L_h(θ_g'; D_V^h), L_m(θ_g'; D_V^m))`) — and blended against the rare-class loss `L_u` on the protected validation set `D_V^n`:

```
L'(w) = (1 − γ) · L_a(θ_g'(w)) + γ · L_u(θ_g'(w)),          γ ∈ [0, 1]
```

**Algorithm 2's weight update (Eq. 10 / Algorithm 2 line 6).** For `E'` epochs per federated round:

```
w ← w − η' ∇_w L'(w)
```

read literally as plain gradient descent on `w` (the paper's prose says Adam — see [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #2 — but neither reading includes a projection step of any kind; this defect is orthogonal to which optimizer is meant). **Algorithm 2 has exactly 8 lines. There is no line 9.** No projection, clipping, or renormalization of `w` appears anywhere in the algorithm box, and a direct grep of this project's line-for-line reproduction (`fedgraa/aggregation.py`) confirms zero matches for any simplex-projection call — the paper's own algorithm, as printed, does not have one.

**The feasible set.** The weights are meant to be a convex-combination coefficient vector — `θ_g = Σ_k w_k θ_k` is only guaranteed to be a point inside the convex hull of the client models `conv{θ_k}` when `w` lies in the probability simplex:

```
Δ = { w ∈ ℝ^K : w_k ≥ 0 for all k,  Σ_k w_k = 1 }
```

Nothing in Eq. 10 or Algorithm 2 constrains `w` to `Δ`. This is the fact §2 shows is consequential, not merely aesthetic.

**Lemma 1's descent guarantee (paper Appendix A.1; see [`understanding/05_convergence_proof.md`](../../understanding/05_convergence_proof.md) §5.3 for the full derivation).** Under the paper's Assumptions 1–3 (composite loss `L'` convex and β-smooth; bounded stochastic-gradient variance/magnitude; step size `η' ≤ 1/β`), each weight-update step is claimed to satisfy:

```
L'(w^t) ≤ L'(w^{t-1}) − (η' − β(η')² / 2) ‖∇L'(w^{t-1})‖²
```

The bracket `(η' − β(η')²/2)` is positive whenever `η' ≤ 1/β`, which is what makes the right-hand side a strict decrease — the paper's basis for claiming `w^t → w*` and, via Lemma 2, that the global model `θ_g^t` converges to the loss-minimizing point. §2 shows this guarantee fails at the very first round on real data, and §3 shows why the proof's own assumptions cannot actually deliver it for the system the paper describes.

---

## 2. The defect: `w` leaves the feasible set at round 1, unconditionally

**Empirical fact (established in [`fedgraa_divergence_diagnosis.md`](fedgraa_divergence_diagnosis.md) §5.2, reproduced in [`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md) §5.2 and independently re-confirmed byte-for-byte by the 2026-07-23 full audit).** Running the unmodified, strictly literal `FedGRAA` update (`scripts/diagnose_fedgraa.py`, no `--project` flag — exactly what `fedgraa/` ships) with `eval_every=1` so every round's weight vector is recorded, `w` violates the simplex constraint (`|Σw − 1| > 10⁻³` or `min(w) < −10⁻⁶`) at **`first_simplex_violation_round = 1`, at every tested `γ ∈ {0, 0.5, 1.0}`, on both datasets, with zero exceptions**:

| Dataset | γ | First simplex violation | Composite-loss blowup | Lemma 1 monotone? |
|---|---|---|---|---|
| CIC-IDS-2017 | 0.0 | round 1 | 3.78× | ❌ No |
| CIC-IDS-2017 | 0.5 | round 1 | 2.66× | ❌ No |
| CIC-IDS-2017 | 1.0 | round 1 | **193×** | ❌ No |
| UNSW-NB15 | 0.0 | round 1 | 1.25× | ❌ No |
| UNSW-NB15 | 0.5 | round 1 | 1.24× | ❌ No |
| UNSW-NB15 | 1.0 | round 1 | 7.6× | ❌ No |

*(Blowup ratio = max finite `L'` / first finite `L'` over a 60-round run, `n=3`, seed 0. Source: `results/diagnostics/{CIC_IDS_2017,UNSW_NB15}_literal_r60.json`.)*

**This table is seed 0; the Stage B campaign (§5) confirmed it at N = 10 seeds.** `first_simplex_violation_round = 1` holds on **all 60 literal-arm cells** (10 seeds × 3 γ × 2 datasets) with zero exceptions — the round-1 feasibility failure is not a seed artifact but an unconditional property of the update rule on this data. The *blowup* is more variable than seed 0 alone reveals: across 10 seeds the literal arm's blowup ranges from `1.00×` (CIC γ=1, seed 6 — an occasional run that happens not to diverge badly despite leaving the simplex in round 1) to `1.86×10⁶×` (CIC γ=0, seed 8). Feasibility failure is universal; its cost is heavy-tailed. §5 and §6 quantify both.

At `γ=1` on CIC-IDS-2017 — the worst case — the composite loss over the first 10 rounds is:

```
round:   0     1     2     3     4      5      6      7      8      9     10
L'(w^t): 2.98  2.77  4.53  6.06  40.52  75.55  261.48 290.24 319.11 205.30 149.79
```

This is not slow numerical drift. `L'` is non-increasing for exactly one step (round 0→1), then increases by more than an order of magnitude within four further rounds. This is a **direct, measured violation of Lemma 1's claimed guarantee**, at the very first opportunity to check it, on real data, regardless of `γ`.

**The mechanism.** Once `w ∉ Δ`, the aggregated parameters `θ_g' = Σ_k w_k θ_k` are no longer constrained to `conv{θ_k}` — they become an **unbounded affine extrapolation** in parameter space. Geometrically: a convex combination can only ever land *between* the client models; an affine combination with `Σw_k = 1` but some `w_k < 0` (or, more severely, `Σw_k ≠ 1` entirely, which is what the empirical trace shows — see [`fedgraa_divergence_diagnosis.md`](fedgraa_divergence_diagnosis.md) §5.2/5.8 for the raw `Σw`/`min(w)` trajectories) can land arbitrarily far *outside* the region the client models actually explored. On UNSW-NB15 at `γ=0.5`, `Σw` drifts to **−0.731** — the aggregated model is no longer even approximately an average of anything. A network evaluated at a point in weight-space it was never trained to be smooth around produces an unpredictably large loss; that unpredictability, replayed every round because nothing pulls `w` back toward `Δ`, is exactly the observed non-monotone, exploding `L'`.

**Confirmation the mechanism, not just its correlation, is causal.** Re-projecting `w` onto `Δ` after every update (`ProjectedFedGRAA` in `scripts/diagnose_fedgraa.py` — diagnostic-only, never imported by `fedgraa/`), with everything else held fixed (same gradient, same `η'=0.1`, same `E'=5`), collapses the blowup: CIC `γ=1` goes from 193× to 4.3×, CIC `γ=0` goes from 3.78× to 1.01×. The same gradient direction, merely re-clipped back into the convex hull every step, does not blow up. This isolates the defect to the missing projection specifically, not to some other property of the gradient computation.

**Formal statement of the defect.** Algorithm 2, taken literally, solves an *unconstrained* optimization problem (`min_w L'(w)` over all of `ℝ^K`) using an update rule whose descent guarantee (Lemma 1) is proved only under an implicit assumption — never stated, never enforced — that `w` remains in the *constrained* feasible set `Δ`. The paper presents Algorithm 2 as if it operates on `Δ` (weights are called a "weighted average," `Σw_k=1` is used informally throughout the exposition) while implementing an algorithm that has no mechanism to keep `w` there. This is the **feasibility-set defect**: the algorithm's implicit domain and its explicit update rule disagree, and on real data this disagreement is not a rare edge case — it fires in round 1, every time.

---

## 3. Why Appendix A.1's proof does not cover this — three compounding gaps

Three separate, already-documented facts about the paper compound to explain why its own convergence proof gives no protection against the round-1 failure in §2. None of these is a new discovery — each is assembled here, for the first time, into a single argument about *why the proof fails specifically at the point where the empirical defect appears*.

**(i) Assumption 1 (`L'` convex and β-smooth) is false for the model the paper actually uses.** Appendix A.1's Assumption 1 states `L'(θ)` is convex and β-smooth. §5 of the paper describes the network as a multi-layer `tanh`-activated dense network (200→100→50 hidden units) with a softmax output — a standard non-convex neural network loss surface. This is [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) item **#13**: *"the proof, as written, does not actually apply to the system the paper claims it applies to."* Without convexity, gradient descent (on `w` or on `θ`) is not guaranteed to reach any global minimizer at all — the entire "weights converge to `w*`, which is the loss-minimizing weight vector" chain in Lemma 1/Theorem 1 loses its foundation before the feasibility question is even reached.

**(ii) `L_a = min(L_h, L_m)` is non-smooth exactly where training pushes it, so no finite β — and therefore no valid `η'` — exists.** Eq. 8 defines `L_a` as the pointwise minimum of two loss functions computed on disjoint validation splits. `min(f, g)` is non-differentiable wherever `f = g`, and — this is the sharper point, [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) item **#6** — the entire purpose of Eq. 8 is to *balance* `L_h` against `L_m`, meaning the optimization is actively driven *toward* the crossing boundary `{L_h = L_m}`, not away from it by chance. At a kink, no finite β-smoothness constant exists in a neighborhood of the crossing, which means **Assumption 3's step-size condition `η' ≤ 1/β` is either vacuous (β = ∞ ⟹ no valid `η'` at all) or arbitrarily restrictive** immediately before/after the crossing. The descent bracket `(η' − β(η')²/2)` in Lemma 1's inequality can go negative right where the algorithm's own design intends to spend time. This gap and gap (i) are independent — either alone breaks the proof — but they compound: the network is non-convex *everywhere*, and additionally non-smooth at a boundary the loss function is explicitly designed to approach.

**(iii) Even granting smoothness by fiat, Lemma 2 tacitly assumes `w ∈ Δ` — an assumption the algorithm never enforces.** This is the gap specific to §2's empirical failure and the one the first two do not by themselves explain. Suppose, for the sake of argument, that Assumption 1 held exactly and (ii)'s non-smoothness boundary were never approached — a hypothetically well-behaved `L'`. Lemma 1's proof would still only bound `L'(w^t)` in terms of `L'(w^{t-1})` and the gradient norm; it says nothing about *where in ℝ^K* `w^t` lives. Lemma 2 then claims `θ_g^t = Σ_k w_k^t θ_k^t → θ_g^*`, a step that is only meaningful — indeed, only makes `θ_g^t` a "weighted average" at all — if `w^t` stays a convex-combination coefficient vector. Assumption 2 bounds the *gradient* norm (`‖∇L'‖ ≤ G`); it never bounds the *iterate* `w` itself, and nothing in Algorithm 2 does either. A bounded-gradient assumption constrains how fast `w` can move each step; it does not constrain where `w` is allowed to end up, and with no projection, `η'=0.1` and 5 epochs per round is enough displacement, given the loss landscape described in (i)–(ii), to leave `Δ` in the very first round. **This is the gap most directly responsible for the specific failure mode measured in §2** — the other two explain why the proof's guarantees are unreliable in general; this one explains why, even setting that aside, nothing in the proof rules out `w` drifting arbitrarily far from the region where `θ_g` is even a sensible object.

**Summary.** (i) and (ii) mean the proof's premises (convexity, β-smoothness, a valid `η'`) do not hold for the loss function the paper actually specifies. (iii) means that even if they somehow did, the proof would still say nothing about feasibility — and feasibility, not smoothness, is what §2 shows breaks first and worst on real data. The paper's convergence proof and the paper's algorithm, as printed, are about two different mathematical objects: the proof reasons about optimization over `Δ` (implicitly, through its "weighted average" framing); the algorithm optimizes over all of `ℝ^K`.

---

## 4. Corrected convergence proposition: projected (sub)gradient descent on `Δ`

**The fixed algorithm.** Replace Algorithm 2 line 6 with a projected update:

```
w^t = Π_Δ( w^{t-1} − η' ∂L'(w^{t-1}) )
```

where `Π_Δ` is Euclidean projection onto the probability simplex (Duchi et al., 2008, "Efficient Projections onto the L1-Ball for Learning in High Dimensions," Fig. 1 — already implemented, diagnostic-only, as `project_to_simplex` in `scripts/diagnose_fedgraa.py`, and independently unit-tested in `tests/test_diagnose_fedgraa.py`). This turns the weight-learner into an instance of **projected (sub)gradient descent on the compact convex set `Δ`** — a well-studied setting in convex optimization, distinct from the paper's implicit-but-unenforced unconstrained gradient descent.

**Proposition (stated honestly for a non-convex objective — no global-minimum claim).** Let `L'` satisfy Assumption 2 (bounded subgradients: `‖g‖ ≤ G` for `g ∈ ∂L'(w)`, `w ∈ Δ`) and take `η' ≤ 1/β_eff` for the largest β-smoothness constant that holds on `Δ` away from measure-zero non-smooth points (see below). Then:

1. **Feasibility is structural, not empirical.** `w^t ∈ Δ` for every `t ≥ 0` by construction of `Π_Δ`, so `θ_g^t ∈ conv{θ_k^t}` always — the round-1 blowup of §2 is **provably impossible** under this update rule, not merely observed-to-be-absent in a particular run. This matches every combined-fix measurement in the completed Stage B campaign, on **both** datasets, with zero exceptions: across all **60** (`both`-arm) cells — 10 seeds × 3 γ × 2 datasets, per-seed diagnostic JSONs under `results/diagnostics/`, consolidated in `results/ablation/{CIC-IDS-2017,UNSW-NB15}_ablation_table.json` — `final_weight_sum` is `1.0` to within float32 round-off (max deviation `5.96×10⁻⁸` on CIC, `1.19×10⁻⁷` on UNSW) and `final_weight_min ≥ 0` exactly, on every single cell — a zero-variance empirical match to a structural guarantee, not a statistical coincidence. **Feasibility alone does not entail a flat loss trajectory, and the data bears this distinction out precisely**: composite-loss blowup is exactly `1.0×` on 29 of 30 CIC-IDS-2017 combined-fix cells (the one exception at `1.016×`, still negligible), but on UNSW-NB15 it ranges from `1.0×` up to `1.87×` across the 30 cells — and, checked per-seed rather than only on means, is **strictly below the literal algorithm's blowup on the same seed at the same γ in all 30 of 30 UNSW cells** (literal's UNSW blowup reaches `46.85×` at γ=1 over the 10-seed set, well above the fix's `1.87×` ceiling). This is fully consistent with the proposition as stated: feasibility bounds `θ_g` to a compact region and rules out the *unbounded* extrapolation of §2, but boundedness is not flatness — within `Δ`, `L'` can still legitimately fluctuate somewhat round to round, more so on UNSW's apparently less favorable curvature at higher γ. The right reading is a **dramatic, structural reduction**, not a claim that the fix makes every run's loss perfectly flat everywhere.
2. **Gradient-mapping stationarity, not global optimality.** Because `L'` is not convex (§3(i)), the projected method cannot be shown to reach the global minimizer of `L'` over `Δ`. What *does* follow from standard projected-(sub)gradient-descent theory on a compact convex set with bounded subgradients is weaker but well-defined: the projected-gradient-mapping norm satisfies `min_{t ≤ T} ‖G_{η'}(w^t)‖² = O(1/T)`, where `G_{η'}(w) := (w − Π_Δ(w − η'∂L'(w))) / η'` is the standard projected-gradient-mapping operator (the natural stationarity measure on a constrained set — it reduces to `‖∇L'(w)‖` when the constraint is inactive). I.e., **the iterates approach a stationary/KKT point of `min_{w ∈ Δ} L'(w)` as `T → ∞`**, and `L'(w^t)` is non-increasing along any sub-sequence where the projection is inactive (interior steps behave exactly as Lemma 1 originally claimed; the projection only ever pulls `w^t` toward the feasible region, and by non-expansiveness — see the proof sketch — never increases the distance-to-optimum bound that drives the `O(1/T)` rate).

**Proof sketch.**
1. *Projection is non-expansive*: for any `x, y ∈ ℝ^K`, `‖Π_Δ x − Π_Δ y‖ ≤ ‖x − y‖`. This is the standard property of Euclidean projection onto any closed convex set (Δ is a closed convex polytope) and is what lets a descent argument survive the extra projection step without picking up an error term.
2. *Standard projected-subgradient-descent rate*: for `L'` with bounded subgradients (Assumption 2) on the compact set `Δ`, and step size `η' = O(1/√T)` (or a fixed small `η'` with the corresponding asymptotic bound), the projected-gradient-mapping norm satisfies the textbook `O(1/T)` rate (e.g., Nesterov 2004 §3.2.3 non-convex extension via the gradient-mapping criterion; Bubeck 2015 §3.2 for the convex case, which upper-bounds the non-convex rate's constants). This is the sense in which "the weights converge" survives the loss of convexity: not to a global minimizer, but to a stationary point of the constrained problem, at a quantifiable rate.
3. *Handling the `min(L_h, L_m)` kink (§3(ii))*: away from the measure-zero set `{L_h = L_m}`, `L_a` is smooth and its gradient is simply the gradient of whichever branch is active — the standard, conventional sub-gradient reading already adopted for strict reproduction ([`STAGE_A_MASTER_REPORT.md`](STAGE_A_MASTER_REPORT.md) §4.1 item 4). At the boundary itself, the sub-differential of a pointwise minimum of two smooth functions is well-defined via the Clarke sub-differential (any convex combination of the two branches' gradients is a valid sub-gradient); a projected-subgradient method is exactly the class of algorithm designed to tolerate this — the classical projected-subgradient convergence theory is stated for sub-differentiable (not necessarily differentiable) objectives from the outset, so this is not an additional assumption beyond what §4's proposition already requires. An even more conservative treatment smooths the kink itself via a Moreau envelope of `L_a`, trading an exact minimum for an arbitrarily close smooth approximation; this is noted as an available hardening step, not required for the stationarity claim above.
4. *Non-convexity of the client-model space (§3(i))* does not invalidate steps 1–3: the projected-gradient-mapping/stationarity argument never invokes convexity of `L'` — it is precisely the tool convex-optimization theory offers for the non-convex, constrained case (the same technique used to give convergence guarantees for projected SGD training neural networks with, e.g., simplex- or norm-ball-constrained parameters elsewhere in the ML literature). **Convexity is not recovered; it is no longer needed for the (weaker, honest) claim being made.**

**Contrast with the paper's Theorem 1, explicitly.** The paper claims `L'(θ_g^t) → L'(θ_g^*) = inf_θ L'(θ)` — global optimality. This proposition claims only that the projected-gradient-mapping norm on `Δ` vanishes — **stationarity**, not global optimality, and only over the constrained set `Δ`, not over all of `ℝ^K`. This is a strictly weaker guarantee than what the paper states, and it is the *correct* guarantee for the algorithm the paper's model class (§3(i)) actually admits. **Feasibility is the part of Theorem 1's failure mode that this proposition closes completely** (claim 1, structural, no asterisk); **global optimality is the part it does not restore** and, given non-convexity, no correction to the algorithm alone could restore it without also changing the model class.

**Empirical corroboration.** All 60 seed×γ combined-fix cells in the completed Stage B campaign (10 seeds × 3 γ × 2 datasets) show `Σw = 1.0000` (to float32 precision) and `min(w) ≥ 0` — the feasibility claim (part 1 above) holding with **zero measured exceptions**, exactly as a structural (not probabilistic) guarantee predicts. This was the pre-registered, falsifiable prediction the campaign was designed to test (feasibility, not a specific blowup value), and it held at full scale: the pattern already visible at n=3 (9 CIC + 9 UNSW seed×γ runs) reproduced identically at n=10 (30 + 30). Composite-loss blowup on UNSW-NB15 at γ≥0.5 lands in the same reduced-but-not-exactly-`1.0×` range predicted from the n=3 data (up to `1.87×`, against the literal algorithm's `46.85×` worst case over the same 10 seeds); CIC-IDS-2017 stays essentially exactly `1.0×` (29/30 cells, one at `1.016×`).

**Rigor flag — what remains for the PhD mentor / a from-scratch proof pass.** This proposition states the correct qualitative guarantee (stationarity, not global optimality) and sketches the standard argument that delivers it, but three things are intentionally left for hardening rather than claimed here: **(a)** an exact, estimated `β_eff` for the paper's specific tanh-MLP architecture (needed to state a concrete valid `η'` range rather than "small enough"); **(b)** a fully formal treatment of the `min(L_h, L_m)` sub-differential selection rule as a measure-theoretic statement rather than the standard-practice sketch in step 3 above; **(c)** an explicit, worked-out `O(1/T)` rate constant rather than the asymptotic order. None of these affect the qualitative claim (feasibility is structural; stationarity replaces global optimality); all three are exactly the kind of thing a from-scratch, referee-grade proof pass would tighten, and are flagged here rather than silently assumed away — consistent with this project's standing no-silent-patching mandate.

---

## 5. The consolidated ablation table

The full 2×2 factorial — literal / projection-alone / η′-alone (canonical, η′=0.01, E′=5) / both — plus FedAvg and Static Reparam. reference rows, per γ ∈ {0, 0.5, 1}, on both datasets, **N = 10 seeds per cell**, mean + bootstrap 95% CI (B=10,000, fixed RNG seed), with the 5 pre-registered pairwise comparisons (both vs. each other arm: paired *t* + Wilcoxon signed-rank, Cohen's *d*, Holm–Bonferroni-corrected across the 5 comparisons within each dataset×γ) and the factorial main-effect/interaction read. Generated by:

```bash
python scripts/run_ablation_seed_sweep.py --dataset CIC-IDS-2017 --seeds 0-9 --workers 3
python scripts/run_ablation_seed_sweep.py --dataset UNSW-NB15   --seeds 0-9 --workers 3
python scripts/run_seed_sweep.py --dataset CIC-IDS-2017 --method fedavg --seeds 0 1 2 3 4 5 6 7 8 9 --rounds 60 --eval-every 1 --out results/ablation/seed_sweep_CIC_IDS_2017_fedavg_r60.json
# ... (fedavg/static x 2 datasets — full list in results/ablation/README.md)
python scripts/build_ablation_table.py --dataset CIC-IDS-2017
python scripts/build_ablation_table.py --dataset UNSW-NB15
python scripts/render_ablation.py      --dataset CIC-IDS-2017
python scripts/render_ablation.py      --dataset UNSW-NB15
```

Raw data: `results/ablation/{CIC-IDS-2017,UNSW-NB15}_ablation_table.json`. Figures: `results/ablation/{CIC-IDS-2017,UNSW-NB15}_ablation.png` (grouped bar chart, rare-class accuracy by arm × γ, with 95% CI error bars).

### 5.1 CIC-IDS-2017

![CIC-IDS-2017 ablation — rare-class accuracy by arm × γ](../../results/ablation/CIC_IDS_2017_ablation.png)

Rounds = 60. Metric of record = rare-class accuracy (mean per-class recall over the protected classes), reported as mean [95% bootstrap CI] over the per-seed values persisted in the same JSON. All-class accuracy is a secondary column. Significance markers on a reference-arm cell denote the paired, Holm–Bonferroni-corrected comparison of **both** vs. that arm at the same γ (`* p_holm<.05  ** p_holm<.01  *** p_holm<.001`); exact p-values and effect sizes are in the JSON's `comparisons` block, never only a star here.

**Composite-loss blowup is reported as median (and max), not mean** — it is a heavy-tailed *ratio*, not a bounded score, and on the literal arm a single diverging seed can exceed the other nine by six orders of magnitude (CIC γ=0 per-seed blowups: `[3.78, 7.59, 29.76, 2.58, 3.62, 3.59, 2.47, 3.14, 1857580.09, 3.87]` — mean 185764×, median 3.70×). A mean there describes no run that actually occurred; the median describes eight of ten, and the max is the striking part, surfaced rather than averaged away. Per-seed arrays and the mean are both retained in the JSON.

| Arm | γ=0.0 | γ=0.5 | γ=1.0 |
|---|---|---|---|
| FedAvg (baseline) | 0.0% [0.0, 0.0] · all 55.7% | 0.0% [0.0, 0.0] · all 55.7% | 0.0% [0.0, 0.0] · all 55.7% |
| Static Reparam. (baseline) | 84.4% [77.2, 91.4] · all 47.7% | 84.4% [77.2, 91.4] · all 47.7% | 84.4% [77.2, 91.4] · all 47.7% |
| Literal (η′=0.1, no proj.) | 0.0% [0.0, 0.0]*** · all 48.9% · blowup 3.70× med, **1.86×10⁶× max** | 0.0% [0.0, 0.0]*** · all 26.7% · blowup 2.71× med, 9.92× max | 10.0% [0.0, 30.0]*** · all 32.9% · blowup 14.19× med, 192.99× max |
| Projection-alone (η′=0.1) | 0.0% [0.0, 0.0]*** · all 59.8% · blowup 1.08× med, 1.18× max | 0.0% [0.0, 0.0]*** · all 54.8% · blowup 2.90× med, 3.91× max | 51.0% [24.2, 77.1] · all 48.4% · blowup 2.21× med, 5.60× max |
| η′-alone (η′=0.01, no proj.) | 27.5% [12.5, 43.3]** · all 71.7% · blowup 1.00× (med) | 78.5% [62.6, 92.2] · all 44.1% · blowup 1.00× (med) | 88.9% [68.9, 100.0] · all 15.2% · blowup 1.00× med, 30.02× max |
| **Both** (proj. + η′=0.01) — the fix | **69.7% [61.1, 78.9]** · all 62.3% · blowup 1.00× (med) | **81.1% [66.4, 92.2]** · all 54.2% · blowup 1.00× (med) | **89.3% [83.6, 94.6]** · all 44.3% · blowup 1.00× (med) |

The blowup column carries an independent finding the accuracy columns cannot show: **only the `both` arm has a flat loss trajectory at every γ** (median 1.00× and max 1.00× — i.e. *every* seed flat), while `projection-alone` and `η′-alone` each still admit substantial excursions somewhere in the γ range (up to 5.60× and 30.02× respectively). Feasibility is what makes the *unbounded* 1.86×10⁶ excursion structurally impossible; the η′ cut is what additionally flattens the bounded residual fluctuation.

**Pairwise comparisons (both vs. reference arm), Holm-corrected within each metric family.** A tie on the metric of record can hide a large difference in *overall* accuracy — an arm that matches `both` on rare-class accuracy while collapsing all-class accuracy has not matched `both` — so both metric families are tested on the same seeds and rows where they disagree are flagged. Positive Cohen's *d* favours `both`; negative favours the reference arm.

| vs. | γ=0.0 rare / all | γ=0.5 rare / all | γ=1.0 rare / all |
|---|---|---|---|
| Literal | 7.97e-07 *** / 0.142 n.s. | 6.17e-06 *** / 0.0031 ** | 1.54e-04 *** / 0.101 n.s. |
| Projection-alone | 7.97e-07 *** / 0.146 n.s. | 6.17e-06 *** / 0.555 n.s. | 0.0628 n.s. / **0.0424 \* (d=−0.88)** |
| η′-alone | 0.00775 ** / 0.0521 n.s. | 1.0 n.s. / 0.285 n.s. | 0.9696 n.s. / **5.62e-04 \*\*\* (d=+1.99)** |
| FedAvg | 7.97e-07 *** / 0.0178 * | 6.17e-06 *** / 0.378 n.s. | 1.28e-09 *** / 1.26e-04 *** |
| Static Reparam. | 0.00775 ** (d=−1.22) / 1.26e-06 *** | 1.0 n.s. / 5.61e-06 *** | 0.0890 n.s. / 0.0163 * (d=−1.15) |

**2×2 factorial read (rare-class accuracy, pp = percentage points):**

| γ | Projection main effect | η′ main effect | Interaction |
|---|---|---|---|
| 0.0 | +21.1pp [11.0, 31.7] | **+48.6pp [41.1, 56.0]** | +42.2pp [21.9, 63.3] |
| 0.5 | +1.3pp [−9.6, 10.9] (n.s.) | **+79.8pp [70.1, 89.3]** | +2.6pp [−19.2, 21.8] (n.s.) |
| 1.0 | +20.7pp [5.3, 35.7] | **+58.6pp [41.5, 75.0]** | −40.6pp [−75.8, −0.5] |

**On CIC-IDS-2017 the η′ cut dominates the rare-class-accuracy factorial — but reading only that metric would badly mislead.** The η′ main effect on rare-acc (+48.6 to +79.8pp) does dwarf projection's (+1.3 to +21.1pp, n.s. at γ=0.5), and `η′-alone` ties `both` on rare-acc at γ=0.5 and γ=1.0 (Holm p = 1.0 and 0.97). At γ=0.5 that tie is genuine on *both* metrics (all-acc Holm p = 0.285) — there, the η′ cut really does carry the arm. **At γ=1.0 it is not**: `η′-alone` matches `both` on rare-acc (88.9% vs 89.3%) while sitting at **15.2% all-class accuracy against `both`'s 44.3%** — Holm p = 5.6×10⁻⁴, Cohen's *d* = +1.99 in `both`'s favour. That is the same over-correction pathology §6 documents for η′=0.001, appearing at η′=0.01 once γ=1 removes any counterweight from `L_a`: the arm buys rare-class recall by starving every frequent class. An ablation reporting only the metric of record would have scored that arm as equivalent to the fix.

The trade-off runs the other way too, and is worth stating plainly rather than buried: at γ=1.0 `projection-alone` is *significantly better than `both` on all-class accuracy* (48.4% vs 44.3%, Holm p = 0.042, *d* = −0.88) while being far worse on rare-acc (51.0% vs 89.3%, though only marginally significant after correction at n=10). **No single arm dominates on both metrics at every γ.** What distinguishes `both` is that it is never *catastrophically* wrong on either metric anywhere — the trade-offs against it are single-digit percentage points on all-acc, while the trade-offs against the single-lever arms reach 29pp (η′-alone's all-acc at γ=1) and 38pp (projection-alone's rare-acc at γ=1).

### 5.2 UNSW-NB15

![UNSW-NB15 ablation — rare-class accuracy by arm × γ](../../results/ablation/UNSW_NB15_ablation.png)

| Arm | γ=0.0 | γ=0.5 | γ=1.0 |
|---|---|---|---|
| FedAvg (baseline) | 0.0% [0.0, 0.0] · all 55.8% | 0.0% [0.0, 0.0] · all 55.8% | 0.0% [0.0, 0.0] · all 55.8% |
| Static Reparam. (baseline) | 48.3% [40.8, 56.0] · all 52.7% | 48.3% [40.8, 56.0] · all 52.7% | 48.3% [40.8, 56.0] · all 52.7% |
| Literal (η′=0.1, no proj.) | 0.1% [0.0, 0.3] · all 54.9% · blowup 5.10× med, 28.16× max | 26.9% [16.8, 36.2] · all 49.6% · blowup 1.50× med, 2.23× max | 27.6% [15.0, 39.9]** · all 29.1% · blowup 8.26× med, 46.85× max |
| Projection-alone (η′=0.1) | 6.4% [0.0, 16.0] · all 59.4% · blowup 1.00× (med) | 47.7% [35.0, 57.6] · all 53.6% · blowup 1.03× med, 1.32× max | **67.2% [59.6, 73.6]** · all 51.8% · blowup 1.36× med, 2.06× max |
| η′-alone (η′=0.01, no proj.) | 3.4% [0.4, 7.8] · all 59.7% · blowup 1.00× (med) | 12.1% [5.2, 19.6]* · all 49.2% · blowup 1.00× med, 1.06× max | 36.4% [33.7, 39.8]*** · all 13.3% · blowup 1.19× med, 1.48× max |
| **Both** (proj. + η′=0.01) — the fix | 1.7% [0.0, 4.6] · all 58.8% · blowup 1.00× (med) | **37.1% [28.4, 45.7]** · all 53.2% · blowup 1.03× med, 1.39× max | **69.2% [63.4, 73.8]** · all 52.2% · blowup 1.32× med, 1.87× max |

**Pairwise comparisons (both vs. reference arm), Holm-corrected within each metric family** (same dual-metric convention as §5.1; positive *d* favours `both`):

| vs. | γ=0.0 rare / all | γ=0.5 rare / all | γ=1.0 rare / all |
|---|---|---|---|
| Literal | 0.606 n.s. / 0.448 n.s. | 0.418 n.s. / 0.121 n.s. | 0.00221 ** / 0.0185 * |
| Projection-alone | 0.606 n.s. / 0.448 n.s. | 0.418 n.s. / 1.0 n.s. | 0.544 n.s. / **0.0188 \* (d=+1.12)** |
| η′-alone | 0.371 n.s. / 0.448 n.s. | 0.0190 * / 0.0734 n.s. | 1.06e-04 *** / 1.61e-05 *** |
| FedAvg | 0.606 n.s. / 0.448 n.s. | 1.16e-04 *** / 0.217 n.s. | 8.87e-09 *** / 0.0188 * |
| Static Reparam. | 3.22e-06 *** (d=−3.88) / 0.168 n.s. | 0.311 n.s. / 1.0 n.s. | 0.00630 ** / 0.555 n.s. |

**2×2 factorial read (rare-class accuracy, pp):**

| γ | Projection main effect | η′ main effect | Interaction |
|---|---|---|---|
| 0.0 | +2.3pp [−1.0, 6.6] (n.s.) | −0.8pp [−4.5, 1.8] (n.s.) | −8.0pp [−17.3, −0.7] |
| 0.5 | **+22.9pp [12.6, 32.5]** | −12.7pp [−22.1, −1.1] | +4.3pp [−10.3, 18.1] (n.s.) |
| 1.0 | **+36.2pp [25.3, 46.3]** | +5.4pp [−1.5, 12.5] (n.s.) | −6.8pp [−18.9, 6.4] (n.s.) |

**On UNSW-NB15, the finding reverses: projection does the great majority of the work.** Its main effect (+22.9 to +36.2pp at γ≥0.5) dominates η′'s (which is *negative or negligible* at every γ — the η′ cut alone does not help this dataset). `projection-alone` is statistically indistinguishable from `both` **on rare-class accuracy at every** γ (Holm p = 0.606, 0.418, 0.544) — adding the η′ cut on top of projection buys no significant additional rare-class accuracy here, and at γ=0.0 the point estimate is even slightly lower (1.7% vs. 6.4%, CIs overlapping heavily, p_holm=0.606).

**The dual-metric check qualifies that at γ=1.0**, though far less dramatically than the mirror-image case on CIC: there `both` *is* significantly better than `projection-alone` on all-class accuracy (52.2% vs 51.8%, Holm p = 0.019, *d* = +1.12). The effect is statistically clean but only **0.4 percentage points** — a consistent per-seed edge rather than a materially different model, and nothing like the 29pp all-acc gap the same check exposes on CIC. The honest reading is that on UNSW the η′ cut is close to free: it neither helps rare-class accuracy nor hurts, and buys a small but reliable all-class improvement at γ=1.

### 5.3 Cross-dataset synthesis: which lever "does the work" is dataset-dependent

The Stage B factorial replaces the earlier, weaker preview-data hypothesis ("both levers are always independently necessary") with a sharper and more honest finding, in three parts.

**(1) The projection fix is the one structurally necessary lever, and it is dataset-independent.** Feasibility is restored with zero exceptions across all 60 `both`-arm cells (§4) on both datasets. This is the part that carries the theory.

**(2) Which lever drives *rare-class accuracy* recovery is dataset-dependent, and the two datasets are near-mirror-images.** On CIC-IDS-2017 the η′ cut alone recovers 27.5–88.9% rare-class accuracy, tying `both` at γ=0.5–1.0; projection contributes a smaller boost, mainly at low γ. On UNSW-NB15 the pattern flips: `projection-alone` ties `both` on rare-acc at every γ while the η′ cut contributes nothing significant (trending negative at γ=0.5). Neither single-lever arm transfers: `η′-alone` is far worse than the alternatives on UNSW (3.4–36.4% vs. 67–69%), and `projection-alone` is far worse on CIC at γ≤0.5 (0.0% vs. 70–89%).

**(3) Two of those "ties" dissolve once all-class accuracy is tested — and this is the finding that most changes the recommendation.** Reading only the metric of record, one could conclude that on CIC at γ=1 the projection lever is dispensable (`η′-alone` ties `both`, p=0.97). Testing the same seeds on all-class accuracy shows why that conclusion would be wrong: `η′-alone` reaches its rare-class parity while sitting at **15.2% all-class accuracy against `both`'s 44.3%** (Holm p = 5.6×10⁻⁴, *d* = +1.99). The arm is not equivalent to the fix; it has traded the majority classes away to buy rare-class recall — the same pathology §6 documents for the η′=0.001 over-correction, reappearing at η′=0.01 when γ=1 removes `L_a`'s counterweight. The mirror check on UNSW at γ=1 flags the same *kind* of divergence in `both`'s favour but at a trivial magnitude (0.4pp), which is itself informative: the dual-metric test is not simply generating flags everywhere, it is isolating the one place where a headline tie is hollow.

**What this licenses, and what it does not.** No arm dominates on both metrics at every γ — at CIC γ=1 `projection-alone` is genuinely better than `both` on all-class accuracy (48.4% vs 44.3%, Holm p = 0.042), and Static Reparam. beats `both` on rare-acc at γ=0 on both datasets. The defensible claim is narrower and, stated precisely, more useful: **`both` is the only factorial arm that is never *catastrophically* wrong on either metric, on either dataset, at any γ.** Every deficit against it is a few percentage points; the deficits against the single-lever arms reach 29pp (η′-alone's all-acc, CIC γ=1) and 38pp (projection-alone's rare-acc, CIC γ=1). For anyone applying this correction without per-dataset, per-γ retuning, that worst-case profile — not a win on any single metric — is the reason to use both levers.

---

## 6. Discussion

**Severity is dataset-dependent; the mechanism is not.** The feasibility-set defect fires at round 1 on both datasets, at every γ, with an identical qualitative signature (negative weight components, `Σw ≠ 1`, non-monotone `L'`) — confirmed at `first_simplex_violation_round = 1` on **all 60 literal-arm cells** of the 10-seed campaign (10 seeds × 3 γ × 2 datasets), not just at seed 0. Its *consequences*, however, differ by orders of magnitude, and the 10-seed data makes that gap far wider than the seed-0 figures suggested: over the full seed set CIC-IDS-2017's worst-case blowup is **1.86×10⁶×** (γ=0, seed 8) against UNSW-NB15's worst case of **46.85×** (γ=1) — roughly four orders of magnitude apart, where the seed-0-only figures (193× vs. 7.6×) suggested about one. The median blowups are far closer (CIC 2.71–14.19× vs. UNSW 1.50–8.26× depending on γ), so the distinguishing feature is not the typical run but **how catastrophic CIC's tail is**: the same defect that costs UNSW a factor of a few costs CIC a factor of a million on an unlucky seed. The most plausible explanation is client-skew severity: CIC's three rarest classes (Heartbleed=11, SQL Injection=21, Infiltration=36 samples) are far more extreme than UNSW's rarest three (Worms=174, Shellcode=1511, Backdoor=2329) — unconstrained extrapolation in weight-space is more punishing the more disparate the client models being (mis-)combined are. Rare-class accuracy does not uniformly collapse on UNSW either: at γ=0.5, despite `Σw` drifting to −0.731 on seed 0 (numerically a *larger* simplex violation than anything measured on CIC), literal FedGRAA still reaches genuinely nonzero rare-class accuracy — and the 10-seed data shows just how unreliably: per-seed values are `[29.6, 42.8, 0.4, 43.3, 44.0, 38.5, 3.4, 31.0, 24.2, 11.8]%` (mean 26.9%, 95% CI [16.8, 36.2]). A spread from 0.4% to 44.0% on nothing but a seed change is the sharpest available statement of the problem: the literal algorithm is not uniformly broken on UNSW, it is *unpredictable*, which for an intrusion-detection deployment is arguably the worse failure mode. The honest one-line summary is **"FedGRAA reproduces on some real IDS data, sometimes, unreliably,"** not the CIC-only headline of a clean, total 0%.

**The η′=0.001 over-correction — documented, not part of the factorial.** An earlier exploratory run (`CIC_IDS_2017_literal_real_smalleta_r60.json`, η′=0.001, E′=1, no projection) shows that shrinking η′ alone, without projection, *can* recover rare-class accuracy (75–100% across most of a 60-round run) — but does so by starving every non-rare client: with nothing pushing back against overweighting the tiny rare-class validation signal, all-class accuracy collapses to **0.8%**. This is retained as a documented secondary row (stage_b_plan.md §2 caveat (a)) precisely because it is instructive: it demonstrates that a naive "just shrink η′ a lot" fix trades one failure mode (blown-up loss) for another (starved majority-class performance), and it is *not* the same intervention as the factorial's canonical η′-alone cell (η′=0.01, E′=5 — a 10× reduction, not 100×). The factorial's whole point is to isolate whether the *moderate*, principled η′ cut and the projection are each independently necessary — the 0.001 row is a cautionary data point about what happens if the η′ cut is too aggressive, not a competing fix.

**Relationship to the paper's own stated hyperparameters.** The paper's own default is `η′=0.1, E′=5` — exactly the configuration §2 shows fails. The sensitivity sweep on synthetic data (`STAGE_A_MASTER_REPORT.md` §5.6) shows this same configuration is monotone and reaches 100% rare-class accuracy *on cleanly-separable synthetic data*, because `L_u` collapses to ≈0 quickly enough there that gradients vanish before `w` can drift far — a reminder that a hyperparameter's apparent safety on an easy synthetic benchmark does not transfer to real, skewed data, and that the failure mode this document formalizes is a property of the *interaction* between `η′=0.1` and real intrusion-detection data's client-size skew, not of either alone.

**What the fix changes, and what it does not.** The combined fix (projection + η′=0.01, E′ unchanged) is, by design, the *minimal* correction implied by §4's proposition — it adds exactly the missing constraint (§4 point 1) and reduces the step size only as far as needed to keep the projected iterates well-behaved in practice. Whether the η′ cut is doing that stabilizing work or the rare-class-accuracy-recovery work varies by dataset (§5.3): on CIC-IDS-2017, `projection-alone` (η′ still at the paper's own 0.1) stays at 0.0% rare-class accuracy at γ≤0.5 and only partially recovers at γ=1.0 (51.0%) — projection alone eliminates the loss blowup (§2's mechanism) but is not, by itself, reliably enough to also fix rare-class accuracy at this dataset's default η′. On UNSW-NB15 the same `projection-alone` arm is already substantially higher throughout (6.4–67.2%, statistically tying `both` at every γ, §5.2) — η′=0.1 is evidently less destabilizing there. The combined fix is the one choice that performs well on both, which is why it — not either single-lever arm — is the recommendation. It does **not** retroactively validate the paper's literal `η′=0.1` claim, and — per this project's standing no-silent-patching mandate — it does **not** change the strict-reproduction default in `fedgraa/`: `ProjectedFedGRAA` remains a diagnostic-only class defined solely in `scripts/diagnose_fedgraa.py`, never imported by the reproduction package.

---

## 7. Honest limitations

- **Stationarity, not global optimality.** §4's proposition explicitly claims less than the paper's Theorem 1. This is presented as the *correct* guarantee for a non-convex loss, not as a lesser consolation prize — but it means this document does not, and cannot without changing the model class, restore the paper's original claim.
- **The fix is diagnostic-only.** `ProjectedFedGRAA` exists solely in `scripts/diagnose_fedgraa.py` for exactly this analysis. The strict-reproduction package (`fedgraa/`) implements Algorithm 2 literally, unprojected, on purpose — changing that default would stop being a reproduction.
- **N = 10 seeds is still a small sample, and the Wilcoxon test is at its resolution floor.** The pre-registered comparisons (§5) use both a parametric (paired *t*) and non-parametric (Wilcoxon) test specifically because n=10 is too small to lean on asymptotic normality alone; effect sizes and CIs are reported alongside every p-value for the same reason. Several comparisons in §5 are not statistically significant after Holm correction despite visually large point-estimate gaps (e.g. CIC γ=1.0 `both` vs. `projection-alone`, p_holm=0.063, d=0.88) — this is good practice for n=10, not a substitute for a larger campaign a later stage could run, and readers should weight the "not significant" comparisons accordingly rather than reading a large Cohen's *d* alone as proof. Note also that the many Wilcoxon p-values reported as exactly `0.001953` are **at the exact test's floor** for n=10 (= 2/2¹⁰, the smallest two-sided p attainable when all 10 paired differences share a sign); they mean "as significant as 10 seeds can demonstrate," not "p = 0.002 precisely," and no Wilcoxon result at this sample size can go below it however large the true effect.
- **CPU-only training.** Every number in this project, Stage A and Stage B alike, comes from CPU-trained runs (no GPU was used or available). This affects wall-clock cost, not correctness, but is disclosed for completeness.
- **The `min(L_h, L_m)` sub-gradient treatment (§4 proof sketch step 3) is a standard-practice sketch, not a from-scratch formal derivation.** Flagged explicitly in §4's "rigor flag" paragraph as one of three items intended for hardening with the PhD mentor, not claimed as fully rigorous here.
- **Which lever drives rare-class-accuracy recovery is dataset-dependent (§5.3), and only two datasets were tested.** The projection fix is structurally necessary and dataset-independent (feasibility, §4); which of projection or the η′ cut does most of the *accuracy recovery* work reversed completely between CIC-IDS-2017 and UNSW-NB15. With only two datasets, it is not possible to say which pattern (if either) is more typical of real IDS data in general — a third dataset could match either, or neither. The combined fix is recommended as the safe default precisely because of this uncertainty, not because both levers were shown to be independently necessary on every dataset tested.
- **A second, independent deviation from Algorithm 1 exists alongside the simplex-projection defect: client-side Adam momentum persists across all 200 rounds while model weights reset every round.** `fedgraa/trainer.py`'s `client_train_round` resets each client's *model* parameters to the current global aggregate every round (as Algorithm 1 specifies), but never rebuilds that client's Adam optimizer, so its momentum/velocity slots carry state across rounds whose starting weights have since changed underneath them. Already identified and deliberately kept as "a known deviation, not a bug" (`docs/archive/v1_regression_root_cause_analysis.md` §"root cause", lines 245-247) — recorded here because, until this note, it was not mentioned in any of this project's current theory/reproduction docs, so a reader relying on this document alone had no way to know it exists alongside the headline defect above. It is orthogonal to the simplex-projection fix (§4) and not corrected by it.
- **This is a critique of one paper's one algorithm, not a general claim about FL aggregation.** Nothing here suggests learned-aggregation-weight schemes are unworkable in general — only that *this* algorithm needs *this* specific, identified, and now-corrected constraint to behave as its own convergence theorem claims.
- **The target manuscript's publication status is unresolved.** The escalation path this project's own literature docs mandate (DBLP, Semantic Scholar) has now been run: DBLP's search API returns a confirmed, machine-checkable zero (`"@total":"0"`) for the manuscript's title; Semantic Scholar's own API was rate-limited on every attempt, but a web search of its index and the general literature likewise found no trace. Four independent sources now agree, one of them a confirmed negative rather than a strong one — but this still does not conclusively establish non-publication, and the branch that actually matters (is this the lab's own submission, or a third party's?) is not something any search resolves. Nothing in §1–§6 depends on the answer — the algorithm's behaviour on real data is what it is, independent of where or whether it appeared in print. What does depend on it is whether this section can stand as a standalone citable critique, versus becoming internal validation and a Method-section ingredient for Stage C. See [`STAGE_B_MASTER_REPORT.md`](STAGE_B_MASTER_REPORT.md) §21 for the full search record, branch analysis, and mentor-conversation status.
