# FedGRAA — Line-by-Line Paper ↔ Code Gap Analysis (Base Paper Only)

**Paper:** *"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection
Systems"* — `PAKDD_Final_Version_Submit_v3.md` **only**.

> **⚠️ ARCHIVED — superseded by `../audit/REPRODUCTION_AUDIT.md`.** Written
> 2026-07-09 against a codebase state that has since changed (e.g. the
> `reading="A"/"B"` ambiguity toggle referenced below has been removed — see
> `strict_reproduction_report.md`). Kept for historical record.

This document
deliberately ignores `privacy_compliant_fedgraa_redesign.md` (the "FedGRAA-v2"
redesign), every other related-work paper cited in §2/References, and every
piece of code that only exists to support v2 (secure aggregation, differential
privacy, non-oracle rare-class discovery, etc.). Scope is strictly: **does the
code in `fedgraa/` reproduce, line for line, what the base paper says?**

**Codebase root:** `PAKDD/fedgraa/`
**Date of this audit:** 2026-07-09
**Method:** every equation (1–10), every line of Algorithm 1 and Algorithm 2,
every sentence of §5 ("Experiments and Results"), and every convergence-proof
assumption in Appendix A.1 was matched against the exact function/line in the
code that is supposed to implement it. Anything the paper does not fully
specify, anything the code does differently, and anything the code adds that
the paper never asked for is logged below as a **gap**.

This supersedes nothing — `reproduction_gaps.md` and `full_reproduction_gap_analysis.md`
already exist and track the *history* of fixes. This document is a from-scratch,
present-day audit of the **current** code state against the **base paper only**,
organized by paper structure rather than by fix history, so every remaining
(and newly introduced) gap can be seen in one place.

> **2026-07-09, second pass ("final grilling").** The instruction for this
> pass was explicit: stop treating the code's choices as acceptable just
> because they're "reasonable." The goal is a **100% ditto** reproduction of
> the base paper — every place the code currently does something the paper
> doesn't literally say is now called out as something that **must change**,
> not as a defensible interpretation. See the verdict list immediately below,
> and Part 13 for issues found only in this second, harder pass.

---

## FINAL VERDICT — Non-negotiable requirements for a 100% ditto reproduction

No hedging. If the goal is a literal, 1-to-1 copy of what Algorithm 1,
Algorithm 2, and §5 say — not "a faithful-in-spirit implementation" — then
the current code must change in the following ways. Each item points to the
full discussion later in this document.

| # | Required change | Why | Discussed in |
|---|---|---|---|
| 1 | **Delete `_project_simplex` from the default path.** Algorithm 2 has 8 lines and ends at `return w^t`. There is no line 9. No projection, no clipping, no renormalization. If `w` goes negative or stops summing to 1 after the Adam/GD step, **that is what the paper's algorithm produces** — a ditto reproduction must let that happen, not silently patch it. | Paper has zero textual basis for this step | §4.3 |
| 2 | **Default `round_zero_zero` to `True`.** The paper's own Figure 1 table prints exactly `0.00` for every method, every γ, both datasets, at round 0. A ditto reproduction's round-0 output must be `0.00`, full stop — not ≈6.7%/≈10% "real" random accuracy. Whatever mechanism is required to force this is what "ditto" demands. | Directly contradicts the paper's own published numbers otherwise | §11.1 |
| 3 | **Pick one optimizer for Eq. 1/Alg.1-L17 and Eq. 10/Alg.2-L6 and stop offering both as if it were a free choice.** The paper's algorithm boxes say plain gradient descent (`θ -= η∇L`, `w -= η'∇L'`); §5's prose says Adam. These cannot both be "the paper." A ditto reproduction has to declare, explicitly and permanently, which one it is reproducing — currently the code has silently picked Adam everywhere without ever surfacing this as the two-sided coin it is. Recommendation if forced to pick one: Adam, because §5 is unambiguous, specific, and gives exact hyperparameters (η, η′) that only make sense for Adam's bias-corrected update rule — but this must be stated as a forced editorial decision, not "a match." | Alg. 1/2 boxes contradict §5's own prose | §0.1, §0.3 |
| 4 | **Remove `fedgraa_reading="A"` and the `"mean"`/`"smooth"` subgradient options entirely, or fence them off from anything called "reproduction."** These are not alternate valid readings of the paper — they are inventions with zero textual basis, sitting in the same class/config as the paper-mandated defaults, one config flag away from silently producing numbers that have nothing to do with the paper. A ditto reproduction codebase should not contain code paths the paper never describes at all, reachable through the same public API as the paper-faithful path. | Paper gives exactly one algorithm, not a family of three | §4.1, §4.2 |
| 5 | **Decide, explicitly and irreversibly, what "accuracy on the last n classes" means** (`metrics.py::rare_metric`) — mean-per-class recall or pooled accuracy — and delete the other option. The paper uses this exact phrase throughout §5 and Figure 1/2/3 and never once defines it mathematically. A ditto reproduction cannot leave this as a runtime toggle; it must commit to one reading everywhere numbers are reported. | Paper's central reported metric has no formula anywhere in the text | §11 (metrics) |
| 6 | **Decide, explicitly, the feature-scaling and frequent-class-capping strategy** (`scale_strategy`, `cap_strategy` in `config.py`/`data.py`) and delete the alternates. The paper's Data Settings paragraph says nothing about scaling at all and only says "capped at 10,000" with no method. Ditto reproduction cannot be a matrix of options; pick global-fit + global-concat-then-cap (closest to "the naive thing a paper wouldn't bother mentioning") and remove the rest. | Paper is silent; silence + config matrix = not ditto | §5.1, §5.2 |
| 7 | **Acknowledge in-repo, permanently, that UNSW-NB15 and Keras/float32 cannot be reproduced at all right now** — no amount of further "interpretation" work touches this. Either obtain the real UNSW-NB15 CSVs and switch the default model backend to `keras_backend.py` in float32, or stop describing any UNSW/Figure-3 number produced by this repo as a "reproduction" of anything. | Structural, not a code-quality issue | Part 12 |

The rest of this document is the full line-by-line evidence trail behind every one of these seven verdicts, plus everything else found along the way.

---

## Severity legend

| Severity | Meaning |
|---|---|
| **BLOCKING** | No amount of code changes can close this gap; it requires data/artifacts the paper never released |
| **INTERPRETATION** | The paper's text is genuinely ambiguous/self-contradictory here; the code makes a specific choice that is *not forced* by the paper |
| **DEVIATION** | The code does something concrete that the paper's text does not describe at all (i.e., code adds or omits a step) |
| **VERIFIED MATCH** | Line-by-line check confirms the code implements exactly what the paper states |

---

## Part 0 — Paper-internal inconsistencies (before any code is even considered)

These are contradictions *inside the paper itself*. They matter here because
whichever side the code picks is automatically a "gap" relative to the other
side of the paper.

### 0.1 Eq. 1 / Algorithm 1 line 17 describe plain SGD; §5 "Training Setup" says Adam

**Paper Eq. 1:** $\theta_k^{t+1} = \theta_k^{t+1} - \eta \nabla L_{\theta_k^{t+1}}(\theta_k^{t+1}; D_k)$
**Algorithm 1, line 17:** `θᵗ_k ← θᵗ_k − η∇L_{θᵗ_k}(θᵗ_k; b)`

Both are literally vanilla (mini-batch) gradient descent: `param -= lr * grad`.
Nothing here mentions momentum or adaptive learning rates.

**§5 "Training Setup and Hyperparameters":** *"The model was trained using the
**Adam optimizer** with a learning rate of η = 0.01."*

These cannot both be literally true — Adam's update rule is not `param -= lr *
grad`. **The code sides with the prose (Adam), not the algorithm-box notation
(plain SGD).**

- `fedgraa/model.py` → `class Adam` (full Adam: β1=0.9, β2=0.999, ε=1e-7, bias-corrected moments).
- `fedgraa/trainer.py` → `_client_update()` line `optimizer.step(params, grads)` calls `Adam.step`, never plain SGD.

**Classification: INTERPRETATION.** Reasonable choice, but it means the code
does not literally implement Eq. 1 / Algorithm 1 line 17 as written — it
implements the prose description instead. A reproduction that took the
algorithm box literally (plain SGD, no momentum) would diverge from this
codebase after the very first local step.

### 0.2 Eq. 2 (uniform $1/K$ average) contradicts Eq. 5 (data-size-weighted average) — both are called "FedAvg"

**Eq. 2:** $\theta_g^{t+1} = \frac{1}{K}\sum_{k\in S_t}\theta_k^{t+1}$ — a plain, unweighted mean.
**Eq. 5:** $\theta_g^{t+1} = \sum_{k\in S_t}\frac{|D_k|}{|D|}\theta_k^{t+1}$ — the standard McMahan et al. data-size-weighted average.

These are different formulas whenever client dataset sizes differ (which they
always do in this experiment — attack classes range from 11 to 10,000
samples). The paper never reconciles them.

- `fedgraa/aggregation.py` → `fedavg_weights()` implements **Eq. 5** (`sizes / sizes.sum()`), never Eq. 2's uniform $1/K$.

**Classification: INTERPRETATION.** Eq. 5 is the historically correct FedAvg
formula (McMahan et al. 2017) and is almost certainly what the authors meant
in Eq. 2 as well (probably a typo/simplification), so the code's choice is
reasonable — but it is a choice, not a literal rendering of Eq. 2.

### 0.3 Eq. 10 / Algorithm 2 line 6 write plain gradient descent for the weight update; §5 says Adam again

**Eq. 10 / Alg. 2 line 6:** $w^{t-1} \leftarrow w^{t-1} - \eta' \nabla_{w^{t-1}} L'(\ldots)$ — again literally plain GD with rate $\eta'$.
**§5:** *"For FedGRAA, we continued using the **Adam optimizer**, with a learning rate η′ = 0.1 and epochs E′ = 5."*

Same contradiction as 0.1, one level up (the weight-learner instead of the
client model).

- `fedgraa/aggregation.py` → `class FedGRAA.__init__`: `self.optimizer = Adam(lr=eta_prime)`. `update()` calls `self.optimizer.step([w], [grad_w])`, i.e. full Adam on the 1-D weight vector, not `w -= eta_prime * grad`.

**Classification: INTERPRETATION.** Same reasoning as 0.1 — sides with prose
over algorithm-box notation, consistently. This is at least *self-consistent*
across the codebase (both client update and weight update use Adam), which is
worth noting as a positive: the code did not pick SGD in one place and Adam in
another.

---

## Part 1 — Section 3 (Background/Equations 1–6): line-by-line

| Paper element | Code location | Verdict |
|---|---|---|
| Eq. 1 (client SGD step) | `trainer.py::_client_update`, uses `Adam.step` per batch | INTERPRETATION (see 0.1) |
| Eq. 2 (uniform FedAvg) | Not implemented anywhere; `fedavg_weights` implements Eq. 5 instead | INTERPRETATION (see 0.2) — **Eq. 2 as literally written is never computed by the code** |
| Eq. 3 (local loss, mean CE over $D_k$) | `model.py::MLP.loss` / `loss_and_grad`: `-logp[arange(N), y].mean()` | VERIFIED MATCH |
| Eq. 4 (global loss = size-weighted sum of local losses) | **Never computed anywhere in the code.** No function evaluates $L(\theta)=\sum_k \frac{|D_k|}{|D|}L_k(\theta)$ as a standalone quantity; it exists only implicitly through the training/aggregation process. | DEVIATION (omission) — harmless since Eq. 4 is descriptive, not prescriptive, but strictly "100% reproduction" would compute and log this quantity for verification and the code never does |
| Eq. 5 (data-size-weighted FedAvg aggregation) | `aggregation.py::fedavg_weights` | VERIFIED MATCH |
| Eq. 6 (global gradient = size-weighted sum of local gradients) | **Never computed anywhere.** No function assembles $\nabla L(\theta)=\sum_k \frac{|D_k|}{|D|}\nabla L_k(\theta)$ as an explicit quantity (it is implicit in how `aggregate()` combines already-locally-optimized parameters, which is not the same mathematical object as this equation, since Eq. 6 is a gradient of a single θ, not a combination of already-descended $\theta_k$ values). | DEVIATION (omission), same caveat as Eq. 4 |

---

## Part 2 — Section 4, Static Reparameterization (Eq. 7)

**Eq. 7:** $\theta_g^{t+1} = \sum_{k=1}^K \dfrac{1/|D_k|}{\sum_j 1/|D_j|}\,\theta_k^t$

- `aggregation.py::static_reparam_weights(client_sizes)`:
  ```python
  inv = 1.0 / sizes
  return inv / inv.sum()
  ```
  This is exactly the weight vector in Eq. 7 (the $\theta_k^t$ multiplication
  happens separately in `model.py::aggregate`).

**Verdict: VERIFIED MATCH.**

One subtlety: Eq. 7 uses $\theta_k^t$ (i.e., the *previous* round's client
params before this round's local update) rather than $\theta_k^{t+1}$ used in
Eq. 5. The paper is inconsistent about the time index here as well, but since
both `static` and `fedavg` methods in the code aggregate the **freshly-updated**
`client_params` returned by `_client_update` in the current round (i.e. always
$\theta_k^{t}$ meaning "params after this round's local training", matching the
Algorithm-1 box's own use of $\theta_k^t$ in line 11), the code is consistent
with **Algorithm 1's** indexing, not with Eq. 7's literal $\theta_k^t$
(pre-update) naming. This is a cosmetic paper-internal notation clash, not a
code bug — flagged for completeness only.

---

## Part 3 — Algorithm 1 (Adaptive Reparameterization), line by line

| Alg. 1 line | Paper text | Code | Verdict |
|---|---|---|---|
| 3 | `initialize global model parameters θ⁰_g` | `trainer.py::train()`: `global_params = init_params(layer_sizes, seed=cfg.seed)` (Glorot-uniform, zero bias) | VERIFIED MATCH |
| 4 | `for each round t = 1 … T` | `for rnd in range(1, cfg.rounds + 1):` | VERIFIED MATCH |
| 5 | `Sₜ ← (fixed set of K clients)` | `self.split.clients` — **all** `K` clients, every round, no sampling. Confirmed: there is no client-subsampling code anywhere in `trainer.py`. | VERIFIED MATCH (paper says "fixed", i.e. full participation every round — code never samples a subset) |
| 6–8 | send `θᵗ⁻¹_g` to each client, client runs `CLIENTUPDATE` | `client_params = [_client_update(global_params, client, cfg, rng, self._client_optimizers[client.client_id]) for client in self.split.clients]` | VERIFIED MATCH (list comprehension = "in parallel" conceptually, executed sequentially — acceptable since the paper doesn't require literal concurrency, only the same math) |
| 9 | `θᵗ_c = [θᵗ_1,...,θᵗ_K]` | `client_params` list itself | VERIFIED MATCH |
| 10 | `wᵗ ← FedGRAA(θᵗ_c, wᵗ⁻¹)` | `w = self._round_weights(w, client_params)` → for `method="fedgraa"`, calls `self.fedgraa.update(client_params, w_prev, self.split.val_split)` | VERIFIED MATCH (only for the FedGRAA method — see note below) |
| 11 | `θᵗ_g = Σ wᵗ_k θᵗ_k` | `global_params = aggregate(client_params, w)` → `model.py::aggregate`: `np.tensordot(weights, stacked, axes=(0,0))` | VERIFIED MATCH |
| 12 | `return θᵀ_g` | `self.final_params = global_params; return history` | VERIFIED MATCH (return value is `TrainHistory`, not raw params, but `final_params` is stored as an attribute — functionally equivalent) |
| 14 | `initialize local model θᵗ_k ← θᵗ⁻¹_g` | `params: Params = [p.copy() for p in global_params]` | VERIFIED MATCH |
| 15 | `for each local epoch from 1 to E` | `for _ in range(cfg.local_epochs):` (default `E=2`) | VERIFIED MATCH |
| 16 | `for each batch b in Dₖ` | `for start in range(0, n, cfg.batch_size): idx = perm[start:start+cfg.batch_size]` (default `B=64`) | VERIFIED MATCH |
| 17 | `θᵗ_k ← θᵗ_k − η∇L(θᵗ_k; b)` | `optimizer.step(params, grads)` = **Adam**, not `params -= eta * grads` | INTERPRETATION (see 0.1) |
| 18 | `return θᵗ_k to server` | `return params` | VERIFIED MATCH |

**Note on line 10 for non-FedGRAA methods:** For `method="fedavg"` /
`"static"`, `_round_weights` **ignores** `w_prev` and `client_params`
entirely and just recomputes `fedavg_weights(self.client_sizes)` /
`static_reparam_weights(self.client_sizes)` from the fixed dataset sizes.
This is mathematically identical to "line 10" for those methods only because
`fedavg`/`static` weights never depend on `θᵗ_c` in the first place — so
Algorithm 1's generic `wᵗ ← FedGRAA(...)` line is really only literally
implemented for the `"fedgraa"` method; for the other two baselines the paper
never actually routes them through Algorithm 1's line 10 at all (Algorithm 1
as written is FedGRAA-specific; FedAvg/Static are separately defined by Eq. 5
/ Eq. 7 in the paper). **No gap** — this is just clarifying that Algorithm 1
line 10 literally only applies to the FedGRAA method, and the code's branching
(`if self.method == "fedavg": ...`) correctly reflects that.

---

## Part 4 — Algorithm 2 (FedGRAA weight learner), line by line

| Alg. 2 line | Paper text | Code (`aggregation.py::FedGRAA.update`, default `reading="B"`) | Verdict |
|---|---|---|---|
| 2 | `for each local epoch from 1 to E'` | `for _ in range(self.weight_epochs):` (default `E'=5`) | VERIFIED MATCH |
| 3 | `θ'_g = w1^{t-1}θ1^t + ... + wK^{t-1}θK^t` (**recomputed every epoch** using the currently-evolving `w`) | `P = aggregate(client_params, w)` — recomputed inside the loop body every iteration | VERIFIED MATCH (this is the "B"/iterative reading, which is the **default**; see 4.1 below) |
| 4 | `L_a = min(L_h(θ'_g; D_V^h), L_m(θ'_g; D_V^m))` | `_composite_grad_P`: computes `L_h, g_h = _loss_and_grad_P(P, val.X_h, val.y_h)`, `L_m, g_m = ...`, then `if L_h <= L_m: L_a, g_a = L_h, g_h else: L_a, g_a = L_m, g_m` (default `subgradient="min"`) | VERIFIED MATCH for the loss value; the *gradient* of a `min(·,·)` is not specified by the paper at all (see 4.2) |
| 5 | `L' = (1-γ)L_a(...) + γL_u(...)` | `loss = one_minus_g * L_a + self.gamma * L_u` | VERIFIED MATCH |
| 6 | `w^{t-1} ← w^{t-1} − η'∇_{w^{t-1}} L'(θ'_g;...)` | `grad_w = _grad_wrt_weights(grad_P, client_params)` (chain rule: $dL'/dw_k = \langle \nabla_{P}L', \theta_k\rangle$) then `self.optimizer.step([w], [grad_w])` = **Adam step**, not `w -= eta_prime * grad_w` | INTERPRETATION (see 0.3) — chain-rule math is exact, optimizer type is a text-vs-notation choice |
| 7 | `w^t ← w^{t-1}` | Implicit — `w` is updated in place across the `E'` loop iterations, and the final `w` after the loop *is* `w^t` | VERIFIED MATCH |
| 8 | `return w^t` | `w = _project_simplex(w); return w` | **DEVIATION** — see 4.3, the simplex projection is not in the paper at all |

### 4.1 `reading="A"` vs `"B"` — config surface not licensed by the paper

`FedConfig.fedgraa_reading` exposes two modes:
- `"B"` (default): recompute $\theta'_g$ and its gradient every one of the `E'`
  epochs — matches Algorithm 2 line 3 literally (it is inside the `for`
  loop in the pseudocode box).
- `"A"`: compute the gradient **once** at the start (using $w^{t-1}$ from the
  *previous round*), then apply that fixed gradient direction via `E'` Adam
  steps.

The default (`"B"`) is the correct literal reading of the algorithm box.
**Classification: DEVIATION (of the non-default path only).** The paper gives
no license for a `"static-gradient"` reading at all — it is a code invention
to explore an alternative interpretation. As long as `fedgraa_reading="B"`
(the default), this is not exercised, but its mere existence as a config knob
means the reproduction has a **second, non-paper-sanctioned code path** that a
future user could silently enable and get a materially different — and
unjustified-by-the-paper — training trajectory.

### 4.2 Sub-gradient of $L_a = \min(L_h, L_m)$ is not specified by the paper at all

The paper writes $L_a = \min(L_h, L_m)$ and then differentiates $L'$ (which
contains $L_a$) with respect to $w$ (Eq. 10) as if this were routinely
differentiable. `min(f,g)` is not differentiable at points where $f=g$, and
even where it is (sub)differentiable, the paper never states which branch's
gradient to use, nor whether to blend them.

`aggregation.py::FedGRAA._composite_grad_P` offers **three** strategies
(`subgradient` config field):
- `"min"` (default): gradient of whichever of $L_h,L_m$ is currently smaller — the textbook sub-gradient choice.
- `"mean"`: average of both branch gradients.
- `"smooth"`: a soft-min (log-sum-exp with temperature $\tau=0.1$) — smooth everywhere, but numerically **a different loss surface than $\min$** (soft-min $\ne$ hard-min except in the limit).

**Classification: INTERPRETATION** (for the default `"min"` branch — this is
the standard, unambiguous convention for sub-gradients of a min of two smooth
functions) **+ DEVIATION** (for the mere existence of `"mean"`/`"smooth"` as
alternate, paper-unsanctioned code paths, exactly analogous to 4.1).

### 4.3 Simplex projection is applied unconditionally — Algorithm 2 has no such step

`aggregation.py::_project_simplex(w)` (Euclidean projection onto the
probability simplex, Duchi et al. 2008) is called at line
`w = _project_simplex(w)` as the **very last operation before `return w`**,
**unconditionally**, for every round, regardless of any config flag.

Algorithm 2 as printed in the paper has exactly 8 lines, ending at line 7
(`w^t ← w^{t-1}`) and line 8 (`return w^t`) with **no projection, clipping, or
renormalization step of any kind**. The paper's Adam-based (or, per Eq. 10,
plain-GD-based) unconstrained update on $w$ has no built-in mechanism to keep
$w_k \ge 0$ or $\sum_k w_k = 1$ — nothing in the paper's own math guarantees
this, yet the convergence proof in Appendix A.1 (Assumption-free text, "ensures
the convergence of aggregation weights & global weights to their optimal
values") implicitly requires $\theta_g = \sum_k w_k\theta_k$ to remain a valid
combination for the argument to be meaningful in the way Lemma 2 is stated.

**Classification: DEVIATION.** This is the single most consequential place
where the code does something **not licensed by the literal text of Algorithm
2** — it silently adds a step the paper never mentions. For a "100%,
1-to-1 identical" reproduction, an unconstrained Adam-updated `w` (i.e., no
projection at all) is what Algorithm 2 literally specifies; the current code
always deviates from this literal specification. (Whether *not* projecting
would make the whole method numerically unstable/nonsensical is a separate
question from whether it is a faithful reproduction — it is not.)

### 4.4 `FedGRAA.__init__` references `Optional` without importing it

```python
# aggregation.py, top of file:
from typing import List, Sequence, Tuple
...
class FedGRAA:
    def __init__(
        self,
        gamma: float,
        eta_prime: float = 0.1,
        weight_epochs: int = 5,
        reading: str = "B",
        subgradient: str = "min",
        grad_scale: Optional[float] = None,   # <-- Optional is never imported
        grad_clip: Optional[float] = None,    # <-- same
    ):
```

This does not currently crash only because `from __future__ import
annotations` (top of `aggregation.py`) makes all annotations lazy strings that
are never evaluated at runtime. It **would** raise `NameError: name
'Optional' is not defined` the moment anything calls
`typing.get_type_hints(FedGRAA.__init__)`, or if the `from __future__ import
annotations` line were ever removed, or on a hypothetical future Python
version where postponed evaluation is not the default fallback behavior in
this exact way. This is a latent bug, unrelated to paper fidelity per se, but
found during the line-by-line pass. **Not a paper-fidelity gap** — a plain
code-correctness defect worth a one-line fix (`from typing import ...,
Optional`).

---

## Part 5 — Section 5, "Data Settings"

| Paper sentence | Code | Verdict |
|---|---|---|
| *"the datasets are divided among clients"* | `partition.py::partition_non_iid` | VERIFIED MATCH |
| *"To prevent highly frequent attacks ... from dominating, their samples are capped at 10,000"* | `data.py::_cap_frequent_classes`: for every class except benign (see next row), if `len(idx) > spec.frequent_cap (=10_000)`, randomly subsample **without replacement** to exactly 10,000 | VERIFIED MATCH on the cap value; **INTERPRETATION** on the *sampling method* — the paper never says whether the retained 10,000 rows are a random subsample, the first N rows, or something else. Code chooses uniform random subsampling (`rng.choice(idx, size=cap, replace=False)`), a reasonable but unstated choice. |
| *"Each client is assigned data for one specific attack type paired with an equal number of Benign samples"* | `partition_non_iid`: `benign_pick = benign_alloc[attack_id]` sized exactly `len(attack_idx)` | VERIFIED MATCH |
| (implicit) Benign is **not** in the "capped at 10,000" list | `_cap_frequent_classes`: `if class_id == spec.benign_id and not spec.cap_benign: keep_idx.append(...); continue` (i.e. skip capping for Benign by default) | VERIFIED MATCH — and independently cross-checked: sum of the paper's Table-1 non-Benign "Used" counts = 63,901 = the paper's Table-1 Benign "Used" count *exactly*, which is only possible because Benign consumption is a *derived* quantity (total attack demand, since every client draws an equal number of benign rows) rather than an independently-imposed cap. The code's real-CSV path reproduces this **structurally**, without ever hard-coding 63,901 anywhere — a genuine, verified match of the paper's (unstated) design. |
| *"Clients use 90% of their data for local training and reserve 10% for validation"* | `partition.py::_train_val_split`: `n_val = max(1, round(n * val_fraction))`, `val_fraction` default `0.10` | VERIFIED MATCH |
| (implicit) the per-client 10% validation slices, concatenated, form $\mathcal{D}_\mathcal{V}$ | `X_val = np.concatenate(val_X_parts, axis=0)` | VERIFIED MATCH |

### 5.1 How the 10,000-row cap interacts with CSV concatenation order — genuinely unresolvable ambiguity

The paper never says whether capping happens **after all 8 CIC-IDS-2017 daily
CSVs are concatenated** (so the same attack type appearing in two different
daily files is pooled before the 10,000 cap is applied) or **per file**. The
default code path (`DatasetSpec.cap_strategy="global"`) concatenates all CSVs
first (`pd.concat(frames, ...)`, sorted alphabetically by filename via
`sorted(os.listdir(data_dir))`) and caps afterward — meaning which exact
10,000 rows survive is a function of alphabetical file-load order. An
alternate `cap_strategy="per_file"` path also exists in the code
(`_load_real_csvs_per_file_cap`) but is **not the default** and the paper
gives no basis for preferring either. **Classification: INTERPRETATION**
(unresolvable from the paper text alone; both are plausible readings of one
sentence).

### 5.2 Standard scaling: fit-on-what?

The paper's "Data Settings" paragraph says nothing about feature scaling at
all — not whether it happens, and if so, whether the scaler is fit on the full
pool or per-client-train-only. Default (`DatasetSpec.scale_strategy="global"`):
`data.py::_load_real_csvs` calls `_standard_scale(X)` on the **entire pooled
dataset** (all classes, all clients) **before** `partition_non_iid` runs — so
every client's local mean/std statistics leak information from every other
client's (and every validation row's) distribution. An alternate
`scale_strategy="per_client_train"` path fits the scaler on each client's 90%
train split only and applies it to that client's val rows
(`partition.py::_standard_scale_fit_transform`), which is more methodologically
sound but again **not the default** and **not stated in the paper at all.**
**Classification: INTERPRETATION** (paper is silent; code's default choice —
global scaling — is a specific, unforced pick that materially changes every
feature value fed into training).

---

## Part 6 — Section 5, "Network Architecture"

| Paper sentence | Code | Verdict |
|---|---|---|
| *"fully connected neural network using Keras Sequential API"* | `model.py`: hand-written NumPy `MLP` class, no TensorFlow/Keras dependency at all in the default path (`fedgraa/keras_backend.py` exists as an **optional** alternate backend, not used by `trainer.py`/`experiment.py` by default) | **BLOCKING** — see Part 9 |
| *"accepts 78 and 43 input features for CIC-IDS-17 and UNSW"* | `config.py`: `CIC_IDS_2017.n_features=78`, `UNSW_NB15.n_features=43` | VERIFIED MATCH |
| *"hidden layers of sizes 200, 100, and 50 with tanh activation"* | `ModelConfig.hidden_sizes=(200,100,50)`, `hidden_activation="tanh"`; `MLP.forward`: `a = np.tanh(z)` for every layer except the last | VERIFIED MATCH |
| *"output layer of 15 and 10 neurons ... with softmax activation"* | `n_classes=15` / `10` from `len(labels)`; `MLP.forward` last layer: `probs = softmax(z)` | VERIFIED MATCH |

---

## Part 7 — Section 5, "Training Setup and Hyperparameters"

| Paper value | Code default (`FedConfig`) | Verdict |
|---|---|---|
| Adam, η = 0.01 | `eta: float = 0.01` | VERIFIED MATCH (value); Adam itself only matches the prose, not Eq. 1/Alg-1-line-17 notation (0.1) |
| E = 2 local epochs | `local_epochs: int = 2` | VERIFIED MATCH |
| B = 64 batch size | `batch_size: int = 64` | VERIFIED MATCH |
| 200 communication rounds | `rounds: int = 200` | VERIFIED MATCH |
| K = 14 (CIC) / K = 9 (UNSW) clients | Derived: `DatasetSpec.n_clients = len(attack_ids)` = 14 for CIC (15 labels − 1 benign), 9 for UNSW (10 − 1) | VERIFIED MATCH |
| sparse categorical cross-entropy loss | `MLP.loss` / `loss_and_grad`: `-logp[arange(N), y].mean()` (log-softmax gather, mathematically identical to sparse categorical CE) | VERIFIED MATCH |
| FedGRAA: Adam, η' = 0.1, E' = 5 | `eta_prime: float = 0.1`, `weight_epochs: int = 5` | VERIFIED MATCH (value); Adam-vs-Eq.10-notation same caveat as 0.3 |
| γ ∈ {0, 0.5, 1} | `experiment.py::run_gamma_sweep(gammas=(0.0,0.5,1.0))` | VERIFIED MATCH |
| n = 3 rare classes (Heartbleed, SQL Injection, Infiltration / Worms, Shellcode, Backdoor) | `config.py`: `CIC_IDS_2017.rare_label_order[:3]` = `["Heartbleed","SQL Injection","Infiltration"]`; `UNSW_NB15.rare_label_order[:3]` = `["Worms","Shellcode","Backdoor"]` | VERIFIED MATCH |

---

## Part 8 — Table 1 / Table 2 exact values

Every "Used" count in `config.py`'s `CIC_IDS_2017.used_counts` and
`UNSW_NB15.used_counts` was checked digit-by-digit against the paper's Table 1
/ Table 2:

- CIC-IDS-2017: BENIGN 63,901 / Bot 1,948 / DDoS 10,000 / DoS GoldenEye 10,000
  / DoS Hulk 10,000 / DoS Slowhttptest 3,219 / DoS Slowloris 5,228 /
  FTP-Patator 5,931 / Heartbleed 11 / Infiltration 36 / PortScan 10,000 /
  SSH-Patator 5,385 / Brute Force 1,470 / SQL Injection 21 / XSS 652 — **all
  15 values match exactly.**
- UNSW-NB15: Normal 56,691 / Generic 10,000 / Fuzzers 10,000 / Backdoor 2,329 /
  Worms 174 / Reconnaissance 10,000 / Analysis 2,677 / Exploits 10,000 /
  Shellcode 1,511 / DoS 10,000 — **all 10 values match exactly.**

**Verdict: VERIFIED MATCH.** These values are used only by `make_synthetic_dataset`
(the fallback when no real CSVs are present) to fabricate class counts; when
real CSVs are supplied, the actual counts come from the CSVs and the code's
capping logic (Part 5), not from these hard-coded numbers — but the numbers
themselves are transcribed correctly and the synthetic path reproduces the
tables exactly.

---

## Part 9 — Section 5, Figures 1–3 and the results table (Abstract + summary table)

### 9.1 Figure 1 structure (γ ∈ {0,0.5,1}, n=3, both datasets)

`experiment.py::run_gamma_sweep(dataset, gammas=(0,0.5,1.0), n_rare=3)` runs
FedAvg + Static + FedGRAA(γ=0) + FedGRAA(γ=0.5) + FedGRAA(γ=1) — 5 curves,
matching the paper's Figure 1 table columns exactly (`FedAvg`, `Static`,
`Adaptive` × 3 γ values × {All classes, Last n classes}). **VERIFIED MATCH**
on experiment *structure*.

### 9.2 Figures 2–3 structure (n ∈ {5,7}, γ ∈ {0,0.5,1})

`experiment.py::run_n_sweep(ns=(3,5,7), gammas=(0.0,0.5,1.0))` now runs all
3 γ values for every n (this was previously a gap, already closed per
`reproduction_gaps.md` C-3). **VERIFIED MATCH** on structure.

### 9.3 Which classes extend the rare set for n=5, n=7 — the paper never says

The paper states only the n=3 rare classes by name (Heartbleed/SQL
Injection/Infiltration; Worms/Shellcode/Backdoor). It never names which two
(for n=5) or four (for n=7) additional classes are added. Code's
`rare_label_order` extends deterministically by ascending original sample
count: `XSS, Brute Force, Bot, DoS Slowhttptest` for CIC (giving n=5,7 candidate
sets), and `Analysis, Exploits` for UNSW. This is the "most natural" reading
(continue adding the next-rarest classes) but it is **not stated by the
paper anywhere** and cannot be verified against the paper's own Figures 2–3
because the paper's sub-figure captions in the provided text use anonymized
color-channel column headers ("Teal/Yellow/Purple/Red/Blue-Green") rather than
explicit class names — there is no way to check which specific classes the
authors chose. **Classification: INTERPRETATION, unverifiable from the paper alone.**

### 9.4 Headline numbers (Abstract: "88% accuracy on minority classes", "outperforming FedAvg by up to 55%"; results table: FedGRAA γ=1 → 88.88% CIC / 21.09% UNSW)

Nothing in the current codebase asserts, checks, or reproduces these exact
numbers end-to-end. This requires:
1. The real CIC-IDS-2017 CSVs (present in `data/CIC-IDS-2017/`) run through
   the **full 200-round** pipeline, and
2. The real UNSW-NB15 CSVs, which **do not exist in the repository**
   (`data/UNSW-NB15/` is empty; the auto-downloader in `data.py::download_unsw_nb15`
   targets URLs that may or may not still resolve and has never been confirmed
   to succeed in this environment).

Without both, the "21.09%" UNSW number in the paper's own results table cannot
be attempted at all (only a synthetic-data proxy run is possible), and the
CIC-IDS-2017 "88.88%" number has never been validated against a completed
200-round real-data run inside this repository (no saved `results/*.json`
artifact from a full real-data run exists anywhere in the repo as of this
audit — only unit-test-scale synthetic checks in `tests/`).

**Classification: BLOCKING** for UNSW-NB15 (no data), **unverified/pending**
for CIC-IDS-2017 (data present, but no completed full-scale run's numbers have
been captured/compared against the paper's table in this repository).

---

## Part 10 — Appendix A.1 (Convergence Proof) — assumptions vs. code

| Assumption | Paper statement | Code reality |
|---|---|---|
| 1. Smoothness/convexity of $L'$ | $L'$ convex, $\beta$-smooth | $L'$ is built from a tanh-MLP's cross-entropy composed with $\theta'_g = \sum w_k\theta_k$ (linear in $w$) and evaluated at whatever $\theta_k$ the clients happen to have produced. A tanh-MLP's loss surface is **not globally convex** in $\theta$, and since $L'$ is evaluated in terms of $w$ but the $\theta_k$ themselves are outputs of a non-convex local optimization, the assumption is not verified, checked, or enforced anywhere in the code — it is simply assumed to hold by fiat, exactly as the paper does. Nothing to fix in the code; noting this is an assumption the paper itself never justifies for a neural network (a curious but common practice in FL convergence papers). |
| 2. Bounded variance/gradients | $\sigma^2$, $G$ bounds | Never measured/asserted in code. `plot.py::plot_composite_loss` and `TrainHistory.is_composite_loss_monotone` (added per `reproduction_gaps.md` 3.7) let a user **empirically observe** whether $L'(w^t)$ is monotone non-increasing (Lemma 1's conclusion), which is the closest the code comes to "checking" this — but it is an outcome check, not an assumption check. |
| 3. Learning rates $\le 1/\beta$ | $\eta,\eta' \le 1/\beta$ | $\beta$ (the smoothness constant) is never computed anywhere; there is no way to check whether $\eta=0.01$ or $\eta'=0.1$ actually satisfy this for the real network/data. Not addressed by the paper either (no numeric $\beta$ is ever given). |
| 4. Fixed client participation | Fixed K clients, sufficient local data | Matches code (Part 3, line 5) for all `K` **except** clients whose attack class happens to have 0 samples after capping/label-matching, which are silently skipped (`partition.py::partition_non_iid`: `if n_attack == 0: continue`) — a defensive branch not discussed by the paper at all (in practice never triggered for the paper's own class list, since every listed class has count ≥ 11, but it is still an unstated code behavior). |

**Classification:** The proof itself is not "implemented" by the code (proofs
aren't code) — this section is included only because the user asked for
line-by-line coverage of "each and every line" and Appendix A.1 is part of
the base paper. No actionable gap beyond what is already noted in Parts 4/9.

---

## Part 11 — Things the paper never mentions that the code adds anyway (full inventory)

These are **DEVIATION**s in the strict sense: concrete mechanisms present in
the code with no textual basis in the paper at all. Grouped by file:

| File | Addition not in paper | Default state |
|---|---|---|
| `aggregation.py` | `_project_simplex` — unconditional simplex projection of `w` after every FedGRAA update | **Always on** (no way to disable) |
| `aggregation.py` | `FedGRAA(reading=...)` — two weight-gradient "readings" (A/B) | Default `"B"` (matches paper); `"A"` is an invented alternative |
| `aggregation.py` | `FedGRAA(subgradient=...)` — three sub-gradient strategies for $\min(L_h,L_m)$ | Default `"min"`; `"mean"`/`"smooth"` are invented alternatives |
| `aggregation.py` | `grad_scale` / `grad_clip` on the weight gradient | Default `None` (off) |
| `aggregation.py` | `RuntimeWarning` emission when $D_V^h$ and $D_V^m$ are both empty | Always on when triggered; paper never discusses this edge case |
| `config.py` (`FedConfig`) | `round_zero_zero` flag to force round-0 accuracy to 0.00 | Default `False` — meaning **the code's default round-0 behavior does not match the paper's own Figure 1 table**, which shows exactly 0.00 for every method at round 0 (see below) |
| `config.py` (`DatasetSpec`) | `cap_benign`, `exclude_monday`, `scale_strategy`, `cap_strategy` | Defaults `False`/`False`/`"global"`/`"global"` |
| `partition.py` | `val_split_strategy="stratified"` alternative to the paper's plain random 90/10 split | Default `"random"` (matches paper's plain statement) |
| `metrics.py` | `rare_metric="pooled"` alternative to `"mean_recall"` for "Last n classes" accuracy | Default `"mean_recall"` — **this is itself an interpretation**, since the paper never defines whether "accuracy on the last n classes" means mean per-class recall or pooled accuracy across all rare-class samples combined (these differ whenever the rare classes have very different sizes, which they do: Heartbleed=11 vs Infiltration=36 vs SQL Injection=21) |
| `partition.py` | Circular/wraparound benign-sample allocation across clients when demand exceeds the benign pool | Not discussed by the paper at all (irrelevant unless the benign pool is smaller than total attack demand, which does not occur for CIC-IDS-2017/UNSW-NB15 at the paper's stated counts) |
| `baselines.py` | FedProx, FedOpt, FedSND implementations | **Not used by the paper's own Figures 1–3 at all** — these are Related-Work (§2) methods added for extra comparison, entirely outside the base paper's actual experiments. Explicitly out of scope for "100% reproduction of the base paper" per the user's request; flagged only so it's clear this file contributes nothing to (and does not detract from) reproduction fidelity of Figures 1–3. |
| `keras_backend.py` | Optional TensorFlow/Keras MLP | **Not used by default** (`trainer.py` always uses the NumPy `MLP`) — exists as an opt-in module but the default reproduction path never touches it, so the "Keras Sequential API" claim in §5 remains unaddressed by default (Part 6). |

### 11.1 Round-0 accuracy: code's default (~1/K random accuracy) vs. paper's literal table (0.00)

The paper's own Figure 1 data table (transcribed in the attached document)
shows **every single method at round 0 reporting exactly 0.00** for both "All
classes" and "Last n classes," across every γ and every dataset. A freshly
Glorot-initialized softmax network does not produce exactly 0% accuracy — on
15 (CIC) or 10 (UNSW) classes, random-guess accuracy is ≈6.7%/≈10%, not 0%.

`FedConfig.round_zero_zero` exists specifically to let a user force-match the
paper's literal round-0 table, but **defaults to `False`**, meaning **the
out-of-the-box reproduction run's round-0 numbers do not match the paper's own
published table by default.** To get an apples-to-apples comparison against
the paper's Figure 1 table, a user must explicitly pass
`FedConfig(round_zero_zero=True)` — this is not mentioned in any top-level
README/run script and is easy to miss.

**Classification: DEVIATION** (default behavior does not match the paper's
literal published numbers at round 0, even though a flag exists to fix it).

---

## Part 13 — Final grilling: three things the first pass missed

A second, harder read of the base paper's exact wording (not just its equations)
turned up three more items — none of these are in `reproduction_gaps.md` or
`full_reproduction_gap_analysis.md` either.

### 13.1 The paper itself uses two different symbols for "how many clients contribute to θ'_g" — `m` in prose, `K` in Eq. 10 and Algorithm 2 line 3

Immediately before Eq. 8, the paper's prose says:

> *"Once the updated local models are received from the clients, the server
> computes an intermediate global model, θ'_g, by using the weights from the
> previous round, w^{t-1}: $\theta'_g = \sum_{i=1}^m w_i^{t-1}\theta_i^t$."*

This explicitly sums **`i = 1` to `m`**, not `K`. But Eq. 10 and Algorithm 2
line 3, describing the *exact same quantity* θ'_g, both sum over **`i = 1` to
`K`** (the full client count):

> Eq. 10: $\ldots\nabla_{w^{t-1}}L'\left(\sum_{i=1}^{K}w_i^{t-1}\theta_i^t;\ldots\right)$
> Alg. 2 line 3: $\theta'_g = w_1^{t-1}\theta_1^t + w_2^{t-1}\theta_2^t + \cdots + w_K^{t-1}\theta_K^t$

`m` is never defined anywhere else in the paper — it could be a plain
typo for `K` (most likely), or it could mean something real (e.g. a random
subsample of `m < K` clients used only for the *weight-refinement* step,
keeping the full `K` for the *local-training* step). The paper gives no way
to tell which.

**Code:** `aggregation.py::FedGRAA.update` always calls
`aggregate(client_params, w)` over **every client in `client_params`** — i.e.
always `K`, never a subsampled `m`. This matches Eq. 10 / Algorithm 2's
notation, not the prose paragraph's `m`. **If `m` was intentional and
smaller than `K`, the code has never implemented that reading at all, and
no config flag exists to try it.**

### 13.2 Algorithm 1's own "Input" line never declares `E`, `E'`, or `B` — but the algorithm body uses all three

Algorithm 1, line 1, states its full input list explicitly:

> *"**Input:** T - Training rounds, η & η' - learning rates, w^0 = [...] -
> initial aggregation weights ..., γ - personalization coefficient, D_V -
> validation dataset, D_V^n - validation dataset of last n underrepresented
> samples."*

That is seven declared inputs: `T, η, η', w^0, γ, D_V, D_V^n`. Nowhere in
that list are `E` (local epochs, used at line 15), `E'` (weight-refinement
epochs, used at Algorithm 2 line 2), or `B` (batch size, used at line 16 —
"each batch of size B"). The algorithm body silently uses three
hyperparameters its own input contract never admits exist. This is purely a
paper-authoring defect (the values are recovered from §5's prose instead),
but it means **there is no single self-contained place in the paper where
Algorithm 1 is fully specified** — a ditto reproduction has to reach outside
the algorithm box into a completely different section (§5) just to know how
many epochs a client trains for. Code handles this correctly in the sense
that `FedConfig` collects `E`, `E'`, `B` from §5's prose values — but this is
logged here because "100% ditto of the paper" is not even well-posed for
Algorithm 1 in isolation; it always required importing §5 by reference, and
that fact was not previously stated anywhere in this audit.

### 13.3 Using Adam for the weight update (§5's own stated method) breaks the applicability of the paper's own convergence proof

This sharpens finding 0.3 into something more damning. Appendix A.1, Lemma 1's
proof derives its central inequality

$$L'(w^t) \le L'(w^{t-1}) - \left(\eta' - \frac{\beta(\eta')^2}{2}\right)\lVert\nabla L'(w^{t-1})\rVert^2$$

directly from the **standard descent lemma for L-smooth functions under plain
gradient descent** (`w^t = w^{t-1} - \eta'\nabla L'(w^{t-1})`). This specific
inequality is **not generally true for Adam** — Adam's update divides each
coordinate by its own bias-corrected second-moment estimate, which is not an
L-smooth-gradient-descent step and does not admit this same one-line descent
bound (this is precisely why the Adam convergence literature, e.g. Reddi et
al.'s original Adam counterexample, needed entirely different proof
techniques). In other words:

- If the weight update is **plain GD** (as Eq. 10 / Alg. 2 line 6 literally
  say): Lemma 1's proof is at least the right *type* of argument.
- If the weight update is **Adam** (as §5's prose says, and as the code
  does): **Appendix A.1's own proof of Lemma 1 no longer applies to the
  method the paper says it ran.** The paper's convergence guarantee and the
  paper's stated implementation are, on this reading, about two different
  algorithms.

This is not a code bug — it is a logical gap **inside the paper itself**
that becomes highly relevant here because the code (matching §5) uses Adam,
which is exactly the choice that invalidates the one piece of formal backing
(Appendix A.1) the paper offers for the whole method. Flagged for
completeness; no code fix resolves a proof gap in someone else's paper.

---

## Part 12 — Blocking gaps (unresolvable without external artifacts)

1. **UNSW-NB15 CSVs are absent from the repository** (`data/UNSW-NB15/` is
   empty). Every UNSW result the code can currently produce uses the
   synthetic Gaussian-cluster fallback (`data.py::make_synthetic_dataset`),
   which matches only the *per-class sample counts* of Table 2, not any real
   feature statistics. None of Figure 1's UNSW panels, Figure 3, or the
   "21.09%" UNSW headline number can be attempted for real.

2. **NumPy/float64 vs. Keras/float32.** The paper states the network was
   built with the Keras Sequential API (§5). The default code path is a
   hand-rolled NumPy MLP in float64. Even with `fedgraa/keras_backend.py`
   opted into, TensorFlow's fused kernels have different rounding behavior
   from any hand-written stable log-softmax, and float32 vs float64 alone
   guarantees numerical divergence from the very first backward pass. No
   config flag makes the default path bit-identical to a Keras/TF run.

3. **No random seed disclosed.** The paper reports specific numbers (88.88%,
   66.66%, 21.09%, etc. — themselves suspiciously round-looking, e.g. exactly
   repeating-decimal-style "88.88"/"66.66"/"33.33"/"77.77" in the results
   table, which do not look like typical ML accuracy outputs at all and may
   themselves be placeholder/template values in this particular version of
   the paper) without stating the seed used for weight init, benign sampling,
   or batch shuffling. `FedConfig.seed` defaults to `0`; there is no way to
   confirm this (or any other seed) reproduces the paper's exact trajectory.

4. **The results table's numbers are suspiciously uniform-looking** (66.66,
   88.88, 77.77, 33.33 for CIC-IDS-17; 00.00, 21.09, 20.59, 00.00 for
   UNSW-NB15) — several of these (66.66, 88.88, 77.77, 33.33) are exact
   repeating-decimal patterns rather than numbers one would expect from a
   real 200-round training run rounded to 2 decimals. This is a property of
   the paper's own published table, not the code, but it materially affects
   what "100% reproduction" can even mean here: if these are placeholder
   values in the paper itself (rather than genuine measured results), no
   code, however faithful, can reproduce them exactly. Flagged for
   completeness/transparency, not as a code gap.

---

## Summary table — every gap found in this pass

| # | Location | Type | Severity | One-line description |
|---|---|---|---|---|
| 0.1 | Eq.1 / Alg.1 L17 vs §5 | Paper-internal | INTERPRETATION | SGD notation vs. Adam prose; code uses Adam |
| 0.2 | Eq.2 vs Eq.5 | Paper-internal | INTERPRETATION | Uniform vs size-weighted FedAvg; code implements Eq. 5 only |
| 0.3 | Eq.10 / Alg.2 L6 vs §5 | Paper-internal | INTERPRETATION | Plain-GD notation vs. Adam prose; code uses Adam |
| P1 | Eq. 4, Eq. 6 | `aggregation.py`, `trainer.py` | DEVIATION (omission) | Global loss/gradient equations never explicitly computed anywhere |
| 4.1 | `aggregation.py::FedGRAA` | `fedgraa_reading="A"` path | DEVIATION | Non-default, paper-unsanctioned alternate weight-gradient reading exists |
| 4.2 | `aggregation.py::FedGRAA._composite_grad_P` | `subgradient` choices | INTERPRETATION + DEVIATION | Sub-gradient of $\min(L_h,L_m)$ unspecified by paper; 2 of 3 code options are invented |
| 4.3 | `aggregation.py::FedGRAA.update` | `_project_simplex` | **DEVIATION** | Unconditional simplex projection not present anywhere in Algorithm 2 |
| 4.4 | `aggregation.py` top-of-file imports | missing `Optional` import | Code bug (non-paper) | Latent `NameError` risk, masked by `from __future__ import annotations` |
| 5.1 | `data.py::_cap_frequent_classes` | cap sampling method | INTERPRETATION | Random-without-replacement subsampling to 10,000; paper doesn't say how |
| 5.1b | `data.py::_load_real_csvs` vs `_load_real_csvs_per_file_cap` | cap ordering (`cap_strategy`) | INTERPRETATION | Global-concat-then-cap (default) vs per-file cap; paper silent |
| 5.2 | `data.py::_load_real_csvs`, `partition.py::_standard_scale_fit_transform` | scaling strategy | INTERPRETATION | Global scaler fit (default, leaks val stats) vs per-client-train scaler; paper never mentions scaling at all |
| 9.3 | `config.py::rare_label_order` | n=5,7 class extension | INTERPRETATION, unverifiable | Paper never names which classes extend n=3 to n=5/7 |
| 9.4 | (whole pipeline) | headline numbers | **BLOCKING** (UNSW) / pending (CIC) | 88%/55%/21.09% claims never end-to-end verified in-repo |
| 11 (metrics) | `metrics.py::last_n_class_accuracy` | `rare_metric` default | INTERPRETATION | "Accuracy on last n classes" = mean-per-class-recall (default) vs pooled; paper never defines this precisely |
| 11.1 | `config.py::FedConfig.round_zero_zero` | default `False` | **DEVIATION** | Out-of-the-box round-0 numbers don't match paper's literal 0.00 table by default |
| 12.1 | `data/UNSW-NB15/` | missing data | **BLOCKING** | No real UNSW-NB15 CSVs in repo |
| 12.2 | `model.py` vs `keras_backend.py` | framework/precision | **BLOCKING** | Default path is NumPy/float64, not Keras/float32, despite §5's explicit claim |
| 12.3 | (whole pipeline) | no seed disclosed | **BLOCKING** | Paper reports numbers with no reproducible seed |
| 12.4 | Fig.1 results table itself | paper artifact, not code | Not fixable | Several headline numbers look like repeating-decimal placeholders (66.66/88.88/77.77/33.33), which — if true — makes bit-exact reproduction impossible in principle, independent of code quality |
| 13.1 | Prose before Eq.8 ("$\sum_{i=1}^m$") vs Eq.10/Alg.2 L3 ("$\sum_{i=1}^K$") | Paper-internal | INTERPRETATION, unverifiable | Paper uses `m` in prose, `K` in the actual equation/algorithm for the identical quantity θ'_g; code always uses `K` (never implements a possible `m<K` subsample reading) |
| 13.2 | Algorithm 1, line 1 ("Input:") | Paper-internal omission | Not a code gap | `E`, `E'`, `B` are used in the algorithm body but never declared as inputs anywhere in Algorithm 1's own contract; only recoverable from §5 prose |
| 13.3 | Appendix A.1, Lemma 1 proof vs §5's stated Adam optimizer | Paper-internal | Not a code gap | The descent-lemma proof of Lemma 1 is a plain-GD argument; using Adam (which §5 mandates and the code does) makes the paper's own convergence proof inapplicable to the algorithm it claims to certify |

---

## Bottom line — no more hedging

The first pass of this audit called several of the items above
"INTERPRETATION" and left it there, as if a reasonable engineering judgment
call were the same thing as reproducing the paper. On a second, harder read,
that framing was too generous. Restated plainly:

- **The single largest, unforced deviation is the simplex projection
  (`_project_simplex`, §4.3).** Algorithm 2 has eight lines and none of them
  project, clip, or renormalize `w`. Every run of this codebase, in every
  configuration, silently does something to the aggregation weights that
  Algorithm 2 never asks for. This is not an edge case or a config default —
  it is unconditional, on every round, of every method, for every dataset.
- **The code's out-of-the-box round-0 numbers do not match the paper's own
  published table** unless a non-default flag is set (§11.1). Anyone running
  this repository today and comparing round-0 against the paper's Figure 1
  table will see a mismatch before a single round of federated training has
  even happened.
- **Three separate config surfaces (`fedgraa_reading`, `subgradient`,
  `rare_metric`, plus `scale_strategy`/`cap_strategy`) exist specifically
  because the paper is ambiguous or silent, and all of them currently ship
  side-by-side with the paper-matching default as if they were equally valid
  options.** For a genuine ditto reproduction, ambiguity should be resolved
  once, explicitly, and the alternate paths removed — not left as toggles.
- **Two gaps are not fixable by any code change at all**: the missing
  UNSW-NB15 data, and the NumPy/float64-vs-Keras/float32 framework mismatch.
  Nothing above changes that; they are listed again here so they aren't lost
  in the list of things that *are* fixable.
- Everything with a single, unambiguous literal reading in the paper **is**
  implemented correctly (Eq. 3, 5, 7; Alg. 1 lines 3, 4, 6–9, 11, 12, 14–16,
  18; Alg. 2 lines 2, 3, 4 (value), 5, 7; the architecture in §5; all of
  Tables 1–2; the hyperparameter list in §5). That was true before this pass
  and remains true — the grilling above is about everything that is *not*
  in that category, and none of it should continue to be described as
  "fine" just because the paper allows more than one reading.
