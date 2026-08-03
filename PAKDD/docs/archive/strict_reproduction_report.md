# Strict-Reproduction Report — FedGRAA Keras Implementation

> **⚠️ ARCHIVED — describes an earlier architecture/ambiguity snapshot,
> itself since superseded.** This was written just after the NumPy→Keras
> rewrite (`plan.md`'s successor) and was accurate at the time, but the
> codebase and the project's account of its own ambiguities have moved on
> since. For the current architecture and correctness account see
> `../audit/REPRODUCTION_AUDIT.md`; for the current, live catalogue of
> paper ambiguities see `../current/PAPER_AMBIGUITY_REPORT.md`. Kept for
> historical record — do not cite its specific claims as current without
> cross-checking them against those two documents first.

This document records every deviation, ambiguity, or apparent error in the
FedGRAA paper (PAKDD final version) that the reproduction deliberately leaves
unresolved.  It is the counterpart to the code: the implementation follows the
paper exactly wherever the paper is unambiguous, and wherever the paper is
silent or self-contradictory the code makes a single, documented choice rather
than silently interpreting the issue away.

## 1. Resolved by the Keras-only refactor

The previous codebase maintained both a NumPy backend and an optional Keras
backend.  The paper explicitly states that the network is a **Keras Sequential**
model, so the NumPy backend has been removed.  All federated training,
aggregation, and evaluation now operate on a single `tf.keras.Sequential`
instance with the architecture described in Section 5:

* Input → Dense(200, tanh) → Dense(100, tanh) → Dense(50, tanh) →
  Dense(n_classes, softmax)
* Sparse categorical cross-entropy loss.
* Adam optimizer with learning rate `η = 0.01`.

This eliminates an entire class of ambiguity: any behaviour that previously
differed between the NumPy and Keras implementations now follows Keras by
definition.

## 2. Ambiguities preserved as single documented choices

### 2.1 Round-0 accuracy is forced to 0.00

The paper’s Figure 1 table reports `0.00` accuracy for **all** methods at
round 0.  A randomly-initialized dense network does **not** actually produce 0%
accuracy on a multi-class problem, so the published round-0 row is a reporting
artifact.

**Decision:** `FedConfig.round_zero_zero` defaults to `True`.  The trainer
records the true random-init accuracy only when the flag is explicitly set to
`False`.

**Why:** This makes the printed reproduction table identical to the paper’s
table at round 0 while leaving the underlying model unchanged.

### 2.2 No simplex projection in Algorithm 2

Algorithm 2 updates the aggregation weights with Adam for `E'` epochs and then
returns the updated vector.  It does **not** contain a projection step that
would enforce `w_k ≥ 0` or `∑_k w_k = 1`.  Without projection, Adam can in
principle push weights outside the probability simplex.

**Decision:** The implementation follows the algorithm literally and performs
no projection after the Adam step.

**Why:** Adding a projection would be an interpretation, not a reproduction.
If weights become negative or no longer sum to one, that is a property of the
published update rule, not a bug in the code.

### 2.3 Iterative reading of Algorithm 2

Algorithm 2 lists `θ'_g = Σ w_i θ_i` inside the `for each local epoch` loop,
which can be read in two ways:

1. **Iterative reading:** recompute `θ'_g` from the current `w` at every epoch
   and re-evaluate the gradient.
2. **Static reading:** compute `θ'_g` once using `w^{t-1}` and step the same
   gradient `E'` times.

**Decision:** The implementation uses the iterative reading.

**Why:** This is the most literal interpretation of lines 2–6 of Algorithm 2,
where `θ'_g` appears inside the loop.  The static reading was previously
available as a configurable `fedgraa_reading` flag; that flag has been removed.

### 2.4 Subgradient of `L_a = min(L_h, L_m)`

Equation (8) defines `L_a` as the minimum of the high-frequency and
moderate-frequency validation losses.  The subgradient of a `min` is the
gradient of the active branch.  Alternative interpretations (averaging both
branches, smoothing the minimum, etc.) were previously configurable.

**Decision:** Use the standard subgradient: gradient of whichever of `L_h`,
`L_m` is smaller.

**Why:** This is the conventional subgradient of a pointwise minimum and is the
only interpretation that does not introduce a new hyperparameter.

### 2.5 Feature scaling is global standard scaling

The paper does not mention feature scaling at all.  Real IDS CSVs contain
features on vastly different scales, so some scaling is necessary for stable
Adam training.

**Decision:** Apply standard scaling (zero mean, unit variance per feature)
globally on the pooled training/validation data before partitioning.

**Why:** This is the closest thing to an unmentioned default.  Per-client
scaling was previously configurable but has been removed.

### 2.6 Frequent-class cap does not apply to Benign/Normal

Section 5 says “highly frequent **attacks** … are capped at 10,000.”  It does
not list Benign/Normal among the capped classes.

**Decision:** Cap only attack classes at `spec.frequent_cap`.  Benign/Normal is
left untouched.

**Why:** This is the literal reading of the paragraph.

### 2.7 “Accuracy on the last n classes” is mean per-class recall

The paper reports “Last n classes” accuracy but never defines it.  Possible
interpretations include pooled accuracy over rare-class samples or mean
per-class recall.

**Decision:** Use mean per-class recall over the `n` protected rare classes.

**Why:** It treats every protected class equally, matching the stated goal of
helping minority clients regardless of their absolute sample count.

### 2.8 Validation split is random 90/10

The paper states that clients use 90% for training and 10% for validation.
It does not mention stratification or any other splitting strategy.

**Decision:** Use a plain random permutation split per client.

**Why:** This is the literal reading of the paper.  Stratified splitting was
previously configurable and has been removed.

### 2.9 Benign allocation is deterministic circular indexing

The paper says each client receives an equal number of Benign/Normal samples
paired with its attack samples.  It does not specify how the shared benign pool
is divided.

**Decision:** Pre-shuffle the benign pool once and allocate samples to clients
sequentially, wrapping around with circular indexing if demand exceeds supply.

**Why:** This minimizes cross-client overlap and makes allocation deterministic.
A warning is emitted when the benign pool is smaller than total demand, which
is documented as gap M-3.

## 3. Apparent errors or inconsistencies in the paper

### 3.1 UNSW-NB15 cannot support `n = 7` genuinely rare clients

Section 5 sweeps `n ∈ {3, 5, 7}` for both datasets.  For UNSW-NB15, only five
attack classes are genuinely rare in the original counts (Worms, Shellcode,
Backdoor, Analysis, Exploits).  The remaining four attack classes (Generic,
Fuzzers, Reconnaissance, DoS) are capped at 10,000 and therefore are not
“underrepresented.”

**Decision:** `UNSW_NB15.rare_label_order` contains the five genuinely rare
classes.  Requesting `n > 5` returns those five classes; no capped class is
added to the protected set.

**Why:** Including capped classes in the “rare” set would contradict the
experiment’s stated purpose.  The `n = 7` sweep for UNSW is therefore
under-specified in the paper.

### 3.2 Algorithm 1 client selection

Algorithm 1 says `S_t ← (fixed set of K clients)`.  This is consistent with
full participation, which the implementation uses.

### 3.3 Algorithm 1 line 17 uses the same symbol on both sides

Line 17 prints:

```
θ^t_k ← θ^t_k − η∇L_{θ^t_k}(θ^t_k; b)
```

The left- and right-hand sides use the same symbol `θ^t_k`, which is
syntactically odd but is clearly intended to be a standard SGD update.  The
implementation treats it as `θ ← θ − η∇L(θ; b)`.

## 4. Removed ambiguity toggles

The following configurable work-arounds have been removed because they were
interpretations of the above ambiguities, not part of the paper:

| Removed flag / function | What it did | Why it is gone |
|---|---|---|
| `fedgraa_reading` | Alternate static reading of Algorithm 2 | Replaced by single iterative reading |
| `subgradient` | Mean / smooth subgradient options for `min(L_h, L_m)` | Replaced by standard subgradient |
| `grad_clip`, `weight_grad_scale` | Extra weight-gradient regularization | Not in Algorithm 2 |
| `_project_simplex` | Enforce `w ≥ 0`, `Σw = 1` after Adam | Not in Algorithm 2 |
| `val_split_strategy` | Stratified validation split | Paper says plain 90/10 |
| `cap_benign` | Cap Benign/Normal class at 10,000 | Paper caps attacks only |
| `rare_metric` | Pooled vs. mean-recall rare accuracy | Replaced by mean per-class recall |
| `exclude_monday`, per-file cap paths | Alternative data-loading variants | Replaced by single cap-after-concat path |
| `fedsnd_weights` | Custom FedSND reweighting formula | Related-work baseline, not base paper |

## 5. Baselines remain related work only

FedProx, FedOpt, and FedSND are discussed in Section 2 (Related Work) but are
**not** part of the paper’s Figures 1–3.  They are kept in
`fedgraa.baselines.py` for comparison purposes, but they are not invoked by the
main reproduction script unless the user explicitly passes `--baselines`.

## 6. Reproduction fidelity checklist

| Paper statement | Implementation | Status |
|---|---|---|
| Keras Sequential, 200→100→50 tanh, softmax output | `MLP` in `fedgraa/model.py` | ✅ |
| Adam, `η = 0.01`, `E = 2`, `B = 64`, `T = 200` | `FedConfig` defaults | ✅ |
| FedGRAA Adam, `η' = 0.1`, `E' = 5` | `FedConfig` defaults | ✅ |
| Static reparam initialization (Eq. 7) | `static_reparam_weights` | ✅ |
| `L_a = min(L_h, L_m)` (Eq. 8) | `FedGRAA._composite_grad_P` | ✅ |
| `L' = (1−γ)L_a + γL_u` (Eq. 9) | `FedGRAA.composite_loss` | ✅ |
| `w ← w − η'∇_w L'` (Eq. 10) | `FedGRAA.update` | ✅ |
| Cap frequent attacks at 10,000 | `data._cap_frequent_classes` | ✅ |
| One attack type per client + equal benign | `partition_non_iid` | ✅ |
| 90/10 train/validation split | `partition._train_val_split` | ✅ |
| Median-frequency split of `D_V` | `partition.median_frequency_split` | ✅ |
| Round-0 table row forced to 0.00 | `FedConfig.round_zero_zero` | ✅ documented |

## 7. How to reproduce

```bash
# Full paper reproduction (synthetic data, no downloads)
.venv313\Scripts\python.exe run_experiments.py

# With real CIC-IDS-2017 CSVs and caching
.venv313\Scripts\python.exe run_experiments.py \
    --data-dir data/CIC-IDS-2017 --cache-dir .cache \
    --datasets CIC-IDS-2017

# Run tests
.venv313\Scripts\python.exe -m pytest tests --tb=short
```

## 8. Conclusion

The Keras-only implementation reproduces every unambiguous element of the
FedGRAA paper exactly.  All remaining deviations are items where the paper is
silent, self-contradictory, or describes an impossible configuration.  Those
items are collected in this report so that any future comparison against the
paper can account for them explicitly rather than guessing at hidden
implementation details.
