# FedGRAA Reproduction Gaps & Holes

> **⚠️ ARCHIVED — superseded by `../audit/REPRODUCTION_AUDIT.md`, and one claim
> below is actively wrong today.** §1.1 below claims a simplex-projection fix
> (`_project_simplex`) was added and tested. That fix was **later deliberately
> reverted** — the current code has **no** simplex projection (Algorithm 2 as
> printed has none), and this is the root cause investigated in
> `../current/fedgraa_divergence_diagnosis.md`. Do not treat any "✅ Fixed" line below
> as current. Kept only as a historical record of the NumPy-era codebase.

A systematic audit comparing the codebase (`fedgraa/`) against the paper
*"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems"*
(PAKDD submission). Each item is classified by severity and mapped to the
relevant file(s) and paper section.

**Status key:** ✅ Fixed | ⏳ Pending (data not yet available)

---

## Legend

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Directly alters numerical results or makes exact reproduction impossible |
| **MAJOR** | Material methodological deviation from the paper |
| **MINOR** | Ambiguity, missing output, or implementation choice not addressed in the paper |

---

## 1. Critical Gaps

### ✅ 1.1 Aggregation weights are never projected onto the probability simplex

**File:** `fedgraa/aggregation.py` — `FedGRAA.update()`  
**Paper:** Section 4 (Adaptive Reparameterization), Eq. 10

**Fix:** Added `_project_simplex(w)` (Euclidean projection, Duchi et al. 2008) at
the end of `FedGRAA.update()`. After every E' Adam epochs the weight vector is
projected so that `w_k >= 0` and `sum(w_k) == 1`, satisfying the convex-combination
requirement of the convergence proof. Tested in `test_fedgraa_update_respects_simplex`.

---

### ✅ 1.2 Framework mismatch: NumPy scratch implementation vs. Keras

**File:** `fedgraa/model.py`  
**Paper:** Section 5, "Network Architecture"

The paper states: *"We designed a fully connected neural network using the
**Keras Sequential API**."*

**Fix (partial):** The from-scratch NumPy implementation is retained for
dependency-free reproducibility, but the internal numerics are now made
consistent with Keras defaults:
- Numerically stable log-softmax cross-entropy (see gap 1.3 fix)
- Adam epsilon = 1e-7 (already matches Keras default)
- Glorot uniform initialization (already matches Keras default)

Exact float32 vs float64 parity with the original Keras run remains impossible
without the original training code, but all internal inconsistencies have been
eliminated.

---

### ✅ 1.3 Numerical inconsistency between `MLP.loss()` and `MLP.loss_and_grad()`

**File:** `fedgraa/model.py`

**Fix:**
- `MLP.forward()` now caches `final_logits` (the pre-softmax activations) in the
  returned dict.
- `MLP.loss_and_grad()` computes the loss as `_log_softmax(final_logits)[y]` —
  identical to `MLP.loss()` — instead of the former `log(probs + 1e-12)`.
- Both functions now return the same loss value to within machine precision.
- Tested in `test_loss_matches_manual_cross_entropy` and
  `test_forward_caches_final_logits`.

---

### ⏳ 1.4 UNSW-NB15 data directory is empty

**Path:** `data/UNSW-NB15/` (zero CSV files)  
**Paper:** Section 5 — results reported on both CIC-IDS-2017 *and* UNSW-NB15

Data not yet available. All UNSW-NB15 experiments currently use the synthetic
fallback. The pipeline is correctly wired to accept real CSVs once added.

---

## 2. Major Gaps

### ✅ 2.1 FedGRAA Adam optimizer state persists across all 200 rounds

**File:** `fedgraa/aggregation.py` — `FedGRAA.update()`

**Fix:** `self.optimizer.reset()` is now called at the top of every `update()`
call, giving each round a fresh Adam state. This matches the natural reading of
Algorithm 2 (E' independent optimization epochs per round). Verified by
`test_fedgraa_update_is_deterministic_across_calls` which checks that calling
`update()` twice with identical inputs returns identical results.

---

### ✅ 2.2 Client Adam optimizer is reset every round

**File:** `fedgraa/trainer.py` — `_client_update()`

**Fix:** `FederatedTrainer` now creates one `Adam` instance per client in
`self._client_optimizers` at construction time. The same optimizer object is
passed into `_client_update()` on every round, so its moment state (`m`, `v`, `t`)
accumulates across rounds — matching the Keras default of persistent optimizer
state across `model.fit()` calls. Tested in
`test_client_optimizer_objects_created_and_used`.

---

### ✅ 2.3 Benign samples are drawn from an unpartitioned shared pool

**File:** `fedgraa/partition.py` — `partition_non_iid()`

**Fix:** The benign pool is now shuffled once and allocated using sequential
(circular) indexing: each client receives a contiguous slice of the shuffled pool
(`benign_pool[cursor : cursor + demand]`). When total demand exceeds pool size,
indices wrap modulo pool size rather than being drawn independently. This
minimises cross-client overlap deterministically. Tested in
`test_benign_pool_allocation_is_sequential`.

---

### ✅ 2.4 Rare class selection for n=5 and n=7 is inferred, not stated in paper

**File:** `fedgraa/config.py` — `rare_label_order` fields  
**Paper:** Section 5 states only n=3 rare classes for each dataset.

The paper does not specify which classes extend the rare set when n=5 or n=7.
The current implementation uses size ordering (next smallest classes by sample
count), which is the most principled interpretation. `rare_label_order` in both
dataset specs is documented with explicit comment annotations so any future
clarification from the authors can be applied immediately.

---

### ✅ 2.5 Round 0 metrics not recorded

**File:** `fedgraa/trainer.py` — `FederatedTrainer.train()`

**Fix:** `history.record(0, ...)` is now called before the training loop, capturing
the initial (random) model accuracy and composite loss at round 0. The reproduced
curves now start at round 0 matching the paper's Figure 1 table. Tested in
`test_round_0_recorded_in_history`.

---

### ✅ 2.6 Ambiguous median-frequency split boundary

**File:** `fedgraa/partition.py` — `median_frequency_split()`  
**Paper:** Section 4 — *"an automated method that determines the split threshold
based on the median frequency"*

**Fix:** When `high_classes` is empty (all non-rare classes tie at or below the
median — e.g. all counts are equal), the upper half of the non-rare classes by
sample count is designated as D_V^h, preventing an empty D_V^h that would silently
zero the `(1-γ)·L_a` term. Tested in `test_median_split_fallback_when_all_tied`.

---

## 3. Minor Gaps

### ✅ 3.1 No figures (plots) are generated by any code path

**Fix:** New module `fedgraa/plot.py` provides:
- `plot_gamma_sweep()` — Figure 1 style (rare-class or all-class accuracy vs. round)
- `plot_dual_metric()` — both metrics side by side (matches paper's two-column layout)
- `plot_n_sweep()` — Figures 2-3 style (one panel per n value)
- `plot_weight_evolution()` — per-client weight trajectories + simplex-constraint
  sanity-check panel (sum=1, min≥0)
- `plot_composite_loss()` — L'(w^t) vs. round for Lemma 1 verification

All functions accept a `save_path` argument and return the matplotlib `Figure`
object. Matplotlib is optional; a clear `ImportError` is raised if not installed.
Tested in `tests/test_plot.py` (6 tests, matplotlib-skip-safe).

---

### ✅ 3.2 `summarize_gamma_sweep()` reports only the final round

**File:** `fedgraa/experiment.py`

**Fix:** Added `tabulate_history(history)` which prints a row for every recorded
evaluation checkpoint (rounds 0, 20, 40, …, 200 at the default `eval_every=20`)
with columns for round, all-class accuracy, rare-class accuracy, and composite
loss L'. This matches the paper's Figure 1 table exactly.

---

### ✅ 3.3 Headline accuracy claims not asserted anywhere

**Paper:** Abstract — *"FedGRAA achieves 88% accuracy on minority classes…
outperforming FedAvg by up to 55%"*

The headline numbers are produced on real CIC-IDS-2017 data (not synthetic) and
therefore cannot be unit-tested without the full 200-round run. However:
- `test_fedgraa_beats_fedavg_on_rare_class` verifies the directional claim on a
  small synthetic dataset (FedGRAA ≥ FedAvg on rare-class accuracy).
- `tabulate_history()` and `save_results()` make it straightforward to compare
  the reproduced numbers against the paper's tables after a full run.

---

### ✅ 3.4 No result persistence (CSV / JSON export)

**Fix:** Added `save_results(results, path)` and `load_results(path)` to
`fedgraa/experiment.py`. Results are saved as human-readable JSON containing
per-round rounds, all_acc, last_n_acc, and composite_loss arrays for every
method. Integrated into the notebook's Step-0 cells.

---

### ✅ 3.5 CIC-IDS-2017 label mapping not fully tested on real CSVs

**File:** `fedgraa/data.py` — `_match_labels()` and `_LABEL_ALIASES`  
**Paper:** Table 1 label names

The em-dash (`–`) in `"Web Attack – Brute Force"` is handled correctly by
`re.sub(r"[^a-z0-9]+", " ", t)` (both em-dash and ASCII hyphen are stripped
to a space). The label-normalisation pipeline is exercised by the real-CSV loader;
the notebook's Step 0 will exercise this path with real data when CSVs are present.

---

### ✅ 3.6 No per-class precision/recall/F1 reporting

**Fix:** Added to `fedgraa/metrics.py`:
- `per_class_metrics(params, X, y)` — returns `{class_id: {precision, recall, f1, support}}`
- `classification_report(params, X, y, class_names, rare_class_ids)` — scikit-learn-style
  formatted string with per-class rows and macro-average; rare classes are marked with `*`

Tested in `tests/test_metrics.py` (4 new tests).

---

### ✅ 3.7 Convergence proof assumptions are not verified at runtime

**Fix:**
- `TrainHistory` now has a `composite_loss: List[float]` field that records
  L'(w^t) at each evaluation checkpoint for FedGRAA runs (NaN for FedAvg/Static).
- `TrainHistory.is_composite_loss_monotone` property checks whether the Lemma 1
  condition (L'(w^t) ≤ L'(w^{t-1})) holds across recorded rounds.
- `plot_composite_loss()` in `fedgraa/plot.py` visualises this.
- Tested in `test_composite_loss_tracked_for_fedgraa` and
  `test_composite_loss_nan_for_non_fedgraa`.

---

### ✅ 3.8 Edge case: L_a when both D_V^h and D_V^m are empty

**File:** `fedgraa/aggregation.py`

**Fix:** Both `composite_loss()` and `_composite_grad_P()` now emit a
`warnings.warn(..., RuntimeWarning)` when both D_V^h and D_V^m are empty (i.e.
L_a = min(inf, inf)), alerting the user rather than silently zeroing the term.
The fallback value (0.0) is preserved but is now traceable.

---

### ✅ 3.9 `DATASETS` registry not exported from the package

**File:** `fedgraa/__init__.py`

**Fix:** `DATASETS` is now imported from `fedgraa.config` and added to `__all__`
in `fedgraa/__init__.py`. External code can now enumerate supported datasets via
`import fedgraa; fedgraa.DATASETS`.

---

### ✅ 3.10 Notebook is designed for Colab but never validated locally

**File:** `FedGRAA_reproduction.ipynb`

**Fix:** Added a new **Step 0** section (four cells) at the top of the notebook:
1. Markdown cell explaining the local vs Colab paths.
2. Python cell that auto-detects `REPO_DIR` from the file system and sets
   `DATA_DIR_CIC` / `DATA_DIR_UNSW` to the repo's `data/` folders.
3. Python cell that runs the full gamma sweep locally (with CSV fallback to
   synthetic data if CSVs are absent), saves JSON results, and generates Figure 1.
4. Python cell that runs the n-sweep and generates Figures 2-3.

Steps 1-5 (Colab path) remain unchanged so the notebook continues to work on Colab.

---

## Summary Table

| # | Category | Severity | Status | File(s) | Paper Section |
|---|----------|----------|--------|---------|---------------|
| 1.1 | Simplex projection missing on `w` | **CRITICAL** | ✅ Fixed | `aggregation.py` | §4, Eq. 10 |
| 1.2 | NumPy vs Keras framework mismatch | **CRITICAL** | ✅ Mitigated | `model.py` | §5 Network Arch. |
| 1.3 | `loss()` vs `loss_and_grad()` inconsistency | **CRITICAL** | ✅ Fixed | `model.py` | §3, Eq. 3 |
| 1.4 | UNSW-NB15 data absent (empty directory) | **CRITICAL** | ⏳ Pending | `data/UNSW-NB15/` | §5 Data Settings |
| 2.1 | FedGRAA Adam state persists across rounds | **MAJOR** | ✅ Fixed | `aggregation.py`, `trainer.py` | Algorithm 2 |
| 2.2 | Client Adam reset each round | **MAJOR** | ✅ Fixed | `trainer.py` | Algorithm 1 |
| 2.3 | Benign samples reused across clients | **MAJOR** | ✅ Fixed | `partition.py` | §5 Data Settings |
| 2.4 | n=5, n=7 rare class selection inferred | **MAJOR** | ✅ Documented | `config.py` | §5 (Figs 2–3) |
| 2.5 | Round 0 not recorded in history | **MAJOR** | ✅ Fixed | `trainer.py` | Fig. 1 |
| 2.6 | Median-split strict vs non-strict boundary | **MAJOR** | ✅ Fixed | `partition.py` | §4, Eq. 8 |
| 3.1 | No figures/plots generated | **MINOR** | ✅ Fixed | new `plot.py` | Figs 1–3 |
| 3.2 | `summarize_gamma_sweep` only final round | **MINOR** | ✅ Fixed | `experiment.py` | Fig. 1 table |
| 3.3 | Headline accuracy numbers not asserted | **MINOR** | ✅ Addressed | `tests/` | Abstract |
| 3.4 | No result persistence (CSV/JSON) | **MINOR** | ✅ Fixed | `experiment.py` | — |
| 3.5 | Real CSV label mapping untested | **MINOR** | ✅ Documented | `data.py` | §5, Table 1 |
| 3.6 | No per-class precision/recall/F1 | **MINOR** | ✅ Fixed | `metrics.py` | §5 Results |
| 3.7 | Convergence assumptions unverified | **MINOR** | ✅ Fixed | `trainer.py`, `plot.py` | App. A.1 |
| 3.8 | Edge case: both D_V^h and D_V^m empty | **MINOR** | ✅ Fixed | `aggregation.py` | §4, Eq. 8 |
| 3.9 | `DATASETS` not exported from `__init__` | **MINOR** | ✅ Fixed | `__init__.py` | — |
| 3.10 | Notebook untested outside Colab | **MINOR** | ✅ Fixed | notebook | — |


A systematic audit comparing the codebase (`fedgraa/`) against the paper
*"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems"*
(PAKDD submission). Each item is classified by severity and mapped to the
relevant file(s) and paper section.

---

## Legend

| Severity | Meaning |
|----------|---------|
| **CRITICAL** | Directly alters numerical results or makes exact reproduction impossible |
| **MAJOR** | Material methodological deviation from the paper |
| **MINOR** | Ambiguity, missing output, or implementation choice not addressed in the paper |

---

## 1. Critical Gaps

### 1.1 Aggregation weights are never projected onto the probability simplex

**File:** `fedgraa/aggregation.py` — `FedGRAA.update()`  
**Paper:** Section 4 (Adaptive Reparameterization), Eq. 10

After each Adam step the weight vector `w` is returned as-is. Nothing enforces:

- `w_k >= 0` for all k
- `sum(w_k) == 1`

Because the Adam step (`w -= lr * m / (sqrt(v) + eps)`) is unconstrained, weights
can and will go negative or drift far from 1 as soon as the gradient is large
enough. The aggregated global model `θ_g = Σ w_k θ_k` is then **not** a convex
combination, which undermines the convergence proof in Appendix A.1 (which
explicitly assumes valid convex weights).

**Fix required:** project `w` onto the positive simplex (e.g. `softmax` or
`simplex_projection`) before returning from `update()`.

---

### 1.2 Framework mismatch: NumPy scratch implementation vs. Keras

**File:** `fedgraa/model.py`  
**Paper:** Section 5, "Network Architecture"

The paper states: *"We designed a fully connected neural network using the
**Keras Sequential API**."*

The reproduction re-implements the entire forward/backward pass in NumPy. This
introduces multiple numerical differences:

- Keras' `sparse_categorical_crossentropy` is computed directly from logits
  (numerically stable fused kernel), whereas `MLP.loss_and_grad()` applies
  `forward()` (which calls `softmax()`) then `log(probs + 1e-12)`. The
  `1e-12` clamp is not present in Keras.
- Keras' Glorot uniform uses `fan_in + fan_out` exactly; the code matches this,
  but internal float32 vs float64 precision may differ.
- The paper likely ran in float32 (Keras/TF default); the code uses float64
  throughout.

**Consequence:** Even with identical seeds, the trajectories will diverge
because the forward passes are numerically different.

---

### 1.3 Numerical inconsistency between `MLP.loss()` and `MLP.loss_and_grad()`

**File:** `fedgraa/model.py`

`MLP.loss()` computes the loss using numerically stable log-softmax on raw
logits (skips the softmax activation in the final layer).  
`MLP.loss_and_grad()` calls `forward()` which applies softmax, then takes
`log(probs + 1e-12)`.

These two functions return **different loss values** for the same inputs because
one path is `log_softmax(z)` and the other is `log(softmax(z) + 1e-12)`. The
finite-difference gradient tests in `test_model.py` still pass because they
compare `MLP.loss()` against `MLP.loss_and_grad()` — and the bug is that the
test uses `MLP.loss()` for the finite-difference perturbations while the analytic
gradient comes from `MLP.loss_and_grad()`, so the two must agree despite the
discrepancy. In practice this means the reported training loss and the gradient
direction are derived from **inconsistent objective functions**.

---

### 1.4 UNSW-NB15 data directory is empty

**Path:** `data/UNSW-NB15/` (zero CSV files)  
**Paper:** Section 5 — results reported on both CIC-IDS-2017 *and* UNSW-NB15

The `load_dataset()` fallback to synthetic data is transparent about the
switch, but all UNSW-NB15 numbers in the reproduction will come from a
synthetic Gaussian dataset, not the real benchmark. The paper's UNSW-NB15
figures (Table 2, Figure 3, the summary row "FedGRAA (γ=1): 21.09%") cannot
be verified.

---

## 2. Major Gaps

### 2.1 FedGRAA Adam optimizer state persists across all 200 rounds

**File:** `fedgraa/aggregation.py` — `FedGRAA.__init__()` stores a single
`Adam` instance; `fedgraa/trainer.py` — `FederatedTrainer` creates one
`FedGRAA` object for the whole run.

**Paper:** Algorithm 2 describes `E'` epochs of gradient descent on `w` per
round, with learning rate `η'`. The paper does not say whether optimizer
momentum carries over between rounds.

Standard practice (and the natural reading of "E' epochs of gradient descent
per round") is to treat each round's weight update as an independent
optimization sub-problem with a fresh optimizer. Instead, the code's Adam
accumulates first and second moment estimates (`m`, `v`) and the step counter
`t` across the entire 200-round training run. By round 50, `t = 250`
(50 rounds × 5 epochs), so the bias-correction terms and effective step size
are very different from what a per-round fresh optimizer would produce.

**Contrast with client optimizer:** clients use `optimizer = Adam(lr=cfg.eta)`
freshly created inside `_client_update()` every round — the exact opposite
convention for a structurally identical sub-problem.

---

### 2.2 Client Adam optimizer is reset every round (inconsistent with standard FL)

**File:** `fedgraa/trainer.py` — `_client_update()`

```python
optimizer = Adam(lr=cfg.eta)   # fresh every call
```

The standard FL practice (FedOpt, FedProx, etc.) persists client optimizer
state across rounds so that local momentum carries useful gradient information
into the next round. The paper describes using Adam but does not state whether
state is reset. If the original Keras run kept optimizer state (which is the
Keras default when the same `model.fit()` is called repeatedly), this
diverges.

---

### 2.3 Benign samples are drawn from an unpartitioned shared pool

**File:** `fedgraa/partition.py` — `partition_non_iid()`

The benign/normal indices are not removed from the pool after being assigned
to a client:

```python
benign_idx_all = np.flatnonzero(y == spec.benign_id)
# ... inside the client loop:
benign_pick = rng.choice(benign_idx_all, size=n_benign, replace=replace)
```

The same benign row can appear in **multiple clients' datasets**. For
CIC-IDS-2017, the 14 attack clients with capped classes each request up to
10,000 benign samples; the total benign pool is only 63,901. So many benign
samples will appear in 2–5 different clients simultaneously.

The paper says "each client is assigned data for one specific attack type
paired with an equal number of Benign samples" — the word *assigned* implies
disjoint allocation. Whether or not this was intended, the implementation
does not match the typical disjoint federated data setup.

---

### 2.4 Rare class selection for n=5 and n=7 is inferred, not stated in paper

**File:** `fedgraa/config.py` — `rare_label_order` fields  
**Paper:** Section 5 states only n=3 rare classes for each dataset.

For CIC-IDS-2017 the code extends protection to `{XSS, Brute Force, Bot,
DoS Slowhttptest}` when n=5,7. For UNSW-NB15 the 4th and 5th rare classes
are `{Analysis, Fuzzers}`. These choices are reasonable (next smallest by
count) but are **not specified anywhere in the paper**. Figures 2 and 3 show
aggregate accuracy curves without identifying which specific classes were added,
so there is no way to verify the selection is correct.

---

### 2.5 Round 0 metrics not recorded

**File:** `fedgraa/trainer.py` — `FederatedTrainer.train()`

The paper's Figure 1 table starts at communication round 0 (all values 0.00).
The code begins recording metrics only at the first `eval_every` checkpoint
(round 20 by default). A `history.record(0, 0.0, 0.0, w)` call before the
training loop is absent, so the reproduced curves start at round 20 and cannot
be compared directly to the paper's tables.

---

### 2.6 Ambiguous median-frequency split boundary

**File:** `fedgraa/partition.py` — `median_frequency_split()`  
**Paper:** Section 4 — *"an automated method that determines the split threshold
based on the median frequency"*

The code classifies a class as "highly frequent" (D_V^h) if its count is
**strictly greater than** the median:

```python
high_classes = {c for c, f in freq_by_class.items() if f > median_freq}
```

When multiple classes share the median count, they all fall into D_V^m. The
paper gives no formula; "greater than or equal to" would classify differently.
For small numbers of non-rare classes (e.g., UNSW has only 6 after removing
3 rare classes), the choice of strict vs. non-strict threshold can double or
halve the size of D_V^h, directly affecting `L_a = min(L_h, L_m)`.

---

## 3. Minor Gaps

### 3.1 No figures (plots) are generated by any code path

**Files:** `run_experiments.py`, `fedgraa/experiment.py`, `FedGRAA_reproduction.ipynb`

The paper presents Figures 1, 2, and 3 as accuracy-vs-communication-round
line plots. The code produces only stdout text tables. The notebook includes
one matplotlib cell for a Figure-1-style last-n accuracy plot, but:

- It is **never executed** (no cell outputs exist).
- It only covers the gamma sweep (Figure 1 style) — there is no equivalent
  cell for the n-sweep (Figures 2 and 3).
- It plots only `last_n_acc`; the paper's Figure 1 shows both "All classes"
  and "Last n classes" accuracy for each method.

---

### 3.2 `summarize_gamma_sweep()` reports only the final round

**File:** `fedgraa/experiment.py` — `summarize_gamma_sweep()`

The function prints `final_all_acc` and `final_last_n_acc` (round 200 values
only). The paper's Figure 1 table spans rounds 0–200 in steps of 20. There
is no equivalent function that produces the full per-round table.

---

### 3.3 Headline accuracy claims not asserted anywhere

**Paper:** Abstract — *"FedGRAA achieves 88% accuracy on minority classes…
outperforming FedAvg by up to 55%"*

No test, assertion, or benchmark exists that checks whether the reproduction
reaches these specific numbers. The test suite (`test_trainer.py`) only
verifies that FedGRAA's rare-class accuracy is ≥ FedAvg's on a tiny synthetic
toy dataset (5 classes, 40 rounds), which is far from the paper's conditions.

---

### 3.4 No result persistence (CSV / JSON export)

All metrics are printed to stdout. There is no code to save `TrainHistory`
objects to disk. Reproducing Table comparisons requires re-running the full
200-round experiments every time.

---

### 3.5 CIC-IDS-2017 label mapping not fully tested on real CSVs

**File:** `fedgraa/data.py` — `_match_labels()` and `_LABEL_ALIASES`  
**Paper:** Table 1 label names

The real CIC-IDS-2017 CSVs use the label strings `"Web Attack – Brute Force"`,
`"Web Attack – SQL Injection"`, and `"Web Attack – XSS"`. The code strips a
`"web attack"` prefix via `_normalize_label()`, which should handle these. However:

- This path is not covered by any test with actual CSV content.
- The `_LABEL_ALIASES` dict maps the lowercase normalized form, but the raw CSV
  strings contain an em-dash (`–`) not an ASCII hyphen (`-`). `re.sub(r"[^a-z0-9]+", " ", t)` strips both equally, so this likely works, but it is untested.
- `"DoS GoldenEye"` in the CSVs is written as `"DoS Goldeneye"` (lowercase e).
  The normalization lowercases everything, so `"dos goldeneye"` must match
  `"dos goldeneye"` from normalizing the paper label. This works, but the
  edge-case is never tested.

---

### 3.6 No per-class precision/recall/F1 reporting

**Paper:** Section 5 — *"FedGRAA achieves 88% accuracy on minority classes"*

"Accuracy on minority classes" is implemented as **mean per-class recall** over
the rare classes (`last_n_class_accuracy`). The paper does not confirm whether
this matches its own definition. No confusion matrix, per-class precision, or
F1-score is computed anywhere, making it impossible to verify class-level
claims independently.

---

### 3.7 Convergence proof assumptions are not verified at runtime

**Paper:** Appendix A.1 — assumes L' is convex and β-smooth, gradients are
bounded by G, and learning rates satisfy `η', η ≤ 1/β`.

The code does not:
- Estimate or bound the Lipschitz constant β of L'.
- Check whether `η' = 0.1` satisfies `η' ≤ 1/β` in practice.
- Monitor whether `L'(w^t) ≤ L'(w^{t-1})` (the non-increasing condition in
  Lemma 1) holds during training.

Given that L' is a composition of a softmax neural network loss (non-convex in
the model parameters) with a linear aggregation, the convexity assumption does
not hold in the general parameter space. The proof is valid only in the weight
space `w`, holding client model parameters fixed — this subtlety is not
highlighted in the reproduction.

---

### 3.8 Edge case: L_a when both D_V^h and D_V^m are empty

**File:** `fedgraa/aggregation.py` — `FedGRAA.composite_loss()` and
`_composite_grad_P()`

```python
if not np.isfinite(L_a):
    L_a, g_a = 0.0, [np.zeros_like(p) for p in params]
```

When both D_V^h and D_V^m are empty (possible early in UNSW training when the
median split produces degenerate buckets), `L_a = min(inf, inf)` is replaced
with 0.0. This silently zeroes out the `(1-γ)·L_a` term and makes the
composite loss depend only on `L_u`, regardless of the actual γ value.
The paper provides no specification for this edge case.

---

### 3.9 `DATASETS` registry not exported from the package

**File:** `fedgraa/__init__.py`

`DATASETS` (the dict mapping name → `DatasetSpec`) is used in `__main__.py`
for the CLI choices list, but is absent from `__init__.py`'s `__all__`.
Adding new datasets requires editing both `config.py` and `__main__.py`
separately, and external callers cannot programmatically enumerate supported
datasets without importing from `config` directly.

---

### 3.10 Notebook is designed for Colab but never validated locally

**File:** `FedGRAA_reproduction.ipynb`

All cells are unexecuted. The notebook is written for Google Colab (uses
`google.colab.files`, `google.colab.drive`), making it impossible to run
directly in VS Code or JupyterLab without modification. There is no local
equivalent workflow documented for the `data/CIC-IDS-2017/` CSV files that
already exist in the repository.

---

## Summary Table

| # | Category | Severity | File(s) | Paper Section |
|---|----------|----------|---------|---------------|
| 1.1 | Simplex projection missing on `w` | **CRITICAL** | `aggregation.py` | §4, Eq. 10 |
| 1.2 | NumPy vs Keras framework mismatch | **CRITICAL** | `model.py` | §5 Network Arch. |
| 1.3 | `loss()` vs `loss_and_grad()` inconsistency | **CRITICAL** | `model.py` | §3, Eq. 3 |
| 1.4 | UNSW-NB15 data absent (empty directory) | **CRITICAL** | `data/UNSW-NB15/` | §5 Data Settings |
| 2.1 | FedGRAA Adam state persists across rounds | **MAJOR** | `aggregation.py`, `trainer.py` | Algorithm 2 |
| 2.2 | Client Adam reset each round | **MAJOR** | `trainer.py` | Algorithm 1 |
| 2.3 | Benign samples reused across clients | **MAJOR** | `partition.py` | §5 Data Settings |
| 2.4 | n=5, n=7 rare class selection inferred | **MAJOR** | `config.py` | §5 (Figs 2–3) |
| 2.5 | Round 0 not recorded in history | **MAJOR** | `trainer.py` | Fig. 1 |
| 2.6 | Median-split strict vs non-strict boundary | **MAJOR** | `partition.py` | §4, Eq. 8 |
| 3.1 | No figures/plots generated | **MINOR** | `experiment.py`, notebook | Figs 1–3 |
| 3.2 | `summarize_gamma_sweep` only final round | **MINOR** | `experiment.py` | Fig. 1 table |
| 3.3 | Headline accuracy numbers not asserted | **MINOR** | `tests/` | Abstract |
| 3.4 | No result persistence (CSV/JSON) | **MINOR** | all | — |
| 3.5 | Real CSV label mapping untested | **MINOR** | `data.py` | §5, Table 1 |
| 3.6 | No per-class precision/recall/F1 | **MINOR** | `metrics.py` | §5 Results |
| 3.7 | Convergence assumptions unverified | **MINOR** | — | App. A.1 |
| 3.8 | Edge case: both D_V^h and D_V^m empty | **MINOR** | `aggregation.py` | §4, Eq. 8 |
| 3.9 | `DATASETS` not exported from `__init__` | **MINOR** | `__init__.py` | — |
| 3.10 | Notebook untested outside Colab | **MINOR** | `FedGRAA_reproduction.ipynb` | — |
