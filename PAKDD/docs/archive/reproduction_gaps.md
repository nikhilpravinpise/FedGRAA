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

---

*(De-duplicated 2026-08-03: this document previously repeated its entire
contents a second time below this point — the same 20 items, verbatim,
minus their "✅ Fixed" status and "Fix"/"Fix required" framing, i.e. an
earlier, pre-fix draft of the exact same audit left in place after the
"Fixed" version above it was written over it. Layered with this file's own
top banner (which separately warns that item 1.1's fix was later reverted),
a reader could end up with three undated, contradictory beliefs about the
same code path with no way to tell which was current. The older draft added
no information not already in the "Fixed" version above — every file/line
reference, severity, and paper-section citation is identical — so it was
removed rather than kept as a second, undated copy. See git history for the
pre-fix wording if it's ever needed.)*
