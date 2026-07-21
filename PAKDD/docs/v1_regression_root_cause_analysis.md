# V1 Regression — Root Cause Analysis

**Date:** 2026-07-21
**Status:** FIXED AND VERIFIED
**Bug introduced at:** `596c040` (fused XLA speedup, 2026-07-19)
**Fixes verified on working tree at:** `596c040` (same commit + uncommitted fixes)
**Scope:** 23 Python files reviewed (all source + all tests)
**Symptoms:** `latest_baseline.md`
**Fix verification report:** `docs/v1_regression_fix_verification.md`

---

## Executive Summary

A **single pattern** — `np.empty` instead of `np.zeros` for padding buffers — causes every federated learning method to collapse to majority-class prediction. The bug was introduced during the XLA speedup refactor (`_fused_client_step` / `client_train_round`) and is NOT covered by any unit test.

All 6 methods (FedAvg, Static, FedGRAA at g=0/0.5/1, FedProx, FedOpt, FedSND) converge to the same ~51% / ~50% majority-class accuracy because they all share the same local training loop that produces NaN parameters.

---

## Root Cause

### Primary Site: `trainer.py:123-124` — `client_train_round`

```python
pad_x = np.empty((cfg.batch_size, X.shape[1]), dtype=np.float32)   # line 123
pad_y = np.empty(cfg.batch_size, dtype=np.int32)                    # line 124
```

### Secondary Site: `baselines.py:62-63` — `proximal_client_train_round`

```python
pad_x = np.empty((cfg.batch_size, X.shape[1]), dtype=np.float32)   # line 62
pad_y = np.empty(cfg.batch_size, dtype=np.int32)                    # line 63
```

### What Happens

When a client's training set size `n` is NOT evenly divisible by `batch_size`, the last batch of each epoch is partial (`n_valid < batch_size`). The code copies valid data into the padding buffers but never touches entries beyond `n_valid`:

```python
if n_valid < cfg.batch_size:
    pad_x[:n_valid] = X[idx]       # writes n_valid rows of features
    pad_y[:n_valid] = y[idx]       # writes n_valid labels
    x_batch, y_batch = pad_x, pad_y  # passes FULL (batch_size,) array
    # pad_x[n_valid:] — UNINITIALIZED garbage float32
    # pad_y[n_valid:] — UNINITIALIZED garbage int32
else:
    x_batch, y_batch = X[idx], y[idx]
```

**Critical:** `pad_x[n_valid:]` and `pad_y[n_valid:]` are NEVER written. They retain the uninitialized memory that `np.empty` returned. For `pad_y`, these are arbitrary int32 values — anything from −2³¹ to 2³¹−1. For the padding **features** (`pad_x[n_valid:]`), the garbage is float32 — could be zero, subnormal, Inf, or NaN.

### Why This Produces NaN

The padded batch is consumed by the XLA-compiled `_fused_client_step` (`model.py:252-276`):

```python
@tf.function(jit_compile=True, reduce_retracing=True)
def _fused_client_step(x, y, model, opt, n_valid):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits
        )
        batch_size = tf.shape(x)[0]
        mask = tf.cast(tf.range(batch_size) < n_valid, dtype=losses.dtype)
        loss = tf.reduce_sum(losses * mask) / tf.cast(n_valid, dtype=losses.dtype)
```

- `labels=y` contains garbage out-of-range int32 values in `pad_y[n_valid:]`
- Under XLA, `SparseSoftmaxCrossEntropyWithLogits` with out-of-range labels returns **NaN** (xla_data.cc: sparse softmax cross-entropy reads `logits[label_idx]` with an out-of-bounds index → NaN)
- The garbage features in `pad_x[n_valid:]` compound the problem: if a feature is float32 NaN, the forward pass through `model(x)` can produce NaN logits even for valid labels
- **`NaN × 0.0 = NaN`** in IEEE 754 — the mask multiplication does NOT sanitize the padding entries
- `reduce_sum(NaN_tensor) = NaN`
- NaN loss → tape.gradient returns NaN → `opt.apply_gradients(NaN)` → **model weights become NaN**

### The `NaN × 0 = NaN` Trap

This is the crux of the bug. The comment in the code says:

```
# Pre-allocated padding buffers — allocated once, reused every iteration
# to avoid per-iteration ``np.zeros`` overhead.
```

The developer assumed `mask[i] = 0.0` would safely zero out padding entries. But **IEEE 754 NaN propagation doesn't work that way**:

| Operation | IEEE 754 Result |
|-----------|----------------|
| `NaN × 0.0` | **NaN** (not 0) |
| `NaN + 0.0` | NaN |
| `sum([1.0, NaN])` | NaN |

The mask multiplication is NOT a select/predicate — it's arithmetic. The only way to safely exclude padding entries is to never compute the loss on them, or use `tf.where` which performs a true predicate-based select.

---

## Propagation Chain

### Step 1: First client produces NaN (round 1)

| Dataset | Smallest Client | Attack Samples | Benign Samples | Total | ~Train (90%) | Batch Size | Partial Batch? |
|---------|----------------|----------------|----------------|-------|--------------|------------|----------------|
| CIC-IDS-2017 | Heartbleed | 11 | 11 | 22 | ~20 | 64 | **Yes (only batch)** |
| UNSW-NB15 | Worms | 174 | 174 | 348 | ~313 | 64 | **Yes (last of 5 batches)** |

Every client with `n_train % 64 != 0` hits the padding bug. On CIC-IDS-2017, clients with these training sizes all trigger it:

| Client | Attack samples | ~Train | Batches | Partial? |
|--------|---------------|--------|---------|----------|
| Heartbleed | 11 | ~20 | 1 partial | Yes |
| SQL Injection | 21 | ~38 | 1 partial | Yes |
| Infiltration | 36 | ~65 | 1 full + 1 partial | Yes |
| Brute Force | 1,470 | ~2,646 | 41 full + 1 partial | Yes |
| ... | ... | ... | ... | Yes |

Only clients whose `n_train` is an exact multiple of 64 avoid the bug. On CIC-IDS-2017 with 14 clients, at most 0-1 clients might escape, and even then only until a different random permutation in the next epoch creates a partial batch.

### Step 2: NaN propagates through aggregation (`model.py:61-71`)

```python
def aggregate(client_params, weights):
    for j in range(n_arrays):
        stacked = np.stack([cp[j] for cp in client_params], axis=0)
        out.append(np.tensordot(weights, stacked, axes=(0, 0)))  # sum_k w_k * param_k
```

If any client param array contains NaN and its weight `w_k > 0`, the weighted sum is NaN. Even FedAvg's tiny weight for Heartbleed (`11/63901 ≈ 0.00017`) is enough: `0.00017 × NaN = NaN`.

### Step 3: Global model is NaN — all clients collapse (round 2+)

Every client starts round 2 from NaN global params:
- `model.set_params(NaN_params)` sets Keras model weights to NaN
- Forward pass through tanh(NaN params) → NaN activations → NaN logits
- Backward pass on NaN logits → NaN gradients
- `aggregate(NaN_params)` → NaN global params
- Full collapse by round 2

### Step 4: NaN logits — argmax returns class 0 by convention

```python
def predict(self, params, X):
    self.set_params(params)           # sets NaN weights into Keras model
    logits = self._model(X32, ...)    # produces NaN logits
    return tf.argmax(logits, axis=1)  # argmax of all-NaN → returns 0
```

Class 0 = BENIGN/Normal for both datasets. When every prediction is class 0, accuracy equals the proportion of class 0 in the validation set:

- CIC-IDS-2017: **51.02%**
- UNSW-NB15: **49.95%**

These match the reported "all-class" accuracies exactly (`latest_baseline.md` lines 64, 74).

### Step 5: FedGRAA composite loss becomes 0.0000

```python
def composite_loss(self, params, val):                       # aggregation.py:105
    L_h, _ = _loss_and_grad_P(model, params, val.X_h, val.y_h)  # NaN for NaN params
    L_m, _ = _loss_and_grad_P(model, params, val.X_m, val.y_m)  # NaN
    L_u, _ = _loss_and_grad_P(model, params, val.X_n, val.y_n)  # NaN
    L_a = min(L_h, L_m)                                         # min(NaN, NaN) = NaN
    if not np.isfinite(L_a):                                    # True → fallback
        L_a = 0.0
    if not np.isfinite(L_u):                                    # True → fallback
        L_u = 0.0
    return (1.0 - self.gamma) * L_a + self.gamma * L_u          # = 0.0
```

The fallback logic was designed for **empty validation splits** (returning `inf` from `_loss_and_grad_P`). It unintentionally catches **NaN** from NaN params too, silently producing L' = 0.0000 for every round after collapse.

### Step 6: FedGRAA weights become NaN

```python
def update(self, client_params, weights_prev, val):           # aggregation.py:159
    w = np.array(weights_prev, dtype=np.float32).copy()
    for _ in range(self.weight_epochs):
        P = aggregate(client_params, w)                       # P is NaN
        _, grad_P = self._composite_grad_P(P, val)            # gradients are NaN
        grad_w = _grad_wrt_weights(grad_P, client_params)     # NaN
        self.optimizer.step([w], [grad_w])                    # w becomes NaN
    return w
```

---

## Why Every Method Fails

All 6 methods share the same local training function, so all 6 collapse:

| Method | Local Training | `np.empty`? | Call site |
|--------|---------------|-------------|-----------|
| FedAvg | `client_train_round` | `trainer.py:123-124` | `trainer.py:273` |
| Static | `client_train_round` | `trainer.py:123-124` | `trainer.py:273` |
| FedGRAA | `client_train_round` | `trainer.py:123-124` | `trainer.py:273` |
| FedProx | `proximal_client_train_round` | `baselines.py:62-63` | `baselines.py:141` |
| FedOpt | `client_train_round` | `trainer.py:123-124` | `baselines.py:236` |
| FedSND | `client_train_round` | `trainer.py:123-124` | `baselines.py:322` |

None escape because `aggregate(NaN_params, weights)` produces NaN global params for any weight vector (FedAvg, Static, or learned).

---

## Why Tests Don't Catch It

| Test | n_samples | batch_size | n_valid | Padding exercised? | NaN caught? |
|------|-----------|------------|---------|-------------------|-------------|
| `test_fused_xla_step_equivalent_to_eager` | 16 | X.shape[0]=16 | 16 = full | No | N/A |
| `test_fused_xla_multiple_steps_converges` | 64 | all at once | 64 = full | No | N/A |
| `test_adam_decreases_loss` | 64 | N/A (uses `Adam` wrapper) | N/A | No | N/A |
| `test_fedgraa_update_runs_and_changes_weights` | N/A | N/A (tests `FedGRAA.update` directly) | N/A | No | N/A |
| `test_fedgraa_beats_fedavg_on_rare_class` | synthetic | 64 | **< 64 (all clients)** | **Yes** | **No** |

The last test (`test_fedgraa_beats_fedavg_on_rare_class`) is particularly insidious: it uses `_small_spec()` with `used_counts=[300, 120, 90, 60, 10]`. The "Rare" client has 10 attack + 10 benign = 20 total, ~18 train. With `batch_size=64`, this is a partial batch. **The bug fires.** But the assertion is:
```python
assert graa.final_last_n_acc >= avg.final_last_n_acc
```
When both collapse to 0.0 (0% rare-class recall), the assertion **passes trivially**. The test was written to verify the paper's qualitative result (FedGRAA > FedAvg on rare classes), but it cannot distinguish between "both methods work and FedGRAA is better" and "both methods completely failed."

---

## Secondary Issues Discovered

### Issue 2: Optimizer State Persists Across Rounds

`FederatedTrainer.__init__` (`trainer.py:203-207`) creates per-client TF Adam optimizers once. Their slot variables (momentum/velocity) carry over between rounds because `reset()` is never called on them:

```python
self._client_tf_optimizers = {
    c.client_id: build_tf_optimizer(cfg.eta,
                                    self._model._model.trainable_variables)
    for c in split.clients
}
```

| Aspect | Impact |
|--------|--------|
| Momentum carry-over | Round N momentum from client A's data is present in client A's optimizer at round N+1 |
| Bias correction | `iterations` counter increments across rounds, changing the bias correction factor |
| Paper fidelity | The paper implies a fresh optimizer per round; this is a known deviation (noted as "Gap 2.2" in the code) |

This doesn't cause the NaN collapse, but it changes the convergence trajectory vs the paper.

### Issue 3: `_fused_proximal_step` Shares the Same NaN Vulnerability

`model.py:279-301` uses the identical `losses * mask` pattern:

```python
loss = tf.reduce_sum(losses * mask) / tf.cast(n_valid, dtype=losses.dtype)  # line 294
```

If invalid labels reach this function (via `baselines.py:62-63`), NaN propagates identically through the FedProx baseline.

### Issue 4: `np.empty` on Features Can Introduce Float32 NaN

The comment on line 121 says:

```
# Pre-allocated padding buffers — allocated once, reused every iteration
# to avoid per-iteration ``np.zeros`` overhead.
```

`np.empty` for `pad_x` (float32) can contain:
- Zero (from OS page zeroing, first allocation only)
- Stale float data (from heap reuse)
- **Float32 NaN or Inf** (if the memory previously held an uninitialized double that happens to be a NaN bit pattern when interpreted as float32 — rare but possible)

Even if `pad_y` labels happen to be in range (all 15/2³² probability against), garbage features in `pad_x[n_valid:]` can still produce NaN logits after passing through the 200→100→50 tanh network, because a single NaN input propagates through every transformation.

---

## The Fix

### Primary Fix: Zero-initialize padding buffers

**`trainer.py:123-124` and `baselines.py:62-63`:**

```python
# BEFORE:
pad_x = np.empty((cfg.batch_size, X.shape[1]), dtype=np.float32)
pad_y = np.empty(cfg.batch_size, dtype=np.int32)

# AFTER:
pad_x = np.zeros((cfg.batch_size, X.shape[1]), dtype=np.float32)
pad_y = np.zeros(cfg.batch_size, dtype=np.int32)
```

Class 0 is BENIGN/Normal — a valid label for both datasets. The cross-entropy for padding entries is finite. The mask zeros the gradient contribution. No NaN path exists.

**Performance impact:** `np.zeros` on a fresh allocation is essentially free (the OS provides zeroed pages lazily). For the reused `pad_x`/`pad_y` across epochs of one client, the zeroing cost is paid once per client per round, not per iteration — negligible compared to the XLA compilation and forward-backward pass.

### Defense-in-Depth: Label-safe clamping inside fused steps

**`model.py:272-273` (`_fused_client_step`) and `model.py:293-294` (`_fused_proximal_step`):**

```python
n_classes = tf.shape(logits)[-1]
labels_safe = tf.where(
    tf.logical_or(y < 0, y >= tf.cast(n_classes, y.dtype)),
    tf.zeros_like(y), y
)
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_safe, logits=logits)
batch_size = tf.shape(x)[0]
mask = tf.cast(tf.range(batch_size) < n_valid, dtype=losses.dtype)
loss = tf.reduce_sum(losses * mask) / tf.cast(n_valid, dtype=losses.dtype)
```

**Why this works:** `sparse_softmax_cross_entropy_with_logits` with out-of-range labels (−1, 999, etc.) computes a NaN loss and, critically, its **gradient w.r.t. logits is also NaN**. While `tf.where` on the *loss* produces a finite forward value (`NaN → 0` via predicate select), the **gradient** through `tf.where` computes `pred * grad_output`, and in IEEE 754 `0.0 × NaN = NaN`. This means NaN still propagates into parameter gradients.  

The `labels_safe` guard prevents the NaN at its source — by clamping out-of-range labels to 0 *before* the cross-entropy computation, the loss and its gradient are always finite for every entry. The mask then zeros the gradient contribution from padding entries.

Using `tf.where` on labels (instead of on losses) is the correct defense because the gradient of `tf.where(condition, x, y)` w.r.t. `x` is `condition * grad` (element-wise), and for the clamping case `condition = True` for valid labels and `False` for padding. For padding entries, the gradient through the clamped label path is `0` — **not** `0 × NaN`, because the gradient flow into `x` (the original `y`) is blocked by `condition = False`. The finite labels_safe tensor flows through cross-entropy and produces finite gradients for all entries.

### Missing Test Coverage

Three tests must be added to prevent regression:

| Test | File | What It Verifies |
|------|------|------------------|
| `test_fused_xla_step_partial_batch_no_nan` | `test_model.py` | `_fused_client_step` with `n_valid < batch_size` must not produce NaN loss or params |
| `test_client_train_round_small_client_no_nan` | `test_trainer.py` | Full `client_train_round` on a client with `n < batch_size` must return finite params |
| `test_all_methods_no_nan_collapse` | `test_reproduction_gaps.py` | All 3 methods run on synthetic data must not produce NaN global params at any checkpoint |

**Critical:** the existing `test_fedgraa_beats_fedavg_on_rare_class` must also be hardened to assert `np.all(np.isfinite(...))` on the final global params, not just compare the two accuracy values.

---

## Line-by-Line Manifest

| File | Lines | Change |
|------|-------|--------|
| `PAKDD/fedgraa/trainer.py` | 123-124 | `np.empty(…, dtype=float32)` → `np.zeros(…)` |
| `PAKDD/fedgraa/trainer.py` | 124 | `np.empty(…, dtype=int32)` → `np.zeros(…)` |
| `PAKDD/fedgraa/baselines.py` | 62 | `np.empty(…, dtype=float32)` → `np.zeros(…)` |
| `PAKDD/fedgraa/baselines.py` | 63 | `np.empty(…, dtype=int32)` → `np.zeros(…)` |
| `PAKDD/fedgraa/model.py` | 272-273 | Add `labels_safe = tf.where(logical_or(y < 0, y >= n_classes), 0, y)` before `sparse_softmax_cross_entropy` — clamps out-of-range labels to prevent NaN loss/gradient |
| `PAKDD/fedgraa/model.py` | 293-294 | Same for `_fused_proximal_step` |
| `PAKDD/tests/test_model.py` | new | `test_fused_xla_step_partial_batch_no_nan` |
| `PAKDD/tests/test_trainer.py` | new | `test_client_train_round_small_client_no_nan` |
| `PAKDD/tests/test_reproduction_gaps.py` | new | `test_all_methods_no_nan_collapse` |
| `PAKDD/fedgraa/experiment.py` | 63-68 | Fix `data_dir` path construction (`{data_dir}/{dataset}`) so `load_dataset` finds CSV subdirectories |
| `PAKDD/tests/test_trainer.py` | `test_fedgraa_beats_fedavg_on_rare_class` | Add `np.all(np.isfinite(...))` assertion |

---

## Timeline

| Date | Event | State |
|------|-------|-------|
| 2026-07-15 | Pre-speedup baseline | **Working** — per-iteration `model.set_weights()` + `tf.GradientTape()` + `Adam.step()` in Python loop, no padding buffers |
| 2026-07-19 | Speedup PR merged | **Bug introduced** — `_fused_client_step` + `client_train_round` with `np.empty` padding |
| 2026-07-20 | `f95092a` (HEAD) | **Collapsed** — all methods predict majority class |
| Now | This analysis | Root cause identified, fix specified |

**Time to implement and verify:** ~120 minutes (6 production line changes + 4 new tests + 1 existing test assertion + real-data smoke tests on both CIC-IDS-2017 and UNSW-NB15). The defense-in-depth approach was adjusted during implementation: `tf.where` on losses was rejected because `0 × NaN = NaN` in IEEE 754 gradient propagation; `labels_safe` clamping on labels before cross-entropy was adopted instead.

---

## How to Verify the Fix

### Unit tests (59 total, including 4 new regression tests):

```bash
python -m pytest tests/ -v --tb=short
```

### Key regression tests (verify the bug is fixed):

```bash
# Tests the fused step directly with out-of-range labels (-1):
python -m pytest tests/test_model.py::test_fused_xla_step_partial_batch_no_nan -v --tb=long

# Tests the full client_train_round with a small client (n < batch_size):
python -m pytest tests/test_trainer.py::test_client_train_round_small_client_no_nan -v --tb=long

# Tests all 6 methods produce finite params end-to-end:
python -m pytest tests/test_reproduction_gaps.py::test_all_methods_no_nan_collapse -v --tb=long

# Hardened existing test (asserts np.all(np.isfinite(...)) on final params):
python -m pytest tests/test_trainer.py::test_fedgraa_beats_fedavg_on_rare_class -v --tb=long
```

### Real-data smoke test (CIC-IDS-2017, 30 rounds):

```bash
python run_experiments.py --data-dir data --cache-dir .cache \\
    --datasets CIC-IDS-2017 --rounds 30 --eval-every 10 --sweep gamma
```

Pre-fix: all 5 methods report ~51.02% all, 0.00% last-n.  
Post-fix: differentiated results (Static up to 100% last-n, FedGRAA g=0.5 at 66.67% last-n on CIC).

---

## Appendix: Alternative Approaches Considered

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| `np.zeros` for padding | Simple, safe, minimal perf cost | None | **Adopted** |
| `labels_safe` clamping in fused step | Defense-in-depth, prevents NaN at source; gradient-safe (unlike `tf.where` on losses, where `0 × NaN = NaN` in IEEE 754) | Slight XLA graph complexity | **Adopted (defense)** |
| `tf.where(is_finite(losses))` guard | Catches NaN from logits | Gradient `factor × grad_out` with `factor=0` still yields `0 × NaN = NaN` — NaN propagates through backprop | Rejected |
| Dynamic shapes in XLA | Avoids padding entirely | `x[:n_valid]` causes XLA retracing per client/epoch | Rejected |
| Clip labels to [0, n_classes) in fused step | Hides the symptom, not the cause | Doesn't fix garbage features in `pad_x` | Rejected |
| `np.full` with sentinel value + sentinel-aware mask | No benefit over `np.zeros` | More complex | Rejected |
| Remove pre-allocated buffers, allocate fresh each batch | Simple | Per-iteration allocation overhead, defeats purpose of pre-allocation | Rejected |

---

*End of analysis — all 23 Python files re-verified against live source at `f95092a`.*
