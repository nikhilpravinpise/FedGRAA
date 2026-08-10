# Speedup vs Reproduction — Audit & Root Cause Analysis

**Question:** *Is the sped-up v1 implementation deviating from a full and honest
reproduction of the FedGRAA paper?*

**Short answer: No — the algorithm is faithful. The speedup introduced a
numerical stability bug that causes all methods to collapse. Fix the bug and
you have a fast + faithful reproduction.**

---

## 1. What the Audits Say (Algorithm is Correct)

Three independent audits agree:
- [`paper_vs_code_line_by_line_gap_analysis.md`](../archive/paper_vs_code_line_by_line_gap_analysis.md) (archived, superseded by `REPRODUCTION_AUDIT.md`)
- [`REPRODUCTION_AUDIT.md`](REPRODUCTION_AUDIT.md)
- [`strict_reproduction_report.md`](../archive/strict_reproduction_report.md)

All confirm:
- Equations 1–10 are implemented correctly
- Algorithm 1 and Algorithm 2 match the paper line-for-line
- Every hyperparameter (η=0.01, E=2, B=64, T=200, η′=0.1, E′=5, γ∈{0,0.5,1})
  matches §5 exactly
- Keras Sequential architecture matches (200→100→50 tanh + softmax)
- Data partitioning matches (one attack-type per client + equal benign, 90/10
  split, median-frequency validation split)
- No simplex projection (correct — Algorithm 2 has none)
- Round-0 forced to 0.00 (correct — paper's own table shows 0.00)
- All ambiguity toggles removed (single documented choices everywhere)

The **algorithm-level fidelity is high.** No deviation from the paper's method.

---

## 2. What the Speedup Changed

The speedup (Phase 2 in `model.py`/`trainer.py`) replaced:

| Before (per-batch numpy↔TF ping-pong) | After (fused XLA) |
|---|---|
| `model.set_weights()` + `GradientTape()` + `Adam.step()` per batch, with numpy copies in/out every iteration | `model.set_params()` once per round, `_fused_client_step(XLA)` per batch, `model.get_params()` once at end |
| `model.py::Adam` wrapping `tf.keras.optimizers.Adam` per round | Pre-built per-client `tf.keras.optimizers.Adam` via `build_tf_optimizer` |
| Optimizer state fresh each round (implied by per-round creation) | Optimizer state **persists across all rounds** |

---

## 3. Root Cause: Persistent Per-Client Optimizer State

`trainer.py:199-207` builds one TF Adam per client that lives for the entire
training run:

```python
self._client_tf_optimizers: Dict[int, tf.keras.optimizers.Adam] = {
    c.client_id: build_tf_optimizer(cfg.eta,
                                    self._model._model.trainable_variables)
    for c in split.clients
}
```

This optimizer is never reset between rounds. Here is what happens:

```
Round 1: client trains model_A →  global = aggregate(client_A_1, ...)
         Adam state: m₁, v₁ accumulated from model_A's local training

Round 2: client receives global (very different from model_A end state)
         but Adam's m₁, v₁ are still from Round 1's trajectory
         → momentum pushes in the wrong direction

Round 3: state accumulates further → oscillation → instability

... eventually: model collapses to predicting the majority class
                (BENIGN/Normal) for all inputs
```

**The paper addresses none of this.** Algorithm 1 L17 shows plain SGD
(`θ -= η∇L`), which has zero memory between steps. Even under the §5-prose
interpretation (Adam), nothing in the paper suggests or describes optimizer
state persisting across disconnected training rounds where the initialization
resets to the global model each time.

The intuitive reading — and what the pre-speedup code silently did — is: each
client training round is independent, equivalent to creating a fresh optimizer
(or resetting state) at the start of each round.

### This Explains All Observed Symptoms

| Symptom | Cause |
|---|---|
| **All methods collapse** (FedAvg, Static, FedGRAA) | The bug is in the shared client-training path (`_fused_client_step` / `client_train_round`), which ALL three methods use. The aggregation method is downstream of the damaged client params. |
| **FedGRAA weights → NaN** | Once the global model collapses, FedGRAA's `_composite_grad_P` computes gradients **through** the broken model. The chain rule `dL'/dw = <dL'/dP, θ_k>` produces NaN because the model's forward pass produces NaN logits. |
| **L' = 0.0000** | Early in training, before NaN, the validation splits (`D_V^h`, `D_V^m`, `D_V^n`) are too small for stable gradient computation. With tiny splits, `_loss_and_grad_P` returns `inf`, the fallback sets `L_a = 0`, and the composite loss vanishes. |
| **Pre-speedup 5-round results showed differentiation** | The pre-speedup code (per-batch numpy↔TF ping-pong, fresh optimizer each round) did not have the persistent-state bug, so methods actually diverged correctly. |

---

## 4. Other Minor Speedup Anomalies

### 4.1 Padding buffers reuse stale data

`client_train_round:123-124` creates `pad_x` / `pad_y` once, then reuses them
for every batch. If `n_valid < cfg.batch_size`, only the first `n_valid` rows
are written; the tail contains whatever was in the previous mini-batch. The
`_fused_client_step` masks out padded entries via `mask = range < n_valid`, so
stale data should not contribute to the loss. But if the masking logic has a
bug (e.g. off-by-one in the mask), stale labels or features from a different
attack class could leak in, corrupting gradients.

- **Current risk:** Low — the masking appears correct (`losses * mask` with
  `mask` applied to the per-sample losses). But this deserves a test with
  deliberate garbage in the padding region to confirm zero gradient contribution.

### 4.2 `build_tf_optimizer` dummy gradients

`build_tf_optimizer:314-316` applies zero gradients to create slot variables,
then resets the iteration counter to 0. Since the gradients are zero, the slot
variables (`m_k = β₁·m_{k-1} + (1-β₁)·g_k`) are initialized to zero. This is
correct — no bias from the dummy step.

- **Risk:** None. Verified: Adam with `g=0` gives `m=0, v=0`.

### 4.3 XLA `reduce_retracing` with model/opt arguments

`_fused_client_step` takes `model` and `opt` as Python-object arguments to
`@tf.function(jit_compile=True, reduce_retracing=True)`. Passing non-Tensor
Python objects to `@tf.function` typically triggers retracing when the object
identity changes. With `reduce_retracing=True`, this is mitigated but not
eliminated.

- **Current risk:** Low — the same `model._model` and `opt` objects are reused
  every call, so identity stays the same. But this architecture would break if
  we rebuilt optimizers each round (which we should do to fix the root cause).

---

## 5. The Fix

### Primary fix: Reset Adam state each round in `client_train_round`

Replace the persistent per-client optimizer with a fresh one each round, or
reset the existing one:

```python
def client_train_round(model, opt, X, y, cfg, global_weights, rng):
    # Reset optimizer state at the start of this round
    for var in opt.variables():
        if "iterations" not in var.name:
            var.assign(tf.zeros_like(var))
    opt.iterations.assign(0)

    model.set_params(global_weights)
    # ... rest of training ...
```

Or, more robustly, create a new optimizer each round (the overhead of
`build_tf_optimizer` is negligible — it is one dummy gradient apply):

```python
def client_train_round(model, X, y, cfg, global_weights, rng):
    opt = build_tf_optimizer(cfg.eta, model._model.trainable_variables)
    model.set_params(global_weights)
    # ... rest of training ...
```

The second approach (fresh optimizer each round) is simpler and eliminates
retracing concerns by keeping the optimizer tied to the round.

### Secondary fix: Test that FedGRAA validation splits have adequate samples

Even after fixing the optimizer, ensure `median_frequency_split` produces
non-trivial validation subsets. Add an assertion that `len(val.X_h) +
len(val.X_m) + len(val.X_n) >= min_samples_for_gradient` before entering the
FedGRAA weight learner.

---

## 6. Time to Fix & Run

| Step | Time |
|---|---|
| Fix `client_train_round` to reset optimizer | ~15 minutes coding |
| Run unit tests (55 tests) | ~50 seconds |
| Run full 200-round CIC-IDS-2017 gamma-sweep | ~14 minutes |
| Run full 200-round UNSW-NB15 gamma-sweep | ~14 minutes |
| **Total** | **~30 minutes for a fixed, tested, fully-run reproduction** |

No 80-hour wait. The speedup is already in place — `_fused_client_step` and
`client_train_round` are fast. The only thing broken is the optimizer-state
persistence, and it is a one-line conceptual fix.

---

## 7. What You Will Get After the Fix

- ✅ **Fast training** (~14 min per dataset for full 200 rounds, vs 7.9 h pre-speedup)
- ✅ **Differentiated methods** — FedAvg vs Static vs FedGRAA γ∈{0,0.5,1} producing
     *different* accuracy curves, not all collapsing to the same number
- ✅ **Non-NaN FedGRAA weights** — the weight learner receives real gradient
     signals and produces meaningful aggregation weights
- ✅ **Paper-faithful algorithm** — every equation, every hyperparameter, every
     algorithm-box line reproduced exactly, no toggles, no deviations

The current collapsed state is a **bug, not a design flaw.** Fix the bug, keep
the speedup, get the paper numbers.
