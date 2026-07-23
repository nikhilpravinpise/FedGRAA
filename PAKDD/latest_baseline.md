# Latest Baseline — Post-Speedup Performance

> **⚠️ SUPERSEDED — historical snapshot, not current state.** This document
> captures a NaN-collapse regression that existed at commit `f95092a`
> (2026-07-20) and was fixed the same day by `2bb548a` ("fix: NaN collapse
> from np.empty padding buffers in XLA fused steps"). The 55/55 test count and
> the 0%-everywhere accuracy table below are **not** current — see
> [`PAKDD_STATUS_REPORT.md`](PAKDD_STATUS_REPORT.md) for the current 79/79
> test count and [`results/`](results/) for current accuracy numbers. Kept for
> historical record of the regression and its diagnosis.

**Date:** 2026-07-20
**Commit:** `f95092a` (HEAD)
**Python:** 3.13.14 · **TensorFlow:** 2.21.0 (CPU, oneDNN on) · **NumPy:** 2.5.1 (OpenBLAS)

---

## Unit Tests

All **55/55** tests pass in 49.6s (vs 291s in the previous report). The speedup comes from the fused XLA client training step (`_fused_client_step` in `model.py:252`).

```
tests/test_aggregation.py::test_fedavg_weights_proportional_to_size          PASSED
tests/test_aggregation.py::test_static_weights_inverse_and_normalized        PASSED
tests/test_aggregation.py::test_fedgraa_weight_gradient_matches_finite_difference PASSED
tests/test_aggregation.py::test_fedgraa_gamma_extremes_select_objective      PASSED
tests/test_aggregation.py::test_fedgraa_update_runs_and_changes_weights      PASSED
tests/test_aggregation.py::test_fedgraa_update_is_deterministic_across_calls PASSED
tests/test_data.py          (4 tests)                                        PASSED
tests/test_metrics.py       (7 tests)                                        PASSED
tests/test_model.py         (9 tests)                                        PASSED
tests/test_partition.py     (7 tests)                                        PASSED
tests/test_plot.py          (6 tests)                                        PASSED
tests/test_reproduction_gaps.py (7 tests)                                    PASSED
tests/test_trainer.py       (8 tests)                                        PASSED
```

Key tests verifying the speedup remain valid:
- `test_fused_xla_step_equivalent_to_eager` — numerical equivalence
- `test_fused_xla_multiple_steps_converges` — convergence behavior
- `test_fedgraa_beats_fedavg_on_rare_class` — qualitative paper result
- `test_build_tf_optimizer_creates_slot_variables` — pre-built optimizer

---

## Training Throughput (Real Data)

| Dataset | Rounds | Total Time | Per Round | vs Pre-Speedup* | Speedup |
|---------|--------|-----------|-----------|-----------------|---------|
| CIC-IDS-2017 | 50 | 211–226 s | **~4.2 s** | ~150 s/round | **~31×** |
| UNSW-NB15 | 50 | 152–173 s | **~3.4 s** | ~8 s/round | **~2.4×** |

*Pre-speedup baseline: eager Keras training loop (no XLA). CIC per-round time measured at 14 clients × ~258 iterations/client. UNSW has fewer samples, so the speedup is smaller (XLA compile overhead is a larger fraction).

Projected full reproduction times (200 rounds, both datasets, gamma + n sweeps):

| Scenario | Pre-Speedup | Post-Speedup |
|----------|-------------|--------------|
| Single 200-round run | ~7.9 h | **~14 min** |
| Gamma sweep (5 runs × 200 rounds) | ~40 h | **~1.2 h** |
| Full reproduction (both datasets) | ~80+ h | **~2.5 h** |

---

## Accuracy Results — Current Regression

Both datasets show all methods converging to the same majority-class accuracy with **NaN aggregation weights** and **0% rare-class recall**.

### CIC-IDS-2017 (50 rounds, real data)

| Method | All-class | Last-n (rare) |
|--------|-----------|---------------|
| FedAvg | 51.02% | 0.00% |
| Static Reparam. | 51.02% | 0.00% |
| FedGRAA g=0 | 51.02% | 0.00% |
| FedGRAA g=0.5 | 51.02% | 0.00% |
| FedGRAA g=1 | 51.02% | 0.00% |

### UNSW-NB15 (50 rounds, real data)

| Method | All-class | Last-n (rare) |
|--------|-----------|---------------|
| FedAvg | 49.95% | 0.00% |
| Static Reparam. | 49.95% | 0.00% |
| FedGRAA g=0 | 49.95% | 0.00% |
| FedGRAA g=0.5 | 49.95% | 0.00% |
| FedGRAA g=1 | 49.95% | 0.00% |

**Diagnosis:** The model collapses to predicting only the majority class (BENIGN/Normal). The final FedGRAA aggregation weights are **NaN**, which propagate through `aggregate()` to produce NaN global parameters, causing the model to output NaN logits and defaulting to class 0.

The `L'=0.0000` composite loss suggests the FedGRAA validation splits (`D_V^h`, `D_V^m`, `D_V^u`) may be producing zero or inf gradients, leading to NaN weight updates.

### Comparison with Pre-Speedup (5-round runs from 2026-07-15)

The pre-speedup 5-round results showed **differentiation between methods** (e.g., Static at 75% last-n on CIC, FedGRAA at 29.92% on UNSW). The current fused-XLA path regresses this behavior.

This is a **post-speedup regression**, likely introduced by the fused XLA client training step (`_fused_client_step`), the padding-batch handling, or the pre-built per-client optimizers.

---

## What Changed (Speedup Implementation)

The speedup replaced the per-iteration numpy↔TF ping-pong with a fused XLA graph:

**Before:** `_client_update` → per-batch `model.set_weights()` + `tf.GradientTape()` + `Adam.step()` with numpy copies

**After:** `client_train_round` → one `model.set_params()` per client + `_fused_client_step(XLA-compiled)` per batch, reading weights back once at the end

Key files modified:
- `model.py:252-276` — `_fused_client_step`: XLA-compiled forward+backward+optimizer
- `model.py:279-301` — `_fused_proximal_step`: FedProx variant
- `model.py:304-317` — `build_tf_optimizer`: pre-built per-client TF Adam
- `trainer.py:90-144` — `client_train_round`: new fused XLA client training loop
- `trainer.py:199-207` — Per-client TF optimizer dictionary

---

## Investigation Needed

1. **NaN aggregation weights** — why does `FedGRAA.update()` produce NaN weights after a few rounds? Suspect validation split data sizes are too small for stable gradient computation.
2. **All methods collapsing** — even FedAvg and Static (which don't use the FedGRAA weight learner) collapse to the same majority-class predictor. This suggests the underlying local training (fused XLA client step) is also affected.
3. **Validation composite loss = 0.0000** — the FedGRAA composite loss is zero across all rounds, indicating the validation splits may be empty or producing no gradient signal.
