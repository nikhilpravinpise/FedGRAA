# Full 100% Reproduction Gap Analysis — FedGRAA

**Paper:** *"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems"*  
**Codebase:** `PAKDD/fedgraa/`  
**Date:** 2026-06-30

This document is a complete, independent audit of everything standing between the
current codebase and a verifiable 100% reproduction of the paper's results.
It extends the existing `reproduction_gaps.md` (which tracks incremental fixes) by
cataloguing **all** outstanding structural, mathematical, data, metric, and
experimental gaps — including gaps that were acknowledged but not fully resolved and
entirely new gaps not previously documented.

---

## Severity Legend

| Severity | Meaning |
|----------|---------|
| **BLOCKING** | Makes 100% numerical reproduction fundamentally impossible regardless of code fixes |
| **CRITICAL** | Directly alters computed numbers; must be resolved to match the paper's tables |
| **MAJOR** | Material methodological deviation that changes the qualitative experimental outcome |
| **MINOR** | Ambiguity or missing output that makes comparison harder but doesn't change training |

---

## Part 1 — Blocking Gaps (100% reproduction is impossible without these)

### B-1. UNSW-NB15 data directory is completely empty

**Files:** `data/UNSW-NB15/` (0 CSV files), `fedgraa/data.py`  
**Paper:** Section 5, Figures 1 (UNSW panels) and 3

Every UNSW-NB15 result currently runs on the synthetic fallback (Gaussian clusters),
not real network traffic. The paper's entire second dataset section — Figure 1 UNSW
panels, Figure 3, and the Table 2 class breakdown — is therefore **not reproduced at
all**. The synthetic generator preserves the per-class sample *counts* from Table 2
but generates linearly separable Gaussian clusters that bear no statistical
resemblance to real PCAP-derived feature distributions.

No fix in the codebase can resolve this without obtaining the actual UNSW-NB15 CSV
exports.

**Status:** ⏳ Pending — data not acquired.

---

### B-2. Framework mismatch: NumPy float64 vs Keras float32

**File:** `fedgraa/model.py`  
**Paper:** Section 5, "Network Architecture" — *"We designed a fully connected neural
network using the **Keras Sequential API**."*

| Dimension | Paper (Keras/TF) | Code |
|-----------|------------------|------|
| Numeric precision | float32 | float64 |
| Backprop engine | TF XLA-compiled fused kernels | Manual NumPy |
| Cross-entropy | Fused `softmax_cross_entropy_with_logits` | Separate `log_softmax` + `mean` |
| Adam epsilon | 1e-7 | 1e-7 (matches) |
| Weight initialization | Glorot uniform float32 | Glorot uniform float64 |

The float32 vs float64 difference alone means that even with the same seed, the same
architecture, and the same optimizer hyperparameters, all intermediate activations,
gradients, and weight vectors will diverge from the paper after the very first
backprop step. This is a fundamental numerical identity gap, not just a tolerance
issue.

Additionally, TF's fused cross-entropy kernel avoids intermediate softmax computation
entirely (operates on raw logits), which has different rounding behaviour from the
code's `_log_softmax(final_logits)` path, even though both are numerically stable.

**Status:** Partially mitigated (internal inconsistencies fixed) but fundamental
mismatch is irresolvable without TensorFlow/Keras and float32.

---

### B-3. No random seed is reported in the paper

**Files:** All experiment files  
**Paper:** Section 5 — no seed is mentioned

The paper reports specific accuracy numbers (e.g., 88%, 0.33, 0.88 in Figure 1's
table) without disclosing the random seed used for:
- Model weight initialization
- Benign sample allocation ordering
- Train/validation splits
- Batch shuffling within each client

The code uses `seed=0` as a default. There is no way to verify that `seed=0` (or
any other seed) reproduces the paper's exact trajectory without the original training
code and environment.

**Status:** Irresolvable without author disclosure.

---

## Part 2 — Critical Gaps (alter computed numbers)

### C-1. Standard scaling is fitted on the full dataset before the train/val split

**File:** `fedgraa/data.py` — `_load_real_csvs()`, `_standard_scale()`  
**Paper:** Section 5, "Training Setup" — no mention of scaling strategy

The real CSV loader applies `_standard_scale(X)` **after** calling
`_cap_frequent_classes()` but **before** `partition_non_iid()`. This means the
mean and variance for each feature are computed from the full pooled dataset,
including every sample that will later become a validation row.

```
load_dataset()
  └─ _load_real_csvs()
       ├─ _cap_frequent_classes(X, y, spec, seed)   ← operates on full data
       └─ _standard_scale(X)                        ← FITTED ON FULL DATA
```

In strict ML practice the scaler should be fitted only on training data (each
client's 90% split) and applied to that client's validation rows. Because the
paper does not specify this, it is unclear whether the original Keras implementation
made the same choice. A scaler fitted on the full data will compute slightly
different mean/std than one fitted on training-only rows, changing every feature
value seen during training and FedGRAA's weight optimization.

For very large datasets (CIC-IDS-2017 has 2.8M rows before capping) the difference
is small but non-zero.

**Status:** Not addressed anywhere in the codebase or `reproduction_gaps.md`.

---

### C-2. Frequent-class cap is applied per-class globally, but paper specifies it per-dataset

**File:** `fedgraa/data.py` — `_cap_frequent_classes()`  
**Paper:** Section 5, Data Settings — *"to prevent highly frequent attacks … their
samples are capped at 10,000"*

The code caps each class at `spec.frequent_cap = 10_000` uniformly by random
subsampling. The paper does not say whether this cap is applied before or after
loading all eight CIC-IDS-2017 CSV files. Because the same attack type can appear
across multiple files (e.g., DoS Hulk appears in Wednesday's CSV), the order of
CSV concatenation before capping changes which exact 10,000 rows are retained.

The eight CIC-IDS-2017 files are loaded alphabetically (`sorted(os.listdir(...))`),
so the cap retains a deterministic but file-order-dependent subset. If the paper
applied the cap differently (e.g., uniformly from each day's file), the resulting
training distributions differ.

**Status:** Not addressed in the codebase.

---

### C-3. Figure 2 and 3 experiment runner only sweeps `n` at a fixed γ=1

**File:** `fedgraa/experiment.py` — `run_n_sweep()`  
**Paper:** Figures 2 and 3

The paper's Figures 2 and 3 each contain **six sub-figures**: (γ=0, n=5),
(γ=0.5, n=5), (γ=1, n=5), (γ=0, n=7), (γ=0.5, n=7), (γ=1, n=7).

The current `run_n_sweep()` implementation runs only:
- FedAvg (γ irrelevant, n=3/5/7)
- Static Reparameterization (n=3/5/7)
- FedGRAA at the **single fixed gamma passed in** (default 1.0)

It does not run FedGRAA at γ=0 or γ=0.5 for n=5 and n=7. To reproduce Figures 2
and 3 fully, `run_n_sweep` would need to sweep `gamma ∈ {0, 0.5, 1}` for each `n`.
The current structure produces 3 curves per n-panel instead of the paper's 5 (FedAvg,
Static, FedGRAA-γ0, FedGRAA-γ0.5, FedGRAA-γ1).

```python
# Current: only one FedGRAA gamma per n-panel
out[n] = {
    "fedavg":   run_experiment(..., method="fedavg"),
    "static":   run_experiment(..., method="static"),
    "fedgraa":  run_experiment(..., method="fedgraa", gamma=gamma),  # fixed gamma
}
```

**Status:** Not addressed anywhere. The full Figure 2/3 structure is not
reproducible with the current experiment runner.

---

### C-4. Round 0 accuracy in the paper's Figure 1 table is 0.00 for ALL classes

**File:** `fedgraa/trainer.py` — `FederatedTrainer.train()`  
**Paper:** Figure 1 table, round 0 row

The paper's Figure 1 data table shows every method (FedAvg, Static, FedGRAA at
every γ) reporting **0.00** accuracy for both "All classes" and "Last n classes" at
round 0.

A freshly initialized Glorot-uniform network with softmax output distributes
probability mass nearly uniformly over all classes. On 15 CIC-IDS classes the
expected random accuracy is ≈ 1/15 ≈ 6.7%, not 0.00%. The code's round 0 record:

```python
history.record(
    0,
    accuracy(global_params, self.split.X_val, self.split.y_val),   # ~0.067
    last_n_class_accuracy(...),                                       # ~0.033
    w,
)
```

will produce non-zero values that contradict the paper's table. Two interpretations:

1. The paper's authors may have defined "round 0" as the first evaluation point
   *after* round 20 of training (i.e., the x-axis starts at 0 but the first
   data row corresponds to post-initialization state where no training has
   occurred yet and all sample-class predictions are collapsed to benign,
   giving 0 on rare classes).
2. The initial Keras model was so poorly calibrated on this data that it
   achieved 0% by chance.

Either way, the code's round-0 values will not reproduce the paper's 0.00 entries.

**Status:** Not addressed in `reproduction_gaps.md`. The `round-0` fix (gap 2.5)
records *actual* model accuracy at initialization, which differs from 0.00.

---

### C-5. `last_n_class_accuracy` metric definition is inferred, not stated in the paper

**File:** `fedgraa/metrics.py` — `last_n_class_accuracy()`  
**Paper:** Figure 1 columns labelled "Last n classes (Accuracy)"

The code implements this as **mean per-class recall** over the rare class IDs:

```python
recalls = [(preds[y==c] == c).mean() for c in rare_class_ids]
return np.mean(recalls)
```

The paper does not define the metric precisely. Two common alternatives:
- **Mean per-class recall** (what the code does): each rare class weighted equally.
- **Pooled accuracy**: total correct / total samples across all rare classes combined
  (classes with more samples get higher weight).

For Heartbleed (11 samples) + SQL Injection (21 samples) + Infiltration (36 samples),
pooled accuracy gives roughly 3× more weight to Infiltration than Heartbleed.
Mean per-class recall gives equal weight to all three.

The paper's claim of "88% accuracy on minority classes" could mean either. The
choice changes the reported number by several percentage points and cannot be
determined from the paper alone.

**Status:** Not addressed in `reproduction_gaps.md`. The metric choice is documented
in `metrics.py` but not cross-validated against the paper.

---

## Part 3 — Major Gaps (qualitative methodological deviations)

### M-1. Algorithm 2's inner-loop gradient recomputation ambiguity

**File:** `fedgraa/aggregation.py` — `FedGRAA.update()`  
**Paper:** Algorithm 2 and Eq. 10

The paper writes the weight update as:

$$w^{t-1} \leftarrow w^{t-1} - \eta' \nabla_{w^{t-1}} L'\!\left(\sum_i w_i^{t-1}\theta_i^t;\ldots\right)$$

with the notation `w^{t-1}` on **both sides**, and this is repeated for E' epochs.
There are two valid readings:

**Reading A (static gradient):** The gradient is computed once at the initial
`w^{t-1}` and the same gradient is applied E' times (equivalent to E' steps of
fixed-direction gradient descent).

**Reading B (iterative descent, what the code does):** Each of the E' epochs:
1. Recomputes `θ'_g = Σ w_k θ_k^t` using the **current** (already-updated) `w`
2. Computes the new gradient at the new `θ'_g`
3. Takes a gradient step

Reading B is the natural interpretation of "for each local epoch from 1 to E'"
in Algorithm 2. The code implements Reading B (the loop recomputes `P =
aggregate(client_params, w)` at the start of each E' epoch). Reading A would give
a different weight trajectory. The paper does not disambiguate.

**Status:** Not addressed in `reproduction_gaps.md`. The code's choice (Reading B)
is the most algorithmically principled but is an assumption.

---

### M-2. Sub-gradient selection for `min(L_h, L_m)` when `L_h ≈ L_m`

**File:** `fedgraa/aggregation.py` — `FedGRAA._composite_grad_P()`  
**Paper:** Eq. 8 — `L_a = min(L_h, L_m)`

`min(L_h, L_m)` is a non-smooth function. At any point where `L_h == L_m` the
gradient is undefined; the valid subgradient is any convex combination
`α ∇L_h + (1-α) ∇L_m` for `α ∈ [0,1]`.

The code always uses the gradient of whichever branch is strictly smaller (`if L_h
<= L_m: use g_h`). This is a valid subgradient but:

1. The paper gives no guidance on subgradient selection.
2. Near the boundary `L_h ≈ L_m`, tiny floating-point differences in the loss
   values will flip which gradient branch is chosen, causing stochastic gradient
   discontinuities even with deterministic data and seeds.
3. This discontinuity is not discussed in the convergence proof (Appendix A.1),
   which assumes smoothness (Assumption 1).

**Status:** Not addressed in `reproduction_gaps.md`.

---

### M-3. Benign sample pool is insufficient for UNSW-NB15

**File:** `fedgraa/partition.py` — `partition_non_iid()`, `fedgraa/config.py`  
**Paper:** Section 5, Data Settings — "paired with an equal number of Normal samples"

For UNSW-NB15 the total attack demand is:

| Class | Used samples |
|-------|-------------|
| Generic | 10,000 |
| Fuzzers | 10,000 |
| Backdoor | 2,329 |
| Worms | 174 |
| Reconnaissance | 10,000 |
| Analysis | 2,677 |
| Exploits | 10,000 |
| Shellcode | 1,511 |
| DoS | 10,000 |
| **Total attack demand** | **66,691** |

But the Normal (benign) pool has only **56,691** used samples (Table 2). Total benign
demand exceeds the pool by ≈10,000 samples. The code's circular indexing wraps
around the benign pool, meaning approximately 10,000 benign samples appear in **two**
different clients' training sets simultaneously.

The paper claims each client has "equal number" of benign samples, implying no
overlap. This constraint cannot be satisfied given the specified sample counts. The
paper does not acknowledge this contradiction.

**Impact:** Clients that share benign samples have non-independent training data,
violating the federated learning assumption of data isolation and potentially
inflating benign detection accuracy.

**Status:** Gap 2.3 describes the circular indexing fix, but does not identify the
root cause (pool shortage) or its impact on the experimental validity.

---

### M-4. Rare class selection for n=5 and n=7 on UNSW-NB15 is undocumented speculation

**File:** `fedgraa/config.py` — `UNSW_NB15.rare_label_order`  
**Paper:** Section 5 — only n=3 rare classes (Worms, Shellcode, Backdoor) are named

The paper's Figures 2 and 3 show results for n=5 and n=7 on UNSW without naming the
additional rare classes. The code assigns:

```python
rare_label_order=[
    "Worms",            # 174
    "Shellcode",        # 1511
    "Backdoor",         # 2329
    "Analysis",         # 2677    ← n=5 adds this
    "Fuzzers",          # 10000   ← n=7 adds this (but it's a frequent class!)
    "DoS",              # 10000
    "Reconnaissance",
],
```

For n=5 the code adds "Analysis" (2,677 samples) which is reasonable. But for n=7
it adds "Fuzzers" (10,000 samples, one of the most frequent classes in the dataset),
which contradicts the concept of "clients with minority samples". Adding a 10,000-
sample client to the protected rare set fundamentally changes the semantics of the
experiment and gives FedGRAA an unfair advantage in the rare-class accuracy metric.

**Status:** Gap 2.4 notes the issue but does not flag the semantic problem with
adding high-frequency classes to the rare set.

---

### M-5. Local optimizer: paper describes SGD in equations but uses Adam in training

**Files:** `fedgraa/trainer.py`, `fedgraa/config.py`  
**Paper:** Eq. 1 (SGD notation) vs. Section 5 "Training Setup" (Adam, η=0.01)

The background section (Eq. 1) describes local updates as plain SGD:
$$\theta_k^{t+1} = \theta_k^{t+1} - \eta \nabla L_{\theta_k^{t+1}}(\theta_k^{t+1}; D_k)$$

But Section 5 explicitly states the model is "trained using the Adam optimizer with
a learning rate of η = 0.01". The code correctly uses Adam for local training.
However, the convergence proof in Appendix A.1 is derived under the SGD-style
update (bounded gradients, fixed step sizes), which is **not** the Adam update used
in practice. The convergence proof is therefore mismatched with the actual
optimization algorithm implemented in both the paper and the code.

**Status:** Not addressed anywhere; it is a paper-level inconsistency that the
reproduction inherits.

---

## Part 4 — Minor Gaps (missing outputs, ambiguities)

### m-1. CIC-IDS-2017 Monday CSV contains only BENIGN traffic

**File:** `fedgraa/data.py` — `load_dataset()`  
**Data:** `data/CIC-IDS-2017/Monday-WorkingHours.pcap_ISCX.csv`

The Monday CSV file from the CIC-IDS-2017 dataset contains only BENIGN traffic
(no attacks). The code loads all CSV files in the directory alphabetically and
concatenates them before capping. This means the Monday file adds a large block of
additional BENIGN rows to the pool before the per-class cap is applied.

If the paper's authors excluded Monday's file or processed each day separately, the
resulting class distributions after capping would differ. The paper provides no
information on which specific CSV files were used.

**Status:** Not addressed in the codebase.

---

### m-2. Tiny clients have near-empty validation sets affecting D_V^n quality

**File:** `fedgraa/partition.py` — `partition_non_iid()`  
**Paper:** Section 5, Data Settings

The three n=3 rare clients for CIC-IDS-2017 have:

| Client | Attack samples | Total (attack + benign) | 10% val ≈ |
|--------|---------------|-------------------------|-----------|
| Heartbleed | 11 | 22 | ~2 samples |
| SQL Injection | 21 | 42 | ~4 samples |
| Infiltration | 36 | 72 | ~7 samples |

With only 2–7 validation samples per rare class, D_V^n is extremely sparse. The
FedGRAA weight optimizer updates `w` by minimizing `L_u` on these tiny sets, which
means gradient estimates have very high variance. The paper does not discuss this
limitation.

Furthermore, `last_n_class_accuracy` on a 2-sample D_V^n can only return 0.0,
0.5, or 1.0 — it has only 3 possible values — making the "88%" claim difficult to
map onto actual run-time measurements.

**Status:** Not addressed anywhere.

---

### m-3. No comparison against FedProx, FedOpt, or FedSND

**File:** `fedgraa/experiment.py`  
**Paper:** Section 2 (Related Work) — cites FedOpt [7], FedProx [5], FedSND [12]

The paper mentions these methods as related work and explicitly discusses FedSND
as a predecessor that "dynamically adjusts aggregation weights based on data
distribution statistics". None of these baselines are implemented or compared
experimentally in the paper or the code. The evaluation compares only against FedAvg
and Static Reparameterization, which are relatively weak baselines.

**Status:** Not in the reproduction codebase (by design), but limits the ability to
assess whether FedGRAA represents a genuine improvement over the state of the art.

---

### m-4. The convergence proof assumes convex L' but gradient steps may not decrease loss monotonically on real data

**File:** `fedgraa/trainer.py` — `TrainHistory.is_composite_loss_monotone`  
**Paper:** Appendix A.1, Lemma 1

The convergence proof (Lemma 1) assumes L' is convex and β-smooth and that the
learning rate satisfies η' ≤ 1/β. The composite loss
`L' = (1-γ)·min(L_h, L_m) + γ·L_u` is:

- **Not convex in general** — cross-entropy over a neural network is non-convex
- **Not smooth at `L_h = L_m`** — the `min` introduces a kink
- **β is never estimated or verified** — the code has no check that η'=0.1 ≤ 1/β

The `is_composite_loss_monotone` property tracks whether the loss decreases each
round, but this is evaluated at the global model parameters **after** client local
training, not at the FedGRAA internal E' sub-problem trajectory. A non-monotone
reading does not indicate a bug but it does show the convergence proof assumptions
are violated in practice.

**Status:** The tracking infrastructure was added (gap 3.7) but the fundamental
mismatch between the proof's assumptions and real-world non-convex networks is not
discussed.

---

### m-5. Weight evolution and simplex constraint plots not integrated into the run_experiments.py pipeline

**File:** `fedgraa/plot.py`, `run_experiments.py`  
**Paper:** Figures 1, 2, 3

`run_experiments.py` prints text tables but does not call any plotting functions. The
`plot_gamma_sweep`, `plot_n_sweep`, `plot_weight_evolution`, and
`plot_composite_loss` functions exist in `plot.py` but are not invoked anywhere in
the main reproduction script. The figures that appear in the paper cannot be
regenerated automatically by running `run_experiments.py`.

**Status:** Not addressed. The notebook (`FedGRAA_reproduction.ipynb`) calls some
plot functions but the standalone script does not.

---

### m-6. Gradient magnitude scaling for the FedGRAA weight update is uncontrolled

**File:** `fedgraa/aggregation.py` — `_grad_wrt_weights()`  
**Paper:** Eq. 10

The weight gradient is computed as:
$$\frac{\partial L'}{\partial w_k} = \left\langle \nabla_P L',\, \theta_k^t \right\rangle$$

This dot product involves the raw parameter arrays of the global model (hundreds of
thousands of floats). The magnitude of `grad_w` scales with the L2 norm of the
client parameters, which grows with the network size and training duration. With
η'=0.1 and parameters of typical magnitude O(0.1), the inner product can be O(10²)
or larger, making the effective gradient step far larger than the nominal η'=0.1
suggests.

The paper provides no discussion of gradient scaling or gradient clipping for the
weight update step. Adam's adaptive learning rate partially mitigates this, but the
uncontrolled scale means different network sizes or training depths could cause the
weight optimizer to diverge.

**Status:** Not addressed in any file.

---

### m-7. The `frequent_cap` is 10,000 but BENIGN/Normal classes are NOT capped at 10,000

**File:** `fedgraa/data.py` — `_cap_frequent_classes()`  
**Paper:** Section 5, Data Settings — *"to prevent highly frequent attacks … their
samples are capped at 10,000"*

The paper specifically mentions capping "highly frequent **attacks**". The BENIGN
(Normal) class is not an attack and is not mentioned in the cap list. The code's
`_cap_frequent_classes()` applies the cap to **all** classes including BENIGN:

```python
for class_id in range(spec.n_classes):       # iterates over ALL classes
    idx = np.flatnonzero(y == class_id)
    if len(idx) > spec.frequent_cap:
        idx = rng.choice(idx, size=spec.frequent_cap, replace=False)
```

If this cap is applied to BENIGN, the benign pool is limited to 10,000 samples
regardless of how many clients need benign pairing. But the `used_counts` in
`config.py` specifies BENIGN as 63,901 (CIC) and 56,691 (UNSW), values much larger
than 10,000. This suggests the code loads all BENIGN samples (up to the used_count)
without capping them at 10,000. The resolution depends on whether the capping loop
runs before or after the benign pool is established — currently it runs on the raw
loaded data, which has BENIGN pre-loaded to the full used_count from the CSV files.

However, the `used_counts` values are only used by the *synthetic* generator; the
real CSV loader does not pre-filter to `used_counts`. In the real loader,
`_cap_frequent_classes` would cap BENIGN at 10,000, leaving far fewer benign samples
than the 63,901 the synthetic path uses.

This creates an asymmetry between synthetic and real data paths for the benign class
count.

**Status:** Not addressed in `reproduction_gaps.md`.

---

## Part 5 — Summary Table

| ID | Category | Severity | Status | Paper Section |
|----|----------|----------|--------|---------------|
| B-1 | UNSW-NB15 data absent | **BLOCKING** | ⏳ | §5, Figs 1 (UNSW), 3 |
| B-2 | float32 vs float64 / Keras vs NumPy | **BLOCKING** | ⚠️ Mitigated | §5 Network Arch. |
| B-3 | No random seed in paper | **BLOCKING** | ⏳ | §5 |
| C-1 | Standard scaler fitted on full dataset | **CRITICAL** | ❌ Unaddressed | §5 Data Settings |
| C-2 | CSV concatenation order affects class cap | **CRITICAL** | ❌ Unaddressed | §5 Data Settings |
| C-3 | Figure 2/3 require γ sweep per n; `run_n_sweep` uses fixed γ | **CRITICAL** | ❌ Unaddressed | Figs 2–3 |
| C-4 | Paper's round-0 accuracy = 0.00 contradicts initialized model | **CRITICAL** | ❌ Unaddressed | Fig. 1 table |
| C-5 | `last_n_class_accuracy` definition inferred, not stated | **CRITICAL** | ❌ Unaddressed | Fig. 1 |
| M-1 | Algorithm 2 inner-loop gradient recomputation ambiguity | **MAJOR** | ❌ Unaddressed | Algo. 2, Eq. 10 |
| M-2 | Sub-gradient of min(L_h, L_m) unspecified | **MAJOR** | ❌ Unaddressed | Eq. 8 |
| M-3 | UNSW benign pool too small; ~10k samples duplicated | **MAJOR** | ⚠️ Partial (gap 2.3) | §5 Data Settings |
| M-4 | n=7 UNSW rare-class selection adds frequent classes | **MAJOR** | ⚠️ Partial (gap 2.4) | §5, Fig. 3 |
| M-5 | Local optimizer: SGD in equations vs Adam in training | **MAJOR** | ❌ Paper inconsistency | Eq. 1, §5 |
| m-1 | Monday CIC CSV is BENIGN-only; effect on cap unspecified | **MINOR** | ❌ Unaddressed | §5 Data Settings |
| m-2 | Rare clients have 2–7 validation samples; D_V^n quality | **MINOR** | ❌ Unaddressed | §5, Fig. 1 |
| m-3 | No FedProx/FedOpt/FedSND baseline implementation | **MINOR** | ❌ By design | §2 Related Work |
| m-4 | Convergence proof assumes convex L'; neural nets are non-convex | **MINOR** | ⚠️ Tracked (gap 3.7) | App. A.1 |
| m-5 | Plots not generated by `run_experiments.py` | **MINOR** | ❌ Unaddressed | Figs 1–3 |
| m-6 | FedGRAA weight gradient magnitude is uncontrolled/unscaled | **MINOR** | ❌ Unaddressed | Eq. 10 |
| m-7 | BENIGN cap asymmetry between synthetic and real data paths | **MINOR** | ❌ Unaddressed | §5 Data Settings |

---

## Part 6 — Claims in the paper that cannot be independently verified

The following quantitative claims from the paper are unverifiable with the current
reproduction:

1. **"FedGRAA achieves 88% accuracy on minority classes"** (Abstract) — requires
   real CIC-IDS-2017 data AND the paper's exact seed AND the Keras float32 environment.

2. **"outperforming FedAvg by up to 55%"** (Abstract) — "up to 55%" is a peak
   difference observed across the 200 rounds on a specific n value; the exact round
   and γ where this occurs are not specified.

3. **"FedGRAA maintains its advantage, outperforming FedAvg by up to 55%"**
   (Section 5) — as above; the specific (n, γ, round) combination is not identified.

4. **"approximately 65% accuracy for n=5, γ=1"** and **"around 80% for n=7, γ=1"**
   (Section 5) — both require real CIC-IDS-2017 data.

5. **"FedAvg and static reparameterization achieved accuracies below 10%"** for n=5,
   **"40% for FedAvg"** for n=7 — these baseline numbers cannot be cross-checked
   without real data.

6. **All UNSW-NB15 numerical results** — entirely from a dataset not present in the
   repository.

---

## Part 7 — What the reproduction CAN currently verify

Despite the gaps above, the codebase can:

- ✅ Confirm the **directional** result: FedGRAA > Static Reparameterization > FedAvg
  on rare-class accuracy on synthetic data.
- ✅ Verify the **mathematical correctness** of the FedGRAA gradient (finite-difference
  tests pass in `test_aggregation.py`).
- ✅ Confirm the **simplex constraint** is maintained on aggregation weights.
- ✅ Verify all **hyperparameters** match the paper's Table in §5.
- ✅ Confirm the **network architecture** (200→100→50, tanh, softmax) matches §5.
- ✅ Confirm the **data partition structure** (one attack per client, equal benign,
  90/10 split, median-frequency D_V split) matches §5.
- ✅ Confirm the **FedGRAA algorithm structure** (Algorithm 1 + Algorithm 2) is
  faithfully implemented.
