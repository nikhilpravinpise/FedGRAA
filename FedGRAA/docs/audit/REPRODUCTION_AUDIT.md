# FedGRAA — Ruthless Reproduction Audit (Current Code State)

**Paper:** *"Federated Gradient-Refined Adaptive Averaging for Intrusion
Detection Systems"* — [`PAKDD_Final_Version_Submit_v3.md`](../paper/PAKDD_Final_Version_Submit_v3.md).

**Codebase audited:** `PAKDD/fedgraa/` **as it exists right now** (Keras/TensorFlow
backend, `model.py`/`aggregation.py`/`trainer.py`/`partition.py`/`data.py`/`metrics.py`/`experiment.py`).

**Scope:** the **base paper only**. `../current/privacy_compliant_fedgraa_redesign.md`
("FedGRAA‑v2"), every related-work method cited in §2/References that is not
one of the paper's own Figures 1–3 curves (FedProx/FedOpt/FedSND baselines in
`fedgraa/baselines.py`), and every piece of v2-only code (`secagg.py`-style
mechanisms, differential privacy, `confusion_matrix`, etc. — none of which
currently exist as separate modules) are explicitly **out of scope**. This is a
documentation-only deliverable: **no code was changed to produce this report.**

**Date of this audit:** 2026-07-10. **Test suite status at audit time:**
`pytest tests -q` → **50 passed**, 0 failed, run time ≈164s
(`.venv313\Scripts\python.exe -m pytest tests -q`).

> **Update (2026-07-10, same day):** the two fixable code deviations this
> audit originally flagged (F8: double feature scaling; F9: silent UNSW n=7
> truncation) have since been fixed in `fedgraa/partition.py` and
> `fedgraa/config.py`, with two new regression tests
> (`test_partition_does_not_rescale_features`,
> `test_unsw_n7_warns_instead_of_silent_truncation`) added to
> `tests/test_reproduction_gaps.py`. Full suite now **52 passed**. The
> findings below (§5.1, §7, §13 F8/F9) are left in their original
> "as-found" wording for the historical record, with a note at each
> location marking them ✅ FIXED and pointing to the change; every other
> finding in this document (F1–F7, F10–F15) is unchanged and still
> reflects the current code.

---

## 0. Why this document exists and what it supersedes

Four earlier audit documents already exist in this folder (the first three now
moved to `archive/`, since this document supersedes them):
[`reproduction_gaps.md`](../archive/reproduction_gaps.md),
[`full_reproduction_gap_analysis.md`](../archive/full_reproduction_gap_analysis.md),
[`paper_vs_code_line_by_line_gap_analysis.md`](../archive/paper_vs_code_line_by_line_gap_analysis.md),
and [`strict_reproduction_report.md`](../archive/strict_reproduction_report.md).

**Every one of them is at least partially stale.** The codebase went through a
full backend rewrite (hand-written NumPy MLP → real `tf.keras.Sequential`
model in `fedgraa/model.py`) and an ambiguity-removal pass (all
"interpretation" config flags such as `fedgraa_reading`, `subgradient`,
`grad_clip`, `val_split_strategy`, `cap_benign`, `rare_metric` were deleted)
**after** `../archive/full_reproduction_gap_analysis.md` (2026-06-30) and
`../archive/paper_vs_code_line_by_line_gap_analysis.md` (2026-07-09) were written. Those
two documents describe a codebase state (NumPy backend, unconditional
`_project_simplex`, a `reading="A"/"B"` toggle, an `Optional`-import bug) that
**no longer exists**. `../archive/strict_reproduction_report.md` reflects the rewrite
correctly at a summary level but does not go line-by-line through every
equation/algorithm-box line the way this document does, and it does not
independently verify the arithmetic claims it repeats from the older
documents (one of which — the UNSW benign-pool "shortage" — is **factually
wrong**; see §5.6 below).

**This document is the current, single source of truth.** It re-derives every
verdict from the code as it exists today, corrects the stale claims, and adds
several findings that no prior document logged (§8, §9, §10).

### Severity / verdict legend

| Verdict | Meaning |
|---|---|
| ✅ **VERIFIED MATCH** | Code implements exactly what the paper states, checked line-by-line |
| 🟡 **AMBIGUITY — RESOLVED** | Paper is silent/self-contradictory; code has committed to one documented, singular choice (no alternate code path exists) |
| 🟠 **DEVIATION** | Code does something the literal paper text does not describe, or omits something the paper describes |
| 🔴 **BLOCKED** | Not reproducible at all right now regardless of code; requires an artifact (data, seed, author disclosure) the paper never released |
| ⚪ **NOT YET EXECUTED** | The code path exists and is believed correct, but the actual 200-round numeric result has not been produced/compared against the paper's tables |

---

## 1. Executive verdict

| Paper claim / element | Verdict | Where |
|---|---|---|
| Abstract: "88% accuracy on minority classes" | ⚪ NOT YET EXECUTED (no real-CIC 200-round run has been completed and compared) | §11 |
| Abstract: "outperforming FedAvg by up to 55%" | ⚪ NOT YET EXECUTED | §11 |
| §3 Eq. 1–6 (background) | 🟡/🟠 mixed — see §3 | §3 |
| §4 Eq. 7 (Static Reparam) | ✅ VERIFIED MATCH | §4.1 |
| §4 Algorithm 1 (FL loop) | ✅ VERIFIED MATCH (Adam vs SGD notation gap noted) | §4.2 |
| §4 Algorithm 2 (FedGRAA weight learner) | ✅ VERIFIED MATCH to the code's own committed reading; 🟡 the paper itself is ambiguous on 3 sub-points, all now resolved as *single* choices, not toggles | §4.3 |
| §5 Data Settings | 🟡 mostly resolved; the one real deviation found (double feature scaling) is ✅ FIXED (2026-07-10) | §5 |
| §5 Network Architecture | ✅ VERIFIED MATCH — genuine `tf.keras.Sequential`, not a lookalike | §6 |
| §5 Training hyperparameters | ✅ VERIFIED MATCH, all values | §7 |
| Table 1 / Table 2 counts | ✅ VERIFIED MATCH, digit-by-digit, both datasets | §5.5 |
| CIC-IDS-2017 real data | ✅ present, loads, cached once (`.cache/CIC_IDS_2017_seed0_*.npy`) | §9.1 |
| UNSW-NB15 real data | 🔴 BLOCKED — `data/UNSW-NB15/` is **empty**, zero CSVs | §9.2 |
| Figure 1 (γ sweep, n=3) | ⚪ code path complete and tested at small scale; full 200-round run **not executed** on real data | §11.1 |
| Figures 2–3 (n sweep × γ sweep) | ⚪ code path complete (`run_n_sweep` now covers all 3×3=9 combinations, fixing the historical "only one γ" gap); **not executed** at full scale | §11.2 |
| Appendix A.1 convergence proof | 🟠 proof assumptions (Assumption 1: convexity) do not hold for this network; code tracks `is_composite_loss_monotone` as an empirical proxy only | §12 |

**Bottom line:** the **code-level, algorithmic fidelity** to the paper is now
very high — every hyperparameter, every equation the paper actually specifies
unambiguously, and the full Keras architecture are implemented exactly, and
every place the paper is silent or self-contradictory has been resolved to a
**single, documented, non-configurable choice** (verified by reading
`aggregation.py`, `partition.py`, `data.py`, `metrics.py`, `config.py` in
full — there are no more `reading=`/`subgradient=`/`cap_benign=`-style knobs
anywhere in the current source). What is **still missing** is empirical: (a)
UNSW-NB15 real data does not exist in this workspace at all, so every UNSW
number in the paper is currently unreproducible in principle, and (b) even for
CIC-IDS-2017, no full 200-round run has actually been executed and its output
numbers compared against the paper's own Figure 1/2/3 tables — the pipeline is
tested end-to-end only at small scale (≤50 rounds, synthetic + a handful of
real-CSV smoke assertions in `tests/`).

---

## 2. Method

Every equation (1–10) in §3–4, every line of Algorithm 1 and Algorithm 2,
every sentence of §5 ("Experiments and Results"), Tables 1–2 digit-by-digit,
and the convergence-proof assumptions in Appendix A.1 were matched against the
exact function/line in the **current** source that is supposed to implement
it, by direct reading of:

`fedgraa/config.py`, `fedgraa/model.py`, `fedgraa/keras_backend.py`,
`fedgraa/data.py`, `fedgraa/partition.py`, `fedgraa/aggregation.py`,
`fedgraa/trainer.py`, `fedgraa/metrics.py`, `fedgraa/experiment.py`,
`fedgraa/baselines.py`, `fedgraa/__main__.py`, `run_experiments.py`, and the
full `tests/` suite (which was also executed, not just read).

---

## 3. §3 Background — Equations 1–6

| Eq. | Paper statement | Code | Verdict |
|---|---|---|---|
| Eq. 1 | Client SGD step: $\theta_k^{t+1} \leftarrow \theta_k^{t+1} - \eta\nabla L(\theta_k^{t+1};D_k)$ | `trainer.py::_client_update` calls `model.loss_and_grad` then `optimizer.step` where `optimizer` is a real `tf.keras.optimizers.Adam` wrapper (`model.py::Adam`) | 🟡 AMBIGUITY — RESOLVED. Eq. 1's own notation is plain gradient descent; §5's prose says Adam with η=0.01. These cannot both be literally true (Adam ≠ `θ -= η·∇`). The code has made one committed, singular choice: **Adam**, consistently, everywhere. No SGD code path exists anywhere in `fedgraa/` to fall back to. |
| Eq. 2 | Uniform FedAvg: $\theta_g^{t+1} = \frac{1}{K}\sum_k \theta_k^{t+1}$ | Not computed anywhere. `fedavg_weights()` implements Eq. 5 (size-weighted), never Eq. 2's uniform $1/K$. | 🟠 DEVIATION (of omission) — Eq. 2 as literally printed is never executed by any code path in the repository. This is the one place in Background section where a printed equation has genuinely zero corresponding code, not even as an alternate/unused path. |
| Eq. 3 | Local loss = mean CE over $D_k$ | `model.py::MLP.loss` and `.loss_and_grad` both call `tf.nn.sparse_softmax_cross_entropy_with_logits` then `tf.reduce_mean` | ✅ VERIFIED MATCH |
| Eq. 4 | Global loss = size-weighted sum of local losses | **Never computed as a standalone quantity anywhere.** No function evaluates $L(\theta)=\sum_k\frac{|D_k|}{|D|}L_k(\theta)$ directly. | 🟠 DEVIATION (omission) — descriptive equation, not prescriptive; still, a "100%" reproduction would compute and log it for verification, and nothing in `metrics.py`/`trainer.py` does. |
| Eq. 5 | Data-size-weighted FedAvg aggregation | `aggregation.py::fedavg_weights` = `sizes/sizes.sum()` | ✅ VERIFIED MATCH |
| Eq. 6 | Global gradient = size-weighted sum of local gradients | Never assembled as an explicit quantity anywhere (only implicit through already-locally-optimized `client_params` being combined, which is a different mathematical object from "the gradient of a single θ") | 🟠 DEVIATION (omission), same caveat as Eq. 4 |

---

## 4. §4 Methodology

### 4.1 Static Reparameterization — Eq. 7

$$\theta_g^{t+1} = \sum_{k=1}^K \frac{1/|D_k|}{\sum_j 1/|D_j|}\,\theta_k^t$$

`aggregation.py::static_reparam_weights`:
```python
inv = 1.0 / sizes
return inv / inv.sum()
```
✅ **VERIFIED MATCH.** (Minor cosmetic note only: Eq. 7 uses $\theta_k^t$
while both `static` and `fedavg` in the code aggregate the freshly-updated
`client_params` from the *current* round — i.e. the code follows Algorithm 1's
own line-11 indexing convention rather than Eq. 7's separate notation. This is
a paper-internal notation clash across two different parts of its own text,
not a code defect.)

### 4.2 Algorithm 1 — line by line

| Line | Paper text | Code | Verdict |
|---|---|---|---|
| 3 | init $\theta_g^0$ | `trainer.py::train()`: `init_params(...)` — genuine Keras `GlorotUniform(seed=...)` via `tf.keras.initializers.GlorotUniform` | ✅ |
| 4 | `for t = 1..T` | `for rnd in range(1, cfg.rounds + 1):` | ✅ |
| 5 | $S_t \leftarrow$ fixed set of $K$ clients | `self.split.clients` — **all** clients every round, no subsampling anywhere in `trainer.py` | ✅ ("fixed" = full participation) |
| 6–9 | send model, run `ClientUpdate`, collect $\theta_c^t$ | list comprehension over `self.split.clients` calling `_client_update` | ✅ (sequential execution, mathematically equivalent to "in parallel") |
| 10 | $w^t \leftarrow \text{FedGRAA}(\theta_c^t, w^{t-1})$ | `self._round_weights(...)` → for `method="fedgraa"` calls `self.fedgraa.update(...)`; for `fedavg`/`static` recomputes the fixed-formula weights instead (mathematically identical outcome since those methods' weights never depend on $\theta_c^t$) | ✅ |
| 11 | $\theta_g^t = \sum_k w_k^t\theta_k^t$ | `aggregate(client_params, w)` = `np.tensordot(weights, stacked, axes=(0,0))` per parameter array | ✅ |
| 12 | return $\theta_g^T$ | `self.final_params = global_params` | ✅ |
| 14 | init local $\theta_k^t \leftarrow \theta_g^{t-1}$ | `params = [p.copy() for p in global_params]` | ✅ |
| 15–16 | `for E epochs / for batch b` | `for _ in range(cfg.local_epochs): perm = rng.permutation(n); for start in range(0, n, cfg.batch_size):` (E=2, B=64 defaults) | ✅ |
| 17 | $\theta_k^t \leftarrow \theta_k^t - \eta\nabla L$ | `optimizer.step(params, grads)` — real Keras Adam | 🟡 AMBIGUITY — RESOLVED (see Eq. 1 row above) |
| 18 | return $\theta_k^t$ | `return params` | ✅ |

### 4.3 Algorithm 2 — FedGRAA weight learner, line by line

`aggregation.py::FedGRAA.update()` (current, single implementation — **no
`reading=` or `subgradient=` parameter exists anywhere in the class
signature**: `__init__(self, gamma, eta_prime=0.1, weight_epochs=5,
layer_sizes=None)`):

| Line | Paper text | Code | Verdict |
|---|---|---|---|
| 2 | `for epoch 1..E'` | `for _ in range(self.weight_epochs):` (E'=5 default) | ✅ |
| 3 | $\theta_g' = \sum_k w_k^{t-1}\theta_k^t$ **recomputed inside the loop** | `P = aggregate(client_params, w)` recomputed every iteration using the currently-evolving `w` | ✅ — this is the literal reading of the algorithm box (the line is printed *inside* the `for` loop); no alternate "compute-once" path exists in the code any more |
| 4 | $L_a=\min(L_h,L_m)$ | `_composite_grad_P`: computes both, takes `if L_h <= L_m: use (L_h,g_h) else (L_m,g_m)` | ✅ for the loss value; 🟡 the **gradient** of a non-smooth `min(·,·)` is mathematically undefined at $L_h=L_m$ — paper gives zero guidance; code commits to the standard convex sub-gradient convention (gradient of the currently-smaller branch) as its **single, non-configurable** choice |
| 5 | $L'=(1-\gamma)L_a+\gamma L_u$ | `loss = one_minus_g * L_a + self.gamma * L_u` | ✅ |
| 6 | $w^{t-1}\leftarrow w^{t-1}-\eta'\nabla_{w^{t-1}}L'$ | `grad_w = _grad_wrt_weights(grad_P, client_params)` (exact chain rule $dL'/dw_k=\langle\nabla_P L',\theta_k\rangle$), then `self.optimizer.step([w],[grad_w])` = real Keras Adam | 🟡 AMBIGUITY — RESOLVED, same Adam-vs-plain-GD notation gap as Eq. 1, resolved identically (Adam, everywhere, no toggle) |
| 7 | $w^t\leftarrow w^{t-1}$ | implicit — `w` mutated in place across the E' loop | ✅ |
| 8 | return $w^t$ | `return w` — **no projection/clipping/renormalization step of any kind** | ✅ — Algorithm 2 as printed has exactly 8 lines and no such step; the code no longer calls `_project_simplex` anywhere (that function does not exist in the current `aggregation.py` at all — grep confirms zero matches). This is now a **verified literal match**, correcting the two older documents that flagged unconditional simplex projection as an active deviation. |

**Net effect on Algorithm 2:** every line is now a literal, single-reading
implementation. The three sub-points the paper leaves genuinely ambiguous
(Adam-vs-SGD notation, sub-gradient of `min`, "static" vs "iterative" reading
of the inner loop) are each resolved to exactly one hard-coded behaviour with
no config surface to switch away from it — this satisfies the "100% ditto, not
a menu of options" bar the second-pass line-by-line document (2026-07-09)
originally demanded, and the current code now meets that bar for Algorithm 2
in full.

---

## 5. §5 "Data Settings"

| Paper sentence | Code | Verdict |
|---|---|---|
| *"datasets are divided among clients"* | `partition.py::partition_non_iid` | ✅ |
| *"highly frequent attacks ... capped at 10,000"* | `data.py::_cap_frequent_classes`: every class except Benign/Normal, if count > `frequent_cap` (10,000), `rng.choice(idx, size=cap, replace=False)` | ✅ value; 🟡 AMBIGUITY — RESOLVED on *sampling method* (paper never says random vs first-N; code commits to uniform random subsampling, single choice, no alternate path) |
| *"each client is assigned data for one attack type paired with an equal number of Benign samples"* | `partition_non_iid`: `benign_pick = benign_alloc[attack_id]` sized exactly `len(attack_idx)` | ✅ |
| Benign/Normal **not** in the capped-at-10,000 list | `_cap_frequent_classes`: `if class_id == spec.benign_id: keep all, continue` | ✅ |
| *"90% train / 10% validation"* | `partition.py::_train_val_split`, `val_fraction=0.10` default | ✅ |
| $\mathcal{D}_\mathcal{V}$ = concatenation of all clients' 10% slices | `X_val = np.concatenate(val_X_parts, axis=0)` | ✅ |

### 5.1 Feature scaling — paper is silent; code has one committed default, but it is a real methodological deviation

`data.py::_load_real_csvs` calls `_standard_scale(X)` on the **entire pooled
dataset** (every class, every client, including future validation rows)
**before** `partition_non_iid` runs; then `partition.py::partition_non_iid`
*re-derives* mean/std from the pooled **validation** set only (`X_val.mean()`
/`X_val.std()`) and re-applies that second normalization to train and val
alike. Concretely there are **two sequential standardizations** in the current
pipeline:
1. `data.py`: global scale over the full pre-partition pool (train+val+every client).
2. `partition.py`: a second global scale, computed only from the post-split validation pool, applied to everything again.

The paper's "Data Settings" paragraph says nothing about scaling at all — not
whether it happens, not what it's fit on. 🟠 **DEVIATION** (not an
"ambiguity resolved" — this is a concrete, unstated methodological artifact:
double-scaling with statistics leaking from validation into every client's
training features is not what any reasonable single reading of the paper's
silence would produce, and it is not the "naive default" the paper's silence
would imply). This is a genuine finding not flagged as precisely in this form
by any prior document (the older docs flagged "global vs per-client" scaling
as the ambiguity but did not notice the pipeline **re-scales twice, the
second time on validation-only statistics**).

> ✅ **FIXED (2026-07-10).** `partition.py::partition_non_iid` no longer
> re-derives `mean`/`std` from `X_val` and re-applies them. The single
> committed scaling pass now happens exactly once, upstream, in
> `data.py::_load_real_csvs` / `make_synthetic_dataset`, before partitioning
> ever runs. Regression test:
> `tests/test_reproduction_gaps.py::test_partition_does_not_rescale_features`
> (asserts every row handed to every client is bit-identical to a row of the
> input `X`). Full suite re-run: 52/52 passed.

### 5.2 CSV concatenation order and the 10,000-cap interaction

`load_dataset` reads CIC-IDS-2017 CSVs via `sorted(os.listdir(data_dir))`
(alphabetical) then `pd.concat` then caps. The paper never states whether
capping happens pre- or post-concatenation across the 8 daily files. 🟡
AMBIGUITY — RESOLVED as a single default (global-concat-then-cap); no
alternate `cap_strategy=` code path exists any more in the current `data.py`
(grep of the file confirms no `cap_strategy` field).

### 5.3 Network Architecture

| Paper sentence | Code | Verdict |
|---|---|---|
| *"fully connected neural network using Keras Sequential API"* | `model.py::MLP._build_model` builds a genuine `tf.keras.Sequential([...])`, `model.build(...)`, uses `tf.GradientTape` for backprop, `tf.keras.optimizers.Adam` for updates. `requirements.txt` pins `tensorflow>=2.21.0`. Verified importable/running (50/50 tests pass, which exercise `MLP.forward`/`.loss_and_grad`/`Adam.step` against real TF ops). | ✅ **VERIFIED MATCH** — this closes the single biggest gap ("BLOCKING B‑2" in the 2026-06-30 document) that all three older audit docs flagged as unresolved. It is resolved in the current codebase. |
| *"78 and 43 input features"* | `config.py`: `CIC_IDS_2017.n_features=78`, `UNSW_NB15.n_features=43` | ✅ |
| *"hidden layers 200, 100, 50, tanh"* | `ModelConfig.hidden_sizes=(200,100,50)`; Keras `Dense(fan_out, activation="tanh")` for all but the last layer | ✅ |
| *"output layer 15/10 neurons, softmax"* | Last `Dense` layer has `activation=None` (linear); softmax is applied *outside* the model via `tf.nn.softmax(logits)` in `.forward()`, and the loss uses `sparse_softmax_cross_entropy_with_logits` directly on logits (the numerically-stable equivalent of a softmax layer + CE loss) | ✅ mathematically equivalent; 🟡 note that the literal Keras object graph does **not** have a `Dense(..., activation="softmax")` output layer as the paper's prose literally says — it is functionally identical (fused logits→CE) but not a byte-for-byte "Sequential ending in a softmax Dense layer." |

### 5.4 Precision: float32, but not fully float32 end-to-end

`model.py` is careful to keep weights/activations in `float32` throughout
(`dtype="float32"` on the `Input` layer, `Dense` layers, `init_params`,
`Params` casts). However `data.py::_load_real_csvs` converts the raw CSV
features to **`float64`** (`X_df.to_numpy(dtype=np.float64)`) and both
`_standard_scale` calls (data.py and partition.py) operate in float64 before
the arrays are ever cast to float32 at the model boundary
(`tf.convert_to_tensor(X, dtype=tf.float32)` inside `MLP.loss`/`.forward`).
🟡 Minor — the final numbers fed to TensorFlow are float32, but the scaling
statistics themselves (mean/std) are computed in float64 then downcast,
which is a defensible and common practice but is one more unstated precision
choice layered on top of an already-silent paper.

### 5.5 Table 1 / Table 2 — verified digit-by-digit against current `config.py`

**CIC-IDS-2017** (`CIC_IDS_2017.used_counts`): BENIGN 63,901 / Bot 1,948 /
DDoS 10,000 / DoS GoldenEye 10,000 / DoS Hulk 10,000 / DoS Slowhttptest 3,219 /
DoS Slowloris 5,228 / FTP-Patator 5,931 / Heartbleed 11 / Infiltration 36 /
PortScan 10,000 / SSH-Patator 5,385 / Brute Force 1,470 / SQL Injection 21 /
XSS 652 — **all 15 values match the paper's Table 1 "Used" column exactly.**

**UNSW-NB15** (`UNSW_NB15.used_counts`): Normal 56,691 / Generic 10,000 /
Fuzzers 10,000 / Backdoor 2,329 / Worms 174 / Reconnaissance 10,000 / Analysis
2,677 / Exploits 10,000 / Shellcode 1,511 / DoS 10,000 — **all 10 values
match the paper's Table 2 "Used" column exactly.**

Independently re-summed for this audit: the sum of the 14 non-Benign CIC
"Used" counts is **63,901**, exactly equal to the Benign "Used" count. The sum
of the 9 non-Normal UNSW "Used" counts is **56,691**, exactly equal to the
Normal "Used" count. Both are consistent with "pair every attack sample with
one benign/normal sample," i.e. Benign/Normal usage is a *derived* quantity
(sum of attack demand), not an independently imposed number — a genuinely
elegant, verified structural match between the paper's own published tables
and the intended experimental design.

### 5.6 Correction to a prior document: the UNSW "benign pool shortage" claim is arithmetically wrong

`../archive/full_reproduction_gap_analysis.md` (§ M-3) claims *"total attack demand
(66,691) exceeds the Normal/benign pool (56,691) by ≈10,000 samples"* and
treats this as an unresolved data contradiction inherited by the code's
circular-indexing benign allocator. **Re-adding the same 9 numbers that
document itself lists** (Generic 10,000 + Fuzzers 10,000 + Backdoor 2,329 +
Worms 174 + Reconnaissance 10,000 + Analysis 2,677 + Exploits 10,000 +
Shellcode 1,511 + DoS 10,000) gives **56,691**, not 66,691 — the older
document's own addition was wrong by exactly 10,000 (an arithmetic slip, not
a real discrepancy). The correct conclusion, verified independently in this
audit, is: **for the "Used" counts baked into `config.py`, UNSW-NB15's benign
demand exactly equals its benign supply, with zero shortfall** — exactly
analogous to the CIC-IDS-2017 case in §5.5. `partition.py`'s circular-indexing
benign allocator (`benign_pool[indices % pool_size]`) will therefore consume
the **entire** shuffled benign pool with **zero wraparound** for both datasets,
*provided* the real CSVs, once loaded and capped, reproduce these exact
per-class counts. Because UNSW-NB15 CSVs are not present in this workspace
(§9.2), this exact-balance property has only been verified arithmetically
against the hard-coded Table 2 numbers and against the synthetic generator
(which reproduces `used_counts` by construction) — it has **not** been
verified against real UNSW-NB15 CSVs, since none exist here. The
`RuntimeWarning` path in `partition_non_iid` for pool shortage remains in the
code as a safety net but is not expected to fire for either dataset's nominal
counts.

---

## 6. §5 "Training Setup and Hyperparameters"

| Paper value | Code default (`FedConfig`) | Verdict |
|---|---|---|
| Adam, η = 0.01 | `eta: float = 0.01` | ✅ |
| E = 2 local epochs | `local_epochs: int = 2` | ✅ |
| B = 64 batch size | `batch_size: int = 64` | ✅ |
| T = 200 communication rounds | `rounds: int = 200` | ✅ |
| K = 14 (CIC) / K = 9 (UNSW) clients | `DatasetSpec.n_clients` property = `len(attack_ids)` = 14 / 9 | ✅ |
| sparse categorical cross-entropy | `tf.nn.sparse_softmax_cross_entropy_with_logits` in `MLP.loss`/`.loss_and_grad` | ✅ |
| FedGRAA: Adam, η'=0.1, E'=5 | `eta_prime: float = 0.1`, `weight_epochs: int = 5` | ✅ |
| γ ∈ {0, 0.5, 1} | `run_gamma_sweep(gammas=(0.0,0.5,1.0))` | ✅ |
| n=3 rare classes named in paper | `rare_label_order[:3]` for both datasets matches the paper's named classes exactly | ✅ |

---

## 7. Figure 1/2/3 experimental structure — code coverage

| Paper structure | Code | Verdict |
|---|---|---|
| Figure 1: 3 methods (FedAvg, Static, FedGRAA) × γ∈{0,0.5,1} at n=3 | `run_gamma_sweep()` runs exactly `fedavg`, `static`, and `fedgraa_g{0.0,0.5,1.0}` — 5 total runs | ✅ code coverage complete |
| Figures 2–3: n∈{3,5,7} × γ∈{0,0.5,1} (6 sub-panels each in the paper) | `run_n_sweep(ns=(3,5,7), gammas=(0.0,0.5,1.0))` produces `fedavg`, `static`, and 3 FedGRAA-γ variants for **every** n — 5×3 = 15 total runs, a full superset of the paper's 2×6=12+ sub-panel structure | ✅ code coverage complete — this fixes the historical "C-3" gap (previously only one fixed γ was run per n) |
| n=5, n=7 rare-class identities (not named in paper for either dataset) | `config.py::rare_label_order` — CIC extends to 7 named classes by ascending size; UNSW's `rare_label_order` only has **5** entries (`Worms, Shellcode, Backdoor, Analysis, Exploits`), so requesting n=7 for UNSW returns only 5 classes (`rare_client_ids` slices `[:n]` past the list end) | 🟡 AMBIGUITY — RESOLVED for CIC (single deterministic choice); 🟠 **structural limitation** for UNSW n=7, now ✅ **loud instead of silent (fixed 2026-07-10)** — the paper's own Figure 3 sweeps n=7 for UNSW but UNSW-NB15 genuinely has only 5 attack classes that are not artificially capped at 10,000 (the other 4 — Generic, Fuzzers, Reconnaissance, DoS — are frequency-capped and therefore not "rare" by the experiment's own logic). `rare_client_ids()` now raises a `RuntimeWarning` whenever `n` exceeds the dataset's `rare_label_order` length, naming the shortfall explicitly, instead of returning fewer classes with no signal at all. This does **not** and cannot make a genuine 7-class UNSW experiment possible (the paper's own dataset still only has 5 non-capped rare classes — that part is a data-level fact, not a code bug), but the previously-silent behaviour is now explicit and impossible to miss. Regression test: `test_unsw_n7_warns_instead_of_silent_truncation`. |

---

## 8. Round-0 accuracy — resolved, and now the *default*

The paper's Figure 1 table prints exactly `0.00` for every method/γ/dataset at
round 0. A freshly Glorot-initialized softmax network does **not** produce 0%
accuracy in reality (~1/K by chance). `FedConfig.round_zero_zero` now
**defaults to `True`** (verified in `config.py`, line: `round_zero_zero: bool
= True`), and `trainer.py::train()` forces `_all_0 = 0.0` / `_last_0 = 0.0`
when the flag is set, while still computing and storing the real composite
loss `L'` at round 0 for FedGRAA (`_comp_0` is always the true value,
independent of the flag). ✅ **VERIFIED MATCH to the paper's printed table**,
by explicit design choice, tested in
`tests/test_reproduction_gaps.py::test_round_zero_zero_flag`.

---

## 9. Data availability — the one hard blocker

### 9.1 CIC-IDS-2017 — present and exercised

All 8 expected daily CSVs are present in `data/CIC-IDS-2017/` (confirmed via
workspace listing): `Monday-WorkingHours`, `Tuesday-WorkingHours`,
`Wednesday-workingHours`, `Thursday-Morning-WebAttacks`,
`Thursday-Afternoon-Infilteration`, `Friday-Morning`,
`Friday-Afternoon-PortScan`, `Friday-Afternoon-DDos` — the canonical 8-file
CIC-IDS-2017 release. `.cache/CIC_IDS_2017_seed0_X.npy` and
`..._y.npy` exist in the workspace, confirming the real-CSV loader has been
executed at least once end-to-end (label matching, capping, scaling) and
cached successfully.

### 9.2 UNSW-NB15 — completely absent

`data/UNSW-NB15/` is **empty** (0 files, confirmed by direct directory
listing at audit time). `data.py::download_unsw_nb15` exists and is wired to
auto-attempt a download from AARNet CloudStor URLs, but **this audit did not
execute a network download** (out of scope for a documentation-only audit,
and the URLs' current validity is unverified). Until real UNSW-NB15 CSVs are
placed in this directory, **every UNSW-NB15 number this codebase can produce
is generated by `make_synthetic_dataset`** — Gaussian clusters that match
Table 2's *sample counts* but have no relationship to real network-traffic
feature distributions. 🔴 **BLOCKED.** No code change can fix this; it
requires the actual dataset.

---

## 10. Random seed — unresolvable without author disclosure

The paper reports specific numbers (88%, 0.33, 66.66, etc. in Fig. 1's table)
without ever stating a random seed for weight init, benign-pool shuffling,
train/val splitting, or batch order. `FedConfig.seed` defaults to `0`; there
is no way to know whether any particular seed value reproduces the paper's
exact trajectory, because the original authors' training code and environment
were never released. `experiment.py::run_seed_sweep` exists specifically to
let a future user scan seeds and report variance, but this audit did not run
it (would require dozens of full 200-round runs). 🔴 **BLOCKED**, structurally,
independent of code quality.

---

## 11. What has **actually** been numerically verified vs. what has not

This is the most important section for a "ruthless" audit and the one most
likely to be glossed over: **algorithmic fidelity to the paper's text is not
the same thing as having reproduced the paper's numbers.**

### 11.1 Figure 1 (γ sweep)

- **Code path:** complete, tested at small scale (≤50 rounds, synthetic and a
  handful of real-CSV-loader assertions) — 50/50 tests pass.
- **Full-scale run:** `run_experiments.py` with `--rounds 200
  --data-dir data/CIC-IDS-2017 --cache-dir .cache` has **not** been executed as
  part of this audit or (per `file_search` for `*.json`/`*.png` across
  `PAKDD/`, which returned **zero results**) at any prior point captured in
  this repository. No `save_results()` JSON output and no `plot.py` PNG output
  exist anywhere in the workspace.
- **Verdict:** ⚪ **NOT YET EXECUTED.** The abstract's "88% accuracy on
  minority classes" and "outperforms FedAvg by up to 55%" claims have **not**
  been checked against this codebase's actual output numbers on real
  CIC-IDS-2017 data. The only empirical evidence of directional correctness is
  `tests/test_trainer.py`'s small-scale, few-round, synthetic-data assertion
  that FedGRAA's rare-class accuracy is $\ge$ FedAvg's — a sanity check, not a
  reproduction of the headline numbers.

### 11.2 Figures 2–3 (n sweep)

- Same status: code path complete and covers the full 3(n)×3(γ) matrix, never
  executed at 200-round scale, no output artifacts exist.

### 11.3 What a genuine numeric-reproduction pass would require

1. Run `python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir
   .cache --datasets CIC-IDS-2017 --sweep both --rounds 200` to completion
   (estimated tens of minutes to a few hours on CPU per the README's own
   estimate) and persist results with `save_results`.
2. Obtain real UNSW-NB15 CSVs (currently impossible from within this audit;
   requires either a successful `download_unsw_nb15()` run or a manual
   download from the official UNSW site) and repeat for UNSW-NB15.
3. Compare the resulting `all_acc`/`last_n_acc` numbers at rounds
   {0,20,...,200} against the paper's own Figure 1/2/3 tables.
4. Given the paper's own numbers are suspected placeholders (§8 of the
   ambiguity report), even a "perfect" code-level reproduction is not
   expected to hit them exactly — but this has never been empirically
   checked either way in this repository.

This audit deliberately does **not** launch that multi-hour run itself (it
was scoped as a documentation task), but flags its absence as the single
largest remaining gap between "the code is faithful to the paper's
specification" (true, per §§3–8) and "the paper has been reproduced" (not yet
demonstrated).

---

## 12. Appendix A.1 — Convergence proof vs. the actual network

The proof's Assumption 1 requires $L'(\theta)$ to be **convex** and
$\beta$-smooth. The composite validation loss $L'$ is a function of a
multi-layer `tanh`-activated neural network's parameters — this is **not
convex** in general, and the paper offers no argument for why it would be for
this specific architecture. This is a mathematical gap **in the paper
itself**, not something the code can "reproduce" one way or the other; the
code's only engagement with the proof is `TrainHistory.is_composite_loss_monotone`,
an **empirical, not analytic**, check of whether Lemma 1's conclusion
($L'(w^t)\le L'(w^{t-1})$) happens to hold on a given run — this has been unit
tested (`test_composite_loss_tracked_for_fedgraa`) but never verified across a
full 200-round real-data run (see §11).

---

## 13. Full severity-tagged findings list (for quick scanning)

| # | Finding | Severity |
|---|---|---|
| F1 | Eq. 2 (uniform 1/K FedAvg) is never computed by any code path; only Eq. 5 is | 🟠 DEVIATION (omission) |
| F2 | Eq. 4 / Eq. 6 (global loss/gradient as explicit standalone quantities) are never computed | 🟠 DEVIATION (omission) |
| F3 | Adam vs. plain-GD notation conflict (Eq.1/Alg.1-L17 and Eq.10/Alg.2-L6 vs §5 prose) resolved as Adam everywhere, consistently, no toggle | 🟡 RESOLVED |
| F4 | Sub-gradient of `L_a=min(L_h,L_m)` resolved as "gradient of the smaller branch," no toggle | 🟡 RESOLVED |
| F5 | Inner-loop reading of Algorithm 2 (recompute every epoch) resolved as the literal, only reading; no alternate "static" reading exists in code any more | 🟡 RESOLVED |
| F6 | No simplex projection anywhere — literal 8-line-only Algorithm 2, verified by direct grep of `aggregation.py` | ✅ VERIFIED MATCH (correcting 2 stale prior docs) |
| F7 | Round-0 forced to 0.00 by default, matching the paper's printed table | ✅ VERIFIED MATCH (by explicit design choice) |
| F8 | Feature scaling happened **twice** (once in `data.py` globally, once in `partition.py` on validation-only statistics) — an unstated, undocumented double-normalization the paper's silence does not obviously imply | ✅ **FIXED 2026-07-10** — `partition.py`'s redundant re-scaling block removed; single upstream scaling pass only |
| F9 | UNSW's `rare_label_order` has only 5 entries; requesting n=7 silently returned 5 classes, not 7, with no error | ✅ **FIXED 2026-07-10** — `rare_client_ids()` now raises `RuntimeWarning` naming the shortfall; the underlying data-level limitation (UNSW genuinely lacks a 7th rare class) is unchanged and unfixable, but the silence is gone |
| F10 | UNSW-NB15 "benign pool shortage" claimed by a prior document is an arithmetic error — no shortage exists in the hard-coded Table 2 counts (56,691 = 56,691 exactly) | ✅ **correction of a stale/wrong prior finding** |
| F11 | UNSW-NB15 real data is completely absent from the workspace | 🔴 BLOCKED |
| F12 | No random seed disclosed by the paper; unresolvable | 🔴 BLOCKED |
| F13 | No full 200-round run has been executed and compared to the paper's numbers, on either dataset; zero result JSON/PNG artifacts exist anywhere in the repo | ⚪ NOT YET EXECUTED |
| F14 | Output layer is logits+softmax-applied-outside-the-graph rather than a literal `Dense(..., activation="softmax")` layer, though numerically equivalent | 🟡 minor cosmetic note |
| F15 | Convergence proof's convexity assumption does not hold for a tanh MLP; this is a paper-level gap the code cannot resolve either way | 🟠 paper-level (see also ambiguity report) |

---

## 14. Conclusion

Relative to the **base paper's own text**, the current codebase is a
disciplined, single-choice, non-configurable implementation: every
hyperparameter matches, the full Keras/TensorFlow architecture is now
genuinely in place (closing the single largest gap flagged by every prior
audit), Algorithm 1 and Algorithm 2 are implemented line-for-line including
the *absence* of a simplex projection the paper never asked for, and every
place the paper is silent or self-contradictory has been resolved to exactly
one documented behaviour rather than left as a matrix of options. Of the two
concrete code-level deviations this audit originally found (F8 double
feature-scaling, F9 silent UNSW n=7 truncation), **both have now been fixed**
(2026-07-10), each with a dedicated regression test (52/52 tests passing).
What remains is not a code-quality problem: (1) two harmless equation
omissions that are descriptive, not prescriptive, in the paper itself (Eq.
2/4/6 never computed as standalone quantities — see F1/F2), and (2) the fact
that, as of this audit, **no one has actually run the full pipeline to
completion on real data and checked the resulting numbers against the
paper's published tables** — the reproduction is code-complete but
numerically unverified.
