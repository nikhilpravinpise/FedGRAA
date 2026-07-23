# Plan — Faithful Reproduction of FedGRAA

> **⚠️ ARCHIVED — describes a codebase state that no longer exists.** This was
> the original 2026-06-28 build plan and proposed a **hand-written NumPy MLP**
> (no TensorFlow). The project was later rewritten Keras-only; see
> `strict_reproduction_report.md` for the current architecture and
> `../../PAKDD_STATUS_REPORT.md` (repo root) for current status. Kept for historical
> record of the original design intent.

> **Goal.** Build a complete, working, tested implementation of the paper
> *"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems"*
> (FedGRAA) that reproduces **exactly** what the paper describes — with **no
> deviations** introduced by the critical questions in
> [understanding/07_my_questions.md](../../understanding/07_my_questions.md). Every
> design choice the paper makes is implemented *as written*; the alternatives
> raised in the critique (ReLU, SMOTE, SGD, stratified split, etc.) are
> deliberately **not** substituted.

---

## 1. Fidelity contract — what "no deviations" means

Every item below is locked to the paper and the understanding notes. The
implementation must match each one.

| Aspect | Paper value (locked) | Source |
|---|---|---|
| Datasets | CIC-IDS-2017 (78 feat, 15 cls), UNSW-NB15 (43 feat, 10 cls) | §5, Tables 1–2 |
| Frequent-class cap | 10,000 samples | §5 Data Settings |
| Client partition | **one attack type per client** + equal Benign/Normal | §5 Data Settings |
| Train/val split | **90% train / 10% validation** (`D_V`) | §5 Data Settings |
| Hidden layers | **200 → 100 → 50**, activation **tanh** | §5 Network Architecture |
| Output layer | 15 (CIC) / 10 (UNSW) neurons, **softmax** | §5 Network Architecture |
| Loss | **sparse categorical cross-entropy** | §5 Training Setup |
| Local optimizer | **Adam**, `η = 0.01` | §5 Training Setup |
| Local epochs / batch | `E = 2`, `B = 64` | §5 Training Setup |
| Rounds | `T = 200` | §5 Training Setup |
| Clients | `K = 14` (CIC), `K = 9` (UNSW) | §5 Training Setup |
| Weight optimizer | **Adam**, `η' = 0.1`, `E' = 5` | §5 Training Setup |
| Personalization knob | `γ ∈ {0, 0.5, 1}` | §5 |
| Protected rare clients | `n ∈ {3, 5, 7}` | §5 |
| Static init weights | inverse-data-size, normalized (eq. 7) | §4 |
| Freq. split threshold | **median frequency** → `D_V^h` / `D_V^m` | §4 |
| Frequent-class loss | `L_a = min(L_h, L_m)` (eq. 8) | §4 |
| Combined loss | `L' = (1−γ)L_a + γ L_u` (eq. 9) | §4 |
| Weight update | `w ← w − η' ∇_w L'` (eq. 10) | §4 |
| Global update | `θ_g = Σ_k w_k θ_k` (Algo 1, line 11) | §4 |

**Rare classes (default n = 3):**
- CIC-IDS-2017: Heartbleed (11), SQL Injection (21), Infiltration (36)
- UNSW-NB15: Worms (174), Shellcode (1511), Backdoor (2329)

---

## 2. Implementation strategy

### 2.1 Framework choice
The paper uses the Keras Sequential API. TensorFlow has no stable wheel for the
installed interpreter (Python 3.13), so we implement the **identical network and
math from scratch in NumPy**:

- Dense layers `200→100→50` with **tanh**, **softmax** output, **sparse
  categorical cross-entropy** loss — byte-for-byte the same model the paper
  specifies.
- A hand-written **Adam** optimizer (same update rule Keras uses) for both the
  local model (`η=0.01`) and the FedGRAA weight learner (`η'=0.1`).
- Manual back-propagation. This is *not* a deviation: the architecture,
  activations, loss, optimizer, and all hyperparameters are identical to the
  paper. Only the execution engine differs.

This choice also makes the whole project dependency-light (numpy/pandas/sklearn
only), deterministic, and fully unit-testable.

### 2.2 The FedGRAA gradient (core math)
The global model is a linear blend of client models:
`P = Σ_k w_k · P_k`, where `P_k` is the flattened parameter vector of client `k`.
Therefore `∂P/∂w_k = P_k`, and for any validation loss `L'(P)`:

```
∂L'/∂w_k = ⟨ ∇_P L'(P), P_k ⟩      (dot product over all parameters)
```

So FedGRAA only needs:
1. One back-prop pass of `L'` w.r.t. the **aggregated parameters** `P` on each
   validation subset (`D_V^h`, `D_V^m`, `D_V^n`).
2. `∇_P L_a = ∇_P L_h` or `∇_P L_m` depending on which of `L_h, L_m` is smaller
   (sub-gradient of `min`).
3. `∇_P L' = (1−γ)∇_P L_a + γ ∇_P L_u`.
4. `grad_w[k] = ⟨∇_P L', P_k⟩` for every client `k`.
5. Adam step on `w` with `η' = 0.1`, repeated `E' = 5` times per round.

This reproduces eq. (8)–(10) exactly.

### 2.3 Data
Real CIC-IDS-2017 / UNSW-NB15 CSVs are large external downloads not shipped in
the repo. We support both:
- **Real loader** (`data.py`): reads CSVs from a configurable directory, cleans
  `inf`/`NaN`, label-encodes, standard-scales — the standard IDS pipeline.
- **Synthetic generator**: reproduces the exact per-class "Used" sample counts
  from Tables 1 & 2 with class-separable Gaussian features, so the *entire*
  pipeline and all experiments run end-to-end with zero downloads and still
  exhibit the FedAvg-fails / FedGRAA-wins behaviour.

---

## 3. Module layout

```
fedgraa/
  __init__.py        # public API
  config.py          # dataclasses: ModelConfig, FedConfig, dataset presets
  model.py           # MLP: glorot init, forward, backprop, loss; Adam optimizer
  data.py            # real CSV loader + synthetic generator + preprocessing
  partition.py       # non-IID one-attack-per-client split, 90/10, median split
  aggregation.py     # fedavg(), static_reparam(), FedGRAA weight learner
  trainer.py         # Algorithm 1 outer loop (server + ClientUpdate)
  metrics.py         # accuracy: all-classes and last-n (rare) classes
  experiment.py      # sweeps over γ, n; collects per-round histories
  cli.py             # `python -m fedgraa ...` entry point
tests/
  test_model.py        # forward shapes, gradient check vs finite differences
  test_data.py         # synthetic counts match Tables 1 & 2
  test_partition.py    # one attack/client, equal benign, 90/10, median split
  test_aggregation.py  # static weights sum to 1; fedavg == size-weighted mean
  test_fedgraa.py      # weight-grad matches finite-diff; γ behaviour
  test_trainer.py      # end-to-end short run improves accuracy
  test_metrics.py      # all/last-n accuracy correctness
run_experiments.py   # reproduce Fig 1 (γ sweep) and Fig 2/3 (n sweep)
requirements.txt
README.md
plan.md
```

---

## 4. Build order (each step verified before the next)

1. **plan.md** (this file).
2. **Scaffold** package, `requirements.txt`.
3. **config.py** — lock every hyperparameter from the fidelity table.
4. **model.py** — MLP forward/backprop + Adam; finite-difference gradient test.
5. **data.py** — synthetic generator (exact Table 1/2 counts) + real loader.
6. **partition.py** — non-IID partition, 90/10 split, median-frequency split.
7. **aggregation.py** — FedAvg, Static (eq. 7), FedGRAA (eq. 8–10).
8. **trainer.py** — Algorithm 1 federated loop.
9. **metrics.py** — all-class and last-n accuracy.
10. **experiment.py** + **cli.py** — γ/n sweeps, result tables.
11. **tests/** — full suite (unit + gradient checks + smoke E2E).
12. **Run** the suite + a smoke experiment; confirm FedGRAA > Static > FedAvg on
    rare classes.
13. **README.md** — usage, math, reproduction instructions, real-data hookup.

---

## 5. Acceptance criteria ("done", not "good enough")

- [ ] All hyperparameters match the fidelity table in §1 exactly.
- [ ] FedGRAA weight gradient matches a finite-difference check to < 1e-4.
- [ ] Model back-prop matches finite differences to < 1e-5.
- [ ] Synthetic dataset reproduces the exact "Used" counts of Tables 1 & 2.
- [ ] Partition yields one attack per client + equal benign, K=14 (CIC)/9 (UNSW).
- [ ] End-to-end run shows the paper's qualitative result: on rare classes
      **FedGRAA(γ=1) > Static > FedAvg≈0**, and raising γ trades a little
      overall accuracy for more rare-class accuracy.
- [ ] `run_experiments.py` reproduces the Figure-1/2/3 style tables.
- [ ] Full test suite passes.
- [ ] README documents the math, the fidelity contract, and real-data usage.
