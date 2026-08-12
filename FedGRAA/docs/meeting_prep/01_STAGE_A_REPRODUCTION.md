# Stage A, Deeply — The Faithful Reproduction

**Primary sources:** `docs/current/STAGE_A_MASTER_REPORT.md` (979 lines),
`docs/current/fedgraa_divergence_diagnosis.md`, `docs/current/
PAPER_AMBIGUITY_REPORT.md`. Status: ✅ complete, completion commit `be4ef03`,
79/79 tests passing at close-out (142 after Stage B added its own suite).

Read this like you already know it — it's written to let you reconstruct
the whole argument from memory, not to be read aloud.

---

## 1. What FedGRAA (v1) actually is

**Target paper:** *"Federated Gradient-Refined Adaptive Averaging for
Intrusion Detection Systems"* — anonymized, "No Author Given" — source:
`docs/paper/PAKDD_Final_Version_Submit_v3.md`.

**The problem it addresses.** In federated intrusion detection, each client
typically holds one attack type plus benign traffic. Standard FedAvg
weights each client by how much data it has — which systematically drowns
out the clients holding the rarest attack types, exactly the classes a real
IDS most needs to catch (Heartbleed: 11 samples; SQL Injection: 21;
Infiltration: 36 — vs. Benign at 2.27M).

**The proposed fix.** Instead of a fixed weighting rule, *learn* the
aggregation weight vector `w` each round by gradient descent on a composite
validation objective that explicitly trades off common-class performance
against rare-class performance via a personalization coefficient `γ∈[0,1]`.

**Architecture:** `Input(d) → Dense(200,tanh) → Dense(100,tanh) →
Dense(50,tanh) → Dense(C,softmax)`. `d=78`, `C=15` on CIC-IDS-2017; `d=39`,
`C=10` on UNSW-NB15. Loss: sparse categorical cross-entropy.

**Hyperparameters, paper value = code default (all "locked to the paper"):**

| Symbol | Meaning | Value |
|---|---|---|
| `η` | local Adam LR | 0.01 |
| `E` | local epochs/round | 2 |
| `B` | local batch size | 64 |
| `T` | federated rounds | 200 |
| `η′` | weight-learner Adam LR | 0.1 |
| `E′` | weight-learner epochs/round | 5 |
| `γ` | personalization coeff. | {0, 0.5, 1} swept |
| `n` | # protected rare classes | {3, 5, 7} swept |
| `K` | # clients | 14 (CIC) / 9 (UNSW) |

**Algorithm 1 (federated loop).** Standard: each round, every client does
`E` local Adam epochs from the current global model, the server calls
Algorithm 2 to get updated weights `w^t`, then `θ_g^t = Σ_k w_k^t θ_k^t`.

**Algorithm 2 (the FedGRAA weight learner) — the actual contribution, and
the whole story.** For `E′` epochs per round:

```
θ'_g = Σ_k w_k^{t-1} θ_k^t
L_a  = min(L_h(θ'_g; D_V^h), L_m(θ'_g; D_V^m))
L'   = (1-γ)L_a + γL_u
w^{t-1} ← w^{t-1} - η'∇_{w^{t-1}} L'
```

**Algorithm 2 has exactly 8 lines. There is no line 9.** No projection,
clipping, or renormalization of any kind — verified by direct grep of the
reproduction (`fedgraa/aggregation.py`), zero matches. This one missing
line is the entire finding.

---

## 2. The datasets — genuine, and why that matters

Both datasets were forensically audited against their raw CSVs specifically
to rule out "the data is corrupted or synthetic" as an alternative
explanation (`docs/audit/data_provenance_audit.md`). **Verdict: both
genuine, complete, correctly labeled, bit-faithful to canonical
distributions.**

**CIC-IDS-2017:** 8 raw CSVs, 2,830,743 rows total — matches the canonical
total exactly. 10.9% of rows are exact duplicates, which is itself evidence
of authenticity (it's the known raw-release duplicate-flow issue; a cleaned
re-upload wouldn't have it). `rare_label_order` (rarest first): Heartbleed
(11) → SQL Injection (21) → Infiltration (36) → XSS (652) → Brute Force
(1,470) → Bot (1,948) → DoS Slowhttptest (3,219).

**UNSW-NB15:** official train/test split, 257,673 rows total, matches
canonical exactly. Only **5** attack classes are non-frequency-capped:
Worms (174), Shellcode (1,511), Backdoor (2,329), Analysis (2,677),
Exploits (10,000 used / 44,525 raw). This is why UNSW **cannot support a
genuine n=7 sweep** — `rare_client_ids(7)` returns the same 5 classes
`rare_client_ids(5)` does, with a `RuntimeWarning` naming the shortfall
rather than silently padding. Documented honest divergence, not a bug.

**Why the data can't be the cause of the central finding:** (1) the
instability reproduces on clean synthetic Gaussian data too — real-data
quirks aren't required to trigger it; (2) real data controls *severity*,
not *existence* — CIC's extreme skew drives 193× blowup, UNSW's milder skew
drives 7.6× (both seed-0 figures; §4 below has the full 10-seed range);
(3) the diagnostic fix recovers accuracy on the *same, unchanged* data —
only the algorithm changed.

An earlier archived document had claimed a ~10,000-sample UNSW benign
shortfall (M-3) — re-checked and found to be a simple arithmetic error in
that document (summed to 66,691 instead of 56,691). No shortfall exists.

---

## 3. The 9 strict-reproduction judgment calls

Wherever the paper is silent or self-contradictory, the codebase commits to
exactly **one** documented, non-configurable choice — no interpretation
flags, no menu of options. (All such toggles that once existed have been
deleted, verified by grep.)

| # | Ambiguity | Decision |
|---|---|---|
| 1 | Paper's Fig.1 shows 0.00 at round 0 for every method, but a fresh network doesn't score 0% | `round_zero_zero` defaults `True` — forces the *reported table* to 0.00, model itself unchanged |
| 2 | No projection step in Algorithm 2 | Implemented literally — **no projection performed** |
| 3 | Algorithm 2's inner loop reuses `w^{t-1}` for two things | Iterative reading: recompute using the current, evolving `w` each inner epoch |
| 4 | Sub-gradient of `L_a=min(L_h,L_m)` at the crossing point | Standard sub-gradient of whichever arm is currently smaller |
| 5 | Paper never mentions feature scaling | Global standard scaling before partitioning |
| 6 | Does the 10,000 cap apply to Benign? | No — literal reading of "highly frequent **attacks**... capped" |
| 7 | "Accuracy on the last n classes" never defined | Mean per-class recall over the `n` protected classes |
| 8 | Validation split method unspecified | Plain random permutation, no stratification |
| 9 | How is the benign pool divided among clients? | Pre-shuffle once, allocate sequentially, wrap circularly if demand > supply |

Two more, documented rather than patched: UNSW genuinely can't support
n=7 (above); and Algorithm 1 line 17 reuses `θ_k^t` on both sides of an
assignment (standard pseudocode shorthand, technically malformed notation).

**A second, separate deviation, worth having ready if asked "what else is
different":** `fedgraa/trainer.py`'s `client_train_round` resets each
client's *model weights* to the global aggregate every round (as
Algorithm 1 specifies) — but never rebuilds that client's Adam optimizer,
so its momentum/velocity state persists across rounds whose starting
weights have since changed underneath it. Deliberately kept as "a known
deviation, not a bug" — orthogonal to the simplex-projection defect, not
fixed by its correction. Confirmed by direct code reading: `trainer.py`'s
`_client_tf_optimizers` dict is built once in `__init__`, outside the round
loop.

---

## 4. The central finding, mechanically, step by step

1. **`w` is supposed to be a probability distribution** over clients
   (`w_k≥0`, `Σw_k=1` — the "probability simplex," `Δ`). Nothing in the
   8-line algorithm keeps it there.
2. **Measured, not inferred:** instrumenting every round
   (`scripts/diagnose_fedgraa.py`, `eval_every=1`), `w` violates the
   simplex — `|Σw−1|>10⁻³` or `min(w)<−10⁻⁶` — at **round 1**, at every
   tested `γ∈{0,0.5,1}`, on **both** datasets, with **zero exceptions**.
   At Stage B's full 10-seed scale this held **120/120** unprojected cells.
   `Σw` ranged `[−2.62,+2.77]` on CIC, `[−1.87,+2.12]` on UNSW — individual
   weight components as negative as −1.157. A negative weight means the
   aggregation is *subtracting* that client's model from the global one.
3. **The consequence:** the paper's own Lemma 1 claims composite loss `L'`
   is non-increasing round-over-round. Violated immediately. At γ=1 on CIC,
   the first 10 rounds of `L'`:
   `2.98, 2.77, 4.53, 6.06, 40.52, 75.55, 261.48, 290.24, 319.11, 205.30,
   149.79` — non-increasing for exactly one step, then up by more than an
   order of magnitude within four rounds.
4. **The downstream effect:** rare-class accuracy collapses to **~0%** at
   the paper's own 200-round horizon, both datasets, under the algorithm
   exactly as printed — while the naive "Static Reparameterization"
   baseline (fixed inverse-size weighting, *no learning at all*) gets
   **84–92% for free**. The learned mechanism actively underperforms doing
   nothing clever.
5. **Confirming causality, not just correlation:** re-projecting `w` onto
   the simplex after every update, everything else held fixed (same
   gradient, same `η′=0.1`, same `E′=5`), collapses blowup: CIC γ=1
   193×→4.3×, CIC γ=0 3.78×→1.01×. The same gradient direction, merely
   re-clipped, does not blow up — this isolates the defect to the missing
   projection specifically.

**Formal statement (this is the sentence to have ready verbatim):**
Algorithm 2, taken literally, solves an *unconstrained* problem over all of
`ℝ^K`, using an update rule whose descent guarantee (Lemma 1) is proved
only under an implicit, never-enforced assumption that `w` stays in the
*constrained* set `Δ`. The paper presents Algorithm 2 as if it operates on
`Δ` (calling weights a "weighted average" throughout) while implementing an
algorithm with no mechanism to keep it there.

---

## 5. Headline numbers table (γ=1, n=3, 200-round literal reproduction)

| | Paper claims | Literal 200-round reproduction | Diagnostic combined-fix (60-round, seed-mean) |
|---|---|---|---|
| CIC-IDS-2017 | **88.88%** | **0.00%** | **94.44% ± 3.93%** |
| UNSW-NB15 | **21.09%** | **33.33%** | **74.79% ± 1.99%** |

Note the UNSW literal number *exceeds* the paper's own claim — a milder,
different-shaped divergence than CIC's, consistent with UNSW's instability
being less catastrophic (7.6× blowup vs. CIC's 193× at seed 0).

**Diagnostic-fix recipe:** Euclidean simplex projection + `η′` cut 10×
(0.1→0.01, `E′=5` unchanged). Stabilizes loss to exactly 1.0× on CIC, ≤1.5×
on UNSW; recovers rare-class accuracy to 72–94% (CIC, γ-dependent) and
0–75% (UNSW, γ-dependent). **Kept strictly diagnostic** — lives only in
`scripts/diagnose_fedgraa.py`'s `ProjectedFedGRAA`, never merged into the
strict-reproduction default in `fedgraa/`. (Stage B later re-validates this
fix at full statistical power — see `02`.)

**Bottom-line sentence, verbatim-ready:** *"Algorithm 2, implemented
exactly as printed, does not reliably converge on real network-intrusion
data — worst on CIC-IDS-2017 (clean 0% at every γ), milder but still
present on UNSW-NB15. The cause is a measured, mechanistically understood
numerical-stability defect: no simplex projection, which interacts
especially badly with the paper's own η′=0.1 on CIC's more extreme client
skew. The algorithm's central idea — gradient-refined adaptive weighting to
protect rare classes — is directionally sound and empirically recoverable;
what the paper omits is one constraint and one hyperparameter sensitivity."*

---

## 6. The 17-issue paper critique — see `05` for the full catalogue

Stage A also independently critiqued the paper's own *text*, with no
reference to the code — 17 distinct issues: 3 direct equation-vs-prose
contradictions (optimizer identity, twice, and two formulas both called
"FedAvg"), a proof-vs-model-class contradiction (convexity assumed, tanh-MLP
used), a sweep-vs-own-data contradiction (UNSW n=7), 6 undefined central
terms, 2 notational issues, and 2 statistical red flags on the paper's own
numbers (the 33.33/66.66/77.77/88.88 repeating-decimal pattern). Full
detail with exact quotes: `05_PAPERS_CODE_AND_AI_INTEGRITY.md` §1.

---

## 7. Verification — why you can say "we checked this," not "we think this"

- **79/79 tests** at Stage A close (later 142, then 563 by Stage C) — unit,
  gradient-vs-finite-difference, fidelity, aggregation, end-to-end.
- **`check_stage_a_complete.py` exits 0** — every paper artifact has a
  persisted counterpart; every figure renders every curve (a real bug —
  the n-sweep plot silently dropped all 3 FedGRAA curves per panel, found,
  fixed, regression-tested: `test_plot_n_sweep_includes_fedgraa_curves`).
- **A full independent re-audit (2026-07-23)** re-derived every headline
  claim from raw sources, not prior docs — confirmed the dataset genuineness
  via a feature-signature check (real CIC post-scaling max|x|=672, 8/78
  all-zero columns — the known real-CIC signature; a synthetic Gaussian
  maxes at ≲6) and spot-checked the loss trace byte-for-byte.

## 8. Environment (for the "how was this run" question)

Python 3.13.14, TensorFlow 2.21.0, NumPy 2.5.1, Keras 3.15.0, scikit-learn
1.9.0, pandas 3.0.3, matplotlib 3.11.1, pytest 9.1.1. **CPU-only throughout
— a GPU attempt (RTX 5050 via WSL2) measured 0.87ms/step vs. CPU's 0.70ms,
i.e. GPU 25% *slower* for a model this small; kernel-launch overhead
dominates at this scale.** (Source: `FedGRAA_STATUS_REPORT.md`'s "GPU
Report" table — this is the actual source of the "25% slower" figure
quoted elsewhere in this project's docs; it is a real local WSL2/RTX5050
benchmark, not the Modal cloud infrastructure doc.)
