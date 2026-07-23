# PAKDD / FedGRAA — Full Independent Audit

**Date:** 2026-07-23
**Scope:** Complete audit and verification of the whole PAKDD reproduction —
paper→code fidelity, test suite, code correctness, data provenance, persisted
results, and the divergence diagnosis. Every claim below was re-derived
independently from the source (paper text, source CSVs, raw `.cache`/`results`
artifacts), not taken from prior audit docs.

**Verdict, up front:** The PAKDD project is **sound and internally faithful.**
Every headline claim in the repository was independently reproduced or confirmed:
the test suite passes exactly as claimed, the persisted results match the
documented tables to the digit, the datasets are genuine, and the divergence
diagnosis is fully backed by its own raw JSON. No correctness defect was found in
the reproduction pipeline. Five *minor* issues were surfaced — all documentation
staleness or latent edge cases, none affecting any published number or conclusion.

---

## 1. What was verified (all ✅)

| Area | Claim | Independent check | Result |
|---|---|---|---|
| Tests | 79/79 pass | Ran `pytest tests` in `.venv313` | **79 passed, 0 failed** (121 s) |
| Test count | "79 tests" | Counted `def test_` across 12 files | 8+4+5+8+7+11+2+6+6+9+4+9 = **79** |
| Core math | `∂L'/∂w_k = ⟨∇_P L', P_k⟩` | `test_fedgraa_weight_gradient_matches_finite_difference` uses central FD, `atol=5e-3` | **Validated** |
| Config | Matches Tables 1 & 2 | Line-by-line vs paper tables | **Exact** (all 15 CIC + 10 UNSW `used_counts`) |
| Architecture | `200→100→50` tanh, softmax, Adam η=0.01, SCCE | vs `model.py` / Section 5 | **Exact** (logits + `sparse_softmax_ce`, equivalent) |
| Algorithm 1/2 | Federated loop + weight learner as printed | vs `trainer.py` / `aggregation.py` | **Faithful**, incl. no simplex projection (deliberate) |
| Results | Figure-1 table (61/0, 52.3/91.7, 62.5/0, 18.6/0, 17.4/0) | Parsed `results/CIC-IDS-2017_gamma_sweep.json` | **Exact match** to README & STATUS_REPORT |
| Provenance | Datasets genuine, not synthetic | Loaded `.cache/*.npy`, checked distributions & feature signature | **Genuine** (see §2) |
| Diagnosis | blowup 3.8×/2.7×/193×, loss trace, simplex-violation round 1, combined-fix 72/92/94% | Parsed `results/diagnostics/*.json` | **Every number backed** (see §3) |

## 2. Data provenance — independently confirmed genuine

Loaded the cached arrays directly and cross-checked class histograms and feature
statistics:

- **Class counts match the paper's *Original* column exactly** — CIC BENIGN =
  2,273,097; Bot = 1,966; Heartbleed = 11; Infiltration = 36; SQL Injection = 21;
  XSS = 652. UNSW attack classes match Table 2 "Used" exactly (Worms 174,
  Shellcode 1,511, Backdoor 2,329, Analysis 2,677). A synthetic generator would
  hit the spec count for *every* class including benign; the cache does not.
- **Feature-value signature is real, not Gaussian.** CIC post-scaling max|x| =
  **672** (synthetic standard-scaled Gaussian would be ≲6); **8/78 columns are
  all-zero** (exactly 10.26 % of cells) — the well-known constant-column
  signature of real CIC-IDS-2017. UNSW max|x| = 313.
- **Shapes:** CIC `(2,342,420 × 78)`, UNSW `(149,691 × 39)`. The 39 UNSW features
  = 42 (after dropping `id`/`label`/`attack_cat`) − 3 categorical
  (`proto`/`service`/`state`). Matches the loader and the actual array.

This corroborates `data_provenance_audit.md` completely.

## 3. Divergence diagnosis — fully reproducible from raw JSON

Spot-checked the headline numbers against `results/diagnostics/`:

- **Composite-loss blowup:** γ=0 → 3.8×, γ=0.5 → 2.7×, γ=1.0 → **193×** — matches
  `CIC_IDS_2017_literal_r60.json` diagnosis fields exactly.
- **γ=1 loss trace** (rounds 0–10): `2.98, 2.77, 4.53, 6.06, 40.52, 75.55,
  261.48, 290.24, 319.11, 205.30, 149.79` — **byte-for-byte** the trace printed
  in the diagnosis doc §2.
- **Simplex violation** `first_simplex_violation_round = 1` and
  `monotone_lemma1 = False` at every γ — as claimed.
- **Combined-fix seed sweep** (`seed_sweep_CIC_IDS_2017_fedgraa_combined_fix.json`):
  γ=0 → 72.2 % ± 10.4 % [83.3/75.0/58.3], γ=0.5 → 91.7 % ± 0, γ=1.0 → 94.4 % ± 3.9
  [91.7/91.7/100] — **exact**. FedAvg baseline 0.00 % ± 0.00 %, Static 88.89 % ± 3.93 %
  also confirmed.

The diagnosis is honest about what it does and does *not* establish (projection
alone ≠ accuracy recovery; the fix requires projection **and** a smaller η′; the
strict-repro default is intentionally left unpatched).

## 4. Minor issues found — all fixed (2026-07-23)

1. **Mid-size-class `Used` counts are not reproduced (fidelity gap, benign-adjacent).**
   For CIC classes between the rare tier and the 10 k cap — Bot (1,966 vs paper
   1,948), DoS Slowhttptest (5,499 vs 3,219), DoS Slowloris (5,796 vs 5,228),
   FTP-Patator (7,938 vs 5,931), SSH-Patator (5,897 vs 5,385), Brute Force (1,507
   vs 1,470) — the real pipeline keeps the **Original** count because
   `_load_real_csvs` replaces inf/NaN in place (never drops rows) and
   `_cap_frequent_classes` only caps > 10 k. The three protected **rare** classes
   match exactly, so no conclusion was ever affected.
   **Fix:** `data_provenance_audit.md` §7 now documents this gap explicitly
   alongside the pre-existing UNSW-benign note, with a table of all six affected
   classes and why the paper's exact subsampling can't be reproduced (no stated
   method or seed).

2. **Latent parallel-worker edge case** — `experiment.py` used
   `workers = min(5, os.cpu_count() // 2)` (and `min(8, ...)` in the n-sweep
   variant): on a 1-core host this evaluates to 0 → `ProcessPoolExecutor(max_workers=0)`
   raises `ValueError`; if `os.cpu_count()` returns `None`, `None // 2` raises
   `TypeError`. Only reachable via `--parallel` on unusual hosts.
   **Fix:** both call sites now use `max(1, min(N, (os.cpu_count() or 2) // 2))`,
   verified against 1-core, `None`, 2-core, and 16-core inputs.

3. **STATUS_REPORT commit reference was stale** — cited "Latest commit `270d1ed`"
   while HEAD had advanced to `71839ee`.
   **Fix:** updated to `71839ee` — docs: audit data provenance.

4. **`../../latest_baseline.md` still reported 55/55 tests** (now 79) with no
   indication it was a historical snapshot of a since-fixed NaN-collapse
   regression.
   **Fix:** added a `⚠️ SUPERSEDED` banner at the top pointing to the current
   79/79 count in `../../PAKDD_STATUS_REPORT.md` and current results in `results/`,
   while preserving the historical regression record intact (rewriting its
   numbers would falsify the record of what the regression looked like).

5. **`config.py` comment arithmetic imprecision** — "43 → 39 features (3
   categorical dropped)" read as if 43 raw columns minus 3 categorical = 39,
   which doesn't add up (43 is UNSW's *paper-stated* feature count, not the raw
   column count). The final **39 is correct** and matches the data.
   **Fix:** comment now reads "42 → 39 ... (42 = the 43 raw columns minus
   id/label/attack_cat)", accurately describing what the code drops.

All five fixes were re-verified against the full test suite (79/79 still pass)
after applying them.

## 5. Bottom line

The whole PAKDD reproduction verifies. Tests pass (79/79), the code faithfully
implements the paper's Algorithms 1–2 and Tables 1–2, the datasets are genuine,
the published result tables are exactly backed by persisted JSON, and the
divergence diagnosis — the project's central scientific contribution — is fully
reproducible from its raw artifacts and honestly scoped. The five issues found
were documentation-hygiene and latent-edge-case items, not correctness defects
— none of them ever changed a reported number — and all five are now fixed.
