# FedPGO faithful reimplementation and rerun — plan and decision record (2026-07-31)

> **Purpose.** The dated decision record and execution plan resolving the open
> decision left by
> [`STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md`](STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md)
> §6: the 20 committed FedPGO cells are a disclosed **proxy**, not a faithful
> reproduction of Bikila & Čapek (*Computer Networks* 280:112210, 2026), whose
> full primary text is now local at
> [`FedPGO/FedPGO.md`](FedPGO/FedPGO.md). This document commits to that update's
> **Path 1 (faithful-baseline)**: reimplement the mechanism from the primary
> source (Eqs. 6–10; Algorithms 1–3), rewrite the tests, rerun the 20 cells on
> Modal, rebuild the frontier artifacts, and land the dated corrections in the
> live documents. It does **not** supersede
> [`Final_Stage_C.md`](Final_Stage_C.md) or
> [`REMAINING_COMPUTE_WORK.md`](REMAINING_COMPUTE_WORK.md); it records the
> decision those documents have been waiting on, and it is written before any
> implementation so the mapping below can be executed against, and audited
> against, a fixed spec.

> **✅ Audited and corrected 2026-07-31 (same day, before any execution).** An
> adversarial audit pass re-verified every claim in this document against the
> repository source and the primary text. Three corrections were made in
> place: **(1) JC2 was wrong** — per-client validation splits already exist in
> the harness (`ClientShard.X_val/y_val`, carved deterministically at
> partition time by `_train_val_split` in `partition_non_iid`
> [`fedgraa/partition.py:164`](../../fedgraa/partition.py) and
> `partition_by_silo` at `partition.py:367`), so no trainer-internal carve is
> needed and there is no data-efficiency deviation against other baselines;
> **(2) JC1's "(90%-train portion)" qualifier removed** — the faithful trainer
> simply trains on `client.X_train`, the same shard every baseline trains on;
> **(3) the `--arm` filter design moved into `enumerate_cells()` as an
> optional parameter** with tests in
> [`tests/test_run_v2_frontier_sweep.py`](../../tests/test_run_v2_frontier_sweep.py),
> because `tests/test_modal_frontier_campaign.py` skips without the `modal`
> package installed (a CLI test there would silently never run in the project
> venv). Everything else survived the audit unchanged; the corrected rows are
> marked ⚠️ below.
>
> **✅ Second audit 2026-07-31 (same day, before any execution).** A repo-wide
> re-scan for surviving "FedPGO full text inaccessible/unobtainable/blocked"
> statements mapped **every** remaining site to §8, which is expanded from
> five to seven items. Sites newly covered: the driver comment block
> `scripts/run_v2_frontier_sweep.py:138–151`; `Final_Stage_C.md` §11
> bibliography entry (line 725), §13.4 item 1 (line 882) and the Verdict
> section (lines 913–915) — §8 item 3 now enumerates all seven FedPGO sites
> in that document; and `STAGE_C_LITERATURE_MASTER.md` §4.4 FedPGO row
> (line 207). Dated pass logs predating the full text
> (`STAGE_C_DEEP_RESEARCH_2026-07-25.md` §14, `STAGE_C_FASTTRACK_FINAL_PASS_2026-07-26.md`
> §§2/7, `STAGE_C_LITERATURE_MASTER.md` §4.12 note, `Final_Stage_C.md` §13.3
> table) were checked and deliberately left untouched: they record the access
> state at the time, and this series' convention (per `Final_Stage_C.md`
> §13.5) is dated corrections retained in place, not silent rewrites of dated
> records. Everything else survived the second audit unchanged.
>
> **✅ Third (final) audit 2026-07-31, same day, before any execution.** Full
> end-to-end re-verification: every anchor in §2–§7 re-read against the
> primary text and the live scripts; the full test suite re-run (**441
> passed**); the completeness gate re-run (**730/730 cells**, all 19 arms).
> Three primary-text anchors corrected: Alg. 3's "Update gradient memory" is
> step **23** (transcribed line 964), not 24 — §2 row 1 and §3 step 9;
> JC7's quoted phrase is Alg. 3 step 12 of §3.9 (line 942), with §3.2
> stating the same phase — not §3.2 alone. The plan doc's own
> `../../fedgraa/partition.py:164` link was the gate's **single doc-link
> failure** (`check_doc_links.py` resolves paths only; inline `file:line`
> anchors survive as code spans, not link targets) — fixed here. §6.2 now
> also covers the two post-rerun count references: the renderer's "13
> external-baseline points" docstring/comment and `run_v2_frontier_sweep.py:588`'s
> "13 of its arms" both become 14 once FedPGO plots
> (`BASELINE_ARM_CLASSES` = 6 T3 + 8 T5).

## 1. Decisions taken (author-approved 2026-07-31)

| # | Decision | Choice |
|---|---|---|
| D1 | Path 1 vs Path 2 (07-30 update §6) | **Path 1 — faithful reimplementation + rerun.** Proxy remains only as historical record (git history + this plan + archived JSONs) |
| D2 | Disposal of the 20 proxy cell JSONs | **Archive** into `results/v2_frontier/_archive_fedpgo_proxy_2026-07-30/` with a dated README, not delete. Verified collision-free: `build_v2_frontier_table.py`'s glob is non-recursive (`scripts/build_v2_frontier_table.py:94`) and `check_stage_c_complete.py` verifies exact `cell.path` (`scripts/check_stage_c_complete.py:115`), so archived files can never contaminate seed counts |
| D3 | Plot FedPGO in the frontier figure | **Yes** — add to `render_v2_frontier.py` `BASELINE_LABELS` + a 14th marker. Exclusion was a proxy artifact; FedPGO is the closest competitor |

## 2. Why the current cells are a proxy (recap, primary-source-anchored)

The committed implementation (`fedgraa/baselines_t5_fedpgo.py`, committed at
`c5c8ac8`/`30fd064`) diverges from the primary text in three named ways:

| Paper element (primary source) | Committed proxy behavior | Location |
|---|---|---|
| `g_G` = **previous round's** global gradient; gradient memory updated `g_G ← g'_G` each round, initialized to `0` (Alg. 3 lines 23, 3; §3.5, Eq. 6 text) | `tau_g` recomputed each round as the **same-round** FedAvg-weighted mean of local deltas | `baselines_t5_fedpgo.py:303–312` |
| Recompose `g'_i = α_i·g∥ + (1−α_i)·g⊥`, `α_i ∈ [0,1]` (Eq. 8, Alg. 1 step 6) | `parallel + orthogonal_weight·orthogonal` | `baselines_t5_fedpgo.py:148–171` |
| PGF score `S_i = λ1·cos(g_i,g_G) + λ2·‖g⊥‖ + λ3·ΔL_val` (Eq. 9; Alg. 3 lines 11–16) | ReLU-clipped cosine + orthogonal *fraction*; no validation-loss-improvement term | `baselines_t5_fedpgo.py:174–211` |

Also outdated: the module docstring's "full text is genuinely inaccessible"
framing is now false (full text local since 2026-07-30) and must be replaced.

## 3. Faithful mechanism spec (the mapping the implementation must match)

All vectors live in **delta space**: `g_i` is approximated by the per-client
per-round shared-parameter delta `tau_k = updated_shared − shared_global`
(repo-wide convention; the paper's own Alg. 3 is internally ambiguous — line 10
trains locally before line 15 computes `∇L_i(w_t)` — and the paper trains with
Adam, so its "gradient" is not recoverable literally without a second backward
pass; disclosed judgment call JC1).

Per round `t` (client side), per client `i`:

1. Train `E` local epochs (existing `client_train_round`) on the client's
   own `client.X_train` — the harness already carves a deterministic 90/10
   train/val split per client at partition time (`partition.py:164`), so this
   is the same `X_train` every other baseline trains on — starting from
   `shared_global + personalized[i]`.
2. `g_i = tau_k = updated_shared − shared_global`.
3. **AGO** (Eqs. 6–7): with `c_i = (g_i·g_G)/(‖g_G‖² + ε)`:
   `g∥ = c_i·g_G`, `g⊥ = g_i − g∥`.
4. **Recompose** (Eq. 8): `g'_i = α_i·g∥ + (1 − α_i)·g⊥`.
5. **Personalization / ΔL_val** (Alg. 3 lines 11–14): before/after the
   head-only fine-tune on the client's existing per-client validation split
   `client.X_val/y_val` (the 10% slice carved at partition time):
   `ΔL_val = L_before − L_after`.
6. **PGF score** (Eq. 9): `S_i = λ1·cos(g_i, g_G) + λ2·‖g⊥‖ + λ3·ΔL_val`,
   where `cos(g_i, g_G) = (g_i·g_G)/(‖g_i‖·‖g_G‖)`; at round 1 (`g_G = 0`) the
   cosine term is defined as `0` (JC4).

Server side:

7. **Weights** (Eq. 10 / Alg. 2): `w_i = S_i/(Σ_j S_j + ε)`, with scores
   clipped at `0` before summing (JC3); uniform fallback if every score ≈ 0
   (existing degenerate-round repo convention).
8. **Aggregate** (Alg. 2 steps 7–8): `g'_G = Σ_i w_i·g'_i`;
   `shared_global ← shared_global + server_lr·g'_G`.
9. **Gradient memory** (Alg. 3 line 23): `g_G ← g'_G` (persisted across rounds;
   initialized to zero vector).

## 4. Judgment calls (each disclosed in the rewritten module docstring)

| ID | Question | Choice | Rationale |
|---|---|---|---|
| JC1 ⚠️ | What is `g_i`? | Per-round shared delta `tau_k = updated_shared − shared_global` (over `client.X_train`), not a raw gradient pass | Repo-wide delta convention; paper ambiguous; "no additional communication overhead" claim in §1 |
| JC2 ⚠️ | Where does per-client validation data come from? | The harness's existing per-client splits — `ClientShard.X_val/y_val`, carved deterministically at partition time by `_train_val_split` in both `partition_non_iid` (`partition.py:164`) and `partition_by_silo` (`partition.py:367`), pooled into the reporting split `FederatedSplit.X_val/y_val` (`partition.py:180–181`) | No carve needed — this is the 60/20/20 train/val analogue already built into the harness (90/10 per client, `config.py:57`); every baseline trains on the same 90% `X_train`, so zero data-efficiency deviation; the head fine-tune uses only the client's own 10% slice; reporting stays on the pooled split (the harness's standing convention — v1/v2 also optimize on D_V and report on it) |
| JC3 | Negative PGF scores? | Clip at 0 before normalization | Paper asserts `w_i ∈ [0,1]` (§3.7), which requires `S_i ≥ 0`; literal formula would produce negative weights, a contradiction with the paper's own stated invariant |
| JC4 | Round-1 degeneracy (`g_G = 0`) | `c_i = 0`, `g∥ = 0`, `g⊥ = g_i`, `g'_i = (1−α_i)·g_i`; cosine term `0` | Direct consequence of Eq. 6 with `g_G = 0`; ε protects the denominator |
| JC5 | `α_i` values | Constructor scalar `alpha` default `0.5` (optional per-client array), validated `∈ [0,1]` | Paper: "client-specific… may be adapted based on data diversity or gradient alignment" — no formula given; tuned there by random search |
| JC6 | `λ1, λ2, λ3` | Constructor defaults all `1.0` | Paper: "tuned via validation performance" — no values given |
| JC7 | Head fine-tune size | `personalize_steps` default `5`, lr `= cfg.eta`, head (personalized) parameters only, on `client.X_val/y_val` | Paper: "Fine-tune personalized head using validation data" (Alg. 3 step 12, §3.9; §3.2 states the same phase) — count unspecified; kept cheap (per client per round) |
| JC8 | Delta-space sign convention | Server update `+ server_lr·g'_G` with `server_lr = 1.0` (current behavior unchanged) | Consistent with the existing proxy and all T3/T5 baselines; the paper's `w_{t+1} = w_t − η·g'_G` maps onto it with the delta sign convention |

Constructor surface (all disclosed defaults, none claimed as paper values):
`alpha=0.5`, `lambda1=lambda2=lambda3=1.0`, `personalize_steps=5`,
`personalized_layers=1`, `server_lr=1.0`, `eps=1e-12`.

## 5. Test spec — rewrite `tests/test_baselines_t5_fedpgo.py` (~20 tests)

The 12 existing tests are proxy-semantics-bound (signatures change) and are
replaced, not preserved. New coverage:

1. AGO exact decomposition matches Eq. 6/7 given a stored `g_G` (incl. ε path).
2. Round-1 degeneracy: `g_G=0` → `g' = (1−α)·g_i`, finite, cosine term 0.
3. **Gradient-memory regression**: `g_G` used in round `t+1` equals round `t`'s
   aggregated `g'` — and is *not* the same-round FedAvg mean (the proxy bug).
4. Recompose matches Eq. 8 at `α ∈ {0, 0.5, 1}`.
5. PGF score: each of the three Eq. 9 terms isolatable via λ toggles
   (cos sign; `‖g⊥‖` magnitude; `ΔL_val`).
6. `ΔL_val` property: `L_after ≤ L_before` after head fine-tune; with
   `λ1=λ2=0`, a high-validation-gain client out-scores a low-gain one.
7. Weights match Eq. 10: non-negative (clip path), sum ≈ 1 (ε-tolerant),
   uniform fallback on degenerate rounds.
8. Anti-aligned client down-weighted vs aligned (cos term, clip path).
9. Personalized heads differ across clients; shared layers aggregated
   (existing parameter-split guarantees preserved).
10. End-to-end: no NaN collapse, valid simplex every recorded round,
    deterministic per seed, accuracies ∈ [0,1].
11. Validation: `alpha` outside `[0,1]` raises; `personalized_layers`
    out-of-range raises.

## 6. Tooling

1. **`--arm` filter** — an optional `arms: Sequence[str] = ()` parameter on
   `enumerate_cells()` (`scripts/run_v2_frontier_sweep.py`), filtering the
   enumerated cells uniformly across tiers (empty = all arms of the tier;
   zero behavior change for existing callers such as
   `check_stage_c_complete.py`), plus a comma-separated `--arm` flag on
   `scripts/modal_frontier_campaign.py` threading it through. Used with
   `--force` (the volume already holds 20 proxy cells that resumability would
   otherwise skip). Tests live in `tests/test_run_v2_frontier_sweep.py` —
   deliberately **not** `tests/test_modal_frontier_campaign.py`, whose
   `pytest.importorskip("modal")` means its tests silently skip in the project
   venv.
2. **Figure**: add `"fedpgo": "FedPGO"` to `BASELINE_LABELS` and a 14th entry
   to `BASELINE_MARKERS` in `scripts/render_v2_frontier.py`; update the two
   "13 external-baseline points" / "13 unfilled points" references (module
   docstring lines 10–16, comment line 68); update
   `run_v2_frontier_sweep.py:588`'s "13 of its arms" → 14
   (`BASELINE_ARM_CLASSES` = 6 T3 + 8 T5 = 14 arms); regenerate.

## 7. Compute rerun (Modal, 20 cells ≈ 5.3 core-h, <$1, ≈ 5–10 min wall clock)

Order matters — containers run the committed repo state and record the git
commit per cell (modal_frontier_campaign.py:506), and the pull must not
overwrite the archived proxies:

1. Commit W1–W3 (implementation, tests, tooling).
2. `modal run scripts/modal_frontier_campaign.py --verify-only`
3. `modal run scripts/modal_frontier_campaign.py --tier T5 --arm fedpgo --smoke`
4. `modal run --detach scripts/modal_frontier_campaign.py --tier T5 --arm fedpgo --force`
5. **Archive first, then pull**: move the 20 proxy JSONs to
   `results/v2_frontier/_archive_fedpgo_proxy_2026-07-30/` + dated README;
   then `modal volume get --force fedgraa-v2-results v2_frontier results/`
6. `python scripts/build_v2_frontier_table.py`
7. `python scripts/render_v2_frontier.py` (figure now gains the FedPGO point)
8. `python scripts/check_stage_c_complete.py --tier T1 T2 T3 T4 T5 T6 --t4-m 3`
   → expect `730/730` again.

## 8. Documentation (dated corrections, retained in place)

1. **This document** — the plan/decision record itself.
2. `STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md` — dated resolution note
   (Path 1 taken, outcome); **correct §4's factual error**: the fedpgo `_r2`
   smoke artefact *is* tracked (committed in `9d20ac9`, alongside `FedPGO.md`;
   other arms' `_r2` files are tracked too).
3. `Final_Stage_C.md` — dated correction (full text local at
   `FedPGO/FedPGO.md` since 2026-07-30), covering **every** FedPGO
   accessibility statement in the document, in place: §4.4 baseline-table row
   (line 222); §9.0 item (a) table (line 580) — the ILL/author-contact
   escalation is retired with the resolution note; §9.1 item 0 (line 592) —
   related-work guidance upgraded from "abstract-level confidence / assert no
   FedPGO limitation" to primary-source confidence; §11 bibliography entry
   (line 725); §13.4 item 1 (line 882); Verdict section (lines 913–915) —
   "the one named exception" no longer applies; the differentiation paragraph
   is now fully writable from the primary text (personalization objective,
   parameter-split, no γ knob, accuracy/robustness evaluation axis,
   CICIoT2023 + CICIDS2017 only). The Verdict's historical saturation
   narrative (line 905) stays untouched.
4. `REMAINING_COMPUTE_WORK.md` — dated correction: §4's "none of their code
   exists anywhere" (line 179) → 13 of 14 remain (FedPGO now implemented,
   cells complete); §3.1 grid T5 row (line 111) — "baseline code does not
   exist" no longer true for the fedpgo arm; §4's recommended order (line
   183) → CELM is now the closest unimplemented baseline (FedPGO was the T5
   closest-per-family pick).
5. `FEDGRAA_V2_MASTER_PLAN.md` — dated note at §10.3 do-not-write item 4
   (line 725): full text now local at `FedPGO/FedPGO.md`; the "abstract-level
   confidence only" constraint is lifted. The item's shape survives: any
   FedPGO limitation sentence must trace to a citeable passage in the local
   primary text, never to secondary summaries.
6. `scripts/run_v2_frontier_sweep.py` — replace the module-top FedPGO comment
   block's "full text remains genuinely inaccessible across six independent
   channels" (lines 138–151) with a pointer to the local primary text
   (`docs/research_c/FedPGO/FedPGO.md`) while keeping the DBLP
   `journals/cn/BikilaC26` / OpenAlex `W7134986038` / Semantic Scholar
   `CorpusId:286471251` triple confirmation; the "disclosed RECONSTRUCTION"
   label becomes a faithful primary-source implementation note. Landed in the
   same commit as W1's `enumerate_cells` arms-parameter edit (same file).
7. `STAGE_C_LITERATURE_MASTER.md` — dated correction on the §4.4 FedPGO row
   (line 207): full text read locally 2026-07-30; mechanism description now
   at primary-source confidence; "must be read in full before Related Work is
   drafted (§9 item 0)" discharged. The §4.12 honest-limitation note (line
   784) is that pass's dated record and stays as-is.

## 9. Verification

- `.venv313/Scripts/python.exe -m pytest tests/ -q` → expect ≈ 450+ green
  (441 baseline + new fedpgo tests + `enumerate_cells` arms-filter tests;
  only `test_baselines_t5_fedpgo.py` and `test_run_v2_frontier_sweep.py`
  change).
- `check_stage_c_complete.py` full pass → 730/730 with the faithful cells.
- Commits: (a) implementation + tests + tooling; (b) results (20 faithful
  JSONs + archive dir + rebuilt table/figure); (c) docs corrections + this
  plan.

## 10. Honest verification bar

**Mechanism-fidelity to Eqs. 6–10 is fully unit-testable** (this plan's test
spec). **Paper-number matching is not possible**: the paper reports
instance accuracy / macro-F1 on CICIoT2023 + CICIDS2017, 10 clients, Conv1D,
Dirichlet(0.3), 60/20/20; the harness runs CIC-IDS-2017 (K=14) and UNSW-NB15
(K=9), MLP, γ=1.0, per-minority-class recall. The paper's *qualitative* claim
(FedPGO ≥ FedAvg under non-IID) remains observable in the rebuilt frontier
table. Both statements are recorded here so no later pass mistakes the
mechanism bar for an experiment bar.

## 11. Final status (2026-07-31) — execution record

All steps executed. Deviations from the plan, each recorded with its cause:

- **§7.2/§7.4 invocation forms.** `--detach` is a `modal run`-level flag and
  must precede the module path: the working form is
  `modal run --detach -m scripts.modal_frontier_campaign --tier T5 --arm fedpgo --force`.
- **First detached run cut short.** The shell's 30-minute timeout killed the
  client mid-campaign; the cancellation propagated to the remote app
  (`ap-JfPkm1bI5CXGpAaECfEmxu`) and aborted the UNSW-NB15 batch after the
  CIC-IDS-2017 batch had finished (10/20 cells). A `--force` rerun
  (`ap-1AcE5ndZ9RzaCB844r4nCY`) completed all 20 cells; every canonical cell
  is stamped `25611ce` with `git_dirty: true` (the then-untracked plan
  document).
- **§7.5 pull.** `modal volume get --force fedgraa-v2-results v2_frontier
  results/` (the plan's exact command) hangs on this Modal client (1.5.3) for
  directory sources and truncates local files (4 cells were zeroed) before
  stalling; `modal volume ls --json` size fields are display-rounded and
  cannot be used for exact comparison. Workaround: per-file
  `modal volume get --force fedgraa-v2-results v2_frontier/<f> results/v2_frontier/`
  for the 22 fedpgo files plus the 4 zeroed cells; the archive-first
  discipline held (archive completed before any pull).
- **§8.2 factual scope.** Only the UNSW-NB15 `_r2` smoke artefact was tracked
  in `9d20ac9`; the CIC-IDS-2017 `_r2` was untracked. Both proxy-era `_r2`
  files were archived; the faithful rerun's CIC-IDS-2017 `_r2` smoke artefact
  is committed at the canonical location.
- **Verification (final):** full suite 453 passed; gate 730/730; doc-link
  checker green. Commits: (a) `25611ce` implementation + tests + tooling;
  (b) `96adb05` results (20 faithful JSONs + archive dir + rebuilt
  table/figure); (c) this commit — docs corrections + this plan.

**Honest bar (reaffirmed):** mechanism fidelity to Eqs. 6–10 is unit-tested
(19 tests); paper-number matching remains impossible (paper: CICIoT2023 +
CICIDS2017, 10 clients, Conv1D, Dirichlet(0.3), 60/20/20 vs harness:
CIC-IDS-2017 K=14 / UNSW-NB15 K=9, MLP, γ=1.0); the paper's qualitative
FedPGO ≥ FedAvg claim is observable in the rebuilt frontier table.
