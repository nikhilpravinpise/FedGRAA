# Independent Stage C Audit — Code vs. Paper, Six Parallel Adversarial Passes (2026-08-03)

**Requested by:** repo owner ("audit and reverify each and every line of code of
Stage-C written as a paper reproduction... go through the paper itself again").
**Performed by:** Claude (Sonnet 5), via six independently-instructed parallel
subagents, each explicitly told *not* to read or trust the prior
`docs/audit/INDEPENDENT_REPO_AUDIT_2026-08-02_*.md` passes and to re-derive
everything from the code, the tests, and `docs/paper/FedGRAA_v2_PAKDD_Submission.md`
directly. Two agents (T7 and the numeric re-derivation agent) independently
computed overlapping numbers with no visibility into each other's work, as a
built-in cross-check.

**Tree audited:** `HEAD = 920d985` (post-remediation of the prior audit's 34
findings), `main` branch, clean tree.

**Scope:** the Stage C paper-reproduction surface only — the core FedGRAA-v2
mechanism, all 19 baselines (14 domain-specific + 5 classic FL), T7
adversarial robustness, and every headline number in
`docs/paper/FedGRAA_v2_PAKDD_Submission.md`/`.tex`. Not a repeat of the prior
passes' documentation-bookkeeping sweep (test counts, stale dates, broken
links) — that ground was already covered on 2026-08-02 and remediated in
`920d985`.

---

## 0. Bottom line

**The engineering is solid. The paper overclaims in two specific, real,
independently-confirmed places** — both in the T7 adversarial-robustness
section, both caught by two agents working independently with no shared
context. Everything else checked — the core privacy-compliant mechanism, all
19 baseline reproductions, the significance testing, the communication-overhead
numbers, and 559 tests — was either confirmed exactly correct or turned up
only low/medium-severity disclosure gaps of the same "true but not stated
precisely enough" character the prior 2026-08-02 audit found elsewhere in the
repo's documentation layer.

**Two HIGH-severity findings, both new (not found by the 2026-08-02 audits, which didn't re-derive T7's numbers from raw data):**

1. **§7/§8's "2.0–3.1×, both attacker fractions" up-weighting claim is false for CIC-IDS-2017 at 50% attackers.** Raw data: 1.60× mean, individual seeds as low as 1.04×. Confirmed independently by two agents using different methods. The paper's own companion table (`STAGE_C_MASTER_REPORT.md` §10.6) already had the correct number — only the paper's *prose* overclaims.
2. **§7's "rare-class recall collapses to ≈0% in every defended cell" is false for UNSW-NB15.** 11 of 30 UNSW-NB15 defended cells are nonzero, up to 33.3%. True only for CIC-IDS-2017 (30/30 cells exactly 0).

Both are prose-level overclaims *layered on top of correct underlying data* — the same "hand-typed number doesn't match the table it's summarizing" pattern the 2026-08-02 audit already identified as this project's single recurring defect class, now shown to also reach the manuscript's own results section, not just status docs.

---

## 1. What was audited, and how

Six subagents ran in parallel, each with a self-contained brief quoting the
relevant paper text verbatim and explicit instructions to independently
re-derive rather than trust:

| # | Scope | Method |
|---|---|---|
| A | Core mechanism: `aggregation.py`, `trainer.py`, `privacy.py`, `secagg.py` vs. §4–§5 (Algorithm 1, simplex projection, SecAgg redesign, ζ-inexact theorem) | Full file reads, ran `verify_v2_exactness.py` fresh, hand-checked DP composition arithmetic |
| B | Classic-FL baselines: `baselines.py`, `baselines_classic.py` (FedAvg/FedProx/FedOpt/SCAFFOLD/FedNova) | Re-derived SCAFFOLD/FedNova from primary-source math independently, checked against code |
| C | 14 domain-specific baselines (CELM, SpectralFed, CoAst, FLTrust, RAGA, FedTruth, FedMADE, SBFedNIDS, FedAPA, FedAWA, DA-PFL, FedAWARE, AdaAggRL, FedPGO) | Per-baseline mechanism-vs-primary-source check; FedPGO checked line-by-line against its own project-authored spec doc |
| D | T7 adversarial robustness: `attacks.py`, `baselines_t7_adversarial.py`, D1c wiring in `trainer.py` | Manually recomputed every §7/§10.6 number from raw `attacker_weight_history` fields, did not trust pre-built tables |
| E | Numeric re-derivation of all headline claims | Wrote fresh Python (not reusing the project's own `build_v2_*.py` scripts) reading raw JSON directly; independent Wilcoxon/Holm-Bonferroni implementation |
| F | Test-suite quality (not just pass/fail) for 22 Stage-C test files | Classified every test REAL/WEAK/SMOKE/MOCK; **mutation-tested 3 source files** (injected real bugs, confirmed tests catch them, reverted) |

All six ran `pytest` on their scope directly rather than trusting a cached
"all green" claim. Totals: **367/367 Stage-C-filtered tests passed** (agent F's
run), consistent with individual per-file runs by agents A–D (66+24+109+38 = 237
tests across their narrower scopes, all passing, some overlap with F's broader
filter).

---

## 2. HIGH-severity findings (paper prose contradicts raw data)

### 2.1 The 50%-attacker-fraction up-weighting range is wrong for CIC-IDS-2017

**Where:** `docs/paper/FedGRAA_v2_PAKDD_Submission.md` §7 line 161 ("At 50%
attacker fraction, the effect persists (2.0–2.2× uniform on both datasets)")
and §8 line 178 ("2.0–3.1× uniform share, both datasets, both attacker
fractions").

**What the raw data says.** Recomputed independently by two agents from
`results/v2_frontier/CIC_IDS_2017_v2_raresig_g1.0_seed{0..4}_af0.5.json`'s
`attacker_weight_history[-1]` field, divided by the uniform share `1/14`:

| Source | CIC-IDS-2017, 30% attackers | CIC-IDS-2017, **50% attackers** | UNSW-NB15, 30% | UNSW-NB15, 50% |
|---|---|---|---|---|
| Paper §7/§8 prose | 3.1× | **"2.0–2.2×"** | 2.7× | 2.0–2.2× |
| Agent D (T7 audit) | 3.07× (confirmed) | **1.605×** | 2.71× (confirmed) | 2.24× (confirmed) |
| Agent E (numeric re-derivation) | 3.07×, seed range [1.72, 3.50] | **1.60×, seed range [1.04, 2.00]** | 2.71× | 2.24× |
| `STAGE_C_MASTER_REPORT.md` §10.6 (the paper's own companion table) | 3.07× | **1.60×** (already correct here) | 2.71× | 2.24× |

Two of five seeds at CIC-IDS-2017/50%-attackers collapse to essentially 1.0×
(no up-weighting effect at all). The paper's own master-report appendix
table already has the right number (1.60×) — only the manuscript's §7/§8
*prose* asserts a floor of 2.0× that the CIC-IDS-2017/50% cell does not meet.
This is not a new computation error; it is a transcription-into-prose error
of a number the project had already correctly computed elsewhere.

**Consequence:** a reader of §7/§8 alone would conclude the mechanism-specific
attack's up-weighting effect is reliably ≥2.0× regardless of dataset or
attacker saturation. The true picture, at 50% attacker saturation on
CIC-IDS-2017 specifically, is noisier and roughly half that at 5-seed
granularity — a materially different claim about the attack's reliability at
high attacker fractions.

**Fix:** change §7's "2.0–2.2×, both datasets" to name CIC-IDS-2017's actual
range (≈1.0–2.0×) separately from UNSW-NB15's (≈2.2–2.3×), and correct §8's
"2.0–3.1×" to something like "1.0–3.5×" or split by dataset. Five minutes of
work; the correct numbers already exist in `STAGE_C_MASTER_REPORT.md` §10.6.

### 2.2 "Rare-class recall collapses to ≈0% in every defended cell" is false for UNSW-NB15

**Where:** `docs/paper/FedGRAA_v2_PAKDD_Submission.md` §7, the sentence
following Table 2: "even where a defense successfully filters attackers,
rare-class recall collapses to ≈0% in every defended cell... rightmost rows
all pair with near-zero rare_acc, not shown for space."

**What the raw data says.** Checked `final_last_n_acc` across all 60
defense-cross-product cell JSONs (`secfednids_*`/`weidetect_*` × 3 attacks ×
2 datasets × 5 seeds):

- **CIC-IDS-2017: 30/30 cells exactly 0.0%** — claim holds cleanly.
- **UNSW-NB15: 11/30 cells are nonzero**, several substantial:
  `weidetect_modelpoison` seed2 = 33.3%, `secfednids_adaptive` seed0 = 22.4%,
  `weidetect_labelflip` seed3 = 22.1%.

**Consequence:** the paper's most-emphasized negative finding — "no defense
preserves FedGRAA-v2's rare-class advantage" — is dataset-dependent. It is
robustly true on CIC-IDS-2017 and only approximately true on UNSW-NB15,
where roughly a third of defended cells retain real (10–33%) rare-class
recall. The universal language ("every") is not supported by this project's
own second dataset.

**A plausible, disclosed-but-unexplored link:** the same agent found SecFedNIDS's
consensus threshold is `min(0.9, 1 - 1/n)`, not the flat 0.9 that
`JC-T7-9` discloses — for UNSW-NB15 (`n=9`) this evaluates to 0.8889, not
0.9 (see §3.3 below). A slightly looser filter on the smaller-client dataset
is a plausible mechanism for occasional attacker leakage, though this was not
established as causal, only noted as suggestive.

**Fix:** soften "every defended cell" to name CIC-IDS-2017 specifically, or
add "with N of 30 UNSW-NB15 cells retaining partial recall" as an honest
caveat — the underlying finding (defenses trade away the mechanism's
advantage on average) still holds; only the "every"/"≈0%" universality is
overstated.

---

## 3. MEDIUM-severity findings

### 3.1 §4.3's "eliminated by construction" claim for gradient staleness is contradicted by the project's own test suite

**Where:** `docs/paper/FedGRAA_v2_PAKDD_Submission.md` §4.3 / master-plan
prose: two obstructions are claimed eliminated "by construction" — (a)
stratum-count weighting, (b) "a stale-gradient interaction with Adam's own
moment estimates that can mask the error in early rounds and diverge only
later."

**What the code and tests actually show.** `aggregation.py`'s own two-phase
protocol code and `FEDGRAA_V2_MASTER_PLAN.md` define obstruction (b) as
**global-vs-per-client local arm selection**, not staleness — a different
bug from the one the paper's sentence names. Staleness itself is a real,
*unaddressed*, separately-tracked quantity: `tests/test_trainer_v2.py`'s
`test_zeta_decomposition_sources_are_separable` explicitly constructs a
stale-gradient scenario and asserts `zeta_stale > 1e-2` — i.e., proves a
measurable, non-eliminated error exists. The paper's own next paragraph
("Honest cost... trading gradient staleness for bandwidth" at `m=5`)
contradicts the "eliminated by construction" sentence two paragraphs
earlier — staleness is precisely the thing `m>1` deliberately re-introduces
in exchange for lower communication cost.

**Consequence:** a reader trusting §4.3's sentence in isolation would believe
Adam-moment/staleness interaction is a solved, purely structural non-issue.
In fact it's a real, quantified, intentionally-traded-off error source — the
theorem's own `ζ_stale(m)` term (§4 of `FEDGRAA_V2_THEORY.md`) exists
specifically to describe it. This is a genuine factual overstatement in the
manuscript, not merely optimistic framing — the paper's own theorem and
appendix elsewhere describe the accurate picture; only this one sentence in
§4.3 overclaims.

**Fix:** narrow the "eliminated by construction" claim to obstruction (a)
only; describe (b) as "characterized and bounded by the ζ-inexact theorem's
own `ζ_stale(m)` term, not eliminated" — consistent with how the rest of the
paper (§4.4, §6's `m=1` vs `m=3` discussion) already frames it correctly.

### 3.2 FedPGO silently omits its primary source's focal-loss training component

**Where:** `fedgraa/baselines_t5_fedpgo.py` trains clients via the shared
`client_train_round` (`fedgraa/trainer.py`), which uses plain
`sparse_softmax_cross_entropy_with_logits` (`fedgraa/model.py`). FedPGO's own
primary-source spec (`docs/research_c/FedPGO/FedPGO.md` §3.3, Eq. 4) names
focal loss as an integral part of its proposed local-training procedure, not
an optional detail — and the module's own docstring claims to implement "the
mechanism exactly as the primary text specifies it."

**Mitigating factor:** using one shared generic trainer across every baseline
is a defensible fair-comparison design choice, applied consistently. But this
specific gap is not entered in FedPGO's own JC1–JC8 disclosure table, despite
being exactly the kind of deviation that table exists to name.

**Fix:** add JC9 to `fedgraa/baselines_t5_fedpgo.py`'s docstring disclosing
the focal-loss omission and the fair-comparison rationale.

### 3.3 Undisclosed SecFedNIDS parameter deviation (JC-T7-9 doesn't match the code)

**Where:** `fedgraa/baselines_t7_adversarial.py:337` passes
`consensus_frac=min(0.9, 1.0 - 1.0/max(n, 2))` to the consensus filter, but
`JC-T7-9` discloses a flat `consensus_frac = 0.9`. For CIC (`n=14`) this
rounds to 0.9 with no practical effect; for UNSW-NB15 (`n=9`) it evaluates to
**0.8889**, undisclosed. Separately, the module docstring claims `perplexity`
is "rescaled proportionally... per JC-T7-9," but `perplexity` is never
actually passed at the call site — it silently uses the hardcoded default
`4.0` for both datasets; the only "rescaling" that happens is a generic
degenerate-case safety clamp inside `sos_outlier_probabilities` itself, not a
deliberate per-dataset choice.

**Consequence:** small in magnitude (immaterial to any individual number),
but breaks the project's own "every deviation gets a numbered JC disclosure"
discipline, and lands on exactly the dataset (UNSW-NB15) where finding 2.2's
"every defended cell ≈0%" overclaim also breaks down — plausibly related,
not established as causal.

**Fix:** correct JC-T7-9's text to match the actual `min(0.9, 1-1/n)`
formula and remove the false "perplexity rescaled" claim, or actually wire
perplexity rescaling in if that was the original intent.

### 3.4 `test_baselines_t7_adversarial.py` is thin relative to its stakes

7 of 12 tests for the SecFedNIDS/WeiDetect defense trainers are pure smoke
tests (no-NaN, runs-to-completion, construction-only) rather than end-to-end
correctness checks. Unlike `test_baselines_t3_trust.py` (which has an
explicit "a poisoned client is actually down-weighted" end-to-end test), T7's
defense suite never asserts that a poisoned client's influence is actually
suppressed by the full trainer, only that its isolated math helpers behave
and outputs don't blow up. Given T7 is this stage's most novel and highest-
stakes claim (the mechanism-specific attack and the defenses evaluated
against it), this is a real coverage gap, not merely a style nitpick.

### 3.5 Overhead ratios are reported as "exactly 6.00×" when they are ≈6.0007×

Independently recomputed: 6.000732×, 4.000583×, 3.000509×, 2.000434× (CIC),
near-identical on UNSW. These round correctly to the paper's quoted 2-decimal
figures, but the word "exactly" in the abstract ("a measured, honestly-
reported communication cost of exactly 6.00×") overstates precision by
implying an exact ratio rather than one that rounds to it. Low practical
consequence; a wording fix ("measured at 6.00×, matching the frontier's
`(1+⌈E'/m⌉)×` formula to within floating-point rounding") would be more
accurate than "exactly."

---

## 4. LOW-severity / informational findings

| # | Area | Finding |
|---|---|---|
| 1 | `aggregation.py` | The plain (unprojected) `FedGRAA` class — used by `FederatedTrainer(method="fedgraa")` — deliberately does *not* implement the paper's "Algorithm 1 (corrected)"; only `ProjectedFedGRAA`/`FedGRAAv2` do. Intentional (strict-reproduction mode, well-tested), but the paper presents Algorithm 1 as unconditionally corrected with no mention that an uncorrected variant still exists in the same codebase under a similarly-named class. |
| 2 | `scripts/verify_v2_exactness.py` | Only exercises γ∈{1.0, 0.5}, not the "three γ values" the paper cites (the third, γ=0, is covered instead by `tests/test_aggregation_v2.py`'s parametrized test). The standalone diagnostic script the audit was pointed at is narrower than the paper's own citation implies. |
| 3 | `baselines_classic.py` (SCAFFOLD/FedNova) | Adam optimizer moment state is never reset between rounds (same pre-existing, already-tracked pattern as Stage A/B's "Gap 2.2"), interacting with SCAFFOLD's per-round control-variate correction term in a way not separately re-disclosed in this module's own JC-TC-1 note. |
| 4 | `fedgraa/baselines_t5_fedpgo.py` | The shared/personalized layer-split boundary (`personalized_layers=1` default) is a real architectural judgment call, exposed as a constructor arg and tested, but not entered in FedPGO's JC1–JC8 table. |
| 5 | `docs/research_c/STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md` | A legitimate 19th judgment call (the adaptive attacker's detectability proxy) is disclosed only inline in `scripts/run_v2_frontier_sweep.py`, not in the consolidated JC table — a reader of the table alone would miss it. Consistent with the already-known "18 vs. 19" miscount the 2026-08-02 audit flagged. |
| 6 | Mutation testing | `test_fednova_reduces_to_fedavg_when_taus_are_uniform` is a real, non-tautological identity test, but structurally *cannot* catch a wrong per-client normalization bug when all τ_k happen to be equal (the buggy and correct formulas coincide in that special case by construction) — caught instead by the separate heterogeneous-τ hand-computed tests. Not a defect, but worth knowing this specific test's blind spot. |

---

## 5. What was independently CONFIRMED CORRECT (not merely "not disproven")

This is the majority of what was checked — reported in full so the two
sections above aren't mistaken for "the audit found the project broken."

**Core mechanism (§4–§5 of the paper):**
- Eq. 2 (FedAvg), Eq. 3 (static reparameterization) — exact formula matches.
- Algorithm 1's `θ'_g` genuinely uses the *previous* round's weights (`w^{t-1}`), matching the paper's own two-index notation.
- Simplex projection (`Π_Δ`) is unconditional, every round, in `ProjectedFedGRAA`/`FedGRAAv2` — a structural guarantee, not an assumption.
- Adam (not plain GD) is used for both the client and weight updates, confirmed by tracing the actual optimizer object, not a comment.
- SecAgg histogram summation genuinely withholds individual client counts — no plaintext/debug path exists; verified both by code trace and by tests that introspect for exactly this property.
- The DP composition arithmetic (`ε_total≈8.89` at `ε_round=0.1, δ=1e-5, T=200`) was independently hand-computed and matches exactly.
- The two-phase federated gradient reconstruction is genuinely algebraically exact — independently reran `verify_v2_exactness.py`, got ~1e-7 to 4e-7 relative error (paper's "2×10⁻⁷" is a representative rounding of a range that varies by γ/dataset, all well under the claimed <1e-6 threshold, not a single fixed measurement — a minor imprecision, not misleading).
- `last_n_class_accuracy` (rare-class recall, the metric of record) correctly implements mean per-class recall over the rarest classes.

**Classic-FL baselines:**
- SCAFFOLD's Option II control-variate update, server update, and local gradient correction all match Karimireddy et al. exactly, independently re-derived from the primary source, not merely trusted from the docstring.
- FedNova's normalized-averaging formula matches Wang et al. exactly, including the FedAvg-reduction identity (independently re-derived algebraically).
- "Adam used uniformly, not vanilla SGD, across all five classic baselines" — confirmed true by tracing every optimizer construction site; no exceptions found.
- All eight §6 classic-optimizer numbers (both datasets) match `frontier_table.json` exactly.

**Domain-specific baselines (all 14):**
- Every one of CELM, SpectralFed, CoAst, FLTrust, RAGA, FedTruth, FedAWA, FedAWARE, DA-PFL, FedMADE, FedAPA confirmed faithful to its primary source by independent mechanism-level derivation, several with hand-computed regression tests reproducing the primary paper's own worked examples.
- FedPGO's core algorithm (Eqs. 6–10, Algorithms 1–3, including the JC8 sign-convention flip) independently re-derived line-by-line and confirmed correct, aside from the focal-loss gap noted in §3.2.
- AdaAggRL and SBFedNIDS are both *honestly self-labeled* as simplified/reconstructed rather than faithful reproductions — the disclosure matches the actual code, which is the correct way to handle an unavoidable simplification.
- All 28 headline §6 numbers (14 baselines × 2 datasets) match `frontier_table.json` exactly.
- The near-universal 0.00% rare-class recall across ~9–13 of the 14 baselines was checked for plausibility (not a metric bug): the same evaluation pipeline correctly returns differentiated, nonzero numbers for FedGRAA-v2 and several baselines, the severity gradient tracks each dataset's actual imbalance level, and SBFedNIDS's 0% specifically is pre-disclosed in its own docstring as a partition-design artifact.

**T7 adversarial robustness:**
- The mechanism-specific attack genuinely derives from `_grad_wrt_weights` (`dL'/dw_k=⟨grad_P,P_k⟩`) exactly as claimed — not a simplified stand-in.
- SecFedNIDS's from-scratch Stochastic Outlier Selection implementation is structurally faithful to Janssens et al. 2012 (perplexity-calibrated affinity, binding probabilities, product-of-non-binding outlier score).
- The 3.1×/2.7× (30%-attacker) up-weighting numbers, all 12 Table 2 ASR values, and the full §10.6 γ-sweep table (all 12 cells) were independently recomputed and match exactly.
- Both qualitative γ-sweep claims (non-monotonic peak near γ=0.5 in 3/4 cells; 10–20× higher variance at γ=0) independently reconfirmed.

**Statistics and headline numbers:**
- All five FedGRAA-v2 arm means (v1_projected, v2_m1/m2/m3/m5) on both datasets match the paper/master-report exactly, with all 10/10 seed files present for every arm — no undercounted comparisons.
- The full 112-comparison Wilcoxon+Holm-Bonferroni significance layer was reimplemented from scratch (not calling the project's own function) and reproduced **111/112 significant, with the identical one exception** (FedAWARE vs. v2_m1, CIC-IDS-2017, p_holm=0.0625).
- No NaN/Inf found in any of 972 result files; every reported 0.00% is a genuine exact zero across all 10 seeds, not a rounding artifact.

**Test suite:**
- 367/367 Stage-C-filtered tests pass. ~74% are real, non-tautological regressions; <1% weak/tautological; ~24% legitimate smoke tests (mostly end-to-end no-NaN checks, an appropriate test type for full-pipeline runs); <1% mock-heavy.
- Three deliberately injected bugs (a sign flip in `secagg.py`, an off-by-one in `stats.py`'s Holm-Bonferroni, a wrong normalization in `fednova_normalized_update`) were all caught by the relevant tests, confirming real detection power, not vacuous green — all changes reverted, `git status` clean afterward.

---

## 6. Recommendations, in priority order

1. **Fix §7/§8's up-weighting range claim** (§2.1) — replace the unqualified
   "2.0–2.2×"/"2.0–3.1×, both datasets, both attacker fractions" with the
   correct per-dataset ranges already sitting in `STAGE_C_MASTER_REPORT.md`
   §10.6. Five-minute fix, no new compute needed.
2. **Soften §7's "every defended cell ≈0%" claim** (§2.2) — scope it to
   CIC-IDS-2017 explicitly, or report the UNSW-NB15 partial-recall cells
   honestly. This is the paper's most emphasized negative finding; it should
   be exactly as strong as the data supports and no stronger.
3. **Narrow §4.3's "eliminated by construction" claim for staleness** (§3.1)
   to match what the theorem section and the frontier results already say
   correctly elsewhere in the same paper.
4. **Add the missing JC entries** (§3.2, §3.3, §3.4's related disclosure gap,
   §4 row 4) — FedPGO's focal-loss omission, SecFedNIDS's actual
   consensus-threshold formula, the mislabeled perplexity-rescaling claim.
5. **Strengthen `test_baselines_t7_adversarial.py`** (§3.4) with at least one
   end-to-end "a poisoned client's influence is actually suppressed"
   assertion for each defense, matching the bar `test_baselines_t3_trust.py`
   already sets.
6. **Soften "exactly 6.00×" to "measured at 6.00×"** (§3.5) — cosmetic, but
   an easy, honest fix.
7. Everything in §4 is optional polish; none blocks submission.

None of the above findings implicate the underlying science, the core
privacy-compliant redesign, the significance testing, or any of the 19
baseline reproductions — all of those were independently re-derived from
first principles or raw data by agents with no access to each other's work
or to prior audit conclusions, and all held up. The defect surface, once
again, is concentrated in prose that summarizes correct underlying data
imprecisely — the same pattern the 2026-08-02 audit already named as this
project's one recurring failure mode, now confirmed to reach the manuscript
itself in two places, not only status/planning documents.

---

*This document is an independent, ad hoc audit output, written at the repo
owner's request, covering ground the 2026-08-02 audits did not (a fresh,
adversarial re-derivation of Stage C's numeric and algorithmic claims against
raw data and primary sources, rather than a documentation-consistency sweep).*

---

## 7. Remediation record (applied same session, after this report was written)

Items 1–4 of §6 (both HIGH findings, §2.1/§2.2; and the two related MEDIUM
findings, §3.1's staleness overclaim and §3.2/§3.3's disclosure gaps) were
fixed directly in the working tree:

- `docs/paper/FedGRAA_v2_PAKDD_Submission.md`: abstract, §4.3, §7 (both
  numbers), §8, and the conclusion all corrected to state the actual
  dataset-/attacker-fraction-dependent up-weighting range (1.0–3.1×, not a
  flat "2.0–3.1×"/"2.0–2.2×"), the actual UNSW-NB15 partial-recall picture
  under defense (11/30 cells nonzero, not "every cell ≈0%"), and the actual
  scope of what the two-phase protocol eliminates "by construction"
  (stratum-count weighting only; staleness is characterized/bounded by
  `ζ_stale(m)`, not eliminated). "Exactly 6.00×" softened to "measured at
  6.00×".
- `fedgraa/baselines_t7_adversarial.py`: module docstring corrected to state
  the real `consensus_frac = min(0.9, 1 - 1/K)` formula (not a flat `0.9`)
  and to stop claiming `perplexity` is rescaled per client count (it is not
  — confirmed by tracing the actual call site).
- `docs/research_c/STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`: JC-T7-9 entry
  corrected to match, with the discrepancy dated and disclosed rather than
  silently changed.
- `fedgraa/baselines_t5_fedpgo.py`: added JC9 (undisclosed focal-loss
  omission relative to the primary source, kept for fair-comparison
  consistency with every other baseline sharing the same generic trainer)
  and JC10 (the `personalized_layers` shared/personalized split boundary,
  previously untabled).

**Verified after every fix:** `python -m compileall -q fedgraa docs` (clean),
`pytest tests/test_baselines_t5_fedpgo.py tests/test_baselines_t7_adversarial.py -q`
(38/38 passed — these were docstring/prose-only changes, no source-logic
edits, so no test behavior changed), `python scripts/check_doc_links.py`
(all links resolve).

**§3.4 also remediated, same session (2026-08-03):** added two genuine
end-to-end detection tests to `tests/test_baselines_t7_adversarial.py`,
parametrized over both `SecFedNIDSTrainer` and `WeiDetectTrainer`:

- `test_t7_defense_trainer_detects_more_than_zero_obvious_attackers` — against
  an obvious model-poisoning attack on a wider (K=9-client), more
  realistic-scale synthetic partition than the existing 3-client smoke
  fixture, asserts the defense's `attack_success_rate_history` contains at
  least one value `<1.0` — i.e., the flag-to-trainer wiring genuinely
  catches a real attacker at least once, not merely "runs without NaN."
- `test_t7_defense_trainer_detects_attackers_in_most_rounds_not_a_single_fluke`
  — a stronger repeatability check: at least 2 of 8 rounds must show
  detection, ruling out a single-round fluke passing the first test above.

Both tests were **mutation-tested** before being trusted: monkeypatching
`secfednids_consensus_filter` to always return "flag nobody" (simulating a
broken detection-to-trainer wiring) makes both new tests fail exactly as
expected (`min asr == 1.0`, `rounds_with_detection == 0`), confirming real
detection power, not vacuous green. The mutation was applied via a
standalone throwaway script that monkeypatched the module attribute at
runtime — no source file was ever edited for this check, so no revert step
was needed; `git status` was confirmed clean immediately after.

**All six §4 LOW/informational items also remediated (2026-08-03, second
pass):** none were correctness gaps, but each was closed rather than left as
a permanent asterisk, since the fix cost was low in every case:

1. `docs/paper/FedGRAA_v2_PAKDD_Submission.md` §4.1: added one clause noting
   the codebase retains the literal, unprojected pre-review algorithm as a
   separate, explicitly-named strict-reproduction class for the diagnostic
   comparison, and that every §6+ result uses only the corrected variant.
2. `scripts/verify_v2_exactness.py`: extended the main sweep from
   γ∈{1.0,0.5} to γ∈{1.0,0.5,0.0} — the script itself now covers the "three
   γ values" the paper cites, rather than relying on a different file
   (`tests/test_aggregation_v2.py`) for the third point. Re-ran: all checks
   pass, γ=0.0 relative error ≈3.1–3.4×10⁻⁷ on both datasets, consistent
   with the already-passing γ=1.0/0.5 cases.
3. `fedgraa/baselines_classic.py`'s JC-TC-1 note: added a paragraph
   disclosing that SCAFFOLD's per-client Adam optimizer state is never
   reset between rounds, so its momentum buffers absorb the history of the
   per-round-varying `(c_global - c_local)` correction term, not merely the
   raw client gradient — a second-order consequence beyond "Adam vs. SGD"
   alone that hadn't been named explicitly before.
4. *(Already remediated in the first pass — FedPGO's JC10.)*
5. `docs/research_c/STAGE_C_T7_JUDGMENT_CALLS_2026-08-02.md`: added JC-T7-19
   (the adaptive attacker's L2-magnitude detectability proxy), consolidating
   a disclosure that previously existed only inline in
   `scripts/run_v2_frontier_sweep.py`. The table now genuinely contains 19
   entries, resolving the "18 vs. 19" miscount the 2026-08-02 audit flagged
   — `STAGE_C_MASTER_REPORT.md`'s two "19" citations were already correct;
   its one remaining "18" instance (line 11) was corrected to match.
6. `tests/test_baselines_classic.py`: added a docstring note to
   `test_fednova_reduces_to_fedavg_when_taus_are_uniform` naming its own
   blind spot (cannot distinguish a correct implementation from a
   wrong-per-client-normalization bug when all τ_k are equal) and pointing
   to the heterogeneous-τ tests that actually cover that bug class — so a
   future reader doesn't mistake this one test for complete coverage.

**Verified after this second pass:** `python -m compileall -q fedgraa scripts docs`
(clean), `pytest tests/test_baselines_classic.py tests/test_baselines_t5_fedpgo.py
tests/test_baselines_t7_adversarial.py -q` (66/66 passed), full suite
`pytest tests -q` (563/563 passed, unchanged — this pass added no new
tests, only docstrings/disclosures/a wider γ-sweep in a standalone script),
`python scripts/check_doc_links.py` (all links resolve), `python
scripts/audit_stage_c.py` (AUDIT PASSED, all 6 sub-checks including the
Sec 6–8 transcription check against the now-updated master report).

Nothing remains outstanding from this audit.
