# The Papers, the Code, and Whether AI Ruined Any of This

**Primary sources:** `docs/current/PAPER_AMBIGUITY_REPORT.md` (402 lines,
read in full), `docs/paper/{draft1.md, PAKDD_Final_Version_Submit_v3.md,
FedGRAA_Simulated_Peer_Review.md, FedGRAA_v2_PAKDD_Submission.md}`,
`fedgraa/*.py`, `scripts/*.py`, `docs/research_c/Final_Stage_C.md`.

Three threads in one file because they reinforce each other: the paper
critique and the code map are both evidence for the same underlying claim
— that this project's rigor is real, checkable, and has caught its own
mistakes (including AI-introduced ones) repeatedly, in public, in the repo
history.

---

## 1. The 17-issue paper critique — the full catalogue

`PAPER_AMBIGUITY_REPORT.md` is a **paper-only** critique — no reference to
the code, no proposed fixes, purely "does the paper's own text hold
together." Every equation, algorithm-box line, §5 sentence, table, figure
caption, and the Appendix A.1 proof were read against each other.

**Severity legend:** CONTRADICTION (two parts state incompatible things) ·
UNDEFINED (a central term is never mathematically defined) · UNDERSPECIFIED
(a procedure is described but not fully enough to execute identically
twice) · SUSPICIOUS DATA (numbers implausible as genuine measurements) ·
NOTATION (malformed but not meaning-changing).

| # | Issue | Type |
|---|---|---|
| 1 | Eq.1/Algorithm 1 line 17 print plain GD (`θ-=η∇L`); §5 prose says Adam | CONTRADICTION |
| 2 | Eq.10/Algorithm 2 line 6 print plain GD on `w`; §5 says Adam, η′=0.1, E′=5 | CONTRADICTION |
| 3 | Eq.2 (uniform 1/K average) and Eq.5 (data-size-weighted) both called "FedAvg," three equations apart, produce different global models whenever client sizes differ (guaranteed here) | CONTRADICTION |
| 4 | "Accuracy on the last n classes" — the paper's central reported metric, behind the 88% headline — never mathematically defined anywhere. At least 2 readings (mean per-class recall vs. pooled accuracy) differ by many points on CIC's extreme skew | UNDEFINED |
| 5 | "Automated method" for the median-frequency split — never specifies which median, tie-breaking, or the degenerate all-tied case | UNDERSPECIFIED |
| 6 | Sub-gradient of `L_a=min(L_h,L_m)` at the crossing point — undefined, and training is *designed* to approach that boundary, not avoid it. Appendix A.1 assumes `L'` convex everywhere — false at the kink | UNDEFINED |
| 7 | Algorithm 2's inner loop reuses the symbol `w^{t-1}` for both "prior round's weight" and "weight as of the current inner epoch" — two non-equivalent readings exist, diverging whenever E′>1 (paper's own default is E′=5) | UNDERSPECIFIED / self-contradictory notation |
| 8 | Fig.1's summary table: FedAvg 00.00/00.00, Static 66.66/00.00, FedGRAA(γ=1) 88.88/21.09, (γ=0.5) 77.77/20.59, (γ=0) 33.33/00.00 — 66.66/88.88/77.77/33.33 are exact repeating-decimal fractions (2/3, 8/9-ish, 7/9-ish, 1/3), far more consistent with hand-picked placeholders than genuine stochastic measurements | SUSPICIOUS DATA |
| 9 | Per-round tables show exact, static values across dozens of rounds (e.g. CIC "Static (All classes)" stays exactly 0.44 from round 60–200) — real training curves essentially never freeze like this | SUSPICIOUS DATA (weaker) |
| 10 | UNSW-NB15 genuinely lacks 7 non-frequency-capped classes — the paper's own n=7 sweep requires including classes its own text elsewhere calls "frequent, capped, non-minority" | CONTRADICTION with own data |
| 11 | `S_t` notated as a time-varying "selection" but defined as always-fixed/full — confusing but not meaning-changing | NOTATION |
| 12 | Algorithm 1 line 17 uses the identical symbol on both sides of an assignment — standard pseudocode shorthand, technically malformed as formal math | NOTATION |
| 13 | Convergence proof assumes `L'` convex and β-smooth (App. A.1); §5's actual network is a multi-layer tanh MLP — a standard non-convex surface. The proof, as written, doesn't apply to the system the paper claims it applies to | CONTRADICTION |
| 14 | The "automated method" for the median split is asserted but never disclosed anywhere — not even a name or citation | UNDERSPECIFIED |
| 15 | Rare-class identities for n=5/n=7 never named, either dataset — compounds #10, since a reader can't tell if a frequency-capped class was quietly included | UNDEFINED |
| 16 | No random seed, no repetition count, no variance/error bars anywhere in Figures 1–3 | UNDERSPECIFIED |
| 17 | B=64 batch size vs. classes as small as 11 total samples (→9-10 training samples after 90/10 split) — undefined whether the partial batch is used as-is, padded, or dropped | UNDERSPECIFIED |

**Summary count, ready to say out loud:** *"Three direct equation-vs-prose
contradictions, one proof-vs-model-class contradiction, one sweep-vs-
own-data-table contradiction, six central-but-undefined terms or
procedures, two purely notational issues, and two independent statistical
red flags on the paper's own numbers — compounded by zero variance/seed
reporting and an unaddressed batch-size edge case. Any reproduction has to
make an explicit, arbitrary choice at every one of these points, and none
can be verified as 'the' correct one without the original authors' code,
seed, and raw logs — none of which the paper discloses."*

## 2. The paper's own lineage — four documents, one story

### 2.1 draft1.md vs. PAKDD_Final_Version_Submit_v3.md

Both 1,738 lines, textually near-identical body (same equations, algorithm
boxes, all data tables, section structure) — but **the abstract and a few
other spots were hand-corrected in draft1.md and left uncorrected in
PAKDD_Final_Version_Submit_v3.md**, which is the file this whole project
treats as "the frozen, anonymized submission draft actually sent out."

- **PAKDD_Final's abstract (the operative claim this project reproduces):**
  *"FedGRAA achieves 88% accuracy on minority classes, significantly
  outperforming FedAvg while preserving client privacy... outperforming
  FedAvg by up to 55%."*
- **draft1.md's abstract (corrected, withdraws the claim):** *"...this
  weight update exactly as specified... does not reliably deliver that
  benefit: the unprojected weight vector leaves the probability simplex in
  round 1 in all 120 tested configurations, driving the composite loss to
  blow up by up to 1.86×10⁶×... we withdraw an 'outperforms FedAvg by up
  to 55%' claim carried in an earlier draft, for which no supporting
  computation could be traced in this project's own results."*
- draft1.md's Appendix A.1 also carries an inserted "Scope disclosure"
  block, absent from PAKDD_Final, stating plainly that the convexity
  assumption doesn't hold for the tanh-MLP actually used.
- draft1.md's bibliography carries five inline correction comments (mis-
  dated, mis-attributed, mis-venued entries, and the fabricated-citation
  replacement — see §3.2 below); PAKDD_Final's bibliography has none —
  the original errors stand.
- **One more discrepancy worth knowing, not yet reconciled anywhere:**
  the original paper states UNSW-NB15 has 43 input features; the v2
  manuscript's Experimental Setup states 39. This project's own Stage A
  report uses 39 (42 raw features minus 3 dropped categorical columns) —
  so 39 appears to be the correct, current number, and 43 is the original
  paper's own (unexplained) figure. Worth a one-line note if the mentor
  asks about feature counts.

### 2.2 The simulated peer review — verdict and full findings

`FedGRAA_Simulated_Peer_Review.md`: a 5-persona panel (EIC — track chair;
R1 — FL-optimization methodology; R2 — network-IDS domain; R3 —
cross-disciplinary privacy/systems; Devil's Advocate — assigned skeptic).
**Explicitly disclosed caveat in the review itself: all five personas ran
in one session, no cross-model dispatch — "persona diversity is not model
diversity."**

**Verdict: Reject — Resubmit Encouraged.** EIC and R1 both Reject
(confidence 5/5); R2 and R3 Major-Revision-leaning-Reject (confidence 4/5).
Estimated revision effort ~10–14 weeks, explicitly stated to exceed the
venue's normal 6–8 week Major-Revision window — part of the basis for
Reject rather than Major Revision.

**Required revisions, verbatim-ready:**
- **R1 (Critical):** add the missing simplex projection; re-run every
  experiment under the corrected update.
- **R2 (Critical):** supply raw seed-level logs or replace unreproducible
  values (the repeating-decimal finding).
- **R3 (Critical):** restate the convergence proof for the non-convex case
  (stationarity, not global optimality).
- R4 (Major): compare against ≥1 adaptive-FL baseline (FedOpt/FedProx/
  FedSND — R2 didn't yet know FedSND was fabricated).
- R5 (Major): report mean±CI over ≥5 seeds with significance testing.
- R6 (Major): resolve the Eq.2-vs-Eq.5 FedAvg contradiction.
- R7 (Major): name UNSW's n=5/n=7 rare-class sets, or drop the panel.
- S1–S8 (Should-fix): threat-model statement; discuss the oracle
  assumption as a limitation; add a softer non-IID partition; report
  overhead; disambiguate notation (twice); define "last-n accuracy" with
  a formula; restructure Related Work.

**The Devil's Advocate's exact words, worth having ready verbatim** — this
is the single most quotable line in the whole project if the mentor wants
the sharpest version of the critique:

> *"Figure 1's headline numbers... are, to two decimal places, simple
> repeating-decimal fractions — 2/3, 8/9, 7/9, 1/3 — rather than the noisy
> values genuine stochastic neural-network training produces... The most
> charitable reading is a rushed final draft where template/placeholder
> values were never replaced; the least charitable is that the reported
> results were not obtained by running the algorithm exactly as
> specified."*

**Consensus structure:** all four standard reviewers agree the empirical
results can't be trusted at face value (CONSENSUS-4); R1/EIC/DA agree the
proof doesn't apply to the model studied (CONSENSUS-3, R2/R3 silent but not
disputing). R1 held Reject specifically because *"no realistic revision
timeline resolves this without effectively producing a new paper."*

### 2.3 The v2 resubmission — point-by-point resolution

`FedGRAA_v2_PAKDD_Submission.md` maps every Critical/Major finding to a
specific fix, explicitly, in its own Introduction:

| Review finding | v2's resolution |
|---|---|
| No mechanism to stay a valid convex combination (R1) | §4.2 — Euclidean simplex projection added to Algorithm 1 |
| Convexity proof for a non-convex model (R3) | §4.4 — restated as PL-condition gradient-mapping stationarity, `O(1/T)+O(ζ²)` |
| Single-run, unseeded, missing adaptive baselines (R5, R4) | §5–7 — 10 seeds, 19 baselines (14 domain + 5 classic), paired Wilcoxon + Holm |
| Undefined "last-n accuracy" (R1/S7) | §6 — defined and tested: mean per-class recall, `last_n_class_accuracy` |
| Eq.2 vs. Eq.5 FedAvg ambiguity (R6) | §3 — one formula used throughout, stated explicitly |
| Oracle rare-class knowledge, server-held `D_V` (S1, S2) | §4.2–4.3 — the entire privacy redesign |
| No overhead reporting (S4) | §8-equivalent — 6.00×/3.00× communication-cost table |
| Fabricated FedSND citation | References section — explicitly noted as removed, not carried forward |

**Not explicitly re-addressed by name:** R7 (UNSW n=7 infeasibility — v2
sidesteps by not sweeping n at all rather than answering it directly) and
S3 (a softer Dirichlet-style non-IID partition — not present in v2's
design). Worth knowing these two are quietly unaddressed rather than
resolved, if the mentor reads the review side-by-side with v2.

**Its own AI disclosure, quoted in full — the single most directly
relevant passage to "how AI didn't ruin this research":**

> *"This manuscript was drafted by an AI assistant (Claude Sonnet 5) at the
> repository author's direction. It is a corrected resubmission... rewritten
> to resolve every Critical and Major finding of the independent simulated
> review panel... using the faithful reproductions, real seed-level compute
> campaign, statistical significance layer, privacy-compliant redesign, and
> adversarial-robustness evaluation implemented in this repository's
> `fedgraa/` and `scripts/` code (all independently tested)... Every number
> in §6–§8 was populated only after the underlying compute campaign
> completed and was independently verified... none was asserted ahead of
> the data that produces it."*

## 3. Did AI ruin this research? — the actual evidence, not a vibe

This is the part worth having the most precise command of, because it's
the one place a skeptical mentor could reasonably push hard. The honest
answer isn't "no mistakes were made" — it's "mistakes were made, and the
project's own structure is what caught every one of them, repeatedly, on
the record."

### 3.1 The structural safeguard: completeness gates vs. correctness audits

Two categorically different checks exist for every "done" claim:

- **Completeness gates** (`check_stage_a/b/c_complete.py`) — does every
  claimed artifact exist and is it shaped right? Confirmed by direct
  reading: these check structure (keys present, files non-empty,
  figures have the right number of plotted lines) — they do **not**
  check whether a number in the prose matches the underlying JSON.
- **Correctness audits** (`audit_stage_a/b/c.py`) — a *second*,
  independent implementation re-derives every reported number from raw
  per-cell JSON (e.g. `audit_stage_c.py` recomputes Wilcoxon/Holm from
  scratch via direct `scipy.stats` calls and a hand-written Holm
  step-down, never importing the project's own `fedgraa/stats.py` or
  table builders — so a bug in the project's own statistics module
  couldn't hide by having the audit trust the same code).

**All three audit scripts are themselves negative-control tested** — a
deliberately corrupted value was injected and confirmed caught before
being trusted for real. This is a load-bearing detail: it's the difference
between "we wrote a checker" and "we verified the checker actually checks."

### 3.2 Real defects caught, on the record

**Citation hallucinations — 7 caught, with a named detection rule.** The
Stage C literature review caught seven fabricated-or-wrong citations
entering the project's own corpus via automated survey tooling, including
**FedSND** — FedGRAA's own reference [12], and a baseline this project had
already *shipped in code* as `FedSNDTrainer` before the fabrication was
caught (renamed to `StaticDistributionAwareTrainer`, F6 in the Stage C
master plan). The detection rule, worth quoting: *"the 'no link, no
citation' test — a fluent, mechanically detailed description in a search
summary whose result list contains no link actually bearing the paper's
name is evidence against existence, not for it."* Critically, this rule
**cuts both ways** — the same escalation procedure *rescued* three real
papers (Fed-LAA, FedPGO, a "Precision-Weighted FL" paper) that initially
looked fabricated before a deeper check confirmed they were real. FedPGO
specifically was later fully reimplemented once its real full text was
found via an institutional repository.

**The same discipline caught a near-miss on the mentor's own record** (see
`04` §5 in full): a claim about the mentor's citation count and venues
looked exactly like a hallucination signature — fluent, repeated, no
matching link — and was checked rather than repeated. It turned out to be
entirely real, confirmed via OpenAlex's structured index and the mentor's
own CV. **Both directions of the failure mode were caught in this project:
fabricated things that looked real, and real things that looked
fabricated.**

**Six reporting defects caught by `audit_stage_b.py`** (Stage B, before
that master report was written) — all correctly-computed numbers, wrongly
*summarized*:
1. Blowup reported as the mean of a heavy-tailed ratio (185,764× — a
   figure describing no run that occurred) → fixed to median+max.
2. Seed-0 worst cases presented as the campaign's worst cases (193×/7.6×
   vs. the real 10-seed maxima of 1.86×10⁶×/46.85×) → sections now
   separate typical-run from tail-severity explicitly.
3. Run-count overstatement — "80 fresh runs" when 71 were fresh, 9 reused.
4. A rare-accuracy-only "statistically indistinguishable" claim that hid a
   29-point all-class-accuracy gap → dual-metric reporting enforced
   everywhere, with automated flagging where the two metrics disagree.
5. *(negative control)* The auditor itself crashed while trying to report
   a caught failure, because a row label containing "η′" broke stdout
   encoding — found and fixed the same day.
6. A malformed significance-marker regex, caught mid-fix of the above.

**A real code bug caught by the same-day independent full-repo audit**
(2026-07-26, `docs/audit/INDEPENDENT_FULL_AUDIT_2026-07-26.md`, referenced
directly in `MENTOR_WALKTHROUGH.md` §6): two "parallel" implementations of
the composite-loss calculation disagreed on which of two `NaN` cases they
silently dropped — one path fed the numbers used to *plot* results, the
other fed the numbers used to *drive the weight update*. A real, reachable
inconsistency (no evidence it corrupted any already-published number, but
a genuine bug). Also caught the same day: a metric that could silently
shrink its own denominator when a rare class had zero samples in a round;
feature-scaling statistics computed before the train/validation split
(small validation leakage into training normalization); and — the
recursive one — a status-tracking document that itself claimed 12 items
were "still open" when `git log` showed they'd all been closed. **All
fixed in the very next commit**, and the audit gates were wired into CI
(`.github/workflows/docs-check.yml`) so this class of drift can't
reappear silently.

**A hallucinated-name baseline shipped in code, caught by cross-referencing
two of the project's own documents against each other** (F6, §3.2 above) —
worth stressing this one specifically: the fabrication was caught in the
*literature review*, but nobody had connected it to the *already-shipped
code* until this project's own Stage C planning pass did the cross-check
explicitly. That's the same discipline the mentor-conversation prep
applied to itself (§3.1's negative controls) applied one level up, across
documents rather than within one script.

**Two internal inconsistencies caught while preparing *this specific
briefing*, tonight, worth naming precisely because they show the audit
discipline continuing to work under fresh scrutiny rather than being a
historical artifact:**
- The v2 manuscript's own communication-overhead claim disagrees with
  itself between §4.3 ("≈2×" at m=5) and §6/abstract ("3.00×" at m=3) —
  `03` §5.
- A stale docstring in `scripts/diagnose_fedgraa.py` still describes
  `ProjectedFedGRAA` as "never imported by `fedgraa/`," which was true when
  written but is now out of date — the class was later promoted into
  `fedgraa/aggregation.py` itself, and a NOTE further down the same file
  says so. Harmless (the code is correct, the top comment isn't), but a
  real, small, still-open documentation drift.

### 3.3 The pattern across all of the above

Every incident above shares a shape: something plausible-looking (a number,
a citation, a status claim, a docstring) turned out to be wrong in a small,
specific, checkable way, and something *other than trust* caught it —
a second independent computation, a negative control, a structured primary
source, or a direct cross-reference between two documents. **None of these
were caught by "re-reading the work more carefully" — they were caught by
building a second, independent way to check, and running it.** That's the
actual argument for "AI didn't ruin this research": not that the AI-assisted
work was error-free (it demonstrably wasn't), but that the project's
verification structure treats "AI wrote this" and "a human wrote this" as
requiring the *same* standard of independent re-derivation — and that
standard is what's visible in the repo's own audit trail, not a claim
resting on trust.

**One more concrete, load-bearing fact for this conversation:** the
project's own AI-disclosure sections (quoted in §2.3 above, and present at
the bottom of `STAGE_A/B/C_MASTER_REPORT.md`) are not boilerplate — each
one makes a specific, checkable claim ("every number is read from
`results/*.json`, generated by the scripts named throughout — none is
hand-transcribed or asserted ahead of the data producing it"), and that
claim is exactly what the correctness audits in §3.1 exist to verify.

---

## 4. The code, mapped — for when "walk me through the implementation" comes up

Confirmed by direct reading of the actual source (not filenames, not
docstrings alone) unless noted.

| File | What it actually does |
|---|---|
| `fedgraa/config.py` | `ModelConfig` (architecture), `FedConfig` (all hyperparameters, `round_zero_zero`, v2's `grad_refresh_every`/`secagg`/`dp_epsilon`/`p_drop`), `DatasetSpec` (per-dataset feature/class counts and the hard-coded `rare_label_order` oracle list v1 uses). `secagg=False` is **rejected** with a `ValueError`, not silently allowed — a previous no-op flag was hardened after an audit flagged it as misleading. |
| `fedgraa/model.py` | The real `tf.keras.Sequential` MLP; an `Adam` wrapper that persists `tf.Variable` moment state across calls (the object at the center of the "momentum persists across rounds" deviation); four `@tf.function(jit_compile=True)` fused XLA training steps — plain, FedProx-style, SCAFFOLD-style (control-variate correction added to the gradient before the shared Adam step — a disclosed judgment call, not silent), and SecFedNIDS's importance-accumulator step. |
| `fedgraa/aggregation.py` | **Confirmed by full-file reading: `FedGRAA.update()` contains no simplex-projection or clipping call anywhere** — a grep for "simplex"/"project"/"clip" across the file matches only inside the two non-default subclasses. `ProjectedFedGRAA` (diagnostic) adds `project_to_simplex` after the same Adam loop. `two_phase_composite_grad` and `FedGRAAv2` are v2's privacy-compliant reconstruction — designed so client contributions pre-combine into a single model-shaped array before SecAgg-summing, which is *why* the measured bytes match the `(1+⌈E'/m⌉)×` formula exactly. |
| `fedgraa/trainer.py` | Algorithm 1's loop. **Confirmed: `client_train_round` resets model weights from the global aggregate every round, but the per-client Adam optimizer object is built once in `__init__` and reused every round** — the exact mechanism behind the documented "momentum persists, weights reset" deviation. `FederatedTrainerV2` adds SecAgg-mediated histograms, client-side DP noise (deliberately never server-side — the server is the threat model's adversary), dropout modeling with a `SecAggDropoutError` that's caught and carries `w` forward rather than crashing, and the one T7 attack (`rare_class_signature_attack`) wired directly into the trainer rather than the generic harness. |
| `fedgraa/metrics.py` | **Confirmed: `last_n_class_accuracy` computes mean per-class recall** — exactly the documented resolution of the paper's undefined "last-n accuracy" (§1, issue #4). |
| `fedgraa/partition.py` | Non-IID partitioning (one client per attack class + disjoint circular-indexed benign draw); `median_frequency_split`/`stratum_tags` shared between v1's pooled computation and v2's per-client SecAgg-histogram tagging — "structurally the same code path, not two implementations kept in sync by hand." |
| `fedgraa/stats.py` | Bootstrap CI, paired t-test, Wilcoxon, Cohen's *d*, Holm–Bonferroni — each with an explicit, tested degenerate-case contract. **Confirmed: `cohens_d_paired` really does return signed infinity** for a zero-variance, nonzero-mean difference — "the mathematically honest value," deliberately not clipped to a large finite number. |
| `fedgraa/secagg.py` / `privacy.py` | `SecAggChannel` models the *ideal functionality* of secure aggregation (not the cryptography) — structurally unable to reveal any individual contribution once submitted. `laplace_mechanism` + two independently-implemented DP composition accountants (advanced composition, RDP), cross-checked against a worked example (`ε_total≈8.89`). |
| `fedgraa/attacks.py` | The 4 T7 attacks — label-flip, the "pill" model-poisoning attack (core mechanism only, disclosed), the project's own novel mechanism-specific attack (derived directly from the weight-gradient formula), and an adaptive random-restart wrapper. |
| Baseline files (`baselines*.py`) | 19 baselines total across 8 files, each carrying an explicit "citation-verification disclosure" naming which equations were read from primary-source full text vs. filled by a disclosed judgment call. |
| `scripts/diagnose_fedgraa.py` | Home of `ProjectedFedGRAA`/`project_to_simplex` (later promoted into `fedgraa/aggregation.py` itself) and the per-round instrumentation (`first_simplex_violation_round`, monotonicity checks) every headline number in Stage A/B traces back to. |
| `check_stage_*_complete.py` / `audit_stage_*.py` | The two-tier verification described in §3.1. |

**One thing worth saying if asked "is the fix actually in the code
somewhere I'd run":** no — and that's deliberate. `ProjectedFedGRAA` and
`FedGRAAv2` are real, tested, non-default classes; `fedgraa.FedGRAA` (the
strict-reproduction default) still implements Algorithm 2 exactly as
printed, with no projection, on purpose — changing that default would stop
being a reproduction. This is the "no silent patching" rule showing up as
an actual, checkable code-structure decision, not just a policy statement.
