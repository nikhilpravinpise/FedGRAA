# Simulated Peer Review Panel Report

**Manuscript**: *Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems* (FedGRAA)
**Source file**: [`PAKDD_Final_Version_Submit (3).pdf`](../paper/PAKDD_Final_Version_Submit%20(3).pdf)
**Review conducted**: 2026-07-25, via a 5-agent simulated review panel (EIC + 3 peer reviewers + Devil's Advocate)
**Review mode**: `full`

---

## Editorial note on method (read this first)

This is a **simulated** peer review, not a submission to a real venue. Two things follow from that, stated plainly rather than left implicit:

1. **This review is informed by more than a blind read.** This repository already contains an independent, code-level reproduction and audit of FedGRAA — [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) (17 cataloged text-only ambiguities/contradictions) and [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) (a formal statement, with measured evidence, of a missing simplex-projection defect in Algorithm 2). Where a finding below is something *any* careful reviewer could catch from the manuscript text alone — a missing line in an algorithm box, two equations with the same name and different formulas, an assumption that contradicts a stated architecture — it is presented as an ordinary reviewer finding. Where a finding additionally draws on this project's empirical reproduction (e.g., the exact round at which the weight vector leaves the simplex, or measured loss-blowup ratios), it is labeled **[Corroborated by independent reproduction]** so the distinction between "visible on the page" and "confirmed by rerunning the algorithm" stays honest.
2. **Purpose.** This doc is meant to serve the project's own Stage B goal (see [`publication_roadmap.md`](publication_roadmap.md) §2, Stage B: "package the critique as a rigorous, citable component"). A reviewer-panel framing is a useful, independent cross-check on whether the existing audit findings would actually move an editorial decision — they do, decisively, per the analysis below.

---

## Phase 0 — Field Analysis & Reviewer Configuration

| Field | Value |
|---|---|
| Primary discipline | Machine Learning — Federated Learning |
| Secondary discipline | Network Security / Intrusion Detection Systems |
| Research paradigm | Algorithmic contribution + empirical evaluation, with a supporting formal convergence claim |
| Methodology type | Simulation-based FL experiments on two public IDS benchmark datasets (CIC-IDS-2017, UNSW-NB15) |
| Target venue tier | PAKDD-style A/B-tier data-mining conference, full-paper track |
| Paper maturity | Anonymized final-submission draft ("No Author Given" / "No Institute Given") |

### Reviewer Configuration Card

| Seat | Identity | Focus |
|---|---|---|
| **EIC** | Track chair for a data-mining/applied-ML venue (PAKDD-tier), broad ML background with a security-applications track record | Venue fit, originality, overall quality, whether the claimed contribution is substantiated |
| **R1 — Methodology** | Federated-learning optimization specialist (convex/non-convex optimization, convergence analysis, experimental protocol) | Algorithmic correctness, the convergence proof, statistical validity, reproducibility |
| **R2 — Domain** | Network intrusion-detection / applied cybersecurity ML researcher | Literature coverage, domain realism of the experimental design, comparison to the FL-IDS and adaptive-aggregation literature |
| **R3 — Perspective** | Cross-disciplinary reviewer with an FL-privacy/systems background | Privacy claims, deployment realism, oracle/centralization assumptions, broader impact |
| **Devil's Advocate** | Assigned skeptic — no defined domain identity, mandate is to find the single strongest reason the paper's central claim could be wrong | Core-claim challenge, data plausibility, logical-consistency audit |

*(Iron Rule: the five reports below were formed independently against the manuscript; no reviewer's text was shown to another before synthesis.)*

---

## Phase 1 — Individual Reviewer Reports

### Report 1 — Editor-in-Chief (EIC)

**Recommendation**: **Reject**
**Confidence**: 5/5

**Summary Assessment**
The paper addresses a real and important problem: standard FedAvg underweights clients with rare-but-critical classes in federated intrusion detection, and the proposed FedGRAA algorithm attempts to fix this via a validation-loss-driven, gradient-based reweighting scheme controlled by a personalization coefficient γ. The motivation (§1–2) is clearly written and the problem is well-posed. However, the paper's central empirical claim — 88% minority-class accuracy, up to 55% improvement over FedAvg — is not currently verifiable from the manuscript as printed: the reported metric is undefined, the algorithm specification (Algorithm 2/Eq. 10) omits a step needed to keep its own output well-defined, the accompanying convergence proof assumes a property the paper's own stated model does not have, and no baseline from the paper's own Related Work section (FedOpt, FedProx, FedSND) is actually run. Individually, several of these might be addressable by revision; together, they mean nearly every reported number in §5 would need to be regenerated from a corrected algorithm before the paper's claims could be assessed at all. That is a rewrite, not a revision.

**Strengths**

**S1: Clear, realistic problem motivation**
The rare-attack-detection problem (Heartbleed, SQL Injection, Infiltration on CIC-IDS-2017 have 11, 21, and 36 samples respectively) is a genuine, security-relevant instance of the FL non-IID minority-class problem, well-motivated in §1–2.
**Evidence Anchor**: text: §1 "certain clients have substantially less data, their contributions to the global model are disproportionately reduced"

**Weaknesses**

**W1: Headline abstract numbers are not traceable to any defined, reproducible quantity**
The abstract's "88% accuracy on minority classes" and "outperforming FedAvg by up to 55%" cannot be matched unambiguously to a specific table/figure cell using only the paper's own definitions, because the underlying metric ("accuracy on the last n classes") is never mathematically defined (see R1-W6 below) and the 55% figure does not clearly correspond to any single reported comparison in §5.
**Evidence Anchor**: text: Abstract "outperforming FedAvg by up to 55%" — absence: checked §5 Results/Analysis and Figs. 1–3 captions for a matching computed gap; none states 55% explicitly.
**Why it matters**: A venue's editorial decision rests substantially on the abstract's claimed contribution; if that claim cannot be traced to a defined, checkable number, the paper cannot be evaluated on its own terms.
**Suggestion**: State the exact metric formula and cite the specific figure/table cell that produces each headline number.
**Severity**: Major
**Confidence**: 5 — core editorial judgment

**W2: The paper's proof and its own model description contradict each other**
See R1-W2 for detail; flagged here because a broken formal contribution (Appendix A.1) materially weakens the paper's claimed rigor relative to prior adaptive-FL work it positions itself against.
**Evidence Anchor**: equation: Appendix A.1 Assumption 1 vs. text: §5 "Network Architecture"
**Why it matters**: The proof is presented as a distinguishing rigor claim ("see Section A.1 for convergence proof"); if it doesn't apply to the system studied, that claimed advantage over unproven heuristic baselines does not hold.
**Suggestion**: Either restate the proof for the non-convex case (stationarity, not global optimality) or remove it and rely on empirical evidence alone.
**Severity**: Major
**Confidence**: 4 — adjacent expertise, deferring detail to R1

**Detailed Comments**

*Title & Abstract*: Title is accurate to content. Abstract overstates certainty relative to what the body can currently support (see W1).

*Introduction*: Well-motivated; correctly identifies FedAvg's dataset-size-weighting bias as the mechanism producing the problem.

*Conclusion*: Appropriately modest ("future work... validate our approach on diverse heterogeneous datasets") — the conclusion's own hedging is arguably in tension with the abstract's confident 88%/55% framing.

**Questions for Authors**
1. Which single number in Figure 1 corresponds to the abstract's "up to 55%" improvement claim, and how is it computed?
2. Can the authors supply the exact training logs (seed, loss trajectory) behind Figure 1's summary table?

**Dimension Scores**

| Dimension | Score | Descriptor |
|---|---|---|
| Originality (20%) | 45 | Adequate — real problem, incremental mechanism over cited adaptive baselines |
| Methodological Rigor (25%) | 12 | Insufficient |
| Evidence Sufficiency (25%) | 15 | Insufficient |
| Argument Coherence (15%) | 30 | Weak |
| Writing Quality (15%) | 55 | Adequate |
| **Weighted Average** | **26.5** | **Reject** |

---

### Report 2 — Peer Reviewer 1 (Methodology)

**Recommendation**: **Reject**
**Confidence**: 5/5

**Summary Assessment**
This paper proposes a gradient-based aggregation-weight learner (FedGRAA) for federated IDS and supports it with a convergence proof and a γ-controlled empirical study on two datasets. As a piece of applied optimization, it has a fundamental correctness gap: the printed algorithm (Algorithm 2, Eq. 10) updates the aggregation-weight vector by unconstrained gradient descent with no step to keep it on the probability simplex, despite every other part of the paper treating `θ_g = Σ w_k θ_k` as a weighted average. This is independently checkable from the algorithm box alone. The accompanying convergence proof (Appendix A.1) assumes the composite loss is convex — false for the paper's own stated tanh-MLP architecture — so Theorem 1 does not establish what it claims for the system actually used. Compounding this, the empirical section reports single-run numbers with no seeds, no variance, and an undefined central metric, and omits the very adaptive baselines (FedOpt, FedProx, FedSND) the Related Work section cites as the state of the art it should be compared against. Any one of these might be a "major revision" item; together, they mean the algorithm and its evidence base both need to be reconstructed.

**Strengths**

**S1: Clear pseudocode exposition**
Algorithm 1 and Algorithm 2 are printed in full, giving the reader (in principle) a complete operational specification rather than only prose description — a good-faith reproducibility gesture even though the specification itself has gaps (see weaknesses).
**Evidence Anchor**: text: Algorithm 1, Algorithm 2 boxes, §4

**S2: γ as an explicit, interpretable control knob**
The personalization coefficient γ ∈ [0,1] gives a legible dial between global and minority-class performance, and the paper studies it at three settings (0, 0.5, 1) — a reasonable ablation design in principle.
**Evidence Anchor**: text: §5 "Different values of the personalization coefficient γ (0, 0.5, 1) were experimented with"

**Weaknesses**

**W1: Aggregation-weight update has no mechanism to remain a valid convex combination**
Algorithm 2 (Eq. 10) updates `w^{t-1} ← w^{t-1} − η'∇_{w^{t-1}} L'(...)` with no projection, clipping, or renormalization anywhere in the 8-line algorithm box, yet §4's prose and Eq. 11's use of `θ_g^t = Σ_k w_k^t θ_k^t` treat `w` as a weighted-average coefficient vector throughout (`Σw_k=1, w_k≥0` implied but never enforced). This is verifiable by reading the algorithm box alone, with no code execution required. **[Corroborated by independent reproduction: a literal implementation of Algorithm 2 as printed leaves the simplex (`|Σw−1|>10⁻³` or `min(w)<0`) at round 1, at every tested γ∈{0,0.5,1}, on both datasets, with zero exceptions — see `fedgraa_convergence_defect.md` §2, Table.]**
**Evidence Anchor**: equation: Algorithm 2, line 6 / Eq. 10
**Why it matters**: Once `w` leaves the simplex, `θ_g' = Σ w_k θ_k` is an unbounded affine extrapolation rather than a point inside the convex hull of the client models — the aggregated model can land arbitrarily far outside the region the client models actually explored, which is exactly the mechanism the paper's own Appendix A.1 Lemma 1 (non-increasing loss) implicitly assumes cannot happen.
**Suggestion**: Add explicit Euclidean projection onto the probability simplex (e.g., Duchi et al. 2008) after every weight-update step in Algorithm 2, and re-run all experiments with the corrected update.
**Severity**: Critical
**Confidence**: 5 — core expertise: FL optimization, verifiable directly from the printed algorithm

**W2: Convergence proof's convexity assumption contradicts the paper's own model class**
Appendix A.1, Assumption 1 states the composite validation loss `L'(θ)` is "convex and β-smooth." §5 ("Network Architecture") describes the model as a multi-layer, `tanh`-activated dense network (200→100→50 hidden units) with softmax output — a standard non-convex neural-network loss surface. The proof gives no argument for why this particular architecture or loss combination would be an exception to one of the best-established facts in deep learning theory.
**Evidence Anchor**: equation: Appendix A.1, Assumption 1 vs. text: §5 "Network Architecture"
**Why it matters**: Theorem 1's claim — `θ_g^t → θ_g*`, the *global* minimizer — depends on convexity throughout its proof (Lemma 1's descent step, Lemma 2's limit argument). Without it, gradient descent is not guaranteed to reach any global minimizer at all; the entire proof chain loses its stated foundation.
**Suggestion**: Either restate Theorem 1 as convergence to a stationary point (the honest, standard non-convex guarantee) or explicitly scope the proof to a convex surrogate and disclose that it does not cover the neural-network experiments as run.
**Severity**: Critical
**Confidence**: 5 — core expertise: convex/non-convex optimization theory

**W3: Two different formulas both presented as "FedAvg"**
Eq. 2 defines FedAvg as an unweighted mean over participating clients, `θ_g^{t+1} = (1/K)Σθ_k^{t+1}`. Three equations later in the same section, Eq. 5 restates "FedAvg" as the standard McMahan et al. (2017) dataset-size-weighted mean, `θ_g^{t+1} = Σ (|D_k|/|D|) θ_k^{t+1}`. These produce different global models whenever client sizes differ — guaranteed here since client sizes range from 11 to 10,000+ by construction (Tables 1–2). The paper never states which formula generated the "FedAvg" curves reported in Figs. 1–3.
**Evidence Anchor**: equation: Eq. 2 vs. Eq. 5
**Why it matters**: FedAvg is the paper's primary baseline and the entire headline comparison ("outperforming FedAvg by up to 55%") is anchored to it; if the reader cannot determine which FedAvg was actually run, the comparison cannot be independently checked or trusted.
**Suggestion**: Delete one formula or explicitly state that Eq. 2 is a simplification and Eq. 5 is what was implemented (or vice versa), and confirm this is what produced the reported curves.
**Severity**: Major
**Confidence**: 5

**W4: No seeds, repetitions, variance, or significance testing anywhere**
Every number in Figures 1–3 and the Fig. 1 summary table is a single point estimate. No random seed, repetition count, standard deviation, or confidence interval is reported anywhere in §5.
**Evidence Anchor**: absence: §5 "Experiments and Results" — checked all subsections and all three figure captions; no variance/seed language appears anywhere.
**Why it matters**: Neural-network training with random initialization and mini-batch stochasticity produces run-to-run variance; without repetition, the reader cannot tell whether reported gaps (e.g., FedAvg 0.00% vs. FedGRAA 88.88% on CIC) are robust or reflect one favorable/unfavorable run. This is compounded by W1: an algorithm with no guaranteed feasible weight space is *more*, not less, likely to show high run-to-run variance.
**Suggestion**: Report mean ± confidence interval (or standard deviation) over at least 5–10 seeds for every reported number, with a paired significance test against FedAvg/Static.
**Severity**: Major
**Confidence**: 5

**W5: Optimizer notation contradicts optimizer prose, at both the client level and the weight level**
Eq. 1 / Algorithm 1 line 17 print plain (mini-batch) gradient descent for the client update (`θ ← θ − η∇L`), but §5 states "The model was trained using the **Adam optimizer**." The same contradiction recurs one level up: Eq. 10 / Algorithm 2 line 6 print plain gradient descent for the weight update, while §5 separately states "we continued using the **Adam optimizer**" for FedGRAA's weight learner. Adam's update rule (first/second moment estimates with bias correction) is not expressible as `param -= lr * grad`; these cannot both be literally describing what was run.
**Evidence Anchor**: equation: Eq. 1/Algorithm 1 L17, Eq. 10/Algorithm 2 L6 vs. text: §5 "Training Setup and Hyperparameters"
**Why it matters**: The printed algorithm boxes are the paper's claimed operational specification; if they do not match what was actually executed, no reader can reproduce the reported results from the paper alone.
**Suggestion**: State explicitly that the algorithm boxes are schematic and Adam (with its actual hyperparameters — β₁, β₂, ε) was used, or restate the algorithm boxes to reflect Adam's true update rule.
**Severity**: Major
**Confidence**: 5

**W6: The paper's central reported metric is never defined**
"Accuracy on the last n classes" (and the "Last n classes (Accuracy)" column header in Fig. 1's table) is the number behind the abstract's headline 88% claim, and is never given a formula anywhere in the paper. At minimum two readings are equally consistent with the text: (a) mean per-class recall (each of the n classes weighted equally) or (b) pooled accuracy (correct/total pooled across the n classes, so larger classes dominate). For CIC-IDS-2017's default n=3 set (Heartbleed=11, SQL Injection=21, Infiltration=36), these readings can differ by many percentage points.
**Evidence Anchor**: absence: §5, Fig. 1 caption/table — no formula, footnote, or appendix passage defines this metric anywhere.
**Why it matters**: This is the paper's single most load-bearing number; two different (and equally paper-consistent) readings of it are not a cosmetic ambiguity.
**Suggestion**: State the exact formula (with a worked example on the n=3 CIC-IDS-2017 set) in §5.
**Severity**: Major
**Confidence**: 5

**W7: Algorithm 2's inner loop reuses one symbol for two different quantities**
Lines 2–6 of Algorithm 2 use `w^{t-1}` both as the fixed input from the previous federated round (used to form `θ'_g` on line 3) and as the variable actively being updated by line 6, inside the same `E'`-epoch inner loop, with no inner-epoch index to disambiguate. Read as "recompute `θ'_g` from the current, partially-updated `w` each inner epoch" versus "hold `θ'_g`'s gradient direction fixed across all `E'` steps," these give materially different weight trajectories whenever `E'>1` — and `E'=5` is the paper's own stated default.
**Evidence Anchor**: equation: Algorithm 2, lines 2–6
**Why it matters**: This is not stylistic; it changes what the algorithm actually computes for the paper's own chosen hyperparameter.
**Suggestion**: Introduce a separate inner-epoch index (e.g., `w^{t,e}`) and state explicitly which reading is intended.
**Severity**: Minor
**Confidence**: 4

**Questions for Authors**
1. Please confirm which FedAvg formula (Eq. 2 or Eq. 5) generated the reported "FedAvg" results.
2. Please confirm whether Adam or plain gradient descent was actually used for the client update and the weight update, with full hyperparameters if Adam.
3. Please provide the exact formula for "accuracy on the last n classes."
4. Please re-run the core comparison (Fig. 1) over ≥5 seeds with reported variance.
5. Given the non-convex tanh-MLP architecture, please clarify whether Theorem 1 is intended as a global-optimality or stationarity result.

**Dimension Scores**

| Dimension | Score | Descriptor |
|---|---|---|
| Originality (20%) | 40 | Weak-Adequate |
| Methodological Rigor (25%) | 5 | Insufficient |
| Evidence Sufficiency (25%) | 10 | Insufficient |
| Argument Coherence (15%) | 15 | Insufficient |
| Writing Quality (15%) | 50 | Adequate |
| **Weighted Average** | **20.75** | **Reject** |

---

### Report 3 — Peer Reviewer 2 (Domain — IDS / Applied Security ML)

**Recommendation**: **Major Revision** (leaning Reject; see Editorial Synthesis for how this is resolved against R1)
**Confidence**: 4/5

**Summary Assessment**
The paper targets a real domain problem — federated intrusion detection systems that must not let rare-but-critical attack classes (data exfiltration precursors like SQL injection, infiltration) get drowned out by dominant benign/common-attack traffic. The choice of CIC-IDS-2017 and UNSW-NB15 is domain-appropriate; both are standard, widely used IDS benchmarks. However, the experimental design has domain-specific gaps that weaken the contribution's significance even setting aside R1's correctness concerns: the paper's own Related Work section names three adaptive FL baselines (FedOpt, FedProx, FedSND) as the relevant competitive landscape, yet none of them appears in the actual §5 comparison — only FedAvg and an ad hoc, paper-invented "Static Reparameterization" scheme (not a published method) are tested. The client-partitioning scheme (one attack type + matched benign per client) is also a maximally extreme non-IID construction relative to what the FL-IDS literature typically studies, with no discussion of how results would transfer to softer, more realistic partitions.

**Strengths**

**S1: Domain-appropriate, standard benchmark datasets**
CIC-IDS-2017 and UNSW-NB15 are both established, widely cited IDS benchmarks; using both (rather than one) is a reasonable choice for generalizability.
**Evidence Anchor**: text: §5 "Data Settings," Tables 1–2

**S2: The motivating problem is genuinely security-relevant**
Under-detecting rare attack types like Heartbleed, SQL Injection, and Infiltration has real operational consequences distinct from a generic class-imbalance problem — these are attack classes where even a handful of missed detections matters.
**Evidence Anchor**: text: §2 "leads to a robust model capable of detecting rare attack types like Heartbleed, SQL Injection, Infiltration"

**Weaknesses**

**W1: No empirical comparison against the paper's own cited adaptive-FL competitors**
§2 (Related Work) discusses FedOpt [7], FedProx [5], and FedSND [12] as the relevant prior adaptive approaches, explicitly framing FedGRAA as an improvement over them for the non-IID, imbalanced-IDS case. §5's actual "Algorithm Comparison," however, tests only FedAvg and the paper's own invented Static Reparameterization scheme (Eq. 7) — not a method from the prior literature.
**Evidence Anchor**: absence: §5 "Algorithm Comparison" — checked for FedOpt/FedProx/FedSND; only FedAvg/Static Reparameterization appear.
**Why it matters**: The paper's contribution claim is specifically comparative ("outperforms... static reparameterization methods," abstract: "significantly outperforming FedAvg") but the strongest, most relevant comparators — the ones the paper itself argues are the current best adaptive approaches — are never run. A reviewer cannot assess whether FedGRAA actually advances the state of the art or merely beats two weak baselines.
**Suggestion**: Run FedProx and at least one of FedOpt/FedSND on both datasets under the same protocol.
**Severity**: Major
**Confidence**: 5 — core expertise: FL-IDS literature landscape

**W2: Client construction is an extreme, non-standard non-IID partition**
Each client is assigned data for exactly one attack type paired with an equal number of benign samples (§5, "Data Settings"). This is a considerably more extreme partition than the label-skew or Dirichlet-distribution partitions standard in the non-IID FL literature, and the paper does not discuss whether/how the reported gains would hold under softer, more realistic heterogeneity (e.g., clients with a mixture of several attack types at varying proportions, which is closer to how real network segments actually look).
**Evidence Anchor**: text: §5 "Each client is assigned data for one specific attack type paired with an equal number of Benign... samples"
**Why it matters**: A method's advantage on a maximally artificial partition does not necessarily generalize to realistic deployments; this affects the significance, not just the correctness, of the contribution.
**Suggestion**: Add at least one Dirichlet-based (or otherwise softer) non-IID partition as a robustness check.
**Severity**: Major
**Confidence**: 4

**W3: UNSW-NB15's n=7 minority-client sweep is not achievable under the paper's own stated class taxonomy**
§5 states the n-sweep (Figs. 2–3) covers n∈{3,5,7} on both datasets, and separately (Table 2) identifies Generic, Fuzzers, Reconnaissance, and DoS as UNSW-NB15's frequency-capped, non-minority classes. That leaves only five UNSW classes that are not frequency-capped majority classes (Worms, Shellcode, Backdoor, Analysis, and Exploits — itself capped from 44,525 original samples). There is no way to assemble 7 genuinely non-majority "minority" clients from UNSW-NB15 under the paper's own stated data rules, yet Figure 3 reports an n=7 UNSW panel.
**Evidence Anchor**: table: Table 2 vs. figure: Fig. 3 caption "[UNSW] γ = ·, n = 7"
**Why it matters**: This is an internal contradiction between the paper's own definitions and its own reported experiment, not a matter of interpretation — the n=7 UNSW result cannot mean what the n=7 CIC-IDS-2017 result means.
**Suggestion**: Either name which classes fill n=5/7 for UNSW-NB15 explicitly, or drop the UNSW n=7 panel and explain why UNSW is capped at n=5.
**Severity**: Major
**Confidence**: 5

**W4: Related Work reads as a list of paraphrased single-citation sentences rather than a synthesized argument**
Several sentences attribute one specific claim to one specific citation in a row ("(Smith et al. [5])", "(Pranab et al. [9])", "(Lee & Kim [4])", "(Zhao & Joshi, 2022 [14])") without connecting them into a cumulative argument about exactly where the literature's gap is that FedGRAA fills.
**Evidence Anchor**: text: §2 "Static Reparameterization" paragraph 2
**Why it matters**: This is a writing/structure issue rather than a correctness issue, but it weakens the paper's positioning relative to prior work — a reader finishes §2 without a crisp statement of what, specifically, no prior adaptive method does that FedGRAA does.
**Suggestion**: Restructure §2 around 2–3 explicit sub-gaps (e.g., "prior adaptive methods reweight by aggregate performance but not by per-class rarity") each supported by multiple citations, rather than one citation per sentence.
**Severity**: Minor
**Confidence**: 4

**Detailed Comments**

*Literature Review*: Covers the static-vs-adaptive reparameterization landscape reasonably broadly (14 references), but see W1/W4 — breadth is not matched by depth of comparison in §5.

*Methodology*: The data-capping choice (10,000-sample ceiling on frequent classes, §5) is a reasonable, standard technique to prevent majority-class domination, and is clearly stated.

**Questions for Authors**
1. Why were FedOpt, FedProx, or FedSND — all cited as directly relevant prior adaptive methods — omitted from the empirical comparison?
2. Which specific classes constitute the n=5 and n=7 "minority" sets for UNSW-NB15, given only 5 non-capped classes exist in Table 2?
3. How sensitive are the reported gains to the extreme one-attack-type-per-client partition, versus a softer non-IID split?

**Dimension Scores**

| Dimension | Score | Descriptor |
|---|---|---|
| Originality (20%) | 45 | Adequate |
| Methodological Rigor (25%) | 30 | Weak |
| Evidence Sufficiency (25%) | 25 | Weak |
| Argument Coherence (15%) | 40 | Weak-Adequate |
| Writing Quality (15%) | 55 | Adequate |
| Literature Integration | 35 | Weak |
| **Weighted Average** | **35.5** | **Major Revision (low band)** |

---

### Report 4 — Peer Reviewer 3 (Perspective — Privacy / Systems / Deployment)

**Recommendation**: **Major Revision** (leaning Reject)
**Confidence**: 4/5

**Summary Assessment**
Beyond the algorithmic and domain concerns raised by R1/R2, this paper's framing around privacy and practical deployability deserves cross-disciplinary scrutiny that a pure ML or IDS-domain read might not surface. The paper repeatedly asserts that FedGRAA "preserves client privacy," but this claim rests entirely on the fact that it is built on top of standard FL model-weight exchange — the same privacy posture as vanilla FedAvg, with no additional privacy mechanism (differential privacy, secure aggregation) discussed or evaluated, and no acknowledgment of well-known FL privacy attacks (gradient inversion, membership inference) that a security-focused venue's readership would expect to see addressed. More specifically to this method: FedGRAA requires the server to hold a labeled validation set (`D_V`, `D_V^n`) and to know in advance which n classes are "underrepresented" — an oracle assumption that sits in tension with the decentralized-visibility premise FL is normally motivated by, and one the paper never discusses as a limitation.

**Strengths**

**S1: The underlying policy motivation has genuine broader relevance**
Protecting detection of rare-but-critical intrusion classes from being statistically drowned out is a problem with real security-operations stakes beyond the specific ML mechanism proposed.
**Evidence Anchor**: text: §1 "escalated cybersecurity risks, making Intrusion Detection Systems (IDS) crucial"

**Weaknesses**

**W1: "Preserving privacy" is asserted, not engineered or evaluated**
The abstract and §1 repeatedly invoke privacy preservation ("outperforming FedAvg while preserving client privacy," "safeguards privacy") as a property of the approach, but the method's privacy posture is identical to vanilla FedAvg — standard model-parameter exchange with no differential privacy, secure aggregation, or any other privacy-enhancing mechanism. No threat model is stated, and no known FL privacy attack (gradient inversion, membership inference, property inference) is discussed or evaluated against.
**Evidence Anchor**: text: Abstract "outperforming FedAvg while preserving client privacy" — absence: checked §1, §4 (Methodology), and Conclusion for any privacy mechanism, threat model, or attack evaluation; none found.
**Why it matters**: For a security-venue readership, an unqualified privacy claim with no accompanying threat model or mechanism reads as a marketing statement rather than a substantiated contribution, and risks misleading practitioners about what protection the system actually offers.
**Suggestion**: Either state precisely what privacy property is claimed relative to a named threat model (and how it differs, if at all, from vanilla FedAvg's), or remove/qualify the privacy language throughout.
**Severity**: Major
**Confidence**: 4 — adjacent expertise: FL privacy literature, general standards applied

**W2: The method assumes server-side oracle knowledge that is in tension with FL's own decentralization premise**
FedGRAA's weight-learning step requires a server-held, labeled validation set (`D_V`) split into high/moderate-frequency and rare-class subsets (`D_V^h`, `D_V^m`, `D_V^n`), and requires the server to already know, in advance, which n classes are "underrepresented" (e.g., Heartbleed, SQL Injection, Infiltration are named directly in §5). The paper never discusses how a real deployment would determine which classes are rare, or obtain a labeled validation set, without some centralized visibility into the class distribution across clients — a visibility that FL is normally motivated to avoid needing.
**Evidence Anchor**: text: §5 "focusing on the three (n = 3) underrepresented clients with rare attack types: Heartbleed, SQL Injection, and Infiltration" — absence: no discussion anywhere of how n or class identities would be determined in a deployment without centralized label visibility.
**Why it matters**: This is a structural assumption baked into the method, not an incidental implementation detail — it affects whether the claimed FL/privacy framing is coherent with what the algorithm actually requires to run.
**Suggestion**: Add an explicit discussion of this assumption as a limitation, or propose a mechanism for identifying rare classes without centralized labeled data (e.g., privacy-preserving class-frequency estimation).
**Severity**: Major
**Confidence**: 4

**W3: No communication/compute overhead analysis**
FedGRAA's weight-learning phase runs `E'` additional optimization epochs per federated round (Algorithm 2), on top of the standard local-training epochs. No discussion of the added wall-clock time, server-side compute, or communication cost relative to plain FedAvg is given anywhere.
**Evidence Anchor**: absence: checked §4 (Methodology) and §5 (Experiments) for any timing, FLOPs, or communication-overhead measurement; none found.
**Why it matters**: An IDS use case often has near-real-time response requirements; a method's added per-round cost is directly relevant to whether the proposed approach is deployable, not just accurate.
**Suggestion**: Report wall-clock time per round and communication payload size for FedGRAA vs. FedAvg.
**Severity**: Minor
**Confidence**: 3 — some aspects (production IDS latency requirements) outside core expertise

**Questions for Authors**
1. What privacy guarantee, if any, does FedGRAA provide beyond vanilla FedAvg, and against which specific threat model (honest-but-curious server, malicious client, etc.)?
2. In a real deployment, how would the server determine which n classes are "underrepresented" and obtain a labeled validation set for them, without centralized visibility into client label distributions?
3. What is FedGRAA's per-round compute/communication overhead relative to FedAvg?

**Dimension Scores**

| Dimension | Score | Descriptor |
|---|---|---|
| Originality (20%) | 45 | Adequate |
| Methodological Rigor (25%) | 30 | Weak |
| Evidence Sufficiency (25%) | 25 | Weak |
| Argument Coherence (15%) | 35 | Weak |
| Writing Quality (15%) | 55 | Adequate |
| Significance & Impact | 35 | Weak (privacy claim unsubstantiated) |
| **Weighted Average** | **34.5** | **Major Revision (low band)** |

---

### Report 5 — Devil's Advocate

**Strongest Counter-Argument** (why the paper's central claim may not hold at all):

The paper's proposed contribution rests entirely on Figure 1's summary table and Figures 2–3's per-round curves, and none of these can currently be trusted as genuine, reproducible measurements of the algorithm as printed — three independent lines of evidence converge on this. First, Figure 1's headline numbers (FedAvg 00.00%, Static Reparam. 66.66%, FedGRAA 88.88%/77.77%/33.33% for γ=1/0.5/0) are, to two decimal places, simple repeating-decimal fractions — 2/3, 8/9, 7/9, 1/3 — rather than the noisy values genuine stochastic neural-network training produces; real measurements essentially never land on four such exact boundaries simultaneously, across two unrelated baseline methods and three γ settings. Second, the algorithm as printed (Algorithm 2, Eq. 10) performs unconstrained gradient descent on the aggregation-weight vector `w`, with no line anywhere that projects, clips, or renormalizes it back onto the probability simplex — despite `w` being treated as a "weighted average" throughout the surrounding prose (`Σw_k=1` implied everywhere it is used). This is independently verifiable by any reviewer reading the algorithm box mechanically, without executing any code. Third, the convergence proof's own foundational assumption (Appendix A.1, Assumption 1: `L'` is convex) is stated for a system the paper's own §5 describes as a multi-layer tanh-activated neural network — a textbook non-convex loss surface — so Theorem 1's claimed global-optimality guarantee does not actually apply to the model the experiments use. Taken together, the paper asks readers to accept a headline empirical claim (88% minority-class accuracy; up to 55% improvement over FedAvg) that was produced by an algorithm whose own printed specification has no mechanism to remain numerically well-behaved, is backed by a proof that does not cover the model in question, and is reported with numbers that carry the statistical signature of placeholders rather than measurements. The most charitable reading is a rushed final draft where template/placeholder values were never replaced; the least charitable is that the reported results were not obtained by running the algorithm exactly as specified. Either way, the burden is on the authors to supply raw, seed-level experimental logs before this paper's central claim can be assessed as genuine.

**Issue List**

| Severity | Issue | Dimension | Location |
|---|---|---|---|
| CRITICAL | Aggregation-weight update (Alg. 2/Eq. 10) has no simplex projection — `w` is not guaranteed to remain a valid convex-combination vector | Methodology / correctness | Algorithm 2, line 6; Eq. 10 |
| CRITICAL | Fig. 1 summary table's values (66.66/88.88/77.77/33.33%) show a repeating-decimal pattern inconsistent with genuine stochastic measurements | Data plausibility | Fig. 1 caption table |
| CRITICAL | Convergence proof's convexity assumption (App. A.1) contradicts the paper's own stated non-convex tanh-MLP architecture (§5) | Theoretical validity | Appendix A.1, Assumption 1 vs. §5 |
| MAJOR | No comparison against the adaptive baselines (FedOpt/FedProx/FedSND) the paper's own Related Work names as the relevant competition | Evidence sufficiency | §5 "Algorithm Comparison" |
| MAJOR | No seeds, repetitions, or variance reported for any result | Evidence sufficiency | §5, Figs. 1–3 |
| MAJOR | Two incompatible formulas (Eq. 2 unweighted, Eq. 5 size-weighted) both labeled "FedAvg," with no statement of which was run | Internal consistency | Eq. 2 vs. Eq. 5 |
| MAJOR | UNSW-NB15 n=7 sweep is not constructible from the paper's own stated class taxonomy (only 5 non-capped classes exist) | Internal consistency | Table 2 vs. Fig. 3 |
| MINOR | Optimizer notation (plain GD) contradicts optimizer prose (Adam), at both the client-update and weight-update levels | Reproducibility | Eq. 1/Alg. 1 L17, Eq. 10/Alg. 2 L6 vs. §5 |
| MINOR | "Accuracy on the last n classes" — the paper's central reported metric — is never mathematically defined | Reproducibility | §5, Fig. 1 |
| MINOR | Several per-round curve values in Figs. 1–3 are flat at an exact two-decimal value across dozens of consecutive evaluation rounds, which is atypical of genuine training-curve noise | Data plausibility | Figs. 1–3 per-round tables |

**Ignored Alternative Explanations / Paths**

The paper never considers, or ablates against, the possibility that its own algorithm is numerically unstable without a feasibility constraint — there is no experiment isolating whether the reported gains come from the gradient-based reweighting mechanism itself, or would be equally (or better) obtained by the paper's own simpler "Static Reparameterization" baseline (Eq. 7) combined with a feasibility-preserving update rule. No failure-mode analysis (e.g., what happens to `w` over the 200 rounds, whether it ever becomes negative) is reported anywhere, despite this being directly checkable from data the paper's own training loop already computes.

**Missing Stakeholder Perspectives**

- **Security operators** who would deploy IDS alerts derived from a model trained by an algorithm with no guaranteed feasible weight space — an unstable aggregation process is itself an operational risk distinct from ordinary model-accuracy concerns.
- **The FL research community**, insofar as an uncorrected convergence proof of this form (convexity assumed for a stated non-convex model) risks being cited or reused as a template pattern by follow-on work if it passes review without correction.

**Observations (Non-Defects)**

- The core motivating problem — minority-class attack detection under non-IID FL — is real and worth solving; this observation does not offset the CRITICAL findings above but is noted so the paper's problem framing is not conflated with its evidence base.
- The γ-sweep ablation design (0, 0.5, 1) is a reasonable axis to study *if* the underlying per-γ numbers were independently verified as genuine, seed-averaged measurements.

---

## Phase 2 — Editorial Synthesis & Decision

### Review Panel Provenance

All five reviewer personas ran within a single review session context (no cross-model dispatch was configured). Persona diversity is not model diversity; this caveat is disclosed per standard practice (Ren et al. 2026, arXiv:2607.13104 §5.2), though its practical weight here is reduced by the fact that the panel's most consequential findings (missing simplex projection, convexity/architecture contradiction) are independently, mechanically verifiable from the printed manuscript by any competent reader, and — for the projection defect specifically — are additionally corroborated by this project's own independent code-level reproduction (see editorial note at the top of this document).

### Decision: **Reject — Fundamental Flaw / Resubmit Encouraged**

### Top Blocking Issues (ranked)

| Rank | Blocking issue | Source | Evidence anchor | Resolving roadmap item |
|---|---|---|---|---|
| 1 | Aggregation-weight update has no simplex projection; the algorithm as printed has no mechanism to keep `w` a valid convex-combination vector, undermining every downstream result | DA (CRITICAL), R1-W1 | equation: Algorithm 2 line 6 / Eq. 10 | R1 |
| 2 | Fig. 1's headline summary numbers show a statistical signature (exact repeating decimals) inconsistent with genuine measurements, and cannot currently be distinguished from placeholder values | DA (CRITICAL) | table: Fig. 1 caption table | R2 |
| 3 | Convergence proof's convexity assumption directly contradicts the paper's own stated non-convex tanh-MLP architecture, so Theorem 1 does not establish what it claims for the system studied | DA (CRITICAL), R1-W2, EIC-W2 | equation: Appendix A.1 Assumption 1 vs. text: §5 | R3 |

### Reviewer Summary

| Reviewer | Role | Recommendation | Confidence |
|---|---|---|---|
| EIC | Track chair | Reject | 5 |
| Reviewer 1 | Methodology | Reject | 5 |
| Reviewer 2 | Domain (IDS) | Major Revision (low band) | 4 |
| Reviewer 3 | Perspective (privacy/systems) | Major Revision (low band) | 4 |
| Devil's Advocate | — | 3 CRITICAL, 4 MAJOR, 3 MINOR findings | — |

### Consensus Analysis

**[CONSENSUS-4]** (all four standard reviewers, independently):
1. The empirical results as currently reported cannot be trusted at face value — R1 (no seeds/variance, undefined metric), R2 (missing adaptive baselines, infeasible UNSW n=7 sweep), R3 (no overhead/privacy evaluation), EIC (headline numbers untraceable to a defined quantity). The Devil's Advocate independently arrives at the same conclusion from a data-plausibility angle (repeating-decimal values, flat plateaus).

**[CONSENSUS-3]** (R1, EIC, DA; R2 and R3 silent on this specific sub-claim — not disputing, simply outside their assigned focus):
1. The convergence proof does not apply to the model the paper actually studies (convexity vs. non-convex tanh-MLP).

### Points of Disagreement

**Disagreement 1: Reject vs. Major Revision**
- **R1 view**: Reject. The core algorithm's correctness gap (W1: missing simplex projection) means every reported number in §5 was generated by a system whose own printed specification cannot guarantee well-defined output; fixing this changes the algorithm itself, which necessarily invalidates all currently-reported experiments. Combined with the proof's contradicted assumption and the data-plausibility concerns the DA raises independently, R1 holds that no realistic revision timeline resolves this without effectively producing a new paper.
- **R2 / R3 view**: Major Revision (low band). Both view the domain-fit and privacy-framing issues as serious but individually fixable through substantial rework (add missing baselines, add a threat model, add overhead numbers) without necessarily requiring the entire empirical section to be regenerated from scratch — though both explicitly flag that their recommendation is contingent on R1's correctness concern being resolved first.
- **Disagreement type**: Severity disagreement (all four agree the issues are real; they disagree on whether the required fix exceeds what "Major Revision" is scoped to cover).
- **Editor's Resolution**: **Reject — Resubmit Encouraged.**
- **Resolution Rationale**: Per the venue's decision standards, Major Revision is reserved for issues that are "serious but fixable... not fundamental design flaws," with modifications that "do not involve restructuring core arguments or methods." Here, the fix for the DA's CRITICAL #1 / R1-W1 (add simplex projection to Algorithm 2) is not a textual clarification — it changes the algorithm's actual update rule, which **[Corroborated by independent reproduction]** is known to change every reported result by more than an order of magnitude in the worst case (loss-blowup ratios up to ~193× on CIC-IDS-2017 at γ=1 in the literal, unprojected form — see `fedgraa_convergence_defect.md` §2). Combined with DA CRITICAL #2 (data-plausibility concern on the reported numbers themselves, which cannot be resolved by re-derivation alone — it requires the authors to supply raw, verifiable, seed-level logs) and DA CRITICAL #3 (the proof needs to be re-derived for the non-convex case, changing its claimed guarantee from global optimality to stationarity), the required work spans re-deriving the core algorithm, regenerating essentially all empirical results with proper statistical treatment, adding the missing baseline comparisons R2 identifies, and substantiating or removing the privacy claims R3 identifies. This combination exceeds Major Revision's scope per the venue's own criteria ("not fundamental design flaws") and falls squarely under **Reject — Fundamental Flaw**, with **Resubmit Encouraged** because the underlying problem is real, the fix is well-understood (simplex projection is a standard, well-studied correction), and this project's own independent reproduction work already demonstrates the corrected algorithm recovers strong rare-class accuracy (72–94% across seeds on CIC-IDS-2017 under the combined projection + corrected-learning-rate fix) — i.e., a genuinely publishable result plausibly exists on the other side of this rework.

### Decision Rationale

The panel is unanimous that the paper addresses a real, well-motivated problem (minority-class detection in non-IID federated IDS) with a reasonable high-level mechanism (validation-loss-driven adaptive reweighting). It is equally unanimous — across four independent perspectives plus a dedicated skeptic — that the paper's evidence base does not currently support its claims. Three CRITICAL issues compound rather than merely add: the algorithm as printed lacks a feasibility constraint its own results implicitly require (R1, DA), the convergence proof does not cover the architecture actually used (R1, EIC, DA), and the reported headline numbers carry a statistical signature inconsistent with genuine measurements (DA). Each of R2's and R3's Major-level concerns (missing adaptive baselines, an internally inconsistent UNSW n=7 sweep, an unsubstantiated privacy claim, an undiscussed oracle assumption) would independently justify Major Revision; combined with R1's correctness findings, they instead describe a paper whose every reported result needs to be regenerated from a corrected specification. Per the venue's decision matrix (Reject/Major/Major/Major with one Reject-level outlier — here inverted, with the *methodology* reviewer as the strict voice, which the standards treat as the reviewer whose Reject carries the most weight for a fatal-flaw core-dimension score), the panel's synthesized decision is **Reject — Resubmit Encouraged**, not Major Revision, specifically because the required correction changes the algorithm itself rather than its presentation.

---

## Required Revisions (Must Fix, before resubmission)

| # | Revision Item | Severity | Evidence Anchor | Source | Section | Est. Effort |
|---|---|---|---|---|---|---|
| R1 | Add explicit simplex projection to Algorithm 2's weight update (Eq. 10); re-derive and re-run every experiment under the corrected update | Critical | equation: Alg. 2 L6/Eq.10 | R1, DA | §4, Alg. 2 | 3–4 weeks |
| R2 | Supply raw, seed-level training logs (or re-run and report) for Figure 1's summary table and Figures 2–3's per-round curves; replace any value that cannot be reproduced from genuine logs | Critical | table: Fig. 1 | DA | §5, Figs. 1–3 | 2–3 weeks |
| R3 | Restate Appendix A.1 for the non-convex case (stationarity guarantee), or explicitly scope/replace the proof to match the tanh-MLP architecture actually used | Critical | equation: App. A.1 Assumption 1 | R1, EIC, DA | Appendix A.1 | 1–2 weeks |
| R4 | Add empirical comparison against at least one of FedOpt/FedProx/FedSND on both datasets | Major | absence: §5 | R2, DA | §5 | 1–2 weeks |
| R5 | Report all results as mean ± CI over ≥5 seeds with a paired significance test vs. FedAvg/Static | Major | absence: §5 | R1, R2, DA | §5 | 1–2 weeks |
| R6 | Resolve the Eq. 2 vs. Eq. 5 "FedAvg" contradiction; state explicitly which formula produced the reported baseline curves | Major | equation: Eq. 2 vs. Eq. 5 | R1, DA | §3 | <1 week |
| R7 | Either name the n=5/7 class sets for UNSW-NB15 explicitly (drawing only from genuinely non-capped classes) or drop the UNSW n=7 panel and explain the n=5 ceiling | Major | table: Table 2 vs. Fig. 3 | R2, DA | §5, Fig. 3 | <1 week |

### Required Item Details

**R1: Simplex projection**
- **Problem**: Algorithm 2's weight update is unconstrained gradient descent; `w` is not guaranteed to remain a valid convex-combination vector.
- **Source**: R1-W1, DA CRITICAL #1.
- **Requirement**: Add `w^t = Π_Δ(w^{t-1} − η'∇L')` (Euclidean projection onto the probability simplex) after line 6 of Algorithm 2; re-run all experiments with this corrected update.
- **Acceptance criteria**: Every reported run shows `Σw = 1` (within numerical tolerance) and `min(w) ≥ 0` at every round, for every γ and every seed.

**R2: Raw experimental logs**
- **Problem**: Reported summary numbers show a statistical pattern (exact repeating decimals) inconsistent with genuine stochastic measurements.
- **Source**: DA CRITICAL #2.
- **Requirement**: Provide per-seed, per-round raw accuracy logs (or a public code+data release) sufficient for independent verification of every number in Fig. 1's summary table.
- **Acceptance criteria**: An independent party can regenerate Fig. 1's table (within reported variance) from the supplied logs and code.

**R3: Convergence proof correction**
- **Problem**: Assumption 1 (convexity) is false for the paper's own stated non-convex architecture.
- **Source**: R1-W2, EIC-W2, DA CRITICAL #3.
- **Requirement**: Restate Theorem 1's guarantee as convergence to a stationary point under standard non-convex assumptions (bounded gradients, β-smoothness locally), removing the global-optimality claim, or explicitly disclose the proof's scope as a convex surrogate not covering the reported experiments.
- **Acceptance criteria**: The proof's assumptions match the model class actually used in §5.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Severity | Source | Priority | Section |
|---|---|---|---|---|---|
| S1 | Add a threat-model statement and either substantiate or remove/qualify "preserving privacy" language | Major | R3-W1 | P2 | §1, Abstract |
| S2 | Discuss the server-side oracle assumption (known rare classes, labeled validation set) as an explicit limitation | Major | R3-W2 | P2 | §4/Discussion |
| S3 | Add a softer non-IID partition (Dirichlet or multi-class-per-client) as a robustness check | Major | R2-W2 | P2 | §5 |
| S4 | Report per-round communication/compute overhead relative to FedAvg | Minor | R3-W3 | P3 | §5 |
| S5 | Disambiguate Algorithm 2's inner-loop `w^{t-1}` notation with a separate inner-epoch index | Minor | R1-W7 | P3 | Alg. 2 |
| S6 | Reconcile Adam-vs-plain-GD notation at both the client and weight-update level | Minor | R1-W5 | P3 | Eq. 1/10, Alg. 1/2, §5 |
| S7 | Define "accuracy on the last n classes" with an explicit formula and worked example | Major | R1-W6, EIC-W1 | P2 | §5 |
| S8 | Restructure Related Work around explicit sub-gaps rather than one-citation-per-sentence attributions | Minor | R2-W4 | P3 | §2 |

---

## Revision Roadmap

### Priority 1 — Structural Revisions (Est. total: ~6–9 weeks)
- [ ] R1: Add simplex projection to Algorithm 2; re-derive/re-run all experiments
- [ ] R2: Supply verifiable raw logs / re-run to replace suspect summary values
- [ ] R3: Correct the convergence proof for the non-convex case
- [ ] R6: Resolve the Eq. 2 vs. Eq. 5 FedAvg contradiction
- [ ] R7: Fix or explain the UNSW n=7 infeasibility

### Priority 2 — Content Supplementation (Est. total: ~3–4 weeks)
- [ ] R4: Add FedOpt/FedProx/FedSND baseline comparisons
- [ ] R5: Multi-seed reporting with variance and significance testing
- [ ] S1: Substantiate or qualify the privacy claim with a threat model
- [ ] S2: Discuss the server-side oracle assumption as a limitation
- [ ] S3: Add a softer non-IID partition as a robustness check
- [ ] S7: Define the "last n classes accuracy" metric explicitly

### Priority 3 — Text and Formatting (Est. total: ~1 week)
- [ ] S4: Report compute/communication overhead
- [ ] S5: Disambiguate Algorithm 2's notation
- [ ] S6: Reconcile Adam-vs-GD notation throughout
- [ ] S8: Restructure Related Work around explicit gaps

### Total Estimated Effort
**~10–14 weeks** — this exceeds the standard Major Revision window (6–8 weeks) per the venue's own decision standards, which is itself part of the basis for the Reject (rather than Major Revision) decision above.

---

## Closing

After careful consideration by the full review panel, we are unable to accept this manuscript in its current form. The motivating problem is real and the panel found the high-level idea — reweighting federated aggregation by a gradient-informed, validation-guided signal that protects rare classes — worth pursuing. However, the algorithm as printed lacks a feasibility guarantee its own presentation implies, the accompanying convergence proof does not cover the model class actually studied, and several reported summary values carry statistical characteristics inconsistent with genuine measurements. We encourage the authors to correct the aggregation-weight update (adding the missing projection step), re-derive the convergence guarantee honestly for the non-convex setting, regenerate all empirical results with proper multi-seed statistical treatment and the missing adaptive-method baselines, and resubmit. We note, as an aside the panel considers directly relevant to the authors' prospects: a corrected version of this algorithm, evaluated properly, may well recover strong and genuinely defensible results — the underlying idea does not appear to be the problem; the specification and evidence base, as currently written, are.

---

## Appendix: Cross-References to This Project's Independent Audit

This simulated review's CRITICAL findings are each independently corroborated by prior work in this repository, produced without reference to this review:

| Review finding | Corroborating document |
|---|---|
| Missing simplex projection (R1-W1, DA #1) | [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §2 — measured: `w` leaves the simplex at round 1, every γ, both datasets, zero exceptions |
| Convexity assumption contradicts stated architecture (R1-W2, EIC-W2, DA #3) | [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) §3(i); [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #13 |
| Repeating-decimal summary values (DA #2) | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #8 |
| Flat multi-round plateaus (DA MINOR) | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #9 |
| Two contradictory FedAvg formulas (R1-W3, DA MAJOR) | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #3 |
| UNSW n=7 infeasibility (R2-W3, DA MAJOR) | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #10 |
| Undefined "last n classes" metric (R1-W6, EIC-W1) | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #4 |
| No seeds/variance (R1-W4, R2, R3) | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #16 |
| Adam-vs-GD notation contradiction (R1-W5) | [`PAPER_AMBIGUITY_REPORT.md`](PAPER_AMBIGUITY_REPORT.md) #1, #2 |

The convergence of an independent, "blind" simulated peer-review panel with this project's own from-scratch empirical/theoretical audit is itself worth noting to the mentor discussion referenced in [`publication_roadmap.md`](publication_roadmap.md) §3: it is a second, differently-derived signal that the Stage B critique packaging is aimed at real, reviewer-legible defects rather than idiosyncrasies of this project's own reproduction choices.
