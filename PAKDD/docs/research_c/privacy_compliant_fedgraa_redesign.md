# Making FedGRAA FL-Compliant: A Complete Redesign Blueprint

**Purpose of this document.** The current FedGRAA paper ([PAKDD_Final_Version_Submit_v3.md](../paper/PAKDD_Final_Version_Submit_v3.md)) has two assumptions that a competent reviewer *will* flag, and that already contradict federated learning's own founding privacy premise:

1. **Oracle knowledge of which classes/clients are rare.** The server hard-codes "the last `n` underrepresented classes" (Heartbleed, SQL Injection, Infiltration, ...) and partitions clients "one attack type per client," implicitly assuming the server (or the experiment designer) knows each client's label distribution. In real FL, the server never sees client labels — assuming this a priori is a privacy violation and also an unrealistic experimental setup (real clients don't hold one attack type each).
2. **A server-held, fully-labelled validation set `D_V`.** Algorithm 2 requires the server to evaluate `L_h`, `L_m`, `L_u` on `D_V` — i.e., the server holds real labelled data. This is precisely the thing FL exists to avoid.

This document is the result of an exhaustive review of the existing project research (`docs/`, `understanding/`) plus a fresh, wide sweep of the current (2019–2026) literature, specifically targeting these two problems. It ends with a concrete, implementable redesign ("FedGRAA-v2") that keeps the paper's core intellectual contribution — *gradient-refined, γ-tunable adaptive aggregation* — while removing both violations, plus a prioritized action plan and a fully-cited reference list.

> **Related project documents** (read first if unfamiliar): [research_followup.md](research_followup.md) (first pass on this exact question), [reproduction_gaps.md](../archive/reproduction_gaps.md) / [full_reproduction_gap_analysis.md](../archive/full_reproduction_gap_analysis.md) (implementation-fidelity audit, archived — see `../audit/REPRODUCTION_AUDIT.md` for the current version), [literature_review_and_publication_strategy.md](literature_review_and_publication_strategy.md) (venue positioning), and [../../understanding/10_SOTA_adaptive_aggregation.md](../../understanding/10_SOTA_adaptive_aggregation.md) (SOTA aggregation comparison). This document supersedes and substantially deepens §1–§3 of `research_followup.md`; it does not repeat the Flower-framework material from §2 of that file (still valid, see there).
>
> **Revision note (2026-07-08, second pass).** A second, independently-run literature sweep (`../archive/sagitra_report.md`, an internally-indexed-corpus search covering ~400 numbered entries) was cross-checked against this document. Every new paper it surfaced was re-verified against the live arXiv API where possible before being folded in below; a few titles could not be located on arXiv under any reasonable query (they may be non-arXiv venue-only papers, e.g. TrustCom/ACM proceedings, or the corpus's title/venue attribution may be imprecise — a known failure mode of automated literature-survey tools). Those are explicitly marked **[unverified]** wherever they appear, in keeping with the project's "no hallucinated citations" standard (see [../../understanding/09_sources.md](../../understanding/09_sources.md)). Everything else below was independently confirmed live via `export.arxiv.org` on 2026-07-08.
>
> **Revision note (2026-07-09, third pass — exhaustive verification + implementation-grounded quantification).** This pass did three things beyond the first two: (1) it re-ran live `arxiv.org/search` queries against every single entry still marked **[unverified]** in the second pass, resolving **6 of 15** to confirmed real papers with correct arXiv IDs (§2 Family 7's **FedLMD** → arXiv:[2409.13136](https://arxiv.org/abs/2409.13136); §3.7's **REVERB-FL** → arXiv:[2512.13647](https://arxiv.org/abs/2512.13647); §9.1's **FedAPA** → arXiv:[2502.07456](https://arxiv.org/abs/2502.07456), **FedAWA** → arXiv:[2503.15842](https://arxiv.org/abs/2503.15842), **FedSWA** → arXiv:[2507.20016](https://arxiv.org/abs/2507.20016), **FedDRL** → arXiv:[2208.02442](https://arxiv.org/abs/2208.02442)) — each corrected entry is annotated **[verified 2026-07-09]** below, with the exact confirmed title, authors, venue, and any discrepancy from how the second pass described it explicitly called out (e.g. FedDRL turns out to be an acronym shared by two unrelated papers — see §9.1). The remaining 9 ([FedRS], [FedCK], [ProtoHAR], [PACT], [CT-BRFL], [FedPRS], [FedMcon], [FedFMD], [FedCoef]) still could not be located on arXiv under any query tried and remain **[unverified]**; (2) it pulled the *exact* architecture, hyperparameter, and dataset-partition constants out of the live `fedgraa/` codebase (`config.py`, `model.py`) rather than relying on paraphrased hyperparameter values, and used them to replace every previously-qualitative communication/overhead claim in §7 with **exact computed byte counts** for both datasets actually used in this project; (3) it added three entirely new sections that a rigorous reviewer or implementer would need and that no prior pass produced: a formal threat-model/privacy-accounting section (§10), a full unit/integration/statistical test specification for FedGRAA-v2 (§11, written as a testable specification — no code — per the instruction that this pass documents only), and a glossary of every new symbol introduced by the redesign (§12). Nothing previously marked **[unverified]** was silently promoted; every status change below is traceable to a specific confirmed arXiv listing quoted at the point of use.
>
> **Revision note (2026-07-09, fourth pass — final exhaustive re-verification, live codebase cross-validation, and a new implementation-mapping section).** This pass, run later the same day as the third pass, did four further things. (1) It independently re-queried arXiv for all 9 still-unverified titles (FedRS, FedCK, ProtoHAR, PACT, CT-BRFL, FedPRS, FedMcon, FedFMD, FedCoef) a second time, using both title-field (ti:) and full-text-field (all:) queries rather than the single query style used in the third pass. Zero of the 9 were found on this second, independent attempt either — the ti:"FedRS" query returned six results, none of which is the reported KDD 2021 paper (the closest lexical matches were an unrelated 2025 remote-sensing FL benchmark also abbreviated "FedRS" and an unrelated LDP paper "FedRE"); ti:"ProtoHAR", all:"FedCK"+AND+all:federated, all:"PACT"+AND+all:"trust"+AND+all:federated, all:"CT-BRFL", all:"FedMcon", and all:"FedFMD" each returned zero total results; all:"FedPRS" returned exactly one unrelated hit ("Learning Federated Visual Prompt in Null Space for MRI Reconstruction", arXiv:2303.16181, ICCV 2023 — a real paper, but not FedPRS); all:"FedCoef" returned zero. This strengthens (does not merely repeat) the third pass's conclusion: these 9 remaining titles are either genuinely non-arXiv venue-only publications (plausible for KDD/IEEE JBHI/TrustCom/ACM TIST, none of which mirror all accepted papers to arXiv) or are corpus mis-attributions from the automated `../archive/sagitra_report.md` survey tool — see the (unchanged) per-entry caveats in §2 Family 7, §3.7, and §9.1. (2) It independently re-confirmed, by fetching the live arXiv abstract pages directly (not just the API), the two quotations already used from FedGreed (arXiv:2508.18060 — the abstract's exact wording, "the central server is trusted and equipped with a reference dataset," matches this document's §3.1 quotation verbatim) and Practical Secure Aggregation (arXiv:1611.04482 — the abstract's exact expansion-factor wording, "1.73x communication expansion for 2^10 users and 2^20-dimensional vectors, and 1.98x expansion for 2^14 users and 2^24-dimensional vectors," matches this document's §7.4 quotation verbatim). (3) It read the entire live `fedgraa/` reference implementation end to end (`config.py`, `model.py`, `aggregation.py`, `trainer.py`, `partition.py`, `metrics.py` — not just the two files consulted in the third pass) and used it to (a) extract the exact Adam-optimizer hyperparameters and reset semantics governing both the local client optimizers and the FedGRAA weight-learner optimizer (new detail folded into §5), (b) compute a genuine, previously-missing UNSW-NB15 analogue of §7.5's worked DP-noise-vs-rare-class example using the dataset's own `rare_label_order` constants (folded into §7.5, with a materially different and non-obvious finding — see there), and (c) build an entirely new §15, Reference Implementation Mapping, which ties every mechanism proposed in §4 to the exact function/class names that already exist in the codebase today (including one previously-unremarked fact: `DatasetSpec.rare_client_ids()` already contains a data-driven, non-oracle fallback path that the current reproduction runs simply never exercise — see §15). (4) It added a pointer from the action checklist (§13, item 5) to this new mapping so the "Should"-priority implementation work is unambiguously scoped at the function level, not just the module level. Nothing previously marked **[unverified]** was promoted by this pass (all 9 remain exactly as unverified as before, now with a second independent negative-result search attached); everything newly added is traceable either to a live arXiv abstract fetch or to a direct line-by-line reading of the checked-in source file named at the point of use.

---

## Table of Contents

1. [Why this matters for reviewers](#1-why-this-matters-for-reviewers)
2. [Taxonomy: detecting/handling class imbalance without an oracle](#2-taxonomy-detecting-class-imbalance-without-an-oracle)
3. [Taxonomy: validating adaptive aggregation without a server-side labelled set](#3-taxonomy-validating-without-a-server-side-labelled-set)
4. [The recommended fix: FedGRAA-v2](#4-the-recommended-fix-fedgraa-v2)
5. [Impact on the convergence proof](#5-impact-on-the-convergence-proof)
6. [Experimental plan: new baselines, ablations, and metrics](#6-experimental-plan-new-baselines-ablations-and-metrics)
7. [Privacy-utility and communication trade-offs (what to report)](#7-privacy-utility-and-communication-trade-offs-what-to-report)
8. [How to reposition the paper's contribution](#8-how-to-reposition-the-papers-contribution)
9. [Beyond the two core gaps: aggregation alternatives and broader research gaps](#9-beyond-the-two-core-gaps-aggregation-alternatives-and-broader-research-gaps)
10. [Threat model and formal privacy accounting](#10-threat-model-and-formal-privacy-accounting)
11. [Testing and validation plan for FedGRAA-v2](#11-testing-and-validation-plan-for-fedgraa-v2)
12. [Glossary of new notation](#12-glossary-of-new-notation)
13. [Prioritized action checklist](#13-prioritized-action-checklist)
14. [Full reference list](#14-full-reference-list)
15. [Reference implementation mapping](#15-reference-implementation-mapping-fedgraa-v2--existing-fedgraa-code)

---

## 1. Why this matters for reviewers

Both assumptions are **exactly the kind of thing PAKDD/AAAI/IJCAI-tier reviewers now check for**, because the FL literature has spent the last five years explicitly hardening this soft spot:

- A 2023 survey, **"A Survey on Class Imbalance in Federated Learning"** (Zhang, Li, Qi, He — arXiv:[2303.11673](https://arxiv.org/abs/2303.11673)), frames the *entire subfield* around exactly this tension: *"we review existing methods for estimating the extent of class imbalance without the need of knowing the actual data to preserve data privacy."* If a paper on FL class imbalance doesn't cite or address this survey's framing, a reviewer familiar with the area will ask why not.
- On the validation side, a 2023 paper attacking a structurally identical assumption (a Byzantine-robust method that also needs a small server-held labelled set, **FLTrust**, NDSS 2021, arXiv:[2012.13995](https://arxiv.org/abs/2012.13995)) is directly criticized by follow-up work: **FedTruth** (Ebron, Zhang, Yang — arXiv:[2311.10248](https://arxiv.org/abs/2311.10248)) states plainly: *"such benign root datasets are impractical."* This is not our (or the mentor's) opinion — it is the field's own consensus about this exact kind of assumption.
- Reviewers at security-adjacent venues (CCS/NDSS/S&P) treat "the server has private labelled data" as a **threat-model admission** that must be justified or removed. Reviewers at ML venues (AAAI/IJCAI/ICML) treat it as an **experimental-design flaw** that breaks the "no raw data leaves the client" claim made in the introduction.

The good news: **both problems have mature, well-cited, directly transplantable solutions.** The rest of this document catalogs them and recommends the specific combination that best preserves FedGRAA's mathematical core.

---

## 2. Taxonomy: detecting class imbalance without an oracle

Six independent families of solution exist. They are **not mutually exclusive** — the recommended redesign in §4 combines Families 1 and 3.

### Family 1 — Infer class composition from gradients (cheapest, drop-in, already-cited-adjacent)

**Core citation: Wang, Xu, Wang, Zhu — "Addressing Class Imbalance in Federated Learning," AAAI 2021** (arXiv:[2008.06217](https://arxiv.org/abs/2008.06217)).

> *"As FL training is executed by exchanging gradients in an encrypted form, the training data is not completely observable to either clients or servers... we propose a monitoring scheme that can infer the composition of training data for each FL round, and design a new loss function — Ratio Loss — to mitigate the impact of the imbalance... Our method is shown to significantly outperform previous methods, while maintaining client privacy."*

**Mechanism.** For cross-entropy loss, the gradient of the loss w.r.t. the output-layer logits is `softmax(z) − y` (a one-hot vector for the true class). Summed over a batch/client update, the induced change in the **output-layer bias** gradient is proportional to `(expected_class_distribution − actual_class_distribution)`. Since the server already knows what the *expected* (uniform-prior or previous-round) distribution looks like, it can **algebraically back out the actual per-round class composition of each client's update purely from the received gradient/weight-delta** — no labels, no raw data, nothing beyond what standard FedAvg already transmits. This is the textbook justification for why label information "leaks" through gradients (see also the gradient-leakage literature below), turned into a *constructive, declared* tool rather than an attack.

**Directly relevant follow-ups (found in this sweep, not in the earlier pass):**

| Paper | Venue/Year | What it adds over Wang et al. 2021 |
|---|---|---|
| **FedImT** — Yan, Li, Ota, Dong, "An Efficient Imbalance-Aware FL Approach for Wearable Healthcare with Autoregressive Ratio Observation" (arXiv:[2310.14784](https://arxiv.org/abs/2310.14784)) | 2023 | Makes the composition estimate **online/recursive**: a *self-attenuating iterative equivalent* tracks how the estimated composition drifts round-to-round (rather than Wang et al.'s round-by-round independent estimate), then "promptly tweaks the balance of the loss computing for minority classes." This is a better fit for FedGRAA's 200-round setting than a one-shot estimate. |
| Yang, Wong, Zhu, Wang, Qian — "Federated learning with class imbalance reduction" (arXiv:[2011.11266](https://arxiv.org/abs/2011.11266)) | 2020 | Uses the inferred composition to drive **client *selection*** (choosing which clients participate each round to minimize aggregate imbalance) rather than reweighting after the fact — a complementary lever FedGRAA doesn't currently use. |
| **FedReLa** — Hu et al., "Imbalanced FL via Re-Labeling" (arXiv:[2606.26037](https://arxiv.org/abs/2606.26037)) | 2026 | Explicitly designed for the case where **global class distribution is agnostic/unknown** ("corrects biased global decision boundaries without requiring knowledge of the global class distribution"). A good citation for "even the very latest 2026 work treats not-knowing-the-distribution as the default assumption." |
| **FedCGNM** — Park, Klabjan, Braun, Li, Ananthanarayanan, ICML 2026 (arXiv:[2607.01474](https://arxiv.org/abs/2607.01474)) | 2026 | Pushes imbalance-correction **entirely to the client side** (a class-grouped, variance-normalized momentum optimizer run locally). The server needs **zero** knowledge of any client's distribution — the most privacy-conservative option in this family, since nothing new is even inferred, just optimized differently where the labels already live. Comes with a convergence proof under time-varying resampling rates, directly relevant to §5. |

**Verdict for FedGRAA:** This is the **most surgical fix** — it requires no new communication round, no cryptography, and replaces exactly one thing: the hard-coded "last `n` rare classes" becomes an **online-inferred** rare-class-and-weight signal computed from the client updates the server already receives in Algorithm 1, line 8 (`θ_k^t`). Caveat to disclose in the paper: this inference degrades under Secure Aggregation or DP noise (a real trade-off, discussed further in §7) and is only as accurate as the gradient-composition assumption (multi-layer, non-convex networks make the exact algebraic inversion approximate rather than exact — say so).

### Family 2 — Cryptographic (homomorphic-encryption / secure-MPC) imbalance measurement

Where Family 1 trades exactness for zero overhead, this family gives **provable** privacy at the cost of crypto compute.

| Paper | Venue/Year | Mechanism |
|---|---|---|
| **Fed-CBS** — Zhang, Li, Tang, Sun, Chen, Zhang, Chen, Chen, Li, ICML 2023 (arXiv:[2209.15245](https://arxiv.org/abs/2209.15245)) | 2023 | Defines an explicit **measure of class-imbalance** and computes it **via homomorphic encryption** so the server never sees anything but the encrypted measure; uses this to run **Federated Class-Balanced Sampling (Fed-CBS)** — actively selecting which clients participate each round so the *aggregate* cohort is class-balanced, with a theoretical guarantee. |
| **FLICKER** — Guleria, Harshan, Prasad, Bharath, CODS-COMAD 2024 (arXiv:[2410.21192](https://arxiv.org/abs/2410.21192)) | 2024 | Uses the **CKKS** homomorphic-encryption scheme so clients can privately share data-attribute summaries (not raw data) and pre-balance before FL, addressing **global** class imbalance specifically. |

**Verdict for FedGRAA:** Heavier (needs an HE library, adds latency), but this is the family to cite if a reviewer explicitly demands a **provable** (not just empirical) privacy guarantee for the "which classes are rare" signal — e.g., if targeting CCS/NDSS per the venue strategy in [literature_review_and_publication_strategy.md](literature_review_and_publication_strategy.md).

### Family 3 — Secure-aggregation / DP-based summary statistics (the "right amount" of machinery)

This is the **standard, minimal-assumption FL primitive** for exactly this need: the server wants one aggregate number (or small vector) it is allowed to know, but no individual client's contribution to it.

| Paper | Venue/Year | Mechanism |
|---|---|---|
| **Practical Secure Aggregation** — Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal, Seth (Google, NeurIPS-W 2016) (arXiv:[1611.04482](https://arxiv.org/abs/1611.04482)) | 2016 | The canonical **SecAgg** protocol: the server learns only `Σ_k x_k` for per-client vectors `x_k`, never any individual `x_k`, and tolerates up to 1/3 client dropout. **Direct application to FedGRAA:** have every client locally compute a length-`C` one-hot label-count vector for its local held-out split and submit it through SecAgg. The server recovers only the **round's aggregate class histogram** — exactly the "is class *c* rare this round" signal FedGRAA needs — with zero visibility into any single client's distribution. |
| Bharadwaj & Cormode — "Sample and Threshold Differential Privacy: Histograms and applications," AISTATS 2022 (arXiv:[2112.05693](https://arxiv.org/abs/2112.05693)) | 2022 | A **federated-analytics** DP mechanism purpose-built for histogram generation with a formal `(ε, δ)`-DP guarantee, without adding noise to *disclosed* data (uses subsampling instead) — directly reusable to publish a DP-safe class-histogram each round, with an explicit, statable privacy budget for the paper's privacy section. |
| **LDP-Fed** — Truex, Liu, Chow, Gursoy, Wei (arXiv:[2006.03637](https://arxiv.org/abs/2006.03637)) | 2020 | Local-DP variant: each client adds calibrated noise to its own histogram **before** sending, so privacy holds even if the aggregator itself is compromised — the strongest guarantee in this family, at the cost of noisier estimates for very rare classes (a real tension with FedGRAA's *raison d'être*, discuss explicitly). |

**Verdict for FedGRAA:** This is the family to **lead with** in the paper's redesigned methodology section — it is the most standard, most easily understood by FL reviewers, and it maps almost one-to-one onto what FedGRAA already needs (a per-round class-frequency signal to replace the hard-coded rare-class list and to compute the median-frequency split of §4 of the paper).

### Family 4 — Loss-only reweighting (no distributional signal at all)

The most privacy-conservative option: use only the **scalar loss value** each client already returns, never any class-level signal.

- **Fed-Focal Loss** — Sarkar, Narang, Rai, FL-IJCAI 2020 (arXiv:[2011.06283](https://arxiv.org/abs/2011.06283)): reshapes local cross-entropy via focal loss (down-weighting well-classified/majority examples) plus a tunable client-contribution sampling scheme; **> 9%** absolute gain on imbalanced MNIST.
- **q-FedAvg** (Li et al., ICLR 2020) and **AFL** (Mohri et al., ICML 2019) — already cataloged in [../../understanding/10_SOTA_adaptive_aggregation.md](../../understanding/10_SOTA_adaptive_aggregation.md); reweight purely by scalar per-client loss `L_k(θ)`.
- **FedNolowe** (Le et al., 2025, bioRxiv) — same family, already cataloged in `research_followup.md` §1.4.

**Verdict for FedGRAA:** Good as an **ablation/baseline**, not as the primary mechanism — a "hard client" (high loss) is not always a "rare-class client," so this family answers a related but distinct question. Include as a comparison point (a client-loss-only variant of FedGRAA, γ-free) to show FedGRAA's class-aware signal beats a class-agnostic one.

### Family 5 — Clustering: sidestep needing a global notion of "rare" entirely

Rather than trying to *learn* which classes are rare, group clients by similarity and let each cluster define its own notion of majority/minority.

| Paper | Venue/Year | Notes |
|---|---|---|
| **IFCA** — Ghosh, Chung, Yin, Ramchandran, NeurIPS 2020 (arXiv:[2006.04088](https://arxiv.org/abs/2006.04088)) | 2020 | Alternates estimating each client's **cluster membership** (by which cluster's current model gives lowest *local* loss — no data leaves the client) and gradient-descent optimization per cluster. Convergence proof included. |
| **Byzantine-Robust Clustered FL** — Tao, Yang, Kulkarni (arXiv:[2306.00638](https://arxiv.org/abs/2306.00638)) | 2023 | Combines IFCA with coordinate-wise trimmed-mean/median aggregation — relevant if FedGRAA wants a robustness story alongside the imbalance story. |
| **EMD-CFL** — Zhang, Williams, Toni (arXiv:[2506.07769](https://arxiv.org/abs/2506.07769)) | 2025 | One-shot clustering via **Earth Mover's Distance** between client embedding distributions. |
| **Clust-PSI-PFL** (arXiv:[2512.20363](https://arxiv.org/abs/2512.20363)) | 2026 | Clusters via a **Population Stability Index**, claimed more informative than Hellinger/Jensen–Shannon/EMD for this purpose; 18% accuracy gain, 37% fairness gain over baselines. |

**Verdict for FedGRAA:** Directly resolves the awkward **gap M-4** already flagged in [full_reproduction_gap_analysis.md](../archive/full_reproduction_gap_analysis.md) (for `n=7` on UNSW, the current rare-class list absurdly includes a 10,000-sample class, "Fuzzers," because someone had to hand-pick n classes). Clustering removes the need to hand-pick *any* fixed-size rare set — clusters emerge from the data itself. Recommended as a **secondary experiment** ("FedGRAA + clustering") rather than a replacement for the core method, since combining it is a natural, reviewer-pleasing extension (ties into the "few-shot / limited-data" positioning already suggested in `literature_review_and_publication_strategy.md` Part 6).

### Family 6 — Heterogeneity-quantification metrics (for reporting/diagnosis, not for training)

These don't solve the training-loop problem, but they let the paper **formally characterize how imbalanced/non-IID the partition really is**, which the current paper never quantifies (a real, separate gap: the paper *asserts* "extreme non-IID" but never measures it).

- Zhao, Li, Lai, Suda, Civin, Chandra — "Federated Learning with Non-IID Data" (arXiv:[1806.00582](https://arxiv.org/abs/1806.00582), already ref [13] in the paper) — introduces the **Earth Mover's Distance (EMD)** between a client's local class distribution and the population distribution, and shows accuracy degradation is up to **55%** and *explained by* this EMD (note: this is likely the actual origin of the paper's own "outperforming FedAvg by up to 55%" framing — worth an explicit citation tie-back).
- Xiao & Wang, IEEE SSCI 2021 (arXiv:[2109.04094](https://arxiv.org/abs/2109.04094)) — proposes **two new metrics**: `MID` (global class-imbalance degree) and `WCS` (local difference of imbalance among clients), specifically because "class imbalance needs to be re-defined in distributed learning environments." Recommended addition to FedGRAA's experiments section: report `MID`/`WCS` (or EMD) for the CIC-IDS-2017 and UNSW-NB15 partitions actually used, quantitatively justifying "this is an extreme non-IID regime" instead of asserting it qualitatively.

### Family 7 — Restricted-softmax, masked-logit, and prototype-based distribution inference (from the second literature pass)

A seventh family, surfaced by the `../archive/sagitra_report.md` sweep and largely complementary to Families 1–3, infers distribution skew from **structured model artifacts** (restricted output layers, class prototypes) rather than raw gradients or cryptographic sums:

| Method | Mechanism | Verification status |
|---|---|---|
| **FedRS** (Li & Zhan, KDD 2021) | Restricts local softmax to only the classes a client actually observes; the *shape* of the restriction (which logits are suppressed) lets the server infer label-presence/absence per client without seeing any label | **[unverified, re-checked 2026-07-09]** — still not located via live `arxiv.org/search` under "FedRS", "restricted softmax", or "label distribution non-IID" queries. KDD proceedings papers are frequently ACM-DL-only with no arXiv mirror, so absence from arXiv is not itself evidence the paper doesn't exist — it is independently well-attested by secondary citation across the FL literature (multiple unrelated survey papers cite "Li & Zhan, FedRS, KDD 2021" consistently). Treat as **plausible but not independently primary-source-confirmed**; verify against `dl.acm.org` before citing in the submitted paper. |
| **FedLMD** — label-masking distillation (Lu et al., ACM MM 2023) | Clients mask logits for missing classes during distillation; the mask pattern is itself a (coarse, privacy-safe) proxy for local class presence | **[verified 2026-07-09]** — confirmed via live arXiv search: **"Federated Learning with Label-Masking Distillation,"** Jianghu Lu, Shikun Li, Kexin Bao, Pengju Wang, Zhenxing Qian, Shiming Ge, arXiv:[2409.13136](https://arxiv.org/abs/2409.13136), explicitly annotated "Accepted by ACM MM 2023" — matches the reported venue/year exactly. Safe to cite directly with this arXiv ID. |
| **FedCK** (Chen et al., 2025) | Clusters clients by **model-weight similarity** as a proxy for label-distribution similarity — no raw statistics exchanged at all, only the models already being aggregated | **[unverified, re-checked 2026-07-09]** — no matching result under "FedCK", "federated clustering label skew", or "federated knowledge distillation clustering" queries. Mechanism remains a plausible, straightforward extension of Family 5 (clustering) using weights instead of embeddings, but do not cite without locating the primary source. |
| **ProtoHAR** (Cheng et al., IEEE JBHI 2023) | Clients share **class prototypes** (mean feature vectors per class); server infers imbalance from which prototypes are present, weak, or entirely absent | **[unverified, re-checked 2026-07-09]** — no matching result under "ProtoHAR" on arXiv (IEEE JBHI is not an arXiv-mirrored venue for most submissions). Prototype-sharing FL as a *paradigm* is well-established and independently verifiable via FedProto (Tan et al., AAAI 2022), which should be cited instead if a primary source is needed and ProtoHAR cannot be confirmed against IEEE Xplore directly. |
| **PROTEAN** — Chennoufi, Han, Blanc, De Cristofaro, Kiennert, ESORICS 2025 | **Confirmed real, arXiv:[2507.05524](https://arxiv.org/abs/2507.05524).** A federated **intrusion-detection** framework (IIoT + 5G datasets) that shares class prototypes of different attack types across organizations specifically to handle "highly non-IID attack distributions" — i.e., the *same problem domain as FedGRAA*, solved via prototype exchange instead of a server-held `D_V`. **This is the strongest same-domain precedent found in either literature pass and should be added to Related Work regardless of which redesign path is chosen.** |
| Data-Free Client Contribution Estimation via Logit Maximization (2026) | Estimates class-wise client contribution purely from **logit analysis** on the server, no auxiliary data of any kind | **[unverified]** exact paper on arXiv; conceptually aligned with Family 1's gradient/logit-inference logic |

**Verdict:** Family 7 is a useful **complementary signal**, not a replacement for the Family 1/3 combination recommended in §4 — prototype- and restricted-softmax-based methods generally require an architectural change (clients must compute and share prototypes, or restrict their output layer), whereas Family 1/3 require zero architectural change to FedGRAA's existing MLP. The one unambiguously strong addition from this family is **PROTEAN** (verified, same-domain), which should be cited in Related Work as a contemporaneous alternative design point: *prototype exchange instead of aggregation-weight learning* as a different way to solve the identical non-IID-IDS problem.

### Summary comparison table

| Family | Server learns | Extra communication | Extra crypto | Privacy strength | Best fit for FedGRAA |
|---|---|---|---|---|---|
| 1. Gradient composition inference | Approximate per-round class histogram | None (reuses `θ_k^t`) | None | Heuristic (weak under SecAgg/DP) | **Primary mechanism** |
| 2. HE-based measurement (Fed-CBS, FLICKER) | Encrypted imbalance measure | Low | High (HE ops) | Provable | Use if reviewers demand formal guarantees |
| 3. SecAgg/DP histograms | Exact/DP-noised aggregate histogram | Low (1 extra SecAgg round) | Medium (SecAgg protocol) | Provable (SecAgg) / Formal DP | **Co-primary mechanism** |
| 4. Loss-only reweighting | Nothing beyond scalar loss | None | None | Strongest (no new signal) | Ablation baseline |
| 5. Clustering (IFCA family) | Cluster IDs only | None | None | Strong | Secondary experiment |
| 6. Heterogeneity metrics | N/A (offline reporting) | N/A | N/A | N/A | Paper's experiments section |
| 7. Restricted-softmax / prototype inference | Prototype presence/absence or masked logits | Low–medium (prototypes) | None | Strong (aggregated, no raw labels) | Related-work positioning (PROTEAN); optional extension |

---

## 3. Taxonomy: validating without a server-side labelled set

### 3.1 The uncomfortable fact worth stating explicitly in the paper

FedGRAA's `D_V` assumption is **structurally identical** to the "root dataset" assumption used by Byzantine-robust FL defenses like **FLTrust** (arXiv:[2012.13995](https://arxiv.org/abs/2012.13995)). The literature has already had this exact argument, publicly, in a citable form:

> **FedTruth** (arXiv:[2311.10248](https://arxiv.org/abs/2311.10248)): *"existing defenses either require a benign root dataset on the server (e.g., FLTrust) or simply use trimmed mean or median as the threshold for clipping (e.g., FLAME). However, such benign root datasets are impractical, and the trimmed mean or median may also eliminate contributions from these underrepresented datasets."*

Notably, **recent (2025) work still makes this same assumption without apology** — e.g., **FedGreed** (arXiv:[2508.18060](https://arxiv.org/abs/2508.18060)) explicitly assumes "the central server is trusted and equipped with a reference dataset." This tells us two things useful for framing the paper: (1) FedGRAA is not alone in this weakness — it is a known, recognized, still-common shortcut; (2) the field also has active, well-cited fixes we should adopt rather than merely disclose the limitation.

### 3.2 Approach A — Federated (distributed) evaluation: the standard, zero-new-machinery answer

This is literally how the original FedAvg paper validates: **McMahan, Moore, Ramage, Hampson, AISTATS 2017** broadcasts the current global model to (a sample of) clients, has each client compute metrics on its own local held-out data, and aggregates only the **scalar metrics** on the server. No raw data, no labels, ever leave a device.

**Key mathematical point (not previously spelled out in the project's docs, worth stating precisely in the paper):** *confusion-matrix counts are additive.* If each client computes its local confusion matrix on its own held-out split and the server sums the matrices cell-by-cell, the result is **exactly** the same confusion matrix — and hence exactly the same accuracy, precision, recall, F1, and macro-F1 — as if all validation data had been pooled centrally. There is **no approximation error** from federating this particular computation, only communication of `C × C` integers per client per round (trivial compared to model parameters). This gives FedGRAA a mathematically lossless, fully FL-compliant replacement for its rare-class accuracy metric.

**Concrete precedent:** **FedDropoutAvg** (Gunesli et al., 2021, arXiv:[2111.13230](https://arxiv.org/abs/2111.13230)) evaluates on **held-out test sets from centers used in training** *and* on **entirely unseen centers**, both without centralizing data — directly modelable for FedGRAA's own "in-distribution vs. cross-client generalization" story.

### 3.3 Approach B — Federate the FedGRAA *gradient* itself (recommended: preserves the paper's exact math)

FedGRAA's actual weight-update signal is **not** just an accuracy number — it's `∂L'/∂w_k = ⟨∇_P L', θ_k⟩`, a gradient of a composite loss with respect to the *aggregated* parameters `P`. Federated evaluation (Approach A) alone gives you scalars, which is enough for a *loss-based* reweighting scheme (Family 4 in §2) but is **not enough**, by itself, to reproduce Algorithm 2's actual gradient-descent-on-`w` mechanism.

The fix that keeps FedGRAA's mathematics **byte-for-byte intact** while removing the privacy violation:

1. The server still forms the candidate aggregate `θ'_g = Σ_k w_k^{t-1} θ_k^t` (this is public math on already-shared parameters — no privacy issue).
2. The server broadcasts `θ'_g` to the (subset of) clients that hold validation data — exactly the same broadcast step already used for `ClientUpdate` in Algorithm 1.
3. **Each client**, using its own local held-out 10% split (which never leaves the device), computes the forward+backward pass of the composite loss terms relevant to its own data (i.e., its contribution to `L_h`, `L_m`, or `L_u` depending on which stratum its held-out classes fall into) **and returns only the gradient vector** `∇_P L_k'(θ'_g)` — the same size as the model, exactly analogous to the model deltas clients already return in Algorithm 1, line 17.
4. The server sums these (optionally through SecAgg, Family 3 of §2, so no individual client's gradient is visible even in the clear) to recover `∇_P L'(θ'_g) = Σ_k ∇_P L_k'(θ'_g)` **exactly**, then proceeds with Eq. 10 unchanged: `grad_w[k] = ⟨∇_P L', θ_k⟩`.

This is a **federated backward pass** — mechanically identical to what every client already does locally every round, just computed on a validation slice instead of a training slice, and it makes Algorithm 2 fully FL-compliant with **zero change to the underlying math** (Eq. 8–10 are untouched; only *where* `∇_P L'` is computed changes, from "on a server-held `D_V`" to "as a sum of client-computed gradients on client-held data"). This is precisely the "federated evaluation" mechanism recommended in `research_followup.md` §3.4, made concrete and precise here with the exact data flow and the SecAgg tie-in.

**Communication cost to disclose honestly:** this adds `E'` (5, per the paper's hyperparameters) extra broadcast/gradient-return round-trips *per FL round*, each the size of one model — a real, quantifiable overhead the paper must report (see §7).

### 3.4 Approach C — Secure multi-party computation to eliminate the root dataset outright (heaviest, most rigorous)

**SLVR — "Securely Leveraging Client Validation for Robust Federated Learning"** (Choi, Rachuri, Wang, Jha, Wang, 2025, arXiv:[2502.08055](https://arxiv.org/abs/2502.08055)) is the most directly on-point recent paper:

> *"Recent efforts attempt to address this tradeoff by enforcing checks on client updates using zero-knowledge proofs, but they support limited predicates and often depend on public validation data. We propose SLVR, a general framework that securely leverages clients' private data through secure multi-party computation. By utilizing clients' data, SLVR not only eliminates the need for public validation data, but also enables a wider range of checks for robustness, including cross-client accuracy validation."*

SLVR is built for Byzantine-robustness rather than class-imbalance-aware aggregation, but its **MPC evaluation primitive is directly reusable**: clients jointly and privately compute validation statistics (accuracy, cross-client checks) via secret-sharing, so the server *never* sees even an aggregate gradient in the clear, only the final scalar/vector result. This is strictly stronger (and heavier) than Approach B's SecAgg-wrapped gradient sum. Cite as the "if a reviewer wants maximal rigor" option; recommend Approach B as the pragmatic default given FedGRAA's existing NumPy/Flower-shaped implementation.

Related, lighter-weight variants worth knowing about (same "remove the trusted root dataset" goal, applied to Byzantine-robust FL specifically, transferable to FedGRAA's validation need):
- **LoByITFL** (Xia, Egger, Hofmeister, Bitar, 2024, arXiv:[2405.19217](https://arxiv.org/abs/2405.19217)) — replaces FLTrust's server root dataset with a **one-time trusted third party at initialization only**, then runs privately ever after.
- **ByITFL** (Xia et al., 2024, arXiv:[2405.08698](https://arxiv.org/abs/2405.08698)) — full information-theoretic-private Byzantine resilience without any root dataset, via Lagrange coded computing + verifiable secret sharing.

### 3.5 Approach D — Generic robust aggregation that needs no validation set at all

A different question worth asking explicitly in the paper: **does FedGRAA need a learned, task-specific weight vector at all**, or would a validation-free robust-statistics aggregator (no `D_V`, no gradient-of-loss trick) already capture much of the benefit? This is a useful **null-model baseline**:

- **Geometric median / coordinate-wise trimmed mean / median** aggregation (Chen, Su, Xu, 2017, arXiv:[1705.05491](https://arxiv.org/abs/1705.05491); Yin et al. 2018) — needs zero validation data, is a pure statistic of the client updates themselves.
- **RAGA** (Zuo, Yan, Fan, Hu, Shan, Quek, Zhao, 2024, arXiv:[2403.13374](https://arxiv.org/abs/2403.13374)) — geometric-median aggregation with a convergence proof under **both** Byzantine attacks *and* data heterogeneity (non-IID), no validation set required — directly relevant as a "does simple robust statistics already fix some of what FedGRAA fixes?" ablation.
- **FedTruth** (arXiv:[2311.10248](https://arxiv.org/abs/2311.10248)) — dynamically-weighted "ground truth" estimate among all updates, explicitly **without** a benign root dataset and without assuming any particular data distribution.

**Verdict:** Include one of these (RAGA or FedTruth) as an additional baseline in the experiments table — if FedGRAA's minority-class gains vanish or shrink dramatically against this validation-free baseline, that is itself an important, honest finding to report; if FedGRAA still wins, it strengthens the paper's claim that *task-aware* (not just distribution-robust) weighting is necessary for the minority-class objective specifically.

### 3.6 Approach E — Metrics that are meaningful under imbalance (applies regardless of which validation approach is chosen)

Independent of *how* validation data is accessed, *what* is measured must be imbalance-aware. Already covered thoroughly in `research_followup.md` §3.2 (macro-F1, balanced accuracy, MCC, G-mean, per-class recall/precision) — this sweep found one additional, directly relevant example of a paper doing this correctly in the *same application domain*:

- **EdgeDetect** (Mohammad, 2026, arXiv:[2604.14663](https://arxiv.org/abs/2604.14663)) — a federated IDS on **CIC-IDS-2017** (same dataset as FedGRAA) reports **macro-F1** (97.9%) and a specific **minority-class F1** (0.95) under "5% poisoning attacks and severe imbalance," alongside Paillier-homomorphic-encryption-protected communication — an excellent same-domain precedent for how to report both a global metric and an explicit minority-class metric side by side, and a paper FedGRAA should now cite/compare against in Related Work (it is a 2026 federated-IDS paper on the identical dataset).

### 3.7 Validation-free *contribution/trust* assessment (from the second literature pass — a distinct but related sub-problem)

The `../archive/sagitra_report.md` sweep surfaced a cluster of papers on **validation-free contribution/trust assessment** — a closely related problem to FedGRAA's `D_V`-based weight learning (both ask "how good/trustworthy is client k's update?", just for different downstream purposes: fair credit-assignment vs. Byzantine defense vs. adaptive aggregation weight). The strongest of these was independently re-verified and is a direct, citable win for the exact critique in §3.1:

> **CoAst — "Validation-Free Contribution Assessment for Federated Learning based on Cross-Round Valuation"** (Wu, Zhang, Li, Xu, Zhong — arXiv:[2409.02495](https://arxiv.org/abs/2409.02495), 2024). **Confirmed real and directly on-point.** Verbatim from the abstract: *"Research works in this field can be divided into two directions based on whether a validation dataset is required. Validation-based methods need to use representative validation data to measure the model accuracy, which is **difficult to obtain in practical FL scenarios**."* CoAst's mechanism: (1) quantize model parameters and count only the most important weights, (2) score each client via a **cross-round valuation** — the similarity between a client's current local parameters and the *global* parameter's evolution over several **subsequent** communication rounds (i.e., "did this client's direction predict where the model was headed anyway?"). Reported to match validation-based methods' reliability while beating prior validation-free baselines. **This is an independent, second, arXiv-confirmed citation making exactly the same point as the FedTruth quote in §3.1** — reviewers should find it very difficult to argue the server-held-`D_V` assumption is acceptable when two unrelated groups of authors (FedTruth in the Byzantine-robustness literature, CoAst in the contribution-assessment literature) both single it out as "impractical"/"difficult to obtain."

Other entries from this cluster, **not** independently re-confirmed on arXiv under any query tried (flagged accordingly; include only with a caveat, or verify against the publisher's site before citing in the paper itself):

| Method (as reported) | Claimed mechanism | Verification status |
|---|---|---|
| PACT — "Passive Accuracy-Based Trust Metric for Malicious Client Detection" | Fully passive detector inferring trust from class-specific accuracy degradation, no probe/validation data | **[unverified, re-checked 2026-07-09]** — still not found on arXiv under "PACT" + federated + trust, or "passive accuracy trust" queries |
| CT-BRFL — client-guided trust | Explicitly argues server validation sets "contradict the privacy-preserving principle of FL"; proposes trust computed from client-to-client signals instead | **[unverified, re-checked 2026-07-09]** — still not found on arXiv under "client-guided trust" + federated + backdoor queries (closest match found was an unrelated ACSAC 2024 paper, "Adversarially Guided Stateful Defense Against Backdoor Attacks in Federated Deep Learning," arXiv:2410.11205, which is **not** the same paper and should not be conflated with CT-BRFL) |
| FedPRS — representation-synthesis contribution evaluation | Synthesizes representative embeddings from client models themselves instead of using a public validation set | **[unverified, re-checked 2026-07-09]** — still not found on arXiv under "representation synthesis" + federated + contribution queries |
| REVERB-FL | Uses a **tiny reserve set (~5% of a full validation set)** for server-side defense — a practical middle ground between "no data" and "full `D_V`" | **[verified 2026-07-09, but domain differs from what was implied]** — confirmed via live arXiv search: **"REVERB-FL: Server-Side Adversarial and Reserve-Enhanced Federated Learning for Robust Audio Classification,"** Sathwika Peechara, Rajeev Sahay, arXiv:[2512.13647](https://arxiv.org/abs/2512.13647), "Published in IEEE Transactions on Audio, Speech and Language Processing" — author names match the second-pass report ("Peechara & Sahay") exactly, confirming the paper is real, but its application domain is **audio classification**, not IDS or class-imbalance-aware validation generically. The *mechanism* (a small server-held reserve/adversarial validation set) is still transferable to FedGRAA as a low-effort interim step, but cite it honestly as a cross-domain mechanism precedent, not a same-domain one. |

**Practical takeaway:** lead the paper's citation of this critique with **FedTruth** (§3.1) and **CoAst** (both independently confirmed, both explicit), and mention PACT/CT-BRFL/FedPRS/REVERB-FL only as "also reported in this space" with the unverified caveat if space allows — do not present them as confirmed facts in the submitted paper without checking the publisher page directly first (ACM/IEEE Xplore, TrustCom/ACM TIST proceedings), since they could not be located via arXiv.

### Summary comparison table

| Approach | What the server ends up seeing | New assumption introduced | Preserves FedGRAA's exact math? | Overhead |
|---|---|---|---|---|
| A. Federated/distributed evaluation | Scalar metrics / summed confusion matrices | None beyond standard FL | Only for loss-based variants, not Eq. 8–10 as-is | Minimal |
| B. Federated gradient-of-validation-loss (recommended) | `∇_P L'` (sum of client gradients) | Clients compute one extra backward pass on local held-out data | **Yes — exact** | +1 broadcast/gradient round × `E'` per FL round |
| C. Secure MPC validation (SLVR et al.) | Only final scalar/vector result | Requires MPC infrastructure | Yes, with cryptographic overhead | High (MPC protocol cost) |
| D. Validation-free robust aggregation | Nothing (no `D_V` at all) | None | No — changes the mechanism entirely | Minimal (baseline only) |
| E. Imbalance-aware metrics | (orthogonal — applies to any of the above) | None | N/A | None |
| F. Validation-free contribution/trust (CoAst et al.) | Cross-round parameter similarity only | None beyond standard FL | No — different mechanism, but a strong citation for §3.1's argument | Minimal |

---

## 4. The recommended fix: FedGRAA-v2

Combining the highest-value items from each taxonomy (Family 1 + Family 3 from §2; Approach B from §3), here is a concrete, implementable redesign that keeps FedGRAA's intellectual contribution intact.

### 4.1 What changes vs. the original Algorithm 1 / Algorithm 2

| Original FedGRAA (paper) | FedGRAA-v2 (proposed) |
|---|---|
| Server hard-codes `n` rare classes by name before training starts | Server maintains an **online-inferred** per-class frequency estimate `\hat{p}_c^t`, updated every round from (a) the SecAgg-summed local label-count histograms of participating clients (§2, Family 3) and (b) a gradient-composition cross-check (§2, Family 1) as a privacy-degradation fallback when SecAgg/DP noise makes (a) too coarse |
| Server holds `D_V` (labelled) and computes `L_h, L_m, L_u` directly | Server never holds labelled data. It broadcasts the candidate `θ'_g` to clients; each client computes its local contribution to whichever loss stratum (`h`/`m`/`u`) its own held-out classes fall into (using the *online-inferred* frequency ranks, not a hard-coded list) and returns only `∇_P L_k'(θ'_g)` |
| `L' = (1−γ)L_a(θ'_g; D_V^h, D_V^m) + γ L_u(θ'_g; D_V^n)` computed centrally | `∇_P L' = Σ_k ∇_P L_k'(θ'_g)` computed via (optionally SecAgg-wrapped) summation of client-returned gradients — mathematically identical value, just computed as a distributed sum |
| Median-frequency split (`D_V^h` vs `D_V^m`) computed on the true global label counts | Computed on the **inferred/SecAgg-aggregated** histogram `\hat{p}_c^t` instead of ground truth — introduces small estimation noise, must be reported (see §7) |
| `w^{t-1} ← w^{t-1} − η'∇_{w^{t-1}} L'(...)` | **Unchanged** — Eq. 10 is applied verbatim to the federated-computed gradient |
| Convergence proof assumes access to true `L'` | Needs an added assumption bounding the estimation error between the true and inferred/federated `L'` (see §5) |

### 4.2 FedGRAA-v2 pseudocode (drop-in replacement for the paper's Algorithm 1 + Algorithm 2)

```
Server executes:
  initialize θ_g^0, w^0 (static reparam, Eq. 7)
  initialize class-frequency estimate p̂ ← uniform prior
  for round t = 1 .. T:
    S_t ← sample of K clients
    for k in S_t (parallel):
      θ_k^t ← ClientUpdate(k, θ_g^{t-1})           # unchanged, Algorithm 1 lines 13-18
      h_k ← client k's local label-count vector for its held-out split   # NEW, stays local until SecAgg
    p̂ ← SecAgg-sum({h_k}) normalized                # NEW: Family 3, §2 — server sees only the aggregate
    rare_classes_t ← BottomN(p̂, n)                  # NEW: replaces the hard-coded rare-class list
    θ'_g ← Σ_k w_k^{t-1} θ_k^t                       # unchanged
    broadcast θ'_g to clients holding validation data
    for k in S_t (parallel):
      g_k ← ∇_P L_k'(θ'_g; local held-out split, rare_classes_t, γ)   # NEW: federated backward pass
    ∇_P L' ← SecAgg-sum({g_k})                        # NEW: server never sees any individual g_k
    for e' = 1 .. E':
      w^{t-1} ← w^{t-1} − η' · project_grad_w(∇_P L', {θ_k^t})   # Eq. 10, unchanged
    w^t ← w^{t-1}; θ_g^t ← Σ_k w_k^t θ_k^t            # unchanged
  return θ_g^T
```

### 4.3 Why this specific combination (not one of the heavier crypto options) is the right default

- It changes **zero** of the paper's core equations (7)–(10) — the contribution (γ-tunable, gradient-refined adaptive averaging) is unchanged, only *where the numbers come from* changes. This is the least risky path for a paper that is close to submission-ready: it is a **methodology patch**, not a rewrite.
- It reuses machinery (SecAgg) that is the **de facto standard** privacy primitive in production FL systems (Google's Gboard FL, Apple's on-device FL), so reviewers will recognize it immediately and not treat it as an ad hoc bolt-on.
- It is implementable on top of the existing NumPy/Flower-shaped codebase (`fedgraa/`) with moderate effort: `trainer.py` needs a client-side gradient-return path (structurally identical to the existing `_client_update` code path), and `aggregation.py`'s `FedGRAA.update()` needs to accept an externally-supplied `∇_P L'` instead of computing it from a local `D_V` — a **contained, well-scoped code change**, not a rearchitecture.
- Fed-CBS-style homomorphic encryption (Family 2) and SLVR-style full MPC (Approach C) are the "maximal rigor" options to mention as **future work / an appendix ablation** if targeting CCS/NDSS, but they are heavier than needed for the AAAI/IJCAI/PAKDD-tier positioning already laid out in `literature_review_and_publication_strategy.md`.

---

## 5. Impact on the convergence proof

The paper's Appendix A.1 proof (Theorem 1 / Lemmas 1–2) currently assumes `L'` is directly observed and is convex/β-smooth. FedGRAA-v2 needs one additional, standard assumption plus a small modification to the theorem statement:

**New Assumption 5 (Bounded estimation error).** The federated/inferred composite loss gradient `\widehat{∇_P L'}` (from SecAgg-summed client gradients and inferred class frequencies) satisfies
```
E[ ‖ \widehat{∇_P L'} − ∇_P L' ‖² ] ≤ ζ²
```
for some `ζ ≥ 0` depending on (a) SecAgg dropout/quantization noise, (b) any DP noise added to the label-count histogram (Family 3, §2), and (c) sampling noise from each client's local held-out split being smaller than the full `D_V`. When `ζ = 0` (no noise, full participation), this reduces exactly to the original theorem.

This is the same style of modification used throughout the **federated-optimization-under-noise** literature (e.g., the DP/SecAgg convergence analyses cited in §2 Family 3, and standard stochastic-approximation arguments): Lemma 1's descent inequality becomes
```
E[L'(w^t)] ≤ L'(w^{t-1}) − (η' − β(η')²/2)‖∇L'(w^{t-1})‖² + η'ζ + O(η'²ζ²)
```
i.e., convergence to a **noise ball** of radius `O(ζ/η')` around `w*` rather than exact convergence — a standard, well-precedented, and honest weakening that a reviewer will accept far more readily than the current proof's silent assumption of a perfectly-known, perfectly-labelled `D_V`.

**Additional, independently valuable improvements to the proof** (carried over and consolidated from [../../understanding/08_answers_researched.md](../../understanding/08_answers_researched.md) Q10, since they compound naturally with the above):
1. Replace global convexity with the **Polyak–Łojasiewicz (PL) condition** (Karimi, Nutini, Schmidt, ECML-PKDD 2016) — makes the proof honest for the neural-network setting actually used.
2. Model **partial/stochastic client participation** per round (as in Li et al., "On the Convergence of FedAvg on Non-IID Data," ICLR 2020) rather than assuming the same fixed `K` clients every round.
3. State an explicit **convergence rate** (`O(1/T)` or `O(1/√T)`), not just a limit.
4. Either analyze the **Adam** update used in practice (both for local training and for the weight learner) or note explicitly that the proof covers a plain-GD variant and flag the Adam/GD proof mismatch (already gap **M-5** in [full_reproduction_gap_analysis.md](../archive/full_reproduction_gap_analysis.md)).

**Exact optimizer detail relevant to Assumption 5 (grounded in the live `fedgraa/model.py` / `aggregation.py` code, not paraphrased).** Both the local client updates and the FedGRAA weight learner use the *same* `Adam` class (`β1=0.9`, `β2=0.999`, `ε=1e-7` — Keras' defaults, per the module's own docstring "same update rule as Keras / Kingma & Ba 2014"), but with **different moment-persistence semantics** that any convergence-rate analysis (item 3 above) must account for separately: (a) each client's local optimizer (`FederatedTrainer._client_optimizers`, one `Adam` instance per client, keyed by `client_id`) is constructed **once** and its `(m, v, t)` moments persist across **all** `T=200` FL rounds, exactly matching Keras' behaviour when `model.fit()` is called repeatedly on the same compiled optimizer (this is the codebase's own "gap 2.2 fix"); (b) the FedGRAA weight learner's optimizer (`FedGRAA.optimizer`, a single `Adam(lr=eta_prime)` instance) instead calls `self.optimizer.reset()` at the **start of every** `FedGRAA.update()` call (the codebase's own "gap 2.1 fix"), so its `(m, v, t)` moments are wiped every round and each round's `E'=5`-epoch weight optimization is a fresh, independent Adam sub-problem with no cross-round momentum carry-over. Consequently, Assumption 5's noise ball `ζ` enters a **per-round-reset** Adam iteration for `w`, not a globally-warm one — so `ζ`'s effect on `w^t` does not compound via stale first/second-moment estimates the way it would for the (non-reset) client optimizers, which is a helpful simplification for the convergence analysis and should be stated explicitly rather than left for a reviewer to have to work out from the pseudocode alone.

---

## 6. Experimental plan: new baselines, ablations, and metrics

To make the redesign (and the paper generally) reviewer-proof, run and report:

### 6.1 New baselines (in priority order)

| # | Baseline | Why | Source |
|---|---|---|---|
| 1 | **FedGRAA-v2** (federated gradient, SecAgg histogram) vs. **FedGRAA-v1** (original, oracle `D_V`) | Directly shows the privacy fix costs little/no accuracy | §4 (this doc) |
| 2 | **AFL** (Mohri et al. 2019) and **q-FedAvg** (Li et al. 2020) | Already recommended as "must-have" in [../../understanding/10_SOTA_adaptive_aggregation.md](../../understanding/10_SOTA_adaptive_aggregation.md); no change needed, just execute |
| 3 | **Fed-Focal Loss** (arXiv:2011.06283) | Class-imbalance baseline needing *zero* distributional signal — cheapest possible comparison point |
| 4 | **RAGA** or **FedTruth** (geometric-median / validation-free robust aggregation) | Tests whether FedGRAA's task-aware weighting adds value *beyond* generic robustness (§3.5) |
| 5 | **Fed-CBS** (client-selection via HE-measured imbalance) | Tests the "prevent imbalance via selection" alternative to "correct imbalance via reweighting" |
| 6 | **IFCA-clustered FedGRAA** | Secondary experiment addressing gap M-4 (n=7 absurdity) by letting clusters emerge instead of hand-picking rare classes |
| 7 | **FedAPA** (arXiv:[2502.07456](https://arxiv.org/abs/2502.07456)) or **FedAWA** (arXiv:[2503.15842](https://arxiv.org/abs/2503.15842)) — both confirmed real, see §9.1 | Tests whether a *general-purpose* adaptive-aggregation SOTA method (not IDS/imbalance-specific) already captures most of FedGRAA's gain, sharpening the novelty claim either way |

### 6.2 New metrics to report (alongside existing accuracy tables)

- **Macro-F1**, **balanced accuracy**, **MCC** — computed via the additive confusion-matrix trick (§3.2) so they remain FL-compliant even during v1-vs-v2 comparison. Implementation note: `fedgraa/metrics.py` already exposes `per_class_metrics` and `classification_report`; extend these with a `confusion_matrix(params, X, y) -> np.ndarray` helper returning the raw `C×C` integer matrix so it can be summed across clients before macro-F1/MCC are derived from the pooled matrix — this keeps every derived statistic (accuracy, macro-F1, balanced accuracy, MCC, per-class recall) provably identical to the centralized computation, not merely close to it.
- **MID** / **WCS** (Xiao & Wang 2021, arXiv:2109.04094) or **EMD** (Zhao et al. 2018, arXiv:1806.00582) — to quantitatively characterize *how* non-IID the one-attack-per-client partition actually is, replacing the current purely qualitative "extreme non-IID" claim. Concretely, for CIC-IDS-2017 (`C=15`, `K=14` clients) each client's local label distribution is a near-degenerate two-point mass (its one attack class + the paired Benign rows), so the EMD between any client's local distribution and the pooled 15-class population distribution is close to its theoretical maximum for that budget — report the exact computed value per client (not just "extreme"), and repeat for the `α`-Dirichlet partition of §6.3 so the paper can show EMD *decreasing monotonically* as `α → ∞`, tying the heterogeneity metric directly to the experimental knob.
- **Communication overhead** (extra rounds/bytes from SecAgg + the federated-gradient exchange in Approach B) — see §7 for exact, codebase-grounded byte counts for both datasets actually used (not order-of-magnitude estimates).
- **Estimation error `ζ`** between the true (oracle) composite-loss gradient and the federated/inferred one, measured directly in simulation (where ground truth is available to the experiment harness, even though it would not be in a real deployment) — this quantifies exactly how much the Assumption 5 noise ball actually costs empirically. **Measurement protocol:** at every evaluation round, compute `∇_P L'` twice — once the "oracle" way (server pools all client held-out splits into a single in-memory `D_V`, exactly as FedGRAA-v1 does today) and once the "federated" way (sum of per-client `∇_P L_k'(θ'_g)`, optionally passed through a simulated SecAgg/DP noise layer) — then log `ζ_t = ‖\widehat{∇_P L'}_t − ∇_P L'_t‖₂` per round. Report the empirical distribution of `ζ_t` over all `T=200` rounds (median, 90th percentile, max) for at least three noise regimes: (a) `ζ=0` exact/no-noise sanity check (must match FedGRAA-v1 bit-for-bit, see the unit test in §11.2), (b) SecAgg-only (integer/fixed-point quantization noise from the masking protocol, no DP), (c) SecAgg+LDP-Fed histogram noise at `ε ∈ {1, 4, 10}` (three points on the privacy-utility curve, matching the `ε` sweep already standard in the DP-FL literature, e.g. Truex et al. 2020's own evaluation grid).
- **Client dropout robustness.** Re-run the FedGRAA-v2 experiment with `p_drop ∈ {0, 0.1, 0.3}` fraction of clients failing to respond in the SecAgg round (Bonawitz et al.'s protocol explicitly tolerates dropout up to 1/3 of participants) and report the accuracy/ζ degradation — a gap no version of the redesign has addressed until this pass and one every SecAgg-based deployment claim must characterize.

### 6.3 A more realistic non-IID partition (independent but compounding improvement)

Both `../../understanding/07_my_questions.md` and this sweep converge on the same point: "one attack type per client" is a synthetic, extreme partition. Consider adding a **Dirichlet-allocation non-IID partition** (parameterized by concentration `α`, the now-standard non-IID benchmark protocol used throughout the FL literature, e.g. in **FedDiverse**, arXiv:[2504.11216](https://arxiv.org/abs/2504.11216), and dozens of others) as a second experimental regime, showing FedGRAA's gains hold under a more realistic heterogeneity pattern where clients hold **mixtures** of attack types at varying concentrations, not a single pure type. Recommended `α` grid: `{0.1, 0.5, 1.0, 5.0, 100.0}` (from near-pathological to near-IID), holding `K` fixed at the dataset's native client count (14 for CIC-IDS-2017, 9 for UNSW-NB15) so the only variable that changes between regimes is the *shape* of each client's label distribution, not the number of clients.

### 6.4 Statistical rigor: seeds, significance testing, and reporting conventions

None of the paper's current tables report variance across random seeds — a real, separate gap from the two privacy violations this document was originally scoped around, but one a rigorous reviewer will flag in the same pass. Fix it alongside the FedGRAA-v2 rewrite:

1. **Run every configuration (method × dataset × `n_rare` × `γ`) across at least 10 independent seeds** (varying `cfg.seed`, which per `fedgraa/config.py` controls both the Glorot initialization of `init_params` and the client-update minibatch permutation `rng`), not the single seed implied by the current reproduction pipeline (`run_experiments.py`).
2. **Report mean ± standard deviation** (or a 95% bootstrap confidence interval, preferable for the small `K=9–14` sample regime) for every accuracy/macro-F1/ζ number in every table.
3. **Use a paired statistical test** — a paired `t`-test if the per-seed metric differences are approximately normal, otherwise a **Wilcoxon signed-rank test** (safer default given only ~10 paired samples) — to test FedGRAA-v2 vs. each baseline in §6.1, and report the exact `p`-value and effect size (Cohen's `d` or matched-pairs rank-biserial correlation), not just "FedGRAA-v2 wins."
4. **Apply a multiple-comparisons correction** (Holm–Bonferroni is simplest and sufficiently conservative) across the full baseline grid in §6.1, since 7+ pairwise comparisons against FedGRAA-v2 are being run per dataset — an uncorrected significance claim across that many comparisons is a classic, easily-caught reviewer objection.
5. **Fix and disclose the seed list** in an appendix/artifact (e.g. `seeds = [0, 1, ..., 9]`) so results are exactly reproducible, consistent with the project's existing "fidelity contract" framing in `fedgraa/config.py`'s own docstring ("Nothing here is tuned or altered relative to the paper").

---

## 7. Privacy-utility and communication trade-offs (what to report)

A reviewer will ask "how much does fixing this cost?" — the honest answer, computed directly from the model architecture, client counts, and hyperparameters already frozen in `fedgraa/config.py` and `fedgraa/model.py` (not order-of-magnitude guesses).

### 7.1 Exact model and message sizes

The network is `input → Dense(200, tanh) → Dense(100, tanh) → Dense(50, tanh) → Dense(n_classes, softmax)` for both datasets (`ModelConfig.hidden_sizes = (200, 100, 50)`), so the only per-dataset variables are `n_features` and `n_classes`. Counting `[W1,b1,W2,b2,W3,b3,W4,b4]` exactly:

| Dataset | `n_features` | `n_classes` | `K` (clients) | Parameters (`W`+`b`, all 4 layers) | Message size @ float32 |
|---|---|---|---|---|---|
| CIC-IDS-2017 | 78 | 15 | 14 | 15,800 + 20,100 + 5,050 + 765 = **41,715** | **166,860 B ≈ 162.95 KiB** |
| UNSW-NB15 | 43 | 10 | 9 | 8,800 + 20,100 + 5,050 + 510 = **34,460** | **137,840 B ≈ 134.61 KiB** |

(Per-layer breakdown, CIC-IDS-2017: `W1` 78×200=15,600 + `b1` 200 → 15,800; `W2` 200×100=20,000 + `b2` 100 → 20,100; `W3` 100×50=5,000 + `b3` 50 → 5,050; `W4` 50×15=750 + `b4` 15 → 765. UNSW-NB15 differs only in `W1` (43×200=8,600+200→8,800) and `W4` (50×10=500+10→510).)

### 7.2 Exact communication overhead of Approach B (federated gradient-of-validation-loss), for the full `T=200`-round run

Every one of FedGRAA's `T=200` rounds already requires, under plain Algorithm 1, one broadcast of `θ_g^{t-1}` and one parameter-return per participating client — call this the **baseline** traffic, unavoidable in *any* FedAvg-style FL system and already paid for by the original (non-privacy-compliant) paper. Approach B adds, **per round**, `E'=5` additional broadcast/gradient-return round-trips (Algorithm 2's inner loop, since `θ'_g` and hence `∇_P L_k'(θ'_g)` change every one of the `E'` weight-learner epochs — confirmed by `aggregation.py`'s `FedGRAA.update()`, which recomputes `aggregate(...)` and `_composite_grad_P(...)` inside the `for _ in range(self.weight_epochs)` loop). Structurally, **extra traffic = `E'` × baseline traffic**, so total training-run traffic becomes `(1 + E') × baseline = 6 × baseline`:

| Dataset | Baseline (Algorithm 1 only): `2·K·T` transfers | Baseline bytes | Extra (Approach B): `2·E'·K·T` transfers | Extra bytes | **Total bytes (`6×` baseline)** |
|---|---|---|---|---|---|
| CIC-IDS-2017 (`K=14`) | `2·14·200 = 5,600` | 5,600 × 166,860 B ≈ **0.934 GB** | `2·5·14·200 = 28,000` | 28,000 × 166,860 B ≈ **4.672 GB** | **≈ 5.61 GB** |
| UNSW-NB15 (`K=9`) | `2·9·200 = 3,600` | 3,600 × 137,840 B ≈ **0.496 GB** | `2·5·9·200 = 18,000` | 18,000 × 137,840 B ≈ **2.481 GB** | **≈ 2.98 GB** |

In words: making FedGRAA FL-compliant via Approach B **increases total federated-training network traffic by roughly 500% (a 6× multiplier)**, concentrated entirely in the weight-learner's inner loop. This is the real, disclosable cost of the recommended fix — a paper that reports "negligible overhead" without this computation will not survive a careful review; a paper that reports **exactly** these numbers, and shows the accuracy delta they buy, will.

### 7.3 The class-histogram signal (Family 3) is comparatively free

Each client's SecAgg-protected label-count vector has length `C` (`15` for CIC-IDS-2017, `10` for UNSW-NB15). At float32, that is `60 B` and `40 B` respectively — **~2,800× and ~3,400× smaller** than one model-sized message. Total histogram traffic over the whole run (`K·T` submissions): CIC-IDS-2017 `14×200×60 B = 168,000 B ≈ 164 KiB`; UNSW-NB15 `9×200×40 B = 72,000 B ≈ 70.3 KiB`. Both are more than **six orders of magnitude** smaller than the §7.2 gradient-exchange overhead — the honest conclusion is that **Family 3's histogram mechanism is essentially free; Approach B's federated gradient is where all the real communication cost lives**, and the paper's privacy-analysis section should say so explicitly rather than lumping both mechanisms into one vague "overhead" line.

### 7.4 SecAgg cryptographic overhead at FedGRAA's small `K`, quantified honestly

Bonawitz et al. (arXiv:1611.04482) report **1.73–1.98×** expansion factors, but that number is amortized over `2^{10}`–`2^{14}` (1,024–16,384) users — FedGRAA's `K∈{9,14}` is **two to three orders of magnitude smaller** than the regime SecAgg was designed for, so the fixed per-pair key-agreement cost dominates rather than amortizes away. Concretely, Bonawitz-style SecAgg requires each client to establish a pairwise masking secret with every other participant (a Diffie–Hellman key exchange, typically ~32 B for a Curve25519 public key, plus a small protocol overhead for the derived-mask commitment — call it **~100 B/peer** as a conservative round-number estimate, not a measured implementation figure):

| Dataset | Peers per client (`K−1`) | Masking bytes/client/round | Total masking traffic over `K·T` submissions | vs. histogram payload (§7.3) |
|---|---|---|---|---|
| CIC-IDS-2017 | 13 | 13×100 B = 1,300 B | 1,300×14×200 ≈ **3.64 MB** | **~21.7× larger** than the 164 KiB histogram itself |
| UNSW-NB15 | 8 | 8×100 B = 800 B | 800×9×200 ≈ **1.44 MB** | **~20.5× larger** than the 70.3 KiB histogram itself |

This is the honest, precise version of the caveat already flagged qualitatively in earlier passes: **at FedGRAA's small `K`, SecAgg's own protocol overhead exceeds the payload it is protecting by roughly 20×** — still utterly negligible in absolute terms (megabytes, not gigabytes) and dwarfed by §7.2's gradient-exchange traffic, but worth stating with the actual multiplier rather than a vague "relatively larger."

### 7.5 A worked numeric example: DP noise vs. FedGRAA's own rarest classes (the real tension, quantified)

Family 3's LDP-Fed option (arXiv:2006.03637) protects the class histogram with local differential privacy by having each client add Laplace noise, calibrated to the histogram's L1 sensitivity `Δ₁ = 2` (one relabelled record moves one unit of count out of one bin and into another), before submission: `b = Δ₁/ε`, giving per-client-per-bin noise `Laplace(0, b)`. After the server SecAgg-sums `K` independent client submissions for one class bin, the aggregate noise is the sum of `K` i.i.d. `Laplace(0,b)` variables, with standard deviation `σ = b·√(2K)`.

Applying this to FedGRAA's own configuration (`val_fraction = 0.10`, so a client's held-out split is ≈10% of its shard) and CIC-IDS-2017's three rarest classes (the exact `n=3` protected set from `config.py`'s `rare_label_order`):

| Class | `used_count` (Table 1) | ≈ held-out (`×0.10`) count | Aggregate DP noise σ, `ε=1` | σ, `ε=4` | σ, `ε=10` |
|---|---|---|---|---|---|
| Heartbleed | 11 | **≈ 1** | `2×√28 ≈ 10.58` | `0.5×√28 ≈ 2.65` | `0.2×√28 ≈ 1.06` |
| SQL Injection | 21 | **≈ 2** | 10.58 | 2.65 | 1.06 |
| Infiltration | 36 | **≈ 4** | 10.58 | 2.65 | 1.06 |
| *(for contrast)* DoS Hulk | 10,000 | ≈ 1,000 | 10.58 | 2.65 | 1.06 |

**The finding:** even at the *loosest* commonly-reported privacy budget (`ε=10`), the aggregate noise standard deviation (`≈1.06` counts) is **comparable to or larger than the true held-out count of every one of FedGRAA's three protected rare classes** (`≈1`, `≈2`, `≈4`) — the DP-noised histogram signal for exactly the classes FedGRAA exists to protect is dominated by noise, while for a majority class like DoS Hulk (`≈1,000` held-out samples) the identical absolute noise is under `0.11%` relative error and completely harmless. This is not a hypothetical tension; it is a **direct, quantified consequence of FedGRAA's own rare-class regime** (classes with `<40` total samples in a 78-feature, 15-class, 700K-row dataset) colliding with the standard local-DP histogram mechanism, and it is exactly the trade-off flagged qualitatively in Family 3's LDP-Fed entry (§2) — now given actual numbers. **Recommendation for the paper:** report this table (or its UNSW-NB15 analogue, now computed below) directly, and adopt the pragmatic resolution already implicit in §4.1's design — use the **SecAgg-only** (non-DP) aggregate histogram for the rare-class *ranking* signal (`BottomN(p̂, n)`, which only needs relative ordering, not exact counts, and for which SecAgg alone already gives a strong empirical privacy guarantee with zero noise), and reserve formal `(ε,δ)`-DP specifically for any histogram value that is *reported or logged* outside the training loop (e.g. in a paper's table or a monitoring dashboard), where exact counts genuinely leave the trust boundary.

**The UNSW-NB15 analogue (previously flagged as an open item — now computed exactly from `config.py`'s `UNSW_NB15.rare_label_order` and `used_counts`).** UNSW-NB15 has `K=9` clients (fewer than CIC-IDS-2017's 14), so the aggregate-noise formula `σ = (Δ₁/ε)·√(2K)` gives a **smaller** noise standard deviation at every `ε`: `σ = 2·√18 ≈ 8.49` at `ε=1`, `≈2.12` at `ε=4`, `≈0.85` at `ε=10` (vs. CIC-IDS-2017's `≈10.58 / 2.65 / 1.06` at the same three budgets, since CIC's `K=14` gives `√28` instead of `√18`). Applying this to the exact `n=3` protected set from `UNSW_NB15.rare_label_order[:3]`:

| Class | `used_count` (Table 2) | ≈ held-out (`×0.10`) count | Aggregate DP noise σ, `ε=1` | σ, `ε=4` | σ, `ε=10` | Relative error at `ε=10` |
|---|---|---|---|---|---|---|
| Worms | 174 | **≈ 17** | 8.49 | 2.12 | 0.85 | **≈ 5.0%** |
| Shellcode | 1,511 | **≈ 151** | 8.49 | 2.12 | 0.85 | ≈ 0.56% |
| Backdoor | 2,329 | **≈ 233** | 8.49 | 2.12 | 0.85 | ≈ 0.36% |
| *(for contrast)* DoS (capped) | 10,000 | ≈ 1,000 | 8.49 | 2.12 | 0.85 | ≈ 0.085% |

**This is a materially different, non-obvious finding compared to CIC-IDS-2017's version of the same table, and it is exactly the kind of dataset-dependent nuance an exhaustive pass is meant to surface:** on CIC-IDS-2017, the DP-noised histogram signal is *unusable* for the rarest protected classes even at the loosest commonly-reported budget (`ε=10` gives roughly 100% or worse relative error against Heartbleed's ≈1 held-out sample). On UNSW-NB15, by contrast, the *smallest* protected class (Worms, ≈17 held-out samples) is only moderately degraded at `ε=10` (**≈5% relative error** — usable, if noisy) and the two other protected classes (Shellcode, Backdoor) are barely affected at all (**well under 1%**). The reason is two-fold and worth stating explicitly in the paper: (i) UNSW-NB15's smaller client count `K=9` (vs. CIC's `K=14`) directly shrinks the aggregate noise via the `√(2K)` term, and (ii) UNSW-NB15's rarest attack class in absolute terms (Worms, 174 total samples) is still **~16× larger** than CIC-IDS-2017's rarest (Heartbleed, 11 total samples), so even before any DP noise is added, UNSW-NB15's held-out counts start from a much less fragile base. **Practical recommendation:** the paper should not present a single blanket "DP noise is incompatible with rare-class ranking" conclusion — the two datasets actually used in this project sit at genuinely different points on that trade-off curve, and reporting both tables side by side (rather than only the CIC-IDS-2017 one) makes a stronger, more precise, and more honest privacy-utility case than either table alone.

### 7.6 Summary trade-off table

| Dimension | Original FedGRAA | FedGRAA-v2 | Notes |
|---|---|---|---|
| Extra rounds of communication | 0 | `+1` broadcast + `+1` gradient-return per inner epoch, `×E'=5` per FL round | Exactly `5×` the per-round baseline traffic (§7.2) |
| Extra bytes over the full `T=200`-round run | 0 | **≈4.67 GB** (CIC-IDS-2017) / **≈2.48 GB** (UNSW-NB15) | Computed exactly from the codebase's own layer sizes (§7.1–7.2), not estimated |
| Histogram (Family 3) overhead | 0 | **≈164 KiB** (CIC) / **≈70 KiB** (UNSW) total | 6+ orders of magnitude smaller than the gradient overhead (§7.3) |
| Cryptographic (SecAgg) overhead | 0 | **≈3.64 MB** (CIC) / **≈1.44 MB** (UNSW) total masking traffic | ~20× the histogram payload itself at this small `K`, still negligible in absolute terms (§7.4) |
| Privacy guarantee strength | None (server literally holds labelled data) | Empirical (Family 1) up to formal DP/SecAgg (Family 3) depending on which sub-mechanism is used | State the exact guarantee chosen (§10.3's accounting), do not oversell |
| Accuracy cost | — (baseline) | Expected small degradation from Assumption 5's noise ball `ζ`; **must be measured empirically**, not assumed away (§6.2, §6.4) | Report explicitly, this is the honest "cost of compliance" the paper should own |

**Framing recommendation for the paper:** present this exactly as a **trade-off curve** (à la the "Recommended SOTA Baselines" already suggested in `../../understanding/10_SOTA_adaptive_aggregation.md`), not as a silent swap — reviewers reward papers that *measure* the cost of their own privacy fix rather than asserting it is free. With the numbers above in hand, the paper can say precisely: *"FedGRAA-v2 increases total federated-training communication by ≈6× (dominated entirely by the federated validation-gradient exchange, not by the histogram or cryptographic machinery, which are jointly responsible for well under 0.1% of the added traffic), in exchange for removing a server-side labelled-data assumption that the field (§1, §3.1) has repeatedly identified as impractical and non-compliant."*

---

## 8. How to reposition the paper's contribution

With FedGRAA-v2, the paper's contribution statement can be **strengthened**, not weakened, because it now answers a harder and more publication-relevant question:

> *Prior gradient-refined adaptive aggregation for FL-IDS assumed server-side knowledge of client class distributions and a server-held validation set (a privacy leak). We show that this assumption can be removed via (a) secure-aggregation/DP-based, privacy-preserving per-round class-frequency estimation, and (b) a federated backward pass that reproduces the exact composite-loss gradient needed for adaptive weight learning without centralizing any labelled data — at a measured, small accuracy cost — making FedGRAA the first fully FL-compliant gradient-refined adaptive-aggregation method evaluated on realistic intrusion-detection heterogeneity.*

This directly plugs the "Key Gap" already identified in [literature_review_and_publication_strategy.md](literature_review_and_publication_strategy.md) Part 4 (*"None of the A*/A venue papers directly propose a gradient-refined adaptive averaging mechanism specifically designed for IDS under heterogeneous data distributions"*) — and now *also* closes the privacy-compliance gap that a security-adjacent reviewer (CCS/NDSS/S&P track, or even AAAI/IJCAI privacy-aware reviewers) would otherwise flag as disqualifying.

---

## 9. Beyond the two core gaps: aggregation alternatives and broader research gaps

The second literature pass (`../archive/sagitra_report.md`) also surfaced material outside the two core privacy questions this document was originally scoped to. It is captured here, clearly separated from the Must/Should items in §13, because it answers a different question — "is FedGRAA's aggregation mechanism itself state-of-the-art, and what else is missing for a maximally strong submission?" — rather than the privacy-compliance question that is this document's primary subject.

### 9.1 Adaptive-aggregation alternatives not yet in the SOTA comparison

[../../understanding/10_SOTA_adaptive_aggregation.md](../../understanding/10_SOTA_adaptive_aggregation.md) already catalogs AFL, q-FedAvg, FedAdp, FedAtt, FedOpt/FedAdam, SCAFFOLD, FedNova, and pFedMe. The second pass surfaced additional, more recent adaptive-aggregation methods worth adding to that comparison table:

| Method | Venue/Year | Mechanism | Verification status |
|---|---|---|---|
| **FedAPA** — server-side gradient-based adaptive personalized aggregation | IJCAI 2025 (as reported) | Updates aggregation weights from the gradients of client-*personalized* models rather than a global validation loss | **[verified 2026-07-09]** — confirmed via live arXiv search: **"FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation for Federated Learning on Heterogeneous Data,"** Yuxia Sun, Aoxiang Sun, Siyi Pan, Zhixiao Fu, Jingcai Guo, arXiv:[2502.07456](https://arxiv.org/abs/2502.07456), submitted 11 Feb 2025. Title/mechanism match exactly; the IJCAI 2025 venue attribution should still be double-checked against the IJCAI proceedings (the arXiv listing itself does not carry a `Comments:` venue tag), but the paper's existence and mechanism are now primary-source-confirmed. |
| **FedAWA** — adaptive aggregation weights from client representation vectors | CVPR 2025 | Learns weights from client *representation vectors* instead of dataset size or validation loss | **[verified 2026-07-09]** — confirmed via live arXiv search: **"FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors,"** Changlong Shi, He Zhao, Bingjie Zhang, Mingyuan Zhou, Dandan Guo, Yi Chang, arXiv:[2503.15842](https://arxiv.org/abs/2503.15842), comment: "Accepted in CVPR 2025" — exact match on title, mechanism, and venue. |
| **FedSWA** — momentum-based stochastic controlled weight averaging | ICML 2025 | Adds momentum to weight averaging for better generalization under heterogeneity | **[verified 2026-07-09]** — confirmed via live arXiv search: **"FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging,"** Junkang Liu, Yuanyuan Liu, Fanhua Shang, Hongying Liu, Jin Liu, Wei Feng, arXiv:[2507.20016](https://arxiv.org/abs/2507.20016), journal ref: ICML 2025 — exact match on title, mechanism, and venue. |
| **FedDRL** — deep-RL-based adaptive aggregation | ICPP 2022 | Learns aggregation weights via a deep reinforcement-learning policy rather than gradient descent on a loss | **[verified 2026-07-09 — note: acronym collision]** — confirmed via live arXiv search: **"FedDRL: Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning,"** Nang Hung Nguyen, Phi Le Nguyen, Duc Long Nguyen, Trung Thanh Nguyen, Thuy Dung Nguyen, Huy Hieu Pham, Truong Thao Nguyen, arXiv:[2208.02442](https://arxiv.org/abs/2208.02442), "Accepted for presentation at the 51st International Conference on Parallel Processing" (= ICPP 2022) — exact match. **Caution:** the acronym "FedDRL" is also used by an unrelated paper, Leiming Chen et al., "FedDRL: A Trustworthy Federated Learning Model Fusion Method Based on Staged Reinforcement Learning," arXiv:[2307.13716](https://arxiv.org/abs/2307.13716) (2023/2024) — cite the Nguyen et al. ICPP 2022 paper specifically for the adaptive-aggregation baseline described here, and disambiguate explicitly in the paper's references if both are ever cited. |
| **FedMcon** — meta-controller for aggregation | 2025 | A meta-learned controller decides aggregation weights/hyperparameters, potentially auto-tuning FedGRAA's own `γ` | **[unverified, re-checked 2026-07-09]** — no matching result located on arXiv |
| **FedFMD** — fairness-driven aggregation via Mahalanobis distance | 2025 | Uses **Mahalanobis distance** between client updates and the global distribution to detect and compensate for shift, without any validation set | **[unverified, re-checked 2026-07-09]** — no matching result located on arXiv under "Mahalanobis distance federated aggregation fairness"; conceptually a validation-free robust-statistics method, so also relevant to §3.5's Approach D regardless of citation status |
| **FedCoef** — aggregation via coefficient of variation | 2024 | Weights clients by the statistical coefficient of variation of their updates — a simple, data-free dispersion measure | **[unverified, re-checked 2026-07-09]** — arXiv search for this query returned no extractable results in this pass; still unconfirmed |

**Recommendation:** treat this list as *candidate additional baselines* (folded into §6.1 item 7), not as confirmed prior art to cite without independent verification — several titles/venues could not be re-confirmed via arXiv in this pass and should be checked against IJCAI/CVPR/ICML proceedings directly before appearing in the submitted paper's reference list.

### 9.2 Broader gaps identified but outside this document's original two-question scope

These are real, legitimate gaps in FedGRAA — but they are **additive future-work directions**, not blockers for the specific privacy-compliance redesign in §4. Listed here for completeness and roadmap planning, grouped by theme (none independently re-verified via arXiv in this pass unless a link is given; treat unlinked entries as reported-but-unconfirmed):

- **Global vs. local imbalance are conflated.** FedGRAA's `γ` treats "imbalance" as one thing; the literature distinguishes *global* imbalance (the pooled dataset itself is skewed) from *local*/label-distribution skew (each client's slice is skewed differently). A paper explicitly targeting this distinction in FL-IDS was reported as addressing "multiple imbalances (size, heterogeneous, global)" (ICASSP 2025) — worth a targeted follow-up search before citing.
- **No few-shot / data-scarcity handling.** FedGRAA assumes each client has "enough" data to compute a meaningful local loss; real novel-attack scenarios may have only a handful of examples client-side. This connects directly to the "Family 1: Federated Few-Shot Learning" material already cataloged in [literature_review_and_publication_strategy.md](literature_review_and_publication_strategy.md) Part 6 (FedFSL-CFRD, Fed-Meta, DFML, FeCoGraph) — no new action needed beyond what that document already recommends.
- **No Byzantine/poisoning robustness story.** FedGRAA has no defense against malicious clients; IDS is an inherently adversarial domain where an attacker controlling a client has every incentive to poison the global model. §3.5 and §3.7 of this document already catalog several validation-free robust-aggregation methods (RAGA, FedTruth, geometric median) that could be composed with FedGRAA-v2 at low additional cost, since both problems (privacy-compliant validation and Byzantine robustness) point toward the same family of solutions (validation-free, statistics-of-updates-only methods). Recommend citing **Armadillo** (CCS 2025) and **FreqFed** (NDSS 2023) — both already cataloged with venue/year in [literature_review_and_publication_strategy.md](literature_review_and_publication_strategy.md) Part 2 — as the security-venue framing if this direction is pursued.
- **No formal differential-privacy integration.** Beyond the DP-histogram mechanisms already recommended in §2 Family 3 (for the imbalance-detection signal specifically), FedGRAA offers no end-to-end DP guarantee on the model updates themselves. This is a standard, well-trodden extension (DP-SGD locally, or a server-side DP mechanism on the aggregate) and is lower priority than the two core fixes in this document, since it is an orthogonal privacy dimension (protecting *individual training examples* rather than *the fact that a server holds any labelled data at all*).
- **Evaluation is limited to CIC-IDS-2017 / UNSW-NB15.** Both datasets are from 2017/2015; the field has moved toward more realistic, more recent benchmarks. If a full re-evaluation is in scope, **CICIoT2023** and an FL-specific benchmark such as **FLNET2023** would strengthen the empirical section considerably — but this is a large undertaking (new data pipeline, re-run all baselines) and should be sequenced *after*, not instead of, the privacy-compliance redesign in §4, per this document's core mandate.

---

## 10. Threat model and formal privacy accounting

Everything in §2–§7 assumes a particular set of actors and trust boundaries. A rigorous reviewer — especially at a security-adjacent venue — will ask "privacy against *whom*, exactly?" This section states the threat model explicitly, quantifies the formal privacy guarantee FedGRAA-v2 can and cannot claim, and lists residual attack surfaces the redesign does *not* close, so none of this is left implicit in the paper.

### 10.1 Actors and trust assumptions

| Actor | Assumed capability | Justification |
|---|---|---|
| **Server (aggregator)** | **Honest-but-curious (semi-honest):** follows the FedGRAA-v2 protocol exactly (correct SecAgg masking, correct aggregation arithmetic, no fabricated messages) but may try to infer private information from anything it legitimately receives | Standard assumption for SecAgg-based systems (Bonawitz et al., arXiv:1611.04482) and for essentially all production FL deployments (Google Gboard FL, Apple on-device FL). A fully **malicious** server (one that deviates from the protocol, e.g. sends different `θ_g^{t-1}` to different clients to break SecAgg's masking, or fabricates dropout claims) is explicitly **out of scope** — defending against that requires *verifiable* SecAgg (e.g., with zero-knowledge proofs of correct protocol execution), which is a substantially heavier mechanism than anything recommended in §4.3 and should be named as future work, not silently assumed away. |
| **Clients (participants)** | A threshold `f < K/3` may be **dropped-out** (network failure, offline device) — tolerated natively by Bonawitz-style SecAgg. Any client may independently be **honest-but-curious** (tries to learn about another client's data from what it legitimately receives, e.g. the broadcast `θ'_g`). **Active/Byzantine** clients (submitting poisoned updates to corrupt the global model) are **not defended against** by anything in §4 — this is the same, already-flagged gap in §9.2 | FedGRAA-v2's SecAgg/DP mechanisms address *privacy* leakage between clients and the server; they say nothing about *integrity* (a malicious client can still submit an arbitrary `θ_k^t` or `g_k`, and FedGRAA will faithfully aggregate it). Composing FedGRAA-v2 with a Byzantine-robust aggregator (RAGA, FedTruth — §3.5) is the natural, low-cost extension if this threat is in scope for the venue. |
| **Network eavesdropper** | Passive or active man-in-the-middle on the wire | Standard FL assumption: assumed defeated by transport-layer security (TLS/mTLS between clients and server); **not** a contribution of this redesign and should not be claimed as one. |

**The one privacy limit that cannot be engineered away.** If the server colludes with `K−1` of `K` clients (or, equivalently, all-but-one client drops out and the server has their true values from a prior round), it can trivially recover the last remaining client's true contribution by subtracting the known `K−1` values from the SecAgg-revealed sum. This is not a flaw specific to FedGRAA-v2 — it is an information-theoretic limit of **any** sum-based secure aggregation scheme (SecAgg, HE-based summation, or MPC) once fewer than 2 honest parties remain in the sum. State this explicitly in the paper's threat model rather than letting a reviewer discover it: FedGRAA-v2's privacy guarantee holds **only while at least 2 clients are both (a) honest and (b) not colluding with the server**, a standard caveat in the SecAgg literature that is nonetheless frequently left unstated in application papers that adopt SecAgg as a black box.

### 10.2 What each actor learns, formally, under FedGRAA-v2 vs. FedGRAA-v1

| Quantity | FedGRAA-v1 (original) | FedGRAA-v2 (redesign) |
|---|---|---|
| Client `k`'s raw labelled data | Never sent (already true in v1) | Never sent (unchanged) |
| Client `k`'s local model update `θ_k^t` | Sent in the clear (Algorithm 1, unchanged in both versions) | **Unchanged** — this channel is identical in v1 and v2, and remains the residual leakage surface discussed in §10.5 |
| Which classes are "rare" globally | Server is **told directly** (hard-coded `rare_label_order`) | Server learns only `p̂^t`, an **aggregate** (optionally DP-noised) histogram over **all** participating clients — no single client's local distribution is ever isolated |
| The composite validation loss `L'` and its gradient `∇_P L'` | Computed **directly** by the server on server-held `D_V` | Server learns only the **SecAgg-summed** `∇_P L' = Σ_k ∇_P L_k'(θ'_g)` — never an individual `g_k` |
| Server's total "extra" information relative to what standard FedAvg already reveals | **Unbounded** — a full labelled dataset | **Bounded by Assumption 5's `ζ`** (empirical/SecAgg-only) or by a stated `(ε,δ)` (if the LDP-Fed histogram variant of Family 3 is used) — see §10.3 |

### 10.3 Formal privacy accounting across the full `T=200`-round run

If the DP-noised histogram variant of Family 3 (LDP-Fed, arXiv:2006.03637) is used for the rare-class ranking signal (rather than the SecAgg-only variant recommended as the pragmatic default in §7.5), the per-round privacy loss **composes** across all `T=200` rounds, and the paper must report the **composed**, full-training `(ε, δ)`, not the per-round `ε_round` — reporting only the latter while running 200 rounds is a well-known, easily-caught privacy-accounting error.

- **Naive/basic composition** (Dwork & Roth 2014, Theorem 3.16): `ε_total = T · ε_round`. At `ε_round = 0.1`, `T = 200`: `ε_total = 20` — a privacy budget large enough to be nearly meaningless.
- **Advanced composition** (Dwork, Rothblum, Vadhan, FOCS 2010; restated as Dwork & Roth 2014, Theorem 3.20): for a target `δ' > 0`,
  ```
  ε_total ≈ ε_round · √(2T · ln(1/δ')) + T · ε_round · (e^{ε_round} − 1)
  ```
  Worked example at `ε_round = 0.1`, `T = 200`, `δ' = 10⁻⁵`: `√(2·200·ln(10⁵)) = √(400 · 11.51) = √4604 ≈ 67.85`; term 1 `= 0.1 × 67.85 ≈ 6.79`; term 2 `= 200 × 0.1 × (e^{0.1} − 1) ≈ 20 × 0.1052 ≈ 2.10`; **`ε_total ≈ 8.89` at `δ = 10⁻⁵`** — a moderate, paper-reportable budget, roughly `2.25×` tighter than naive composition in this regime.
- **Rényi Differential Privacy (RDP) composition** (Mironov, CSF 2017, arXiv:1702.07476; operationalized via the moments-accountant technique of Abadi et al., CCS 2016, arXiv:1607.00133) composes **additively in RDP order `α`** and converts back to `(ε, δ)` only once, at the end, typically yielding a **substantially tighter** bound than either composition theorem above, especially for the Gaussian-mechanism variant of the histogram release. **Recommendation:** use an off-the-shelf RDP accountant (the same technique underlying `tensorflow-privacy`'s and `opacus`'s accountants) rather than hand-deriving advanced composition, and report the *converted* `(ε, δ)` after all `T=200` rounds in the paper's privacy-analysis subsection.
- **No subsampling amplification applies to FedGRAA's current design.** Privacy amplification via subsampling (a client that participates only in a random subset of rounds gets a strictly better composed `ε`) is a standard trick in the DP-FL literature, but `fedgraa/trainer.py`'s `S_t` currently selects **all** `K` clients every round (no partial participation is implemented) — so every one of the `T=200` rounds counts fully against the composed budget above, with **no discount**. If partial participation is added as a future extension (§9.2's future-work list), the composed `ε_total` would improve and should be recomputed, not assumed to inherit the full-participation number above.

### 10.4 SecAgg's dropout/collusion threshold, stated precisely

Bonawitz et al.'s protocol (arXiv:1611.04482) is built on a `(t, K)`-Shamir secret-sharing scheme: as long as at least `t` of the `K` clients successfully complete the protocol (the rest may drop out due to network failure), the server can reconstruct the **sum** but not any individual **summand**. The threshold `t` is a **deployment-chosen parameter**, not a fixed law — the paper's own evaluation used `t` in the range of roughly half to two-thirds of `K`. For FedGRAA's already-small `K ∈ {9, 14}`, choosing `t = ⌈2K/3⌉` gives `t = 10` (CIC-IDS-2017, tolerating up to 4 dropouts) or `t = 6` (UNSW-NB15, tolerating up to 3 dropouts) — small absolute dropout tolerances that should be explicitly named in the paper rather than left as "SecAgg tolerates ~1/3 dropout" in the abstract, since at `K=9` that translates to a **tolerance of only 3 clients**, a materially different practical guarantee than at Google-Gboard-scale (`K≈10⁴`–`10⁶`).

### 10.5 Attack surface explicitly *not* addressed by FedGRAA-v2 (state this, do not let a reviewer discover it)

- **Gradient inversion / "Deep Leakage from Gradients"** (Zhu, Liu, Han, NeurIPS 2019, arXiv:[1906.08935](https://arxiv.org/abs/1906.08935)). FedGRAA-v2's SecAgg wrapping protects the **validation** gradient `g_k` introduced by Approach B (§3.3) — but it does **nothing** to protect the **training** update `θ_k^t` that every client already sends **in the clear** in Algorithm 1, completely unchanged by this redesign. DLG-style attacks reconstruct training examples pixel/token-accurately from exactly this kind of shared gradient/weight-delta. This is a **pre-existing** limitation of the original FedGRAA paper that this redesign does not claim to fix (it was never in scope — the two violations this document targets are the rare-class oracle and the server-held `D_V`, not gradient leakage of `θ_k^t`) — but the paper's threat-model section should say so explicitly rather than implying "FedGRAA-v2 is now fully private."
- **Malicious/Byzantine clients corrupting the global model.** Already flagged in §9.2 and §10.1 — composing with RAGA/FedTruth/geometric-median aggregation (§3.5) is the recommended low-cost extension if this is in scope.
- **A fully malicious (not just curious) server.** SecAgg's guarantees assume the server correctly executes the masking protocol; a server that deviates (e.g., sends inconsistent broadcasts to split clients into groups it can individually unmask, the classic "Sybil-and-split" attack on SecAgg) is not defended against by anything in §4. Defenses require *verifiable* SecAgg — out of scope, name as future work.
- **Side-channel leakage from `w^t` itself.** Even after adopting FedGRAA-v2, the **learned aggregation weights** `w^t` are a (small, `K`-dimensional) quantity derived in part from the (now privacy-protected) validation signal — but `w^t` is typically published/logged for reproducibility (as the project's own `TrainHistory.weights` already does in `trainer.py`). A large `w_k` for a known client `k` is itself a (weak, coarse) signal that client `k`'s data was informative/rare that round. This is a low-severity, previously-undocumented residual channel worth naming explicitly rather than silently logging `w^t` without comment.

---

## 11. Testing and validation plan for FedGRAA-v2

This section specifies the **test suite** FedGRAA-v2's implementation must pass before its results can be trusted — written as a testable specification (assertions, tolerances, and inputs), not as code, so it can be handed directly to whoever implements §4's `trainer.py`/`aggregation.py` changes. Every test below follows the project's existing convention in `PAKDD/tests/` (one file per module, `pytest`-style, fixtures from `conftest.py`) and should be added as `test_aggregation_v2.py`, `test_privacy.py`, and `test_trainer_v2.py` alongside the existing `test_aggregation.py`, `test_trainer.py`, etc.

### 11.1 Unit tests — confusion-matrix additivity (Approach A, §3.2)

1. **`test_confusion_matrix_additive_over_clients`**: generate a synthetic multi-class prediction/label array, split it into `K` arbitrary (possibly overlapping-in-class, unequal-size) shards, compute the confusion matrix on each shard independently and sum them cell-by-cell; assert the result is **exactly** (bit-for-bit, integer equality — no floating-point tolerance needed since confusion-matrix entries are integer counts) equal to the confusion matrix computed on the pooled data in one call. This is the core mathematical claim of §3.2 and must be a hard equality assertion, not an approximate one.
2. **`test_macro_f1_from_pooled_confusion_matrix_matches_centralized`**: derive macro-F1, balanced accuracy, and MCC from the pooled confusion matrix of test 1, and assert each equals (to floating-point tolerance `1e-12`, since these ratios are computed from exact integers) the same metrics computed directly by `sklearn.metrics` (or the existing `fedgraa/metrics.py` helpers) on the pooled data.
3. **`test_confusion_matrix_handles_missing_classes_per_client`**: a client whose shard contains none of some class `c` (the common case under FedGRAA's one-attack-per-client partition) must still produce a well-formed `C×C` matrix with all-zero rows/columns for `c`, summable with other clients' matrices without shape errors.

### 11.2 Unit tests — FedGRAA-v1 / FedGRAA-v2 equivalence in the zero-noise limit (the single most important test in this plan)

4. **`test_fedgraa_v2_matches_v1_at_zeta_zero`**: with SecAgg simulated as a plain (unmasked) sum, no DP noise added to the histogram, and full client participation (`p_drop = 0`), run one full round of FedGRAA-v1 (server-side, oracle `D_V`) and FedGRAA-v2 (federated gradient sum) from the **identical** `θ_g^{t-1}`, `w^{t-1}`, and validation data partitioned identically across clients vs. pooled server-side. Assert the two methods produce:
   - identical `∇_P L'` (element-wise, tolerance `1e-9`, since both are the same sum computed in a different order — floating-point summation order can introduce tiny non-associativity error, which is why the tolerance is not exactly `0`),
   - identical `grad_w` (same tolerance),
   - identical `w^t` after the Adam step and simplex projection (same tolerance).
   This is the empirical proof that FedGRAA-v2 is a **refactor**, not a **different algorithm**, in the idealized no-noise case — directly operationalizing the "when `ζ=0` this reduces exactly to the original theorem" claim in Assumption 5 (§5).
5. **`test_fedgraa_v2_multi_round_trajectory_matches_v1`**: extend test 4 to a short (`T=10`) multi-round run and assert the two methods' `TrainHistory.all_acc`, `last_n_acc`, and `composite_loss` sequences match within tolerance `1e-6` at every evaluation round — catching any accumulated floating-point or state-management divergence that a single-round test could miss (analogous in spirit to the existing `is_composite_loss_monotone` check in `TrainHistory`).

### 11.3 Unit tests — the new mechanisms in isolation

6. **`test_secagg_histogram_sum_correct_no_dropout`**: with a mock SecAgg (unmasked sum for test purposes — a real SecAgg implementation is out of scope for this project's NumPy-based simulation, but its *interface contract*, "server receives only `Σ_k h_k`," must be exactly honored by the mock), assert the server-recovered `p̂` equals the true pooled class-frequency vector to floating-point tolerance.
7. **`test_secagg_histogram_dropout_tolerance`**: simulate `p_drop ∈ {0, 0.1, 0.3}` (per §6.2's dropout-robustness experiment) and assert the aggregation still succeeds (returns a valid, correctly-renormalized probability vector over the *surviving* clients) whenever the number of surviving clients is `≥ t` (the configured Shamir threshold from §10.4), and raises a clear, explicit error (not a silent wrong answer) when surviving clients `< t`.
8. **`test_bottom_n_rare_classes_matches_hardcoded_list_at_convergence`**: after a full `T=200`-round run with the online-inferred `p̂^t`, assert `BottomN(p̂^T, n)` recovers the **same** rare-class set as the paper's hard-coded `rare_label_order[:n]` for at least `n ∈ {3, 5}` on both datasets — this is the regression test proving the online-inference mechanism (Family 1/3, §2) is a faithful, privacy-preserving replacement for the oracle it replaces, not merely a plausible-looking substitute.
9. **`test_dp_histogram_noise_matches_closed_form_std`**: with the LDP-Fed variant (§7.5) enabled at a fixed `ε`, run the noised-histogram mechanism `N ≥ 1,000` times on a fixed true histogram and assert the empirical standard deviation of the aggregate noise matches the closed-form `σ = (Δ₁/ε)·√(2K)` prediction from §7.5 within `10%` relative error — a statistical (not exact) assertion, since this test is inherently about a random mechanism; use a fixed `numpy.random.default_rng` seed for reproducibility and re-run at 3+ seeds in CI to catch flaky failures rather than trusting a single draw.
10. **`test_simplex_projection_unaffected_by_v2_refactor`**: the existing `_project_simplex` unit tests (already implicitly covered by `test_aggregation.py`'s gap-1.1 checks) must continue to pass unmodified after `FedGRAA.update()` is changed to accept an externally-supplied `∇_P L'` — a regression guard ensuring the v2 refactor does not silently reintroduce the pre-gap-1.1 non-simplex weight bug.

### 11.4 Integration tests — end-to-end training runs

11. **`test_v2_full_run_reaches_comparable_accuracy_to_v1`**: run both FedGRAA-v1 and FedGRAA-v2 for the full `T=200` rounds on a **held-out fixed seed** (not one used for the paper's reported results, to avoid seed-selection bias) on both CIC-IDS-2017 and UNSW-NB15, and assert `|final_all_acc(v2) − final_all_acc(v1)| < 0.02` (2 percentage points, a threshold to be tightened once real empirical numbers are available — this is a smoke test, not the final reported comparison) and that `final_last_n_acc(v2)` is not **catastrophically** worse than `final_last_n_acc(v1)` (e.g. `> 0.5×`), guarding against a regression where the rare-class objective silently stops working under the federated redesign.
12. **`test_v2_communication_byte_count_matches_analytical_formula`**: instrument the simulated client/server message-passing in the v2 trainer to count actual bytes transferred over one full `T=200`-round run, and assert the measured total matches the analytical formula of §7.2 (`(1+E')×baseline`) within `1%` — closing the loop between the paper's disclosed communication-overhead claim and what the reference implementation actually does, so the number in the paper is not just a hand-derived estimate but a **measured, tested** fact about the code.
13. **`test_dropout_robustness_end_to_end`** (operationalizes §6.2's client-dropout experiment): run the full v2 pipeline at `p_drop ∈ {0, 0.1, 0.3}` and assert training completes without error at every level and that accuracy degrades **monotonically** (or stays flat) as `p_drop` increases — a sanity check that dropout never *improves* results (which would indicate a bug in how dropped clients' weights are renormalized).

### 11.5 Statistical / reproducibility tests (operationalizing §6.4)

14. **`test_ten_seed_variance_reported`**: a "meta-test" over the experiment harness (`run_experiments.py`), not the library code — assert that the results artifact for every (method, dataset, `n_rare`, `γ`) configuration contains **at least 10** distinct seed entries before any comparison table is generated, failing loudly (not silently defaulting to `n=1`) if the seed list is incomplete.
15. **`test_wilcoxon_or_ttest_applied_before_significance_claim`**: assert the reporting pipeline raises an explicit error if a "method A beats method B" claim is emitted without an attached `p`-value and effect size computed via the paired test specified in §6.4 — turning the statistical-rigor requirement from a documentation aspiration into an enforced pipeline invariant.

### 11.6 Property-based / fuzz tests (recommended, not just example-based)

16. **`test_confusion_matrix_additivity_property`** (using `hypothesis` or equivalent): generate random partitions of random prediction/label arrays into `1 ≤ K ≤ 50` shards of random unequal sizes, and assert additivity (test 1's property) holds for **all** generated partitions, not just one hand-picked example — this is the kind of test that catches off-by-one and edge-case (empty-shard, single-class-shard) bugs that example-based tests miss.
17. **`test_dp_laplace_sensitivity_property`**: for randomly generated pairs of "neighboring" histograms (differing by exactly one record's label), assert the L1 distance between them is always exactly `Δ₁ = 2` (property underlying the §7.5 sensitivity calculation) — a direct, automated check of the assumption the entire DP-accounting section (§10.3) is built on, so a future change to the histogram construction can never silently invalidate the paper's stated privacy budget without a test failing.

---

## 12. Glossary of new notation

Every symbol introduced by this redesign that does not appear in the original paper, defined in one place for the paper's notation table:

| Symbol | Meaning | First introduced |
|---|---|---|
| `\hat{p}_c^t` (`p̂`) | Server's online-inferred (SecAgg-summed, optionally DP-noised) estimate of class `c`'s frequency at round `t`, replacing the hard-coded rare-class list | §4.1 |
| `h_k` | Client `k`'s local length-`C` label-count vector for its held-out validation split, submitted through SecAgg | §4.2 |
| `rare_classes_t` | `BottomN(p̂^t, n)` — the `n` least-frequent classes *as currently estimated*, replacing the paper's static `rare_label_order[:n]` | §4.1–4.2 |
| `g_k` | Client `k`'s returned gradient `∇_P L_k'(θ'_g)` of its local contribution to the composite loss, w.r.t. the aggregated parameters `P` | §3.3, §4.2 |
| `\widehat{∇_P L'}` | The federated/inferred estimate of the composite-loss gradient (SecAgg-sum of `{g_k}`), as opposed to the true `∇_P L'` a server-held `D_V` would compute exactly | §5 |
| `ζ` | Bound on `E[‖\widehat{∇_P L'} − ∇_P L'‖²]^{1/2}` — the "noise ball" radius introduced by Assumption 5, empirically measured per §6.2's protocol | §5, §6.2 |
| `Δ₁` | L1 sensitivity of the label-count histogram to one record's label changing (`= 2`) | §7.5 |
| `b` | Laplace-mechanism scale parameter, `b = Δ₁/ε`, for the LDP-Fed histogram variant | §7.5 |
| `t` (threshold) | The Shamir secret-sharing reconstruction threshold in Bonawitz-style SecAgg — the minimum number of surviving clients needed to recover the sum | §10.4 |
| `p_drop` | Fraction of clients simulated as dropping out of a given round, for the robustness experiments of §6.2 and the test in §11.4 | §6.2 |
| `α` (Dirichlet) | Concentration parameter of the optional Dirichlet-allocation non-IID partition (§6.3), controlling how mixed vs. pure each client's attack-type composition is | §6.3 |

---

## 13. Prioritized action checklist

1. **[Must, before any resubmission]** Rewrite Algorithm 1/2 and §4 of the paper per §4.1–4.2 above (FedGRAA-v2). This alone resolves both flagged violations.
2. **[Must]** Add Assumption 5 and the modified Theorem 1 statement (§5) to Appendix A.1.
3. **[Must]** Add a new "Privacy Analysis" subsection quantifying the trade-offs in §7 (exact communication byte counts, SecAgg overhead, the §7.5 DP-vs-rare-class worked example) and a "Threat Model" subsection per §10 (actors, what each learns, the `T=200`-round composed `(ε,δ)` of §10.3, and the explicit residual-risk list of §10.5).
4. **[Must]** Implement and pass the full test suite in §11 before reporting any FedGRAA-v2 number in the paper — in particular tests 4/5 (v1≡v2 at `ζ=0`) and test 12 (measured communication matches the analytical §7.2 formula) are non-negotiable prerequisites for trusting any subsequently-reported accuracy or overhead figure.
5. **[Should]** Implement FedGRAA-v2 in `fedgraa/` (extend `trainer.py` with a client-side validation-gradient return path; change `aggregation.py`'s `FedGRAA.update()` to accept an externally-supplied `∇_P L'`; add a `confusion_matrix()` helper to `metrics.py` per §6.2) — see §15 for the exact function-by-function mapping between this item and the code that exists today.
6. **[Should]** Run the baseline set in §6.1 (at minimum items 1–3) across the ≥10-seed protocol of §6.4 and report the comparison table with significance tests, not point estimates alone.
7. **[Should]** Add MID/WCS or EMD heterogeneity quantification (§6.2) to the experiments section, computed per-client and shown decreasing monotonically across the Dirichlet-`α` sweep (§6.3).
8. **[Nice to have]** Add the Dirichlet-partition secondary experiment (§6.3) and/or the IFCA-clustered variant (§6.1 item 6) as extensions.
9. **[Nice to have]** Cite **EdgeDetect** (arXiv:2604.14663), **PROTEAN** (arXiv:2507.05524), **CoAst** (arXiv:2409.02495), **FedLMD** (arXiv:2409.13136), and the class-imbalance survey (arXiv:2303.11673) explicitly in Related Work — all five are directly on-topic, independently arXiv-confirmed, and were not in the paper's current reference list.
10. **[Optional / future work]** Run one additional baseline from §9.1 (now-confirmed **FedAPA**, arXiv:2502.07456, or **FedAWA**, arXiv:2503.15842) to sharpen the novelty claim against general-purpose adaptive aggregation, and consider the Byzantine-robustness composition sketched in §9.2/§10.1 as a follow-up paper rather than a resubmission requirement.
11. **[Optional / future work]** If targeting a security-adjacent venue (CCS/NDSS/S&P), upgrade the SecAgg-only histogram mechanism to the full RDP-accounted LDP-Fed variant per §10.3, and add the verifiable-SecAgg / malicious-server defense named as out-of-scope in §10.1/§10.5 as an appendix ablation.

---

## 14. Full reference list

Entries from the first two passes were verified via the arXiv API on 2026-07-08. Entries newly confirmed or re-checked in the third pass were verified live on 2026-07-09 (see the revision note at the top of this document for the exact methodology and outcome count).

**Class imbalance detection without violating privacy**
- Zhang, Li, Qi, He. "A Survey on Class Imbalance in Federated Learning." arXiv:[2303.11673](https://arxiv.org/abs/2303.11673) (2023).
- Wang, Xu, Wang, Zhu. "Addressing Class Imbalance in Federated Learning." AAAI 2021. arXiv:[2008.06217](https://arxiv.org/abs/2008.06217).
- Yan, Li, Ota, Dong. "An Efficient Imbalance-Aware Federated Learning Approach for Wearable Healthcare with Autoregressive Ratio Observation" (FedImT). arXiv:[2310.14784](https://arxiv.org/abs/2310.14784) (2023).
- Yang, Wong, Zhu, Wang, Qian. "Federated learning with class imbalance reduction." arXiv:[2011.11266](https://arxiv.org/abs/2011.11266) (2020).
- Hu, Menéndez, Liu, Gong, Wang, Peng. "FedReLa: Imbalanced Federated Learning via Re-Labeling." arXiv:[2606.26037](https://arxiv.org/abs/2606.26037) (2026).
- Park, Klabjan, Braun, Li, Ananthanarayanan. "Class-Grouped Normalized Momentum and Faster Hyperparameter Exploration to Tackle Class Imbalance in Federated Learning" (FedCGNM/FedHOO). ICML 2026. arXiv:[2607.01474](https://arxiv.org/abs/2607.01474).
- Zhang, Li, Tang, Sun, Chen, Zhang, Chen, Chen, Li. "Fed-CBS: A Heterogeneity-Aware Client Sampling Mechanism for Federated Learning via Class-Imbalance Reduction." ICML 2023. arXiv:[2209.15245](https://arxiv.org/abs/2209.15245).
- Guleria, Harshan, Prasad, Bharath. "On Homomorphic Encryption Based Strategies for Class Imbalance in Federated Learning" (FLICKER). CODS-COMAD 2024. arXiv:[2410.21192](https://arxiv.org/abs/2410.21192).
- Bonawitz, Ivanov, Kreuter, Marcedone, McMahan, Patel, Ramage, Segal, Seth. "Practical Secure Aggregation for Federated Learning on User-Held Data." arXiv:[1611.04482](https://arxiv.org/abs/1611.04482) (2016).
- Bharadwaj, Cormode. "Sample and Threshold Differential Privacy: Histograms and applications." AISTATS 2022. arXiv:[2112.05693](https://arxiv.org/abs/2112.05693).
- Truex, Liu, Chow, Gursoy, Wei. "LDP-Fed: Federated Learning with Local Differential Privacy." arXiv:[2006.03637](https://arxiv.org/abs/2006.03637) (2020).
- Sarkar, Narang, Rai. "Fed-Focal Loss for imbalanced data classification in Federated Learning." FL-IJCAI 2020. arXiv:[2011.06283](https://arxiv.org/abs/2011.06283).
- Ghosh, Chung, Yin, Ramchandran. "An Efficient Framework for Clustered Federated Learning" (IFCA). NeurIPS 2020. arXiv:[2006.04088](https://arxiv.org/abs/2006.04088).
- Tao, Yang, Kulkarni. "Byzantine-Robust Clustered Federated Learning." arXiv:[2306.00638](https://arxiv.org/abs/2306.00638) (2023).
- Zhang, Williams, Toni. "Clustered Federated Learning via Embedding Distributions" (EMD-CFL). arXiv:[2506.07769](https://arxiv.org/abs/2506.07769) (2025).
- Jimenez-Gutierrez, Hassanzadeh, Solans, Elbamby, Kourtellis, Anagnostopoulos, Chatzigiannakis, Vitaletti. "Clust-PSI-PFL: A Population Stability Index Approach for Clustered Non-IID Personalized Federated Learning." IPDPS 2026. arXiv:[2512.20363](https://arxiv.org/abs/2512.20363).
- Zhao, Li, Lai, Suda, Civin, Chandra. "Federated Learning with Non-IID Data." arXiv:[1806.00582](https://arxiv.org/abs/1806.00582) (2018). *(already ref [13] in the paper)*
- Xiao, Wang. "An Experimental Study of Class Imbalance in Federated Learning." IEEE SSCI 2021. arXiv:[2109.04094](https://arxiv.org/abs/2109.04094).
- Németh, Fanì, Ng, Caputo, Lozano, Oliver, Quadrianto. "FedDiverse: Tackling Data Heterogeneity in Federated Learning with Diversity-Driven Client Selection." FLTA 2025. arXiv:[2504.11216](https://arxiv.org/abs/2504.11216).
- Chennoufi, Han, Blanc, De Cristofaro, Kiennert. "PROTEAN: Federated Intrusion Detection in Non-IID Environments through Prototype-Based Knowledge Sharing." ESORICS 2025. arXiv:[2507.05524](https://arxiv.org/abs/2507.05524). *(same-domain prototype-exchange alternative; see Family 7, §2)*

**Validation without a server-side labelled set**
- McMahan, Moore, Ramage, Hampson. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017. *(already ref [6] in the paper)*
- Cao, Fang, Liu, Gong. "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping." NDSS 2021. arXiv:[2012.13995](https://arxiv.org/abs/2012.13995).
- Ebron, Zhang, Yang. "Identifying the Truth of Global Model: A Generic Solution to Defend Against Byzantine and Backdoor Attacks in Federated Learning" (FedTruth). ACISP 2025. arXiv:[2311.10248](https://arxiv.org/abs/2311.10248).
- Kritharakis, Makris, Jakovetic, Tserpes. "FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning." arXiv:[2508.18060](https://arxiv.org/abs/2508.18060) (2025).
- Choi, Rachuri, Wang, Jha, Wang. "SLVR: Securely Leveraging Client Validation for Robust Federated Learning." arXiv:[2502.08055](https://arxiv.org/abs/2502.08055) (2025).
- Xia, Egger, Hofmeister, Bitar. "LoByITFL: Low Communication Secure and Private Federated Learning." arXiv:[2405.19217](https://arxiv.org/abs/2405.19217) (2024).
- Xia, Hofmeister, Egger, Bitar. "Byzantine-Resilient Secure Aggregation for Federated Learning Without Privacy Compromises" (ByITFL). arXiv:[2405.08698](https://arxiv.org/abs/2405.08698) (2024).
- Gunesli, Bilal, Raza, Rajpoot. "FedDropoutAvg: Generalizable federated learning for histopathology image classification." arXiv:[2111.13230](https://arxiv.org/abs/2111.13230) (2021).
- Chen, Su, Xu. "Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent." arXiv:[1705.05491](https://arxiv.org/abs/1705.05491) (2017).
- Zuo, Yan, Fan, Hu, Shan, Quek, Zhao. "Federated Learning Resilient to Byzantine Attacks and Data Heterogeneity" (RAGA). IEEE TMC 2025. arXiv:[2403.13374](https://arxiv.org/abs/2403.13374).
- Mohammad. "EdgeDetect: Importance-Aware Gradient Compression with Homomorphic Aggregation for Federated Intrusion Detection." arXiv:[2604.14663](https://arxiv.org/abs/2604.14663) (2026).
- Wu, Zhang, Li, Xu, Zhong. "CoAst: Validation-Free Contribution Assessment for Federated Learning based on Cross-Round Valuation." arXiv:[2409.02495](https://arxiv.org/abs/2409.02495) (2024).

**Threat model, privacy accounting, and testing-plan foundations (new in the third pass, §10–§11)**
- Zhu, Liu, Han. "Deep Leakage from Gradients." NeurIPS 2019. arXiv:[1906.08935](https://arxiv.org/abs/1906.08935) — verified 2026-07-09; grounds §10.5's residual gradient-inversion risk on the unchanged `θ_k^t` channel.
- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang. "Deep Learning with Differential Privacy." ACM CCS 2016. arXiv:[1607.00133](https://arxiv.org/abs/1607.00133) — verified 2026-07-09; the moments-accountant technique underlying §10.3's recommended RDP-based composition.
- Mironov. "Rényi Differential Privacy." IEEE CSF 2017. arXiv:[1702.07476](https://arxiv.org/abs/1702.07476) — verified 2026-07-09; the RDP composition rule cited in §10.3.
- Dwork, Rothblum, Vadhan. "Boosting and Differential Privacy." FOCS 2010. *(canonical source for the advanced-composition theorem used in §10.3's worked example; not on arXiv — cite via the FOCS 2010 proceedings or the restatement in Dwork & Roth 2014 below)*
- Dwork, Roth. "The Algorithmic Foundations of Differential Privacy." Foundations and Trends in Theoretical Computer Science, 2014. *(monograph; Theorem 3.8 grounds the Laplace-mechanism-for-histograms claim in §7.5, Theorems 3.16/3.20 ground the basic/advanced composition formulas in §10.3; not on arXiv, standard NOW Publishers monograph — no substitute citation needed, this is the canonical reference the field itself uses)*
- Kairouz, McMahan, Avent, Bellet, Bennis, Bhagoji, Bonawitz, et al. "Advances and Open Problems in Federated Learning." Foundations and Trends in Machine Learning, 2021. arXiv:[1912.04977](https://arxiv.org/abs/1912.04977) — verified 2026-07-09; standard FL threat-model taxonomy reference underpinning §10.1's actor definitions.

**Convergence-proof improvements**
- Karimi, Nutini, Schmidt. "Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak–Łojasiewicz Condition." ECML-PKDD 2016. *(already cited in `../../understanding/09_sources.md`)*
- Li, Sahu, Talwalkar, Smith. "On the Convergence of FedAvg on Non-IID Data." ICLR 2020.

**Already covered in depth elsewhere in this repo (not repeated here, see linked docs)**
- AFL (Mohri et al. 2019), q-FedAvg (Li et al. 2020), FedAdp (Wu & Wang 2022), FedAtt (Ji et al. 2019), FedOpt/FedAdam (Reddi et al. 2021), SCAFFOLD (Karimireddy et al. 2020), FedNova (Wang et al. 2020), pFedMe (Dinh et al. 2020) — see [../../understanding/10_SOTA_adaptive_aggregation.md](../../understanding/10_SOTA_adaptive_aggregation.md).
- Astraea (Duan et al., ICCD 2019, arXiv:[1907.01132](https://arxiv.org/abs/1907.01132)), FedNolowe, DFedReweighting — see [research_followup.md](research_followup.md) §1.
- FedMADE, SBFedNIDS, FD-IDS — see [literature_review_and_publication_strategy.md](literature_review_and_publication_strategy.md) Part 4 (additionally found in this pass, same family: **Sentinel**, personalized federated IDS with dual-model knowledge distillation, arXiv:[2510.23019](https://arxiv.org/abs/2510.23019), 2025; **F-ACVAE**, federated conditional VAE for IDS with constrained-momentum Gaussian aggregation, arXiv:[2607.04698](https://arxiv.org/abs/2607.04698), 2026 — both worth adding to Related Work as recent same-domain competitors).

**From the second literature pass (`../archive/sagitra_report.md`), independently arXiv-confirmed**
- Chennoufi, Han, Blanc, De Cristofaro, Kiennert. "PROTEAN" — arXiv:[2507.05524](https://arxiv.org/abs/2507.05524) (2025). *(Family 7, §2.3)*
- Wu, Zhang, Li, Xu, Zhong. "CoAst" — arXiv:[2409.02495](https://arxiv.org/abs/2409.02495) (2024). *(§3.7)*

**Confirmed in the third pass (2026-07-09) — moved out of the "reported but unverified" list below with corrected arXiv IDs, authors, and venues**
- Lu, Li, Bao, Wang, Qian, Ge. "Federated Learning with Label-Masking Distillation" (FedLMD). ACM MM 2023. arXiv:[2409.13136](https://arxiv.org/abs/2409.13136). §2 Family 7.
- Sun, Sun, Pan, Fu, Guo. "FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation for Federated Learning on Heterogeneous Data." arXiv:[2502.07456](https://arxiv.org/abs/2502.07456) (2025). §9.1, §6.1 item 7. *(IJCAI 2025 venue attribution not carried by the arXiv listing itself — confirm against the IJCAI proceedings before citing the venue specifically.)*
- Shi, Zhao, Zhang, Zhou, Guo, Chang. "FedAWA: Adaptive Optimization of Aggregation Weights in Federated Learning Using Client Vectors." CVPR 2025. arXiv:[2503.15842](https://arxiv.org/abs/2503.15842). §9.1, §6.1 item 7.
- Liu, Liu, Shang, Liu, Liu, Feng. "FedSWA: Improving Generalization in Federated Learning with Highly Heterogeneous Data via Momentum-Based Stochastic Controlled Weight Averaging." ICML 2025. arXiv:[2507.20016](https://arxiv.org/abs/2507.20016). §9.1.
- Nguyen, Nguyen, Nguyen, Nguyen, Nguyen, Pham, Nguyen. "FedDRL: Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning." ICPP 2022. arXiv:[2208.02442](https://arxiv.org/abs/2208.02442). §9.1. *(Disambiguation: do not confuse with the unrelated Chen, Zhang, Dong, Qiao, Huang, Nie, Hou, Tan paper that shares the same "FedDRL" acronym, arXiv:[2307.13716](https://arxiv.org/abs/2307.13716), "A Trustworthy Federated Learning Model Fusion Method Based on Staged Reinforcement Learning" — a Byzantine-robust model-fusion method, not an adaptive-aggregation-for-non-IID-data method.)*
- Peechara, Sahay. "REVERB-FL: Server-Side Adversarial and Reserve-Enhanced Federated Learning for Robust Audio Classification." IEEE Transactions on Audio, Speech and Language Processing. arXiv:[2512.13647](https://arxiv.org/abs/2512.13647) (2025/2026). §3.7. *(Real and author-match-confirmed, but the application domain is audio classification, not IDS or class-imbalance validation — cite the mechanism, not a same-domain precedent.)*

**Still [unverified] on arXiv after the third pass (2026-07-09) — cite only after confirming against the original venue (KDD/ACM MM/2025 preprint servers/TrustCom/ACM TIST proceedings, etc.)**
- Li & Zhan. "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data." KDD 2021 (as reported). §2 Family 7. *(Widely cross-cited by secondary FL-survey sources under this exact title/venue despite no arXiv mirror — plausible but not primary-source-confirmed in this pass; verify against `dl.acm.org` before citing.)*
- Chen et al. "FedCK" — clustering + knowledge distillation for label skew. 2025 (as reported). §2 Family 7.
- Cheng et al. "ProtoHAR" — prototype-guided personalized FL for HAR. IEEE JBHI 2023 (as reported). §2 Family 7.
- Sheykholeslamzadeh & Alhadidi. "PACT" — passive accuracy-based trust metric. TrustCom 2025 (as reported). §3.7.
- Wang et al. "CT-BRFL" — client-guided trust. 2024 (as reported). §3.7. *(Not to be confused with Ali, Nepal, Kanhere, Jha, "Adversarially Guided Stateful Defense Against Backdoor Attacks in Federated Deep Learning," ACSAC 2024, arXiv:[2410.11205](https://arxiv.org/abs/2410.11205) — a real, different paper surfaced by the same search query, verified 2026-07-09, but not a match for CT-BRFL's reported mechanism.)*
- Fan et al. "FedPRS" — privacy-preserving representation synthesis for contribution evaluation. ACM TIST 2025 (as reported). §3.7.
- Shen et al. "FedMcon" — meta-controller for aggregation. 2025 (as reported). §9.1.
- Weng et al. "FedFMD" — fairness-driven aggregation via Mahalanobis distance. CIKM 2025 (as reported). §9.1.
- Xie et al. "FedCoef" — aggregation via coefficient of variation. ASCC 2024 (as reported). §9.1.

*(All 9 entries in this list were independently re-queried a second time in the fourth pass, 2026-07-09, using both `ti:` and `all:` field-restricted live arXiv API queries — see the fourth-pass revision note at the top of this document for the exact query results. All 9 remain unverified after two independent search attempts across two passes.)*

---

## 15. Reference implementation mapping (FedGRAA-v2 ↔ existing `fedgraa/` code)

This section did not exist before the fourth pass. It exists because §13, item 5 ("Implement FedGRAA-v2 in `fedgraa/`") is otherwise only scoped at the file level, which is not precise enough to hand to an implementer without re-deriving exactly what already exists vs. what is genuinely new. Every fact below was obtained by reading the live source files in full (`config.py`, `model.py`, `aggregation.py`, `trainer.py`, `partition.py`, `metrics.py`), not by inference from the paper or from earlier passes of this document.

### 15.1 What already exists and can be reused as-is

| FedGRAA-v2 need (§4) | Existing code | Notes |
|---|---|---|
| A non-oracle, data-driven rare-class selection fallback | `config.py`, `DatasetSpec.rare_client_ids(n)`: when `rare_label_order` is empty, it already falls back to `attacks_by_size = sorted(self.attack_ids, key=lambda i: self.used_counts[i]); return attacks_by_size[:n]` | **Previously-unremarked fact:** this fallback is *already implemented* today, but neither `CIC_IDS_2017` nor `UNSW_NB15` (`config.py`) ever exercises it, because both `DatasetSpec` instances are constructed with a non-empty `rare_label_order` (the paper's hard-coded oracle list), and `run_experiments.py` calls `get_dataset_spec(name)`, which returns these pre-built instances unchanged. FedGRAA-v2 does **not** need to invent a new "pick the `n` smallest classes" mechanism — it needs to (a) stop hard-coding `rare_label_order`, and (b) feed `rare_client_ids()`'s size-sorting logic **online-inferred, SecAgg-aggregated counts** (`p̂^t`, §4.1) instead of the true, centrally-computed `used_counts`. This is a smaller, more surgical change than "add a new selection algorithm." |
| Simplex-projected weight updates | `aggregation.py`, `_project_simplex(w)` — the Duchi et al. (2008) `O(K log K)` Euclidean projection onto the probability simplex, already used inside `FedGRAA.update()` | No change needed; FedGRAA-v2's externally-supplied `∇_P L'` still flows through the identical `_grad_wrt_weights` → `Adam.step` → `_project_simplex` pipeline. |
| Per-round-reset Adam for the weight learner | `aggregation.py`, `FedGRAA.update()` calls `self.optimizer.reset()` before its `weight_epochs` loop | See the new paragraph in §5 — this reset semantics is unaffected by whether `∇_P L'` is computed centrally (v1) or as a federated sum (v2); the only change is the *source* of the value returned by `_composite_grad_P`. |
| Median-frequency validation split | `partition.py`, `median_frequency_split(X_val, y_val, rare_class_ids)` returns a `ValidationSplit` with `(X_h, y_h)`, `(X_m, y_m)`, `(X_n, y_n)` — computed today from the **true, centrally-pooled** `y_val` | FedGRAA-v2 needs this function's *median-over-remaining-classes* arithmetic unchanged; what must change is that the `counts` it implicitly operates on must come from the SecAgg-aggregated `p̂^t` rather than from a server that can see `y_val` directly, since in FedGRAA-v2 no server-pooled `y_val` should exist at all (§4.1's whole point). This is the single largest required refactor: `median_frequency_split` must be split into (a) a client-side per-client class-count/threshold computation that never leaves the device except through SecAgg, and (b) a server-side `X_h/X_m/X_n` *tagging* step that a client can locally apply to its own held-out rows once it knows the (privacy-safe) global median/rare-set. |
| Per-class precision/recall/F1 reporting | `metrics.py`, `per_class_metrics()` and `classification_report()` (the module's own "gap 3.6 fix") | Already implemented and directly reusable for §6.2's macro-F1/balanced-accuracy reporting once fed a *pooled* (summed) confusion matrix — see §15.2 for what is still missing. |
| Composite-loss monotonicity check (operationalizes Lemma 1) | `trainer.py`, `TrainHistory.composite_loss` (recorded every eval round) and `TrainHistory.is_composite_loss_monotone` (property) | Directly reusable, unmodified, as the assertion inside test 5 (§11.2) — no need to write a new monotonicity checker, the codebase already has one. |
| Disjoint-ish benign sample allocation per client | `partition.py`, `partition_non_iid()`'s benign allocation uses a single shuffled pool with **circular indexing** (`indices = np.arange(cursor, cursor + demand) % pool_size`), not independent per-client draws (the module's own "gap 2.3 fix") | Relevant to §6.3's Dirichlet-partition extension: note explicitly in the paper that even today's one-attack-per-client partition does **not** guarantee disjoint benign rows across clients whenever total attack-class demand exceeds the benign pool size — the pool wraps and reuses rows deterministically. This is a pre-existing, minor cross-client sample-reuse subtlety (the *same* benign row can appear in two different clients' shards) that is orthogonal to the two privacy violations this document targets, but should be disclosed alongside them since a careful reviewer auditing the partition code will find it. |
| Adam local-training optimizer state | `trainer.py`, `FederatedTrainer._client_optimizers: Dict[int, Adam]`, one persistent instance per `client_id` (the module's own "gap 2.2 fix") | Unaffected by the FedGRAA-v2 redesign; this is the client's own local-training optimizer, not the weight-learner optimizer, and nothing in §4 changes local training. |

### 15.2 What is genuinely new and must be written

| FedGRAA-v2 need (§4) | Status | Where it plugs in |
|---|---|---|
| SecAgg simulation (mock or real) | **New.** Nothing in `fedgraa/` today simulates any cryptographic protocol; all aggregation (`aggregate()` in `model.py`) is a plain, server-visible weighted sum. | A new module (e.g. `fedgraa/secagg.py`) implementing at minimum the mock interface contract of test 6 (§11.3): "server receives only `Σ_k h_k`, never an individual `h_k`." |
| Client-side per-round label-count vector `h_k` | **New.** No function today computes a per-client label histogram; `median_frequency_split` only ever sees the pooled `y_val`. | New method on `ClientShard` (`partition.py`) or a new free function taking a `ClientShard` and returning a length-`C` count vector from `y_val` (or, if the held-out labels themselves should never be centrally visible even during simulation bookkeeping, from a client-local slice never passed to `partition_non_iid`'s current centralized `y_val` concatenation). |
| Federated backward pass returning `g_k = ∇_P L_k'(θ'_g)` | **New.** `aggregation.py`'s `_composite_grad_P` already computes the *loss and gradient* for the composite objective, but it does so on the server, in one call, over the pooled `ValidationSplit`. FedGRAA-v2 needs this computation split **per client**, on each client's own held-out rows, with only the resulting gradient array returned. | Refactor `_composite_grad_P(params, val: ValidationSplit)` into a per-client variant taking `(params, client_val_slice)` and returning `(loss_contribution, grad_contribution)`; the existing server-side loop simply becomes a sum over per-client calls instead of one pooled call — the arithmetic (`_loss_and_grad_P`, the `min(L_h, L_m)` sub-gradient selection) is unchanged, only its scope shrinks from "whole `D_V`" to "one client's held-out rows." |
| `confusion_matrix()` helper | **New** (confirmed absent from `metrics.py` by direct reading — `per_class_metrics`/`classification_report` exist but no raw `C×C` integer matrix constructor does). | Add `confusion_matrix(params, X, y, n_classes) -> np.ndarray` to `metrics.py`; derive `per_class_metrics`'s precision/recall/F1 from the *summed* matrix in the FL-compliant evaluation path (§3.2, §6.2), while leaving the existing single-call `per_class_metrics` path available for the non-federated (simulation ground-truth) comparison used in tests 4/5 (§11.2) and the `ζ` measurement protocol (§6.2). |
| DP-noised histogram mechanism (LDP-Fed style) | **New.** No differential-privacy code exists anywhere in `fedgraa/` today. | A new small utility (e.g. `fedgraa/privacy.py`) implementing the Laplace mechanism of §7.5 (`b = Δ₁/ε`) plus, if the RDP-composition route of §10.3 is adopted, a thin wrapper around a standard accountant rather than a hand-rolled one. |

### 15.3 Net scope assessment

Counting only the "genuinely new" row in §15.2 against the "already exists" row in §15.1, the FedGRAA-v2 redesign touches **five** small, well-isolated additions (a SecAgg mock, a per-client histogram function, a per-client gradient-contribution refactor of one existing method, one new metrics helper, and one new privacy utility) plus **zero** changes to the mathematical core of `model.py`'s forward/backward pass, `aggregation.py`'s simplex projection, or `trainer.py`'s main training-loop structure. This directly substantiates the claim already made qualitatively in §4.3 ("a contained, well-scoped code change, not a rearchitecture") with a concrete, itemized accounting — exactly the level of detail a reviewer or a future implementer needs to trust that estimate rather than take it on faith.
