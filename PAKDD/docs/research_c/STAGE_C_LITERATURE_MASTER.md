# FedGRAA — Master Literature Review (Consolidated, Final, Stage-C Foundation)

> **What this document is.** A single, complete, deduplicated, re-verified consolidation of every paper, dataset, venue, and claim surfaced across the five documents in [`../research_c/`](../research_c/) — `FedGRAA_Deep_Research_Compendium.md`, `literature_review_and_publication_strategy.md`, `privacy_compliant_fedgraa_redesign.md`, `research_followup.md`, and `sagitra_report.md` — plus a fresh verification/search pass run on **2026-07-24** specifically to close out every item those documents left in an "unverified" state and to check for anything published since their last pass (2026-07-23).
>
> **Why it exists.** Those five documents were written across several independent passes (2026-06-28 → 2026-07-23), each narrower in scope than the last, with real but scattered overlap (the same ~15 papers — FedAPA, FedAWA, FedNolowe, FedGMKD, FedMADE, SBFedNIDS, PROTEAN, CoAst, FedLMD, etc. — are independently re-verified three or four times across the five files). Per the request to "start again with a new doc," this file supersedes them as the **single entry point** for Stage C literature work. The five source documents are **not deleted** — they remain in `../research_c/` as the full, itemized verification trail (methodology, exact search queries, arXiv-API timestamps) behind every claim made here — but this document, not those, should be read, cited from, and updated going forward.
>
> **What "verified" means here.** Every entry below carries one of four tags: **✅ Confirmed** (primary source — arXiv abstract page, DBLP, official PDF, publisher DOI — fetched directly, either in an earlier pass or in this one), **⚠️ Corrected** (real paper, but the earlier review had a wrong year/venue/title/author, now fixed), **❌ Hallucination** (actively searched for and not found, across two independent tools — arXiv API and general web search — in two separate passes; do not cite), or **❓ Unverified** (plausible, reported by an automated survey tool, but never independently located by either arXiv or web search across up to four attempts; cite only after checking the publisher page directly).
>
> **Compiled:** 2026-07-24. **Last updated: 2026-07-25** (deep-research pass — see §12 and the companion [`STAGE_C_DEEP_RESEARCH_2026-07-25.md`](STAGE_C_DEEP_RESEARCH_2026-07-25.md); the earlier addendum [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md) also remains in place). Supersedes all prior compilations. FedGRAA's own draft is at [`../paper/PAKDD_Final_Version_Submit_v3.md`](../paper/PAKDD_Final_Version_Submit_v3.md).
>
> ⚠️ **If you read only one thing before working on Stage C:** the 2026-07-25 pass found that the central novelty claim in earlier versions of this document — an "unoccupied" gradient-refined × FL-IDS × non-IID intersection — **is false**. **FedPGO** (*Computer Networks* 2026) occupies it on FedGRAA's own dataset. §1 and §8 carry the corrected claim; §9 item 0 makes reading FedPGO and CELM in full a blocking prerequisite for any Related Work.

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [What FedGRAA claims to contribute](#2-what-fedgraa-claims-to-contribute)
3. [Master verification ledger](#3-master-verification-ledger)
4. [Complete reference catalog, by family](#4-complete-reference-catalog-by-family)
5. [Intrusion-detection datasets](#5-intrusion-detection-datasets)
6. [Venue rankings, confirmed](#6-venue-rankings-confirmed)
7. [The two structural privacy gaps and the FedGRAA-v2 redesign](#7-the-two-structural-privacy-gaps-and-the-fedgraa-v2-redesign)
8. [Synthesized positioning and differentiation narrative](#8-synthesized-positioning-and-differentiation-narrative)
9. [Gap-closing checklist for an A*/A submission](#9-gap-closing-checklist-for-an-aa-submission)
10. [Stage C decision framing](#10-stage-c-decision-framing)
11. [Full alphabetized bibliography](#11-full-alphabetized-bibliography)
12. [Document provenance](#12-document-provenance)

---

## 1. Executive summary

Across ~185 distinct papers surveyed over five passes plus three consolidation/audit passes, **the central strategic claim as previously worded no longer holds.** Prior versions of this document asserted that *"no A\*/A-venue paper combines gradient-refined adaptive aggregation weighting, applied specifically to intrusion detection, framed around non-IID/limited-data clients."* A deep-research saturation pass on **2026-07-25** (full write-up: [`STAGE_C_DEEP_RESEARCH_2026-07-25.md`](STAGE_C_DEEP_RESEARCH_2026-07-25.md)) found the intersection **occupied**:

- **FedPGO** — Bikila & Capek, *"FedPGO: A personalized federated learning framework with gradient orthogonalization for IoT intrusion detection,"* **Computer Networks 280:112210, 2026**, DOI 10.1016/j.comnet.2026.112210 (✅ DBLP exact-title hit). Its Adaptive Gradient Orthogonalization decomposes each local gradient into components parallel and orthogonal to the global gradient and **adaptively reweights them**; its Personalized Gradient Fusion aggregates clients by a score over gradient directionality and orthogonality. Evaluated on **CICIoT2023 and CIC-IDS2017 — FedGRAA's own primary dataset** — under non-IID conditions. The literal claim now survives only through the "A\*/A venue" qualifier (*Computer Networks* is Q1 but not a CORE-ranked A\*/A conference), which is a weak position to argue from a **CORE B** target venue (§6).
- **CELM** — Ukaye, Tastan, Abdu-Aguye, Nandakumar, *"Data-Free Client Contribution Estimation via Logit Maximization for Federated Learning,"* arXiv:2605.18892, May 2026. Opens on FedGRAA's own premise (*"standard aggregation protocols … degrade minority-class performance"*), pursues FedGRAA's own objective (*"upweight clients providing strong, discriminative evidence for underrepresented classes"*), requires **no raw data, no client metadata and no auxiliary public dataset** — partly pre-empting the FedGRAA-v2 pitch in §7 — and stabilizes aggregation with **simplex constraints and momentum smoothing**, i.e. exactly the projection [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) found *missing* from Algorithm 2. Domain is vision, not IDS.

The previously-nearest works remain relevant but are no longer the nearest: **NIDS-FGPA** (PLOS ONE 2024, gradient-similarity aggregation for federated NIDS, communication-reduction objective + Paillier), **FedVG** and the Khan et al. layer-wise paper informally tracked here as **"Fed-LAA"** (*Applied Soft Computing* 201:115676, 2026, DOI 10.1016/J.ASOC.2026.115676 — see §4.7 note), and **Jiang et al.** (*Neurocomputing* 604:128317, 2024), which makes FedGRAA's *philosophical* argument — protect rare/valuable client contributions instead of filtering them as outliers — via DRL on general data.

**The claim that does survive, and the one to defend:** adaptive aggregation for federated intrusion detection is an active and, as of 2026, **occupied** space; FedGRAA's residual contribution is a **single-global-model, γ-tunable, gradient-refined aggregation rule whose explicit objective is per-class recall on rare attack types**, evaluated on rare-attack detection rather than aggregate accuracy or personalization gain — a different point in an occupied design space, with an evaluation axis no adjacent work reports. That is a defensible PAKDD-tier claim; it is not an A\*/A-tier claim without the additional work in §9. See §8 for the rewritten differentiation narrative.

**However, the corpus also surfaces two problems that are more consequential for Stage C than the novelty gap itself:**

1. **FedGRAA's own design has two privacy-structural flaws** that are not stylistic nitpicks but violations of FL's founding premise: the server hard-codes which classes/clients are rare (an oracle it should not have), and Algorithm 2 requires a server-held, fully-labelled validation set `D_V`. The field has repeatedly and explicitly called out structurally identical assumptions elsewhere as "impractical" — **FedTruth** (ACISP 2025, arXiv-confirmed), critiquing FLTrust's server-held root dataset, and **CoAst** (arXiv:2409.02495, confirmed), stating validation-based contribution methods need data that is "difficult to obtain in practical FL scenarios." (An earlier candidate third citation for this same point, "CT-BRFL," could not be confirmed as a real paper on re-check this pass — see §3.3 — and should not be used; FedTruth and CoAst alone, both independently arXiv-confirmed, already make the point without it.) A concrete, fully-worked redesign (**FedGRAA-v2**, §7) exists that removes both flaws while leaving the paper's core equations untouched, at a precisely quantified (not estimated) communication cost of ≈6× baseline traffic.
2. **FedGRAA's true competitor set is a specific, small group of 2024–2025 papers**, not the field in general: **FedAPA** (IJCAI 2025 — closest mechanism), **FedAWA** (CVPR 2025 — closest mechanism, opposite philosophy), **FedMADE** (ISC 2024) and **SBFedNIDS** (ICASSP 2025 — closest in-domain), and **DA-PFL** (AAAI 2024/TNNLS 2025 — closest on the imbalance objective specifically). Every one of these must be a head-to-head baseline; none currently is.

**Two findings added 2026-07-25 that lead the list and modify findings 1 and 11 below** (numbered N1/N2 so the existing 1–13 numbering, which is cross-referenced elsewhere in this document, stays stable):

**N1. FedPGO occupies the three-way intersection this document previously called unoccupied.** Bikila & Capek, *Computer Networks* 280:112210, 2026, DOI 10.1016/j.comnet.2026.112210 — gradient-orthogonality-based adaptive reweighting, IoT IDS, non-IID, **on CIC-IDS2017**. It differs from FedGRAA on objective (personalization via parameter-split architecture, not single-global-model minority-boosting), has no counterpart to the γ coefficient, and reports accuracy/robustness rather than per-minority-class recall — real distinctions, but *refinements within an occupied space*, not the occupation of an empty one. **Top-priority baseline and Related Work citation; must be read in full before any Related Work is drafted** (§9 item 0). Note that FedPGO would have been filed as a hallucination under acronym-only search — it was described across searches with zero links bearing its name, and was recovered only by the descriptive-phrase DBLP retry now codified in §3.3.

**N2. CELM pursues FedGRAA's objective in general FL, data-free, with the simplex constraint FedGRAA lacks.** arXiv:2605.18892, 2026. Three consequences: (a) "up-weight clients holding underrepresented classes" can no longer be presented as a new idea — only the gradient-refinement route to it; (b) the §8 contribution statement's **"first fully FL-compliant"** must be dropped or narrowed to *"first for intrusion detection"*, since CELM is explicitly data-free; (c) when FedGRAA adds the missing simplex projection, that fix should be framed as **standard practice with a citation**, not as a contribution — a 2026 paper in the same objective space treats it as a basic design requirement.

**Thirteen most load-bearing findings for Stage C**, carried forward and re-verified across two audit passes (2026-07-24, same day):

1. FedGRAA's differentiation narrative is philosophical, not just mechanical: existing adaptive aggregators reward alignment with the global direction (FedAWA), favour low-loss clients (FedNolowe), or filter statistical outliers (MESAS/CrowdGuard/FreqFed) — all of which **suppress** exactly the divergent, high-loss, minority-class clients that hold the rarest attacks. FedGRAA inverts this logic — **but see finding 11 below and the 2026-07-25 amendment: this argument is no longer uniquely FedGRAA's.** *Amended 2026-07-25:* a counter-current now exists in general FL and must be cited rather than contrasted against — **FedAWARE** (arXiv:2310.02702) reweights to **maximize gradient diversity**, the direct opposite of FedAWA and the same philosophy as FedGRAA, *with convergence guarantees attached*; **AdaAggRL** (AAAI 2025) and Jiang et al. (2024) learn rare-protecting weights via RL; **CELM** (2026) up-weights underrepresented-class evidence data-free. The "existing aggregators all suppress minorities" framing is now false as stated and will be read as a straw man if left unrevised (§8).
2. Three security papers had wrong years in the earliest review, now fixed everywhere in this document: **EIFFeL is CCS 2022**, **FreqFed is NDSS 2024**, **CrowdGuard is NDSS 2024** (all previously misdated one year earlier).
3. **PAKDD is CORE B** — confirmed independently across two sources (resurchify.com and portal.core.edu.au, both citing CORE2023) matching the original ICORE2026 finding — not A, despite being A under the older CORE2018 edition. Any framing claiming "PAKDD = CORE A" is wrong and must not appear in a submission.
4. **"FedSND"** (cited as prior art in FedGRAA's own reference [12]) and **"FeCWGAN-GP"** (an augmentation-family citation) remain **confirmed hallucinations** — now checked against **three independent bibliographic tools** (arXiv API, general web search, and DBLP's publication-search API), all returning zero trace. Both must be removed or replaced (FedSND → FedAWA/FedNova; FeCWGAN-GP → IFL-GAN/FedCG-WGAN/Fed-WGCA) before any resubmission.
5. **"CIC-IDS2023" is not a real dataset** — the correct 2023 CIC release is **CICIoT2023**. This error must not propagate into a new manuscript.
6. Of the 9 previously-"unverified" papers flagged across four prior arXiv-only search passes, **8 of 9 are now confirmed real** across two audit passes: **FedRS**, **ProtoHAR**, **FedPRS**, **FedFMD**, **FedCK** (resolved via general web search), then **PACT** and **FedCoef** (resolved via a direct **DBLP publication-search API** query, after general web search had failed on both). Only **CT-BRFL** remains unresolved, and — after the same DBLP escalation found nothing — should now be treated as functionally equivalent to a confirmed hallucination (§3.3, §3.4).
7. **The escalation path that closed out PACT and FedCoef — querying `dblp.org/search/publ/api` directly — is now the standard next step for any future "unverified" item in this document**, since general web search demonstrably fails to surface recent (2024–2025) ACM/IEEE conference papers that DBLP finds in seconds.
8. Two genuinely new, same-domain, 2026 papers surfaced in the first audit pass that were not in any prior document: **XAI-SOH-FL** (arXiv:2606.00134, Aslam et al., NUST — adaptive similarity-thresholded aggregation + SHAP-based explainability for heterogeneous IoT IDS), and a UAV-focused federated-IDS aggregation benchmark in *Scientific Reports* (2026, nature.com/articles/s41598-026-50865-9) whose proposed architecture is named **FedDrone-Shield** — these are one paper, not two (an error in an earlier draft, corrected).
9. **A fresh snowball pass (this pass) surfaced four more same-domain IDS competitors not in any prior document**: a rare-attack-class transfer-learning FL-IDS method (Chunduru Sri Abhijit et al., *Scientific Reports* 2025), **DS-FedIDS** (dynamic minority up-/majority down-sampling on FedPer, MDPI *Applied Sciences* 2025), **CACFL** (curriculum learning + geometric-median aggregation + CVAE-GAN, *Neural Comp. & Applications* 2026), and a second corroborating 2026 survey (*Artificial Intelligence Review*). All four must be added to Related Work and the baseline table (§9).
10. FedGRAA has **no formal convergence proof that survives its own non-convex tanh-MLP**, **no adversarial evaluation**, and uses only two legacy (2015/2017) datasets — the three gaps separating it from A*/A acceptance, independent of the privacy-redesign question.
11. **The single most important finding of the snowball pass**: Jiang et al. (*Neurocomputing* 604:128317, 2024) — "Preventing harm to the rare in combating the malicious" — makes almost exactly FedGRAA's own "protect, don't filter, the rare" argument, via DRL-learned filtering-and-voting aggregation weights, on general (non-IDS) FL data. This does not eliminate FedGRAA's novelty (mechanism and domain both still differ), but it does mean the differentiation narrative in §8 needed rewriting to cite this paper as convergent evidence rather than risk a reviewer discovering the overlap unprompted. Read this paper in full before finalizing any Related Work section.
12. A **separate, already-diagnosed implementation defect** (documented in [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) and the Stage A/B work) shows the paper's Algorithm 2 lacks a simplex projection on the aggregation weights, causing real divergence on real data — this is an empirical/implementation finding, orthogonal to but compounding the literature-positioning gaps catalogued here.
13. The literature itself has already produced the two most defensible fixes for FedGRAA's structural privacy flaws: (a) **infer** rather than hard-code which classes are rare, via SecAgg-summed label histograms and/or gradient-composition inference (zero new assumptions, near-zero communication cost); (b) **federate the validation backward pass** so no `D_V` is ever centrally pooled (exact, not approximate, preservation of FedGRAA's Eq. 8–10, at a precisely quantified ≈6× communication cost). Both are detailed in §7.

---

## 2. What FedGRAA claims to contribute

**Title:** *Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems* (PAKDD submission draft).

**Core problem.** Standard **FedAvg** aggregates client models in proportion to dataset size: `θ_g^{t+1} = Σ_{k∈S_t} (|D_k|/|D|) θ_k^{t+1}`. Under non-IID intrusion-detection data, clients holding **rare attack classes** (Heartbleed, SQL Injection, Infiltration on CIC-IDS-2017) usually also hold **little data**, so the `|D_k|/|D|` weight crushes their contribution and the global model fails to detect exactly the attacks that matter most.

**Proposed mechanism.** FedGRAA replaces static size-proportional weights with an **adaptive, gradient-based weight update** governed by a personalization coefficient `γ ∈ [0,1]`: as `γ → 1` the aggregator prioritizes minority-class clients; as `γ → 0` it reverts toward conventional emphasis. It is contrasted against a static reparameterization baseline (weights `∝ 1/|D_k|`) that over-corrects and cannot adapt over time.

**Reported results.** ~88% accuracy on minority classes; outperforms FedAvg by up to 55% as the number of minority-holding clients grows; evaluated on CIC-IDS-2017 and UNSW-NB15.

**Honest gaps, already catalogued before this pass:** no formal convergence theorem robust to the actual non-convex architecture; no adversarial/poisoning evaluation; only two datasets (both legacy, 2015/2017); small client counts (`K=14`, `K=9`); the "gradient refinement" mechanism is described but never ablated component-by-component; the paper cites a reference ("FedSND", [12]) that could not be verified as described (§3); and two structural assumptions (oracle rare-class knowledge, server-held `D_V`) conflict with FL's privacy premise (§7).

---

## 3. Master verification ledger

### 3.1 Corrections applied (real papers, wrong metadata in earlier drafts of this review)

| # | Item | Was cited as | Verified reality | Status |
|---|------|--------------|-------------------|--------|
| 1 | EIFFeL | CCS 2021 | **CCS 2022** (DOI 10.1145/3548606.3560611; a 2021 arXiv preprint caused the confusion) | ⚠️ Corrected |
| 2 | FreqFed | NDSS 2023 | **NDSS 2024** (DOI 10.14722/ndss.2024.23620) | ⚠️ Corrected |
| 3 | CrowdGuard | NDSS 2022 | **NDSS 2024** (DBLP conf/ndss/RiegerKMDS24) | ⚠️ Corrected |
| 4 | Mimicry / Prov-HIDS | NDSS 2023 | **NDSS 2024** | ⚠️ Corrected |
| 5 | AutoAdapt | "…Adversarial Adaptation…" | Official title uses **"Adaption"** | ⚠️ Corrected |
| 6 | Armadillo | "…Secure Aggregation for FL" | Full title ends **"…with Input Validation"**; CCS 2025 | ⚠️ Corrected |
| 7 | MPFT (multi-domain prototypes) | ICLR 2024 | **ICLR 2025** (arXiv:2410.07738) | ⚠️ Corrected |
| 8 | "Intrusion Detection with Neural Networks" (1997) | attributed loosely | **Ryan, Lin, Miikkulainen** (NIPS 1997), not Cannady | ⚠️ Corrected |
| 9 | SelfFed | "Expert Systems 2023", IoT | **ESWA 2025** (arXiv 2023 preprint); domain is **IoMT/medical imaging**, not generic IoT | ⚠️ Corrected |
| 10 | FedCGS | "Capturing Global Feature Stats" | Title verb is **"Capture"**; AAAI 2025 | ⚠️ Corrected |
| 11 | RepTreeFL | "Neural Networks 2023" (journal) | Confirmed **arXiv Dec 2023**; journal placement unconfirmed | ⚠️ Corrected |
| 12 | "CIC-IDS2023" | dataset name | **No such official dataset** — re-point to **CICIoT2023** or **CSE-CIC-IDS2018** | ⚠️ Corrected |
| 13 | PAKDD rank | implied A/A* | **CORE B** (ICORE2026 and independently reconfirmed CORE2023 today) | ⚠️ Corrected |
| 14 | FedGMKD | "GMM-based clustering" | **Cluster Knowledge Fusion + Discrepancy-Aware Aggregation**; NeurIPS 2024 main track | ⚠️ Corrected |
| 15 | SBFedNIDS | paper title | Real title *"A Federated Learning NIDS for Multiple Imbalances"*; SBFedNIDS is the method acronym; ICASSP 2025 | ⚠️ Corrected |
| 16 | FedDLAD | short title | Subtitle *"…for Enhancing Resilience Against Backdoor Attacks"* — a **backdoor defence**, not IDS-accuracy work | ⚠️ Corrected |
| 17 | DiffuPac | short title | *"…Adversarial Packets **Generation** via Diffusion Model"* | ⚠️ Corrected |
| 18 | DA-PFL | "Dynamic Affinity-**based**" | Official: *"Dynamic Affinity **Aggregation**"*; AAAI 2024 **and** TNNLS 2025 (both exist) | ⚠️ Corrected |

**Name-collision warnings (do not conflate):** ⚠️ **FedAWA (CVPR 2025) ≠ FedAWARE (arXiv:2310.02702)** — near-homonyms with *opposite* mechanisms: FedAWA rewards client-vector **alignment** with the global direction, FedAWARE reweights to **maximize gradient diversity** (i.e. rewards divergence, FedGRAA's own philosophy). Conflating them inverts the meaning of §8's differentiation argument; a reviewer who knows both will notice. DWAFL (ICRAIE 2023, IDS) ≠ FedDWA (IJCAI 2023, personalization). FedDRL (ICPP 2022, adaptive aggregation, Nguyen et al.) ≠ FedDRL: staged RL model fusion (arXiv:2307.13716, Chen et al.). FedPKD (ICDCS 2023, Lyu–Tang) ≠ FedPKD (Springer *Computing* 2025, Yang & Ge) ≠ FedProtoKD (arXiv:2508.19009). TriFusion-IDS (AAAI 2026) ≠ an unrelated namesake at a different 2026 venue. CT-BRFL ≠ the real, unrelated ACSAC 2024 paper "Adversarially Guided Stateful Defense Against Backdoor Attacks in Federated Deep Learning" (arXiv:2410.11205), which a similar search surfaces but is not a match.

### 3.2 Confirmed hallucinations — do not cite

| Item | Claimed as | Search attempts | Verdict |
|------|-----------|------------------|---------|
| **FedSND** (FedGRAA's own ref [12]) | "Adaptive FL adjusting aggregation weights by data-distribution statistics, Zhang et al." | arXiv API + general web search + **DBLP publication-search API, zero hits (2026-07-24 re-audit)** | **❌ Hallucination.** Three independent bibliographic tools, zero trace. The only real "FedSND" is an unrelated encryption method ("Symmetric Noise Decentralized Orthogonal Encryption"). Replace with **FedAWA** or **FedNova**. |
| **FeCWGAN-GP** | "IEEE TNNLS 2026", conditional-GAN augmentation for vertical FL | arXiv API + general web search + **DBLP publication-search API, zero hits (2026-07-24 re-audit)** | **❌ Hallucination.** Three independent bibliographic tools, zero trace. Nearest real analogues: **FedCG** (IJCAI 2022, arXiv:2111.08211), **FedCG-WGAN**-style gout-staging application (MDPI 2025), or **IFL-GAN** (TNNLS 2022). |
| **Light-FFSL** *(added 2026-07-25)* | "IEEE ~2025/26 (verify IoTJ)" — compression + trust-based client selection + robust aggregation + few-shot learning | Literal web search: zero results bearing the name. **Broadened descriptive search** (step 4 of the escalation path): returned FedMADE, EdgeDetect, Tri-LLM and a TinyML siamese/RL-sparsification FL-IDS paper — all real, all differently named. **DBLP: ZERO HITS** | **❌ Hallucination-equivalent.** Nearest real substitutes with the same components: **EdgeDetect** (arXiv:2604.14663) for compression + robust aggregation; the CICIoT/siamese TinyML few-shot FL-IDS line for the FSL element. |
| **FedCCW** *(added 2026-07-25)* | "Zhang et al., 2025" — class-balanced adaptive aggregation weights | Literal search returns three real, differently-named papers — **FedCBA** (sample-level class-balancing aggregation, DOI 10.1145/3654446.3654454), **FedCW** (client selection with adaptive weight, CMC 86(1)), **FedLWS** (arXiv:2503.15111). **DBLP: ZERO HITS** | **❌ Hallucination-equivalent.** If a class-balanced-aggregation citation is needed, **FedCBA** is the real paper matching the description. |
| **FedHDE** *(added 2026-07-25)* | "Zhang et al., 2026" — hyper-knowledge distillation + tensor optimization + dynamic client selection | Broadened search returns a fluent, detailed description with **no link bearing the name**; results resolve to FedHGN, FedPH, Fed-GDBD. **DBLP: ZERO HITS** | **❌ Hallucination-equivalent.** |

### 3.3 Resolved in the 2026-07-24 re-audit pass (previously "unresolved after every search strategy tried")

All three items this document previously flagged as unresolved turned out to be a **search-tool limitation, not non-existence** — general web search does not reliably surface recent (2024–2025) ACM/IEEE conference papers, but **DBLP's own publication-search API** (`dblp.org/search/publ/api`), queried directly, found two of the three immediately:

| Item | Confirmed as | Source |
|------|-----------|---------|
| **PACT** | Sayedali Sheykholeslamzadeh, Dima Alhadidi. *"PACT: A Passive Accuracy-Based Trust Metric for Malicious Client Detection in Federated Learning."* **TrustCom 2025**, pp. 1775–1781. DOI: 10.1109/Trustcom66490.2025.00206. | ✅ DBLP exact-title hit |
| **FedCoef** | Yunpeng Xie, Fujun Pei, Mingjie Shi. *"FedCoef: An Adaptive Aggregation Algorithm Based on Coefficient of Variation for Federated Learning."* **ASCC 2024**, pp. 1302–1306. | ✅ DBLP `conf/ascc/XiePS24`, IEEE Xplore document 10665350 |
| **CT-BRFL** | "Client-guided Trust," explicitly argues server validation sets contradict FL's privacy principle | **Still not found — and now the strongest negative result in this document.** DBLP was queried both for the literal acronym (`q=CT-BRFL`) and the underlying concept (`q=client-guided trust federated Byzantine`); both returned **zero hits** — the same escalation path that resolved PACT and FedCoef in seconds found nothing here. Combined with two prior web searches whose fluent-sounding summaries cited no source actually titled CT-BRFL, **treat this as functionally equivalent to a confirmed hallucination** (§3.2) — the strongest evidence tier this document has for "does not exist," short of literally paging through every 2024 TrustCom/ACM-adjacent proceedings by hand. If a citation is needed for "server validation sets contradict FL's privacy principle," use **FedTruth** (arXiv:2311.10248) or **CoAst** (arXiv:2409.02495) instead — both independently confirmed and making the identical point. |

**Escalation path adopted going forward (standing procedure, amended 2026-07-25):**

1. General web search on the acronym/title.
2. arXiv API title search (`export.arxiv.org/api/query`).
3. **DBLP direct query** on the literal acronym (`dblp.org/search/publ/api`).
4. ⚠️ **MANDATORY before any non-existence verdict — DBLP retry with a *descriptive phrase* rather than the acronym.** Acronyms are frequently the search tool's own invention rather than the authors'; a failed literal-acronym search is **not** sufficient grounds to declare a hallucination.

Step 4 is not a formality. It has now changed a conclusion three times: it rescued **Fed-LAA** (2026-07-25 re-audit, §4.7), it rescued **FedPGO** — *the single most consequential paper in this entire corpus* (§1 finding N1; described across multiple searches with **zero links bearing its name**, recovered instantly by the query `gradient orthogonalization personalized federated intrusion detection`) — and, applied symmetrically in the same sitting, it **confirmed** the non-existence of Light-FFSL, FedCCW and FedHDE (§3.2), which returned nothing but adjacent real papers under the same broadened treatment. The rule cuts both ways, which is why it is trustworthy: it kills false positives *and* rescues real papers.

**A "no link, no citation" test applies at every step:** a fluent, mechanically detailed description in a search summary, whose result list contains no link actually bearing the paper's name, is evidence *against* existence, not for it. This is the signature shared by all seven confirmed-or-equivalent hallucinations in this corpus (FedSND, FeCWGAN-GP, CT-BRFL, pFedCross, Light-FFSL, FedCCW, FedHDE) — every one of which entered via automated survey tooling or acronym-first search, never via a primary source.

The following five were also confirmed in the 2026-07-24 pass:

| Item | Confirmed as | Source |
|------|--------------|--------|
| **FedRS** | Li, X.-C. & Zhan, D.-C. *"FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data."* **KDD 2021.** | Official PDF (lamda.nju.edu.cn), DBLP `conf/kdd/LiZ21`, GitHub `lxcnju/FedRepo` |
| **ProtoHAR** | Cheng, Zhang, Bu, Wang, Wu, Song. *"ProtoHAR: Prototype Guided Personalized Federated Learning for Human Activity Recognition."* **IEEE JBHI 2023.** | GitHub `cheng-haha/ProtoHAR`, Semantic Scholar, ResearchGate |
| **FedPRS** | *"FedPRS: A Privacy-preserving Representation Synthesis Framework for Federated Contribution Evaluation."* **ACM TIST**, DOI 10.1145/3779055. | ACM DL (doi.org/10.1145/3779055) |
| **FedFMD** | *"FedFMD: Fairness-Driven Adaptive Aggregation in Federated Learning via Mahalanobis Distance."* **ACM CIKM 2025** (34th ACM Int'l Conf. on Information and Knowledge Management, Nov 2025), DOI 10.1145/3746252.3761316. | ACM DL |
| **FedCK** | Chen, Li, He. *"FedCK: addressing label distribution skew in federated learning via clustering-efficient and knowledge distillation."* **J. Supercomputing, 2025.** | Springer Nature Link |

Two items independently re-confirmed via a *second*, different-tool search today (already ✅ from earlier arXiv-only passes, now doubly attested): **FedMcon** (FITEE 2025 + sibling ACM MM Asia Workshops paper, DOI 10.1145/3700410.3702124) and **PAKDD's CORE-B ranking** (resurchify.com independently states CORE2018=A → CORE2023=B, matching the ICORE2026=B finding exactly).

---

## 4. Complete reference catalog, by family

Every paper below was independently verified against arXiv, DBLP, Semantic Scholar, OpenReview, IEEE Xplore, ACM DL, or official venue/dataset portals across one or more of the five source passes plus this consolidation pass. "Rel. to FedGRAA" is a one-line differentiation, not a summary of the paper.

### 4.1 Foundational federated-learning aggregation methods

| Method | Authors / Venue / Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **FedAvg** | McMahan et al. (Google), AISTATS 2017, arXiv:1602.05629 | Size-weighted averaging of client SGD updates | **The direct baseline/namesake** — FedGRAA replaces the fixed `n_k/n` weight | ✅ |
| **FedProx** | Li et al. (CMU), MLSys 2020, arXiv:1812.06127 | Proximal term bounds client drift | Client-side regularization vs. FedGRAA's server-side reweighting; standard baseline | ✅ |
| **SCAFFOLD** | Karimireddy et al., ICML 2020, arXiv:1910.06378 | Control variates cancel client drift | Corrects update *direction*; FedGRAA adjusts aggregation *weights* | ✅ |
| **FedNova** | Wang et al., NeurIPS 2020, arXiv:2007.07481 | Normalizes updates by local-step count | **Closest foundational analogue** — both are weight-normalization schemes; FedNova by step count, FedGRAA by gradient/minority signal | ✅ |
| **FedOpt** (FedAdam/Yogi/Adagrad) | Reddi et al. (Google), ICLR 2021, arXiv:2003.00295 | Server-side adaptive optimizer on the pseudo-gradient | Orthogonal, combinable — FedOpt adapts the update rule, FedGRAA adapts the pre-averaging weights | ✅ |
| ~~FedSND~~ | claimed "Zhang et al." | claimed: adaptive weights from data-distribution stats | **❌ Hallucination** — see §3.2 | ❌ |

### 4.2 A* security venues (CCS / NDSS / IEEE S&P)

| Method | Authors / Venue / Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **Entente** | Xu, Li, Zheng, Li, NDSS 2025 | Cross-silo graph-NIDS; "adaptive contribution scaling" | **The single closest published analogue** — same reweighting family, but graph-structured cross-silo vs. FedGRAA's tabular minority-boosting | ✅ |
| **EIFFeL** | Roy Chowdhury, Guo, Jha, van der Maaten, CCS 2022 | Secure aggregation with integrity checks (secret-sharing + ZK) | Cryptographic integrity/filtering layer, complementary, different pipeline stage | ✅⚠️ |
| **Cerberus** | Naseri, Han, Mariconti, Shen, Stringhini, De Cristofaro, CCS 2022, arXiv:2209.03050 | FL feasibility study for forecasting security events | Cross-silo security-FL motivation; not an aggregation algorithm | ✅ |
| **MESAS** | Krauß & Dmitrienko, CCS 2023 | Metric-cascades poisoning defense vs. adaptive attackers | Filter-and-remove defense; risks penalizing benign minority clients FedGRAA protects | ✅ |
| **FreqFed** | Fereidooni, Pegoraro, Rieger, Dmitrienko, Sadeghi, NDSS 2024, arXiv:2312.04432 | DCT-domain clustering filters poisoned updates; evaluated incl. IoT-IDS | Removes suspicious updates vs. FedGRAA's up-weighting of benign minority clients | ✅⚠️ |
| **CrowdGuard** | Rieger, Krauß, Miettinen, Dmitrienko, Sadeghi, NDSS 2024 | Client-assisted backdoor detection via SGX-protected activation clustering | Same mislabeling risk as MESAS for benign minority clients | ✅⚠️ |
| **Kitsune** | Mirsky, Doitshman, Elovici, Shabtai, NDSS 2018, arXiv:1802.09089 | Ensemble-of-autoencoders online NIDS | Not federated — foundational centralized NIDS architecture/dataset source | ✅ |
| **Mimicry vs. Provenance-Graph HIDS** | Goyal, Wang, Bates, Han, NDSS 2024 | Mimicry evasion of provenance-graph HIDS | Tangential — host-based, non-federated, attack-side motivation | ✅⚠️ |
| **AutoAdapt** | Krauß, König, Dmitrienko, Kanzow, NDSS 2024 | Constrained-optimization poisoning evasion of arbitrary FL defenses | A threat FedGRAA must acknowledge — gradient-driven reweighting is a manipulable signal | ✅⚠️ |
| **Armadillo** | Ma, Guo, Karthikeyan, Polychroniadou (UPenn/JPMorgan), CCS 2025 | Single-server secure aggregation with input validation | Cryptographic infra — complementary; constrains admissible inputs, not weighting | ✅⚠️ |

### 4.3 A* AI venues (AAAI / IJCAI / NeurIPS / ICLR)

| Method | Authors / Venue / Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **FedDLAD** | Ding, Yang et al., IJCAI 2025 (#559) | Dual-layer statistical + fine-grained backdoor-update screening | Filters malicious clients (security) vs. FedGRAA up-weighting disadvantaged benign clients | ✅⚠️ |
| **TriFusion-IDS** | Zhao & Zhong (Nanjing U.), AAAI 2026, 40(19):16433–16440 | Graph+tabular+text contrastive fusion, CLIP-style zero-shot generalization | Centralized, multimodal — orthogonal axis; federatable with a FedGRAA aggregator | ✅ |
| **Poisoning with a Pill** | Guo, Wang, Song, Zheng, Hua, Guan, Zhang, AAAI 2026, arXiv:2407.15389 | Model-redundancy-exploiting stealthy poisoning | Robustness stress test for gradient-reweighting aggregators | ✅ |
| **DiffuPac** | Bin Jasni, Manada, Watabe, NeurIPS 2024 | BERT+diffusion evasive-packet generator | Attack/generation, non-federated — defines the evasive-traffic threat landscape | ✅⚠️ |
| **Intrusion Detection with Neural Networks** | Ryan, Lin, Miikkulainen, NIPS 1997 | Backprop NN over per-user command profiles | Historical seminal origin of ML-for-IDS | ✅⚠️ |
| **MPFT** | Zhang et al. (NTU), ICLR 2025, arXiv:2410.07738 | Prototype-generation + global adapter init + few-shot fine-tuning | Prototype-exchange + fine-tuning (representation side), not gradient-weighted aggregation | ✅⚠️ |

### 4.4 Direct FL-IDS competitors (the niche FedGRAA occupies)

| Method | Authors / Venue / Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **SBFedNIDS** | Zeng, Zheng, Li (Sun Yat-sen U.), ICASSP 2025 | Self-balancing local rebalance + balance-aware aggregation vs. size/local-class/global-class imbalance | **Arguably the single most direct competitor** — same goal, different signal (balance statistics vs. gradient) | ✅⚠️ |
| **FD-IDS** | Peng, Wu, Xiao (Zhejiang U.), Sensors 2025, 25(14):4309 | Knowledge distillation for non-IID IoT-IDS | Distillation-side vs. aggregation-side; combinable | ✅ |
| **DWAFL** | Umair et al., IEEE ICRAIE 2023 | Dynamic weights from client validation accuracy/loss | Direct competitor — performance/loss signal, not gradient, no explicit minority focus | ✅ |
| **FedMADE** | Sun, Sharma, Nwodo, Stavrou, Wang (Virginia Tech), ISC 2024, arXiv:2408.07152 | Clusters by traffic pattern, weights by measured contribution to global performance; robust to poisoning | **Closest IDS competitor after SBFedNIDS** — up to 71.07% minority-attack improvement | ✅ |
| **FedPGO** *(new 2026-07-25 — ⚠️ highest-priority competitor in this table)* | Dawit Dejene Bikila, Jan Capek. *"FedPGO: A personalized federated learning framework with gradient orthogonalization for IoT intrusion detection."* **Computer Networks 280:112210, 2026**, DOI 10.1016/j.comnet.2026.112210 | **Adaptive Gradient Orthogonalization (AGO)** decomposes local gradients into components parallel and orthogonal to the global gradient and **adaptively reweights them**; **Personalized Gradient Fusion (PGF)** aggregates clients via a scoring function over gradient directionality and orthogonality; parameter-split model structure separates shared from personalized layers. Evaluated on **CICIoT2023 and CIC-IDS2017** under non-IID | **The closest paper to FedGRAA in the entire corpus, and the one that occupies the three-way intersection §1 previously called unoccupied** — gradient-derived adaptive aggregation weighting × FL-NIDS × non-IID, on FedGRAA's own primary dataset. Differentiate on: (a) FedPGO optimizes **personalization** via parameter splitting, FedGRAA produces a **single global model**; (b) no counterpart to FedGRAA's γ trade-off coefficient; (c) reports accuracy/robustness, not per-minority-class recall. **Must be read in full before Related Work is drafted (§9 item 0); top-priority head-to-head baseline** | ✅ *(DBLP exact-title hit; Elsevier full text paywalled — mechanism description is from consistent secondary summaries corroborated by the DBLP record, sufficient for existence/venue/topic, NOT sufficient for writing differentiation prose)* |
| **PerFedHypID** *(❓→✅ 2026-07-25)* | *"A personalized federated hypernetworks based aggregation approach for intrusion detection systems,"* *Scientific Reports* 2025, `s41598-025-11659-7`, PMC12485126 (PerFedHypID is the method name, used in the Research Square preprint rs-4767476) | Hypernetwork-based personalized aggregation driven by **client embedding vectors** rather than model weights; 0.62 M params/round vs 1.17 M baseline | Same domain, aggregation-side, non-IID, evaluated on **CSE-CIC-IDS2018** — embedding-vector signal vs. FedGRAA's gradient signal; personalization objective. Add to the baseline grid | ✅ |
| **FedKSam\*** *(❓→✅ 2026-07-25)* | *"FedKSam\*: A Two-Phase Federated Learning Intrusion Detection Algorithm."* IEEE Xplore document **10661778** (asterisk is part of the name) | Phase 1: SAM optimizer + knowledge distillation against non-IID drift. Phase 2: re-optimizes **classifier decision boundaries** | Same domain, explicitly targets **data heterogeneity *and* class imbalance** — client-side optimization + boundary correction vs. FedGRAA's server-side reweighting. Directly comparable minority-class claims | ✅ |
| **G-PFL-ID** *(❓→✅ 2026-07-25)* | *"G-PFL-ID: Graph-Driven Personalized Federated Learning for Unsupervised Intrusion Detection in Non-IID IoT Systems,"* **MDPI *IoT* 7(1):13, 2026-01-29** | GCN/GAE + DeepSVDD one-class objective + **FedReg** (proximal term + embedding-variance penalty); 99.46% AUROC on IoT-23, 97.74% on N-BaIoT | Same domain and non-IID framing, but **unsupervised/one-class** — a different problem formulation from FedGRAA's supervised multi-class rare-attack recall. Cite as adjacent, not as a head-to-head baseline | ✅ |
| **AMAFed** *(new 2026-07-25)* | Izadi & Ahmadi. *"Adaptive Meta-Aggregation Federated Learning for Intrusion Detection in Heterogeneous Internet of Things."* arXiv:2602.12541, 2026-02-13 | Meta-learning dynamic weighting: assigns adaptive importance to local models by **data quality and contribution**. ToN-IoT / N-BaIoT / BoT-IoT | Same family (adaptive aggregation × IDS × heterogeneity) but **no class-imbalance framing** and no dataset overlap with FedGRAA. Medium-priority citation | ✅ |
| **Federated transfer learning for rare attack class detection** *(new this pass)* | Chunduru Sri Abhijit, Y. Annie Jerusha, S. P. Syed Ibrahim, Vijay Varadharajan (VIT Chennai + U. Newcastle), *Scientific Reports*, 2025-09-30, `s41598-025-02068-x` / PMC12484838 *(DOI therefore 10.1038/s41598-025-02068-x — derived from the article ID, verify once before use)*. **Note: evaluated on CIC-IDS2018 + UNSW-NB15 + Edge-IIoT — NOT on CIC-IDS-2017** | Adaptive personalized client layers + transfer learning to reduce false-alarm rate on zero-day/rare attack types; server updates global model from client-specific gradients | **Same domain, same objective (rare/zero-day attack detection in FL-IDS)** — personalization + transfer learning vs. FedGRAA's aggregation-weight refinement; must-cite competitor | ✅ *(confirmed via Nature/PMC listing)* |
| **DS-FedIDS** *(new this pass)* | "Enhancing Federated Intrusion Detection with Class-Specific Dynamic Sampling," *Applied Sciences* (MDPI) 15(9):5067, DOI 10.3390/app15095067 | Extends FedPer (personalization layers) with dynamic per-round up-sampling of minority attacks / down-sampling of dominant classes during local client training | **Same domain, same objective** — client-side dynamic resampling vs. FedGRAA's server-side aggregation-weight refinement; directly comparable minority-class-recall claims | ✅ *(confirmed via DOI)* |
| **CACFL** *(new this pass)* | "Multi-modal class-aware curriculum federated learning fusion for advanced intrusion detection using conditional generative augmentation and real-time neural optimization," *Neural Computing and Applications*, DOI 10.1007/s00521-026-11845-2, 2026 | Curriculum learning + class-dependent scheduling + **geometric-median aggregation** + CVAE-GAN augmentation + Bat–Kestrel–Antlion hyperparameter optimizer + XGBoost/1D-CNN/EWC ensemble | Multi-component, heavier system; uses robust-statistics aggregation (geometric median) rather than gradient-based reweighting — a validation-free alternative worth citing as an §7.2-style baseline (§4.10 Approach D family) as well as a competitor here | ✅ *(confirmed via DOI)* |

### 4.5 Federated few-shot / meta-learning

| Method | Authors / Venue / Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **FedFSL-CFRD** | Wang et al. (Xidian), AAAI 2025 | Client-wise + class-wise feature disentanglement | Representation-side; complementary | ✅ |
| **Few-Shot Model Agnostic FL** | Huang, Ye, Gao, Du (Wuhan U.), ACM MM 2022, pp. 7309–7316 | Heterogeneous architectures + public-set distillation | Distant premise (public data, model-agnostic) | ✅ |
| **Fed-Meta FD** | IEEE 2023 (verify TNSE) | Meta-learning for fast fault-category adaptation | IIoT fault diagnosis, not IDS; complementary | ✅ |
| **REFML** | Cui, Li, Mei et al., IEEE TIM 2023, arXiv:2310.09002 | Representation encoding + meta-learning | Heterogeneity-aware, complementary | ✅ |
| **DFML** | Chen, Zeng, Zhang, Lu, Hou, Zhou, Helal, IEEE/ACM TCBB 2023 | Dynamic attention over meta-tasks + accuracy-based dynamic client weighting | **Closest-in-spirit** — adaptive client weighting by accuracy, not gradient | ✅ |
| **FeCoGraph** | Mao, Su et al., IEEE TIFS 2025, 20:2266–2280 | Line-graph flow embeddings + label-aware contrastive loss | **Same domain (few-shot NIDS)**, representation-side — must-cite complement | ✅ |
| **Light-FFSL** | IEEE ~2025/26 (verify IoTJ) | Compression + trust-based selection + robust aggregation + FSL | Partly aggregation-side → closer competitor; trust-based, not gradient | ✅ |
| **Fed-Siamese** | Yang, Mao et al., IEEE IoTJ 2025, 12(24):52886–52899 | Meta-learned Siamese embedding + personalized classifier | RFID-HAR domain; complementary, robustness theme | ✅ |

### 4.6 Prototype-based FL

| Method | Authors / Venue / Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **FedGMKD** | Zhang, NeurIPS 2024 (main) | Cluster Knowledge Fusion + Discrepancy-Aware Aggregation + KD | **Closest prototype analogue** — Discrepancy-Aware Aggregation is aggregation-side like FedGRAA | ✅⚠️ |
| **FedRFQ** | Yan et al., IEEE TC 2024, 73(4):1086–1098 | SoftPool redundancy reduction + BFT-detect Byzantine-tolerant aggregation | Robust aggregation angle; prototype/BFT vs. gradient reweighting | ✅ |
| **FedPKD** (Lyu, Tang) | IEEE ICDCS 2023 | Prototype-based ensemble distillation | Complementary; disambiguate from Springer-2025 "FedPKD" | ✅ |
| **DeProFL** | Li, Gao et al., IEEE IoTJ 2023/24 | Server-free prototype exchange over time-varying topology | Decentralized — different setting from FedGRAA's central aggregation | ✅ |
| **FedFSA** (Cross-Silo Feature Space Alignment) | Qi, Meng, Li, Hu, Meng, AAAI 2025 | Implicit Semantic-Preserving Space Learning for imbalance | **Same problem (class imbalance/minority)** — fixes representation, FedGRAA fixes weights | ✅ |
| **FedMPS** | Yang, Hu et al., IEEE TNNLS 2025 | Multi-level prototype contrastive learning + soft labels | Representation-side; complementary | ✅ |
| **RefProtoFL** | Wu, Li, Fan, Zhu, Chen, Feng (Tianjin U.), arXiv:2601.14746, 2026 | External-referenced prototype alignment + adaptive probabilistic update dropping | Requires server public data (FedGRAA does not); preprint only | ✅⚠️ |
| **ProtoHAR** | Cheng, Zhang, Bu, Wang, Wu, Song, IEEE JBHI 2023 | Global activity prototypes correct local representation for personalized HAR | Different domain (HAR); confirms prototype-presence as an imbalance-inference signal (Family 7, §7) | ✅ *(confirmed this pass)* |
| **FedCK** | Chen, Li, He, J. Supercomputing 2025 | Fuzzy-C-Means clustering by label-distribution similarity + conditional-generator KD | Clustering-as-imbalance-proxy, no raw statistics exchanged | ✅ *(confirmed this pass)* |
| **PROTEAN** | Chennoufi, Han, Blanc, De Cristofaro, Kiennert, ESORICS 2025, arXiv:2507.05524 | Class-prototype exchange for federated IDS under highly non-IID attack distributions | **Strongest same-domain precedent in the entire corpus** — solves FedGRAA's exact problem via prototype exchange instead of aggregation-weight learning | ✅ |
| **FedP3E** ⚠️ *(venue corrected 2026-07-25)* | Rami Darwish, Mahmoud Abdelsalam, Sajad Khorsandroo, Kaushik Roy. *"FedP3E: Privacy-Preserving Prototype Exchange for Non-IID IoT Malware Detection in Cross-Silo Federated Learning."* **arXiv:2507.07258 (CoRR 2025) — PREPRINT ONLY** | GMM-based class-prototype exchange + SMOTE for non-IID IoT malware detection | Prototype-exchange, malware- (IDS-adjacent) not NIDS-specific | ✅⚠️ **Earlier drafts of this document attributed FedP3E to *IEEE TIFS 2025*. DBLP indexes no journal version — only the CoRR record. Cite as an arXiv preprint until a published version is located.** Same error class as the EIFFeL/FreqFed/CrowdGuard corrections in §3.1 |

### 4.7 Adaptive weighted aggregation — FedGRAA's true competitor set

**This is the highest-priority family for baselines.** The comparison matrix follows; one entry newly found in this pass (Jiang et al., Neurocomputing 2024) is now the single closest *philosophical* precedent in the entire corpus and must lead any related-work discussion of FedGRAA's core differentiation claim — see §8.

| Method | Authors / Venue / Year | Weight signal | Objective | Directness vs. FedGRAA |
|---|---|---|---|---|
| **Jiang et al. — "Preventing harm to the rare in combating the malicious"** *(new this pass, ✅ confirmed via DBLP)* | Yanna Jiang, Baihe Ma, Xu Wang, Guangsheng Yu, Caijun Sun, Wei Ni, Ren Ping Liu (UTS + Data61/CSIRO + Zhejiang Lab), *Neurocomputing* 604:128317, 2024 | Deep-RL-learned aggregation weights, explicitly designed to **protect and prioritize rare/valuable client contributions** inside a filtering-and-voting framework, instead of excluding them as outliers | Robustness + fairness for rare/minority contributions in general FL | **Highest-priority related-work citation found in any pass.** Makes almost exactly FedGRAA's own "protect, don't filter, the rare" argument (§8) via DRL instead of gradient refinement, on general (non-IDS) data — distinguish on (a) IDS-specific vs. domain-general, (b) gradient-refinement vs. DRL, (c) no explicit malicious-client filtering layer in FedGRAA. Read in full before finalizing §8. |
| **FedAWARE** *(new 2026-07-25, ✅ arXiv abstract fetched)* | Dun Zeng, Zenglin Xu, Shiyu Liu, Yu Pan, Qifan Wang, Xiaoying Tang. *"On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond."* arXiv:2310.02702 (v4, Dec 2024). **FedAWARE is the method name inside the paper — cite by title** | Server-side adaptive reweighting that **maximizes gradient diversity**: the convex combination of client updates that *least conflicts* in direction. Introduces a new heterogeneity measure, **client consensus dynamics**, and proves FedAvg handles heterogeneity given an appropriate aggregation strategy. Plug-in module | Robustness + generalization under statistical/system heterogeneity | **High threat, and philosophically inconvenient.** ⚠️ FedAWARE does the **opposite of FedAWA** — it rewards divergence, which is FedGRAA's own stated philosophy — in general FL, *with convergence theory attached*. See the FedAWA/FedAWARE name-collision warning in §3.1. Also the best available template for the §9 gap-#1 theorem (§4.11a) |
| **CELM** *(new 2026-07-25, ✅ arXiv abstract fetched verbatim)* | Asim Ukaye, Nurbek Tastan, Mubarak Abdu-Aguye, Karthik Nandakumar (MBZUAI). *"Data-Free Client Contribution Estimation via Logit Maximization for Federated Learning."* arXiv:2605.18892, 2026-05-17 | Server probes each client model class-by-class during a warm-up window by optimizing noisy inputs to **maximize class logits**, yielding a client-class evidence matrix; contribution weights **up-weight clients with strong discriminative evidence for underrepresented classes**. Requires **no raw data, no client metadata, no auxiliary public dataset**. Aggregation stabilized by **simplex constraints + momentum smoothing** | Minority-class performance under label skew | **Highest-threat non-IDS paper in the corpus.** Shares FedGRAA's motivating sentence *and* its objective, in a data-free formulation that partly pre-empts the §7 FedGRAA-v2 pitch — and already has the simplex projection FedGRAA's Algorithm 2 omits. Differentiate on domain (vision) and signal (logit-probing vs. gradient refinement). See §1 finding N2, §4.10, §7 |
| **AdaAggRL** *(new 2026-07-25)* | *"Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning."* arXiv:2406.14217; **AAAI 2025**, DOI 10.1609/aaai.v39i24.34733 | MMD-based pairwise similarity of local data distributions → **policy learning determines aggregation weights** adaptively | Poisoning robustness | Direct — the Jiang et al. (Neurocomputing 2024) RL-aggregation family **at an A\* venue**, which strengthens rather than contradicts finding 11. Note its premise (*benign clients show higher distribution stability than malicious ones*) is in direct tension with FedGRAA's (*the most valuable clients are the divergent ones*) — see §8 and §4.11b |
| **Byzantine-Robust FL with Learnable Aggregation Weights** *(new 2026-07-25, ✅ arXiv abstract fetched)* | Parsa, Daghestani, Teixeira, Johansson. arXiv:2511.03529 (submitted Nov 2025, revised May 2026) | Treats **aggregation weights as learnable parameters jointly optimized with the model**; alternating minimization; **strong convergence guarantees under adversarial attack**; robust under highly heterogeneous data and a large fraction of malicious clients | Byzantine robustness + convergence | **Prior art for FedGRAA's entire method class.** FedGRAA cannot claim the *idea* of gradient-optimized aggregation weights — only the rare-class objective and IDS instantiation. Simultaneously the best available template for §9 gaps #1 and #3 (§4.11a/b) |
| **FedAPA** | Sun, Sun, Pan, Fu, Guo, IJCAI 2025, arXiv:2502.07456 | Gradient of param-change w.r.t. weights | Personalization | **Most similar mechanism** — top-priority baseline |
| **FedAWA** | Shi, Zhao, Zhang, Zhou, Guo, Chang, CVPR 2025, arXiv:2503.15842 | Client-vector alignment with global direction | Global generalization | **Very direct — opposite philosophy** (rewards alignment, FedGRAA rewards divergence) |
| **FedSWA/FedMoSWA** | Liu et al., ICML 2025, arXiv:2507.20016 | Weight-space SWA (temporal) + momentum control variates | Flat-minima generalization | Orthogonal lever, not minority-aware |
| **DA-PFL** | Yang et al., AAAI 2024 (arXiv:2403.09284) + IEEE TNNLS 2025 | Client affinity/complementarity graph | Class-imbalance-aware personalized FL | **Direct — leading imbalance-aware citation** |
| **FedNolowe** | Le et al., bioRxiv 2025 → PLOS ONE 2025 | Normalized training loss (favours low-loss clients) | Robust non-IID | Direct — **opposite for minorities** (high-loss minority clients get down-weighted) |
| **FedMcon** | Shen, Li et al., FITEE 2025 + ACM MM Asia Workshops, DOI 10.1145/3700410.3702124 | Learned meta-controller on a proxy dataset | Learned aggregation | Direct (learned vs. analytic); FedGRAA needs no proxy dataset |
| **FBLG** | Xu, Y. Li et al., IJCAI 2024, pp. 5289–5297 | Local-loss + Jensen–Shannon-divergence graph | Dual-skew (label+size) non-IID | Direct (dual-skew) |
| **FedDRL** | Nguyen et al., ICPP 2022, arXiv:2208.02442 | DRL-learned per-client "impact factor" | Cluster-skew non-IID | Direct (RL vs. gradient); *disambiguate from the unrelated Chen et al. "FedDRL" model-fusion paper, arXiv:2307.13716* |

**Additional adaptive-aggregation entries found across later passes** (candidate baselines, verify venue before citing where flagged):

| Method | Venue/Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **NIDS-FGPA** | PLOS ONE, Oct 2024 | Gradient Similarity Aggregation (GSA) + Paillier encryption, federated NIDS with incomplete features | **The single closest paper in the entire corpus on the exact (gradient-weighting × federated-NIDS) intersection** — must be read in full and explicitly distinguished | ✅ |
| **FedVG** | arXiv:2602.21399, 2026 | Validation-set layer-wise gradient-norm weighting ("flatter landscape → more influence") | Structurally closest *general-FL* analogue; not IDS-specific, needs a public validation set | ✅ |
| **Fed-LAA** | Khan, Rizwan, Ahmad, Khan, Li, Kim, *Applied Soft Computing* 201:115676, 2026, DOI 10.1016/J.ASOC.2026.115676 *(re-verified 2026-07-25 — see [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md))* | Layer-wise weights from gradient angular similarity to global descent direction; data-volume weighting prevents small-node domination | Same "aggregation weight from gradient info" family; layer-level, not IDS-specific | ✅⚠️ *(title/authors/DOI confirmed via DBLP phrase search; the acronym "Fed-LAA" itself is this document's own shorthand, not independently confirmed as the authors' own — cite by title/DOI, same convention as the "FedAgile"→title correction above)* |
| **FedAAW** *(❓→✅ and promoted 2026-07-25)* | *"Tackling data-heterogeneity variations in federated learning via adaptive aggregate weights,"* **Knowledge-Based Systems, 2024**, DOI 10.1016/j.knosys.2024.112484 | ⚠️ **Aggregation weights are theoretically *derived* by minimizing the convergence upper bound of FL on non-IID data across nodes**, yielding a tighter bound; global/local model information dynamically combined at the start of local training. ~3× faster to target accuracy than adaptive-local-aggregation FL, negligible added communication | General FL; addresses aggregation-error drift, not minority-boosting | ✅ — **the single highest-leverage theoretical direction found in any pass.** It inverts FedGRAA's current approach: rather than proposing weights heuristically and then seeking a theorem that tolerates them, choose the weights that minimize the bound. If FedGRAA's γ-tuned weights can be shown to approximate a bound-minimizing choice under a rare-class-weighted objective, the §9 gap-#1 theorem stops being a bolt-on and becomes the derivation. See §4.11a |
| **FedAgg** | arXiv:2303.15799, 2023 | Adaptive per-client learning rate from aggregated-gradient mean-field term | Learning-rate not weight adaptation | ✅ |
| **Aggregated Gradients-based Adaptive Learning Rate Design in FL** | Wenhao Yuan, Xuehe Wang, ACM CIKM 2025, DOI 10.1145/3746252.3761234 | Aggregated-gradient-driven adaptive per-round learning rate | Same family as FedAgg; *note: earlier drafts of this document referred to this paper by the placeholder acronym "FedAgile," which is not confirmed as the authors' own name for it — cite by title until an author-given acronym is verified* | ✅ *(exact title/authors/DOI confirmed via DBLP, replacing the earlier unverified "FedAgile" placeholder)* |
| **ECGR (Exploratory–Convergent Gradient Re-aggregation)** *(new this pass)* | "Local Gradient Regulation Stabilizes Federated Learning under Client Heterogeneity," arXiv:2601.03584, 2026-01 | Swarm-intelligence-inspired **client-side** decomposition of mini-batch gradients into aligned/misaligned components, re-aggregated to stabilize local update direction | **Term-adjacent scooping-risk watch item**, not a direct competitor: general FL (not IDS), operates client-side on mini-batch gradients (not server-side on client updates/weights), and stabilizes convergence rather than boosting minority contributions — but the "gradient re-aggregation"/"refin[ing] their contributions" language is close enough to FedGRAA's own terminology that it should be read and explicitly distinguished in Related Work to pre-empt a reviewer conflating the two | ✅ *(confirmed via arXiv + emergentmind.com summary)* |
| **FedAWR** | MDPI Future Internet, 2026 | Joint client-weight + per-client-learning-rate optimization by compute capability | Weighting signal is compute capability, not gradient | ✅ |
| **FedFMD** | ACM CIKM 2025, DOI 10.1145/3746252.3761316 | Mahalanobis distance between client updates and global distribution; integrates Task Arithmetic | Validation-free, robust-statistics alternative to gradient refinement | ✅ *(confirmed this pass)* |
| **FedCoef** | Yunpeng Xie, Fujun Pei, Mingjie Shi, ASCC 2024, pp. 1302–1306 | Coefficient-of-variation weighting | Statistically-principled alternative to gradient refinement | ✅ *(resolved via DBLP — see §3.3)* |
| **FedRS** | Li & Zhan, KDD 2021 | Restricts local softmax to observed classes; restriction shape signals label presence | Not an aggregation-weight method per se, but a validated Family-7 imbalance-inference signal | ✅ *(confirmed this pass)* |

### 4.8 Data augmentation, knowledge transfer, one-shot / communication-efficient FL

| Method | Authors / Venue / Year | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **AugFL** | Yue, Qin, Deng, Ren, Zhang, Zhang, IEEE/ACM ToN 2025 (in-press), arXiv:2503.02154 | Server-held private pretrained model as a meta-learning regularizer, inexact-ADMM | Injects external PM knowledge; FedGRAA is data-free/PM-free — complementary | ✅ |
| ~~FeCWGAN-GP~~ | claimed TNNLS 2026 | claimed CGAN augmentation for vertical FL | **❌ Hallucination** — see §3.2 | ❌ |
| **SEM** | Pham, Nguyen et al., CANDAR 2023, pp. 120–126 | Lightweight model-agnostic local-training add-on | Client-side, stackable with FedGRAA | ✅ |
| **SelfFed** | Khowaja, Dev, Anwar, Linguraru, arXiv 2023 → ESWA 2025 | Swin-Transformer self-supervised pretraining + aggregation regularizer | Representation-side, **IoMT** domain; complementary | ✅⚠️ |
| **FedCGS** | Guan, Zhou, Gu (CAS), AAAI 2025, arXiv:2503.06962 | Global feature statistics for training-free one-shot FL | One-shot/statistics-based vs. FedGRAA's iterative/gradient approach | ✅⚠️ |
| **FedLMD** | Lu, Li, Bao, Wang, Qian, Ge, ACM MM 2023, arXiv:2409.13136 | Label-masking distillation; mask pattern proxies local class presence | Family-7 imbalance-inference signal, privacy-safe, coarse | ✅ |
| **OT One-Shot FL** | Chiang, Terai, Chiang, Lin, Ji, Lui, IEEE IoTJ 2024, 11(2):2166–2180 | Optimal-transport neuron alignment before single-round fusion | Solves parameter misalignment in one fuse vs. FedGRAA's multi-round contribution imbalance | ✅ |
| **RepTreeFL** | Ghilea & Rekik, arXiv Dec 2023 | Replica clients + diversity-based tree aggregation, discrepancy-driven dynamic weights | **Strongest methodological neighbour in this family** — discrepancy-driven weights ≈ FedGRAA's adaptive reweighting | ✅⚠️ |
| **Model Fusion via OT** | Singh & Jaggi, NeurIPS 2020 | Optimal-transport model fusion (foundation for OT One-Shot FL above) | Foundational reference only | ✅ |

### 4.9 Detecting class imbalance without a server-side oracle (7 sub-families)

The full taxonomy, worked examples, and communication-cost math live in [`privacy_compliant_fedgraa_redesign.md`](../research_c/privacy_compliant_fedgraa_redesign.md) §2 and are summarized in §7 below. Reference list:

| Family | Method | Authors / Venue / Year | Mechanism | Status |
|---|---|---|---|---|
| 1 — Gradient-composition inference | **Ratio Loss** | Wang, Xu, Wang, Zhu, AAAI 2021, arXiv:2008.06217 | Infers per-round class composition from output-layer gradients | ✅ |
| 1 | **FedImT** | Yan, Li, Ota, Dong, arXiv:2310.14784, 2023 | Online/recursive composition estimate (vs. one-shot) | ✅ |
| 1 | Class-imbalance-reduction via selection | Yang, Wong, Zhu, Wang, Qian, arXiv:2011.11266, 2020 | Uses inferred composition to drive client *selection* | ✅ |
| 1 | **FedReLa** | Hu et al., arXiv:2606.26037, 2026 | Corrects biased boundaries without knowing global class distribution | ✅ |
| 1 | **FedCGNM/FedHOO** | Park, Klabjan, Braun, Li, Ananthanarayanan, ICML 2026, arXiv:2607.01474 | Client-side class-grouped variance-normalized momentum; zero server inference needed | ✅ |
| 2 — HE-based measurement | **Fed-CBS** | Zhang et al., ICML 2023, arXiv:2209.15245 | HE-computed imbalance measure drives class-balanced client sampling | ✅ |
| 2 | **FLICKER** | Guleria, Harshan, Prasad, Bharath, CODS-COMAD 2024, arXiv:2410.21192 | CKKS-HE client-attribute summaries pre-balance before FL | ✅ |
| 3 — SecAgg/DP summary stats | **Practical Secure Aggregation (SecAgg)** | Bonawitz et al. (Google), NeurIPS-W 2016, arXiv:1611.04482 | Server learns only `Σ_k x_k`, tolerates ~1/3 dropout | ✅ |
| 3 | Sample-and-Threshold DP histograms | Bharadwaj & Cormode, AISTATS 2022, arXiv:2112.05693 | Formal `(ε,δ)`-DP histogram via subsampling, no noise on disclosed data | ✅ |
| 3 | **LDP-Fed** | Truex, Liu, Chow, Gursoy, Wei, arXiv:2006.03637, 2020 | Local-DP noise added client-side before submission | ✅ |
| 4 — Loss-only reweighting | **Fed-Focal Loss** | Sarkar, Narang, Rai, FL-IJCAI 2020, arXiv:2011.06283 | Focal-loss reshaping + tunable client-contribution sampling | ✅ |
| 4 | **q-FedAvg** | Li et al., ICLR 2020 | Scalar per-client-loss reweighting | ✅ |
| 4 | **AFL** | Mohri et al., ICML 2019 | Agnostic FL, scalar-loss reweighting | ✅ |
| 5 — Clustering | **IFCA** | Ghosh, Chung, Yin, Ramchandran, NeurIPS 2020, arXiv:2006.04088 | Cluster-membership estimation by local loss + per-cluster GD | ✅ |
| 5 | Byzantine-Robust Clustered FL | Tao, Yang, Kulkarni, arXiv:2306.00638, 2023 | IFCA + trimmed-mean/median aggregation | ✅ |
| 5 | **EMD-CFL** | Zhang, Williams, Toni, arXiv:2506.07769, 2025 | One-shot clustering via Earth Mover's Distance | ✅ |
| 5 | **Clust-PSI-PFL** | Jimenez-Gutierrez et al., IPDPS 2026, arXiv:2512.20363 | Population Stability Index clustering | ✅ |
| 6 — Heterogeneity metrics | EMD non-IID metric | Zhao, Li, Lai, Suda, Civin, Chandra, arXiv:1806.00582, 2018 (already ref [13] in the paper) | EMD between local and population class distribution | ✅ |
| 6 | MID/WCS metrics | Xiao & Wang, IEEE SSCI 2021, arXiv:2109.04094 | Two new imbalance-quantification metrics for distributed settings | ✅ |
| 7 — Restricted-softmax / masked-logit / prototype inference | **FedRS** | Li & Zhan, KDD 2021 | Restricted softmax; restriction shape signals label presence | ✅ *(confirmed this pass)* |
| 7 | **FedLMD** | Lu et al., ACM MM 2023, arXiv:2409.13136 | Masked-logit distillation proxies class presence | ✅ |
| 7 | **FedCK** | Chen, Li, He, J. Supercomputing 2025 | Weight-similarity clustering as label-distribution proxy | ✅ *(confirmed this pass)* |
| 7 | **ProtoHAR** | Cheng, Zhang, Bu, Wang, Wu, Song, IEEE JBHI 2023 | Prototype presence/absence signals imbalance | ✅ *(confirmed this pass)* |
| 7 | **PROTEAN** | Chennoufi et al., ESORICS 2025, arXiv:2507.05524 | Same-domain (federated IDS) prototype exchange under non-IID attacks | ✅ |
| — | Class-imbalance survey | Zhang, Li, Qi, He, arXiv:2303.11673, 2023 | Frames the whole subfield around "estimate imbalance without knowing the data" | ✅ |
| — | Data-Free Client Contribution Estimation via Logit Maximization (**CELM**) | Ukaye, Tastan, Abdu-Aguye, Nandakumar, **arXiv:2605.18892**, 2026-05-17 | Server-side logit-maximization probing yields a client-class evidence matrix; zero auxiliary data, zero client metadata | ✅ *(❓→✅ 2026-07-25 — see §4.7, §4.10, §1 finding N2)* |

### 4.10 Validating adaptive aggregation without a server-held labelled set (6 approaches)

Full mathematical detail (confusion-matrix additivity proof, the federated-backward-pass design, SecAgg/MPC options) is in §7 and the source redesign document. Reference list:

| Approach | Method | Authors / Venue / Year | Status |
|---|---|---|---|
| A — Federated/distributed evaluation | Original FedAvg evaluation protocol | McMahan et al., AISTATS 2017 | ✅ |
| A | **FedDropoutAvg** | Gunesli et al., arXiv:2111.13230, 2021 | ✅ |
| B — Federated gradient-of-validation-loss | *(FedGRAA-v2's own construction, §7 — not a prior-art citation)* | — | — |
| C — Secure MPC validation | **SLVR** | Choi, Rachuri, Wang, Jha, Wang, arXiv:2502.08055, 2025 | ✅ |
| C | **LoByITFL** | Xia, Egger, Hofmeister, Bitar, arXiv:2405.19217, 2024 | ✅ |
| C | **ByITFL** | Xia, Hofmeister, Egger, Bitar, arXiv:2405.08698, 2024 | ✅ |
| D — Validation-free robust aggregation | Geometric median / trimmed mean | Chen, Su, Xu, arXiv:1705.05491, 2017 | ✅ |
| D | **RAGA** | Zuo, Yan, Fan, Hu, Shan, Quek, Zhao, IEEE TMC 2025, arXiv:2403.13374 | ✅ |
| D | **FedTruth** | Ebron, Zhang, Yang, ACISP 2025, arXiv:2311.10248 | ✅ |
| D | **FLTrust** *(the critiqued structural precedent)* | Cao, Fang, Liu, Gong, NDSS 2021, arXiv:2012.13995 | ✅ |
| D | **FedGreed** *(still makes the same D_V assumption, 2025)* | Kritharakis, Makris, Jakovetic, Tserpes, arXiv:2508.18060 | ✅ |
| F — Validation-free contribution/trust | **CoAst** | Wu, Zhang, Li, Xu, Zhong, arXiv:2409.02495, 2024 | ✅ |
| F | **FedPRS** | ACM TIST, DOI 10.1145/3779055 | ✅ *(confirmed this pass)* |
| F | **PACT** | Sayedali Sheykholeslamzadeh, Dima Alhadidi. *"PACT: A Passive Accuracy-Based Trust Metric for Malicious Client Detection in Federated Learning."* **TrustCom 2025**, pp. 1775–1781, DOI 10.1109/Trustcom66490.2025.00206 | ✅ *(resolved via DBLP in the 2026-07-24 re-audit — see §3.3. **This row previously read ❓ "(unresolved)", contradicting §3.3 of this same document; corrected 2026-07-25.**)* |
| F — **data-free contribution estimation** *(new sub-family, 2026-07-25)* | **CELM** | Ukaye, Tastan, Abdu-Aguye, Nandakumar (MBZUAI). *"Data-Free Client Contribution Estimation via Logit Maximization for Federated Learning."* arXiv:2605.18892, 2026 | ✅ — **the state of the art on this exact problem.** Class-wise server-side probing by logit maximization; no raw data, no metadata, no auxiliary set; explicitly up-weights underrepresented-class evidence; simplex-constrained. Its abstract states the §7.1 critique independently: *"metadata-driven weighting can be unreliable or strategically manipulated, while validation-based scoring may be unavailable, biased, or costly to maintain"* |
| F — data-free | **SpectralFed / SpectralFuse** | Ukaye, Abdu-Aguye, Tastan, Nandakumar. *"Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy."* arXiv:2604.22562; **CVPR 2026 FedVision Workshop** | ✅ — matrix von Neumann (spectral) entropy of final-layer updates as a data-free contribution signal; SpectralFed uses normalized entropy **directly as aggregation weights**. Same group as CELM; general FL, no minority framing |
| F | **CT-BRFL** | Wang et al., 2024 (as reported) | ❌ *(hallucination-equivalent — reconfirmed 2026-07-25, 4th independent negative DBLP result; see [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md))* |
| F | **pFedCross** *(surfaced 2026-07-25 saturation check, not previously in this document)* | claimed: "personalized federated cross learning," gradient-approximation α-fairness aggregation for imbalanced federated IDS | ❌ *(hallucination-suspect — zero hits on arXiv title search and DBLP across two query strategies; every web-search description traces to the same paywalled 2026 AI Review survey, never to a primary source; likely a conflation with the real, general-FL, non-IDS **FedCross** (arXiv:2210.08285). Do not cite. See reaudit doc §4.)* |
| F | **REVERB-FL** | Peechara & Sahay, arXiv:2512.13647, IEEE TASLP | ✅ *(real, but audio domain — cross-domain mechanism precedent only)* |
| E — Imbalance-aware metrics (orthogonal, applies to all) | **EdgeDetect** | Mohammad, arXiv:2604.14663, 2026 | ✅ |

### 4.11 Threat model, differential privacy, and testing-plan foundations

| Item | Authors / Venue / Year | Role | Status |
|---|---|---|---|
| Deep Leakage from Gradients (DLG) | Zhu, Liu, Han, NeurIPS 2019, arXiv:1906.08935 | Grounds the residual gradient-inversion risk on the (unchanged) `θ_k^t` training-update channel | ✅ |
| iDLG | Zhao, Mopuri, Bilen, 2020 | Exact single-example label recovery from output-layer gradient sign | ✅ |
| Inverting Gradients | Geiping et al., NeurIPS 2020 | Label-set/per-class-count recovery from aggregated gradients | ✅ |
| GradInversion | Yin et al., CVPR 2021 | Same family, image domain | ✅ |
| DP-SGD / moments accountant | Abadi et al., ACM CCS 2016, arXiv:1607.00133 | Underlies RDP-based privacy composition | ✅ |
| Rényi Differential Privacy | Mironov, IEEE CSF 2017, arXiv:1702.07476 | Tighter composition than basic/advanced composition theorems | ✅ |
| Boosting and Differential Privacy | Dwork, Rothblum, Vadhan, FOCS 2010 | Advanced-composition theorem | ✅ |
| Algorithmic Foundations of DP | Dwork & Roth, monograph, 2014 | Canonical composition-theorem source | ✅ |
| Advances and Open Problems in FL | Kairouz, McMahan et al., arXiv:1912.04977, 2021 | Standard FL threat-model taxonomy | ✅ |
| PL-condition convergence | Karimi, Nutini, Schmidt, ECML-PKDD 2016 | Replaces global convexity for the actual non-convex tanh-MLP | ✅ |
| FedAvg non-IID convergence | Li, Sahu, Talwalkar, Smith, ICLR 2020 | Partial/stochastic-participation convergence framing | ✅ |
| Flower framework | Beutel et al., arXiv:2007.14390 | Framework-agnostic (incl. raw NumPy) FL library — recommended implementation target | ✅ |

### 4.11a Convergence theory for adaptive-weight FL *(new family, 2026-07-25)*

§9 gap #1 asks for a convergence theorem covering the actual non-convex tanh-MLP. Prior versions of this document stocked that gap with generic foundations (PL-condition, FedAvg-non-IID) but no work on *adaptive-weight* convergence specifically. Three usable templates, in increasing order of fit:

| Item | Citation | What it gives FedGRAA |
|---|---|---|
| **FedAAW** — derive the weights *from* the bound | Knowledge-Based Systems 2024, DOI 10.1016/j.knosys.2024.112484 | **The best framing available.** Weights are derived by minimizing the convergence upper bound rather than proposed heuristically. Reframes the theorem from a bolt-on into a derivation (§4.7) |
| **Learnable aggregation weights, jointly optimized** | Parsa, Daghestani, Teixeira, Johansson, arXiv:2511.03529 | A ready-made proof architecture: alternating minimization over model and weights, with convergence guarantees **that hold under adversarial attack**. Also prior art — the *idea* of gradient-optimized aggregation weights is not FedGRAA's to claim |
| **FedAWARE** — fix the heterogeneity measure | Zeng et al., arXiv:2310.02702 | Argues the FL theory/practice gap stems from the *heterogeneity measures* used in analysis; introduces **client consensus dynamics**. Relevant because FedGRAA's one-attack-per-client partition is an extreme-heterogeneity regime where standard bounded-dissimilarity assumptions are close to vacuous — a theorem under them risks being technically true but empty |

**The recurring assumption set** across this family for non-convex adaptive-weight FL: *L*-smooth possibly-non-convex local objectives; bounded stochastic-gradient variance; bounded learning rates; partial participation; and — critically — **aggregation weights bounded away from zero and from above**.

⚠️ **That last condition is the theoretical statement of the simplex projection Algorithm 2 omits.** The empirical divergence documented in [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) and the missing theorem are therefore **the same defect seen from two directions** — the projection is required for the proof, and its absence is observable in the runs. This is a clean and honest framing for the paper. Combined with §7.3's note that FedGRAA-v2 needs one new bounded-estimation-error assumption `ζ`, the target theorem is: **PL-condition convergence to a noise ball under bounded weight deviation and bounded composite-gradient estimation error.** All four ingredients now have citable precedent.

### 4.11b Adversarial robustness of reweighting aggregators *(new family, 2026-07-25)*

§9 gap #3 asks for adversarial evaluation. §4.2 already flags that *"gradient-driven reweighting is a manipulable signal"*; this family is the literature that addresses it.

| Item | Citation | Role |
|---|---|---|
| **SecFedNIDS** | Zhao, Zhang, Guo, Yao, Li. *"SecFedNIDS: Robust defense for poisoning attack against federated learning-based NIDS."* **FGCS 134:154–169, 2022**, DOI 10.1016/j.future.2022.04.010; code at `github.com/zhangzhao156/SecFedNIDS` | **The must-cite in-domain baseline.** Model-level poisoned-model detection + data-level poisoned-traffic filtering by class-path similarity. **+48% accuracy under attack on UNSW-NB15, +36% on CIC-IDS2018** — i.e. evaluated on FedGRAA's own datasets, with public code |
| **WeiDetect** | K. M. Sameera, Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti. *J. Inf. Secur. Appl.* 2025, DOI 10.1016/j.jisa.2025.104275; arXiv:2504.04367 | Weibull-distribution-based poisoning defense for FL-NIDS ✅ *(❓→✅ 2026-07-25)* |
| **Parameterizing poisoning attacks in FL-based intrusion detection** | **ARES 2023**, DOI 10.1145/3600160.3605090 | Attack-side anchor — a parameterized attack model to evaluate against |
| **AdaAggRL** | AAAI 2025, arXiv:2406.14217 | RL-learned aggregation weights *as a poisoning defense* — the adversarial mirror-image of FedGRAA's mechanism (§4.7) |
| **SoK: Benchmarking Poisoning Attacks and Defenses in FL** | arXiv:2502.03801 | Standard evaluation harness |

⚠️ **The structural tension FedGRAA must argue, not merely disclose.** Poisoning defenses identify malicious clients by **statistical deviation** — MESAS, CrowdGuard, FreqFed and AdaAggRL all rest on the premise that anomalous updates are suspect (AdaAggRL states it outright: *benign clients exhibit significantly higher data-distribution stability than malicious clients*). FedGRAA's founding premise is that **the most valuable clients are precisely the divergent, high-loss, statistically anomalous ones.** These premises are in direct conflict. FedGRAA is therefore not merely *undefended* against poisoning — it is a mechanism that **structurally amplifies exactly the signal a poisoning defense suppresses**, and an adversary who makes an update look like a rare-class contribution gets up-weighted by construction.

This recasts **Jiang et al. (Neurocomputing 2024)**: their *filtering-and-voting* layer is a proposed **resolution** of this tension, not merely a parallel philosophical argument. FedGRAA's omission of any such layer is a design decision that needs defending. Minimum credible robustness section: label-flipping and model-poisoning under the SecFedNIDS protocol; **an adaptive attacker that fabricates rare-class gradient signatures to farm up-weighting** (the mechanism-specific attack nobody else can run for you); AutoAdapt as the adaptive-threat limit; and an explicit statement of whether γ trades detection performance against poisoning exposure.

### 4.12 Broader FL-IDS context, surveys, and very recent (2026) same-domain work

> ⚠️ **Survey-evidence caveat, added 2026-07-25 — read before relying on any survey row below.** The novelty claim has leaned on *"no FL-IDS aggregation survey lists a gradient-refinement method."* That claim must now hold against **six** independent surveys, not three (three were added on 2026-07-25, below). **No survey taxonomy in this table has been read in primary text by any pass** — all six are paywalled (ACM DL returns HTTP 403; Springer redirects to an identity provider), so every "does not list a gradient-refinement method" assertion rests on secondary description. Given that FedPGO (§4.4) exists and postdates most of them, **the absolute framing is no longer defensible.** Use the comparative framing instead: *"surveys of FL-IDS aggregation organize the field around size-, loss-, similarity- and cluster-based weighting; gradient-refinement approaches appear only in the most recent work (FedPGO 2026, NIDS-FGPA 2024)."*

| Item | Authors / Venue / Year | Relevance | Status |
|---|---|---|---|
| **Survey: FL for IDS — Concept, Architectures, *Aggregation Strategies*, Challenges, Future Directions** | ACM Computing Surveys, 2024, **DOI 10.1145/3687124** | The original taxonomy-gap citation | ✅ *(DOI added 2026-07-25)* |
| **Survey: Intrusion Detection Based on FL — A Systematic Review** | ACM Computing Surveys, 2025, **DOI 10.1145/3731596** | Corroborating systematic review | ✅ *(DOI added 2026-07-25)* |
| **Survey: privacy-preserving FL for IDS** | *Artificial Intelligence Review*, DOI 10.1007/s10462-026-11519-4, 2026 | Third corroborating survey | ✅ *(confirmed via DOI)* |
| **Survey: Taxonomy and Survey of Collaborative IDS using FL** *(new 2026-07-25)* | Aulia Arif Wardana, Parman Sukarno. **ACM Computing Surveys 57(4), Article 88, Dec 2024**, 36 pp., **DOI 10.1145/3701724** | Fourth survey — **was absent from all prior passes of this document** | ✅ |
| **Survey: federated learning in intrusion detection** *(new 2026-07-25)* | *J. Parallel and Distributed Computing*, **DOI 10.1016/j.jpdc.2024.104976** | Fifth survey — absent from all prior passes | ✅ |
| **Survey: Model optimization in federated IDS — *aggregation*, communication, personalization** *(new 2026-07-25)* | *J. Cloud Computing*, **DOI 10.1186/s13677-026-00901-w**, Apr 2026 | ⚠️ **The most on-point survey in existence for the novelty claim** — its entire first axis is aggregation in FL-IDS, and it postdates this document's original compilation. Absent from all prior passes. Obtain and read before finalizing Related Work | ✅ |
| **Sentinel** | arXiv:2510.23019, 2025 | Personalized federated IDS, dual-model knowledge distillation | ✅ |
| **F-ACVAE** | arXiv:2607.04698, 2026 | Federated conditional VAE for IDS, constrained-momentum Gaussian aggregation | ✅ |
| **StatAvg** | arXiv:2405.13062 / IEEE journal 2025 | Per-feature statistics sharing corrects non-IID feature skew for FL-IDS | ✅ |
| **PerFedHypID** | Scientific Reports, 2025 | Hypernetwork-based personalized aggregation via client embeddings, IDS | ✅ |
| **G-PFL-ID** | IoT (MDPI), 2026 | Graph-driven personalized FL for unsupervised IDS, non-IID IoT | ✅ |
| **FSFL-IDS** | Springer chapter, ~2025 | Few-shot FL-IDS for UAV/FANET, notes non-IID sensitivity | ✅ |
| **Tri-LLM Cooperative Federated Zero-Shot IDS** | arXiv:2602.00219, 2026 | LLM-based zero-shot IDS, trust-aware aggregation | ✅ |
| **FedNIDS** | Quoc H. Nguyen, Soumyadeep Hore, Ankit Shah, Trung Q. Le, Nathaniel D. Bastian. *"FedNIDS: A Federated Learning Framework for Packet-Based Network Intrusion Detection System."* **DTRAP 6(1):1–23, 2025**, DOI 10.1145/3696012 | Packet-based FL-IDS framework | ✅ *(❓→✅ 2026-07-25, DBLP)* |
| **FedKSam\*** | *"FedKSam\*: A Two-Phase Federated Learning Intrusion Detection Algorithm."* IEEE Xplore doc. **10661778** | Two-phase: SAM + KD against non-IID drift, then classifier decision-boundary re-optimization; targets heterogeneity **and class imbalance** | ✅ *(❓→✅ 2026-07-25 — promoted to §4.4 as a direct competitor)* |
| **WeiDetect** | K. M. Sameera, Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti. *J. Inf. Secur. Appl.* 2025, **DOI 10.1016/j.jisa.2025.104275**; arXiv:2504.04367 | Weibull-distribution poisoning defense for FL-NIDS | ✅ *(❓→✅ 2026-07-25 — see §4.11b)* |
| **FADngs** *(moved here 2026-07-25)* | Boyu Dong, Dong Chen, Yu Wu, Siliang Tang, Yueting Zhuang. *"FADngs: Federated Learning for Anomaly Detection."* **IEEE TNNLS 36(2):2578–2592, 2025**, DOI 10.1109/TNNLS.2024.3350660 | Federated anomaly detection | ✅ *(❓→✅ 2026-07-25, PubMed 38241100)* |
| **XAI-SOH-FL** *(new this pass)* | Aslam et al. (NUST), arXiv:2606.00134, 2026-05-28 | Similarity-thresholded dynamic aggregation (replaces SOH-FL's manually-tuned coefficient) + SHAP-based explainability, for heterogeneous IoT IDS | ✅ *(confirmed via arXiv abstract + PDF)* |
| **FedDrone-Shield** *(new this pass)* | *"A federated learning-benchmarking framework for privacy-preserving UAV intrusion detection using adaptive aggregation algorithms,"* Scientific Reports, 2026, nature.com/articles/s41598-026-50865-9 | Benchmarks FedAvg/FedProx/FedAdam/FedMedian/ClusterAvg for UAV intrusion detection; FedAdam and ClusterAvg win | ✅ *(confirmed via Nature/Scientific Reports listing — note: FedDrone-Shield is this paper's proposed architecture name, not a separate publication; an earlier draft of this table listed it as a second, distinct row — corrected)* |

---

## 5. Intrusion-detection datasets

| Dataset | Creator / Paper | Size / Features | Classes | FL relevance |
|---|---|---|---|---|
| **CIC-IDS-2017** ✅ *(FedGRAA's primary dataset)* | CIC (UNB); Sharafaldin, Lashkari, Ghorbani, ICISSP 2018 | ~50 GB PCAP, 80+ CICFlowMeter features, ~2.8M labeled flows | Brute Force, DoS variants, Heartbleed, Web (incl. SQLi), Infiltration, Botnet, PortScan, DDoS | De-facto benchmark; days/attack-types map naturally to non-IID client shards — exactly FedGRAA's setting |
| **UNSW-NB15** ✅ *(FedGRAA's secondary dataset)* | Moustafa & Slay, MilCIS 2015 (+ IS Journal 2016) | 2,540,044 records, 49 features, 175,341/82,332 train/test split | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms | Train/test mismatch and "Generic" dominance are known issues |
| **NSL-KDD** (+ KDD Cup 99 / DARPA 1998) ✅ | Tavallaee, Bagheri, Lu, Ghorbani, CISDA 2009 | 41 features, ~125,973/22,544 train/test | DoS, Probe, R2L, U2R + Normal | Very dated (late-1990s), legacy comparability only |
| **CSE-CIC-IDS2018** ✅ | CSE + CIC/UNB, ICISSP-2018 methodology, AWS Open Data | 80 features (CICFlowMeter-V3), 10M+ records | Brute-force, Heartbleed, Botnet, DoS, DDoS, Web, Infiltration | Larger successor to CIC-IDS-2017 |
| **"CIC-IDS2023"** ⚠️ | — | — | — | **Does not exist.** Re-point to CICIoT2023 or CSE-CIC-IDS2018 |
| **CICIoT2023** ✅ | CIC/UNB; Neto et al., Sensors 23(13):5941, 2023 | 105 real IoT devices, 46+ features, millions of flows | 33 attacks in 7 categories incl. Mirai | Modern IoT benchmark; device heterogeneity → natural non-IID FL testbed |
| **FLNET2023** ✅ *(FL-native)* | Kumar, Liu, Tayeen, Misra, Cao, Harikumar, Perez, IEEE MILCOM 2023, pp. 345–350 | Naturally distributed across nodes | — | Most on-point benchmark for FedGRAA's FL claim; niche/low-citation |
| **NF-UNSW-NB15-v2 / NF-BoT-IoT-v2 / NF-ToN-IoT-v2 / NF-CSE-CIC-IDS2018-v2, merged as NF-UQ-NIDS-v2** ✅ *(new 2026-07-25 — **the recommended upgrade**)* | Sarhan, Layeghy & Portmann, *"Towards a Standard Feature Set for Network Intrusion Detection System Datasets,"* **Mobile Networks and Applications 2022**, arXiv:2101.11315. **NetFlow V3** (UQ Cyber Research Centre, 2025) adds 10 temporal features | Four independently-collected NIDS datasets re-expressed in **one shared 43-feature NetFlow schema** (53 under V3) | Union of the four sources' attack taxonomies | ⚠️ **The single best experimental upgrade available to FedGRAA.** A shared feature schema across four independently-collected datasets *is* a cross-silo federated setting: assign one dataset per silo and the non-IID structure is **real rather than synthetically partitioned** — no Dirichlet-α simulation, no one-attack-per-client artifice. It therefore answers §9 gap #4 (modern datasets), gap #5 (scale + realistic non-IID), **and** §7.1's structural objection that the current partition presupposes server-side knowledge of client label distributions — all at once. Cost: a feature-extraction pass, **no algorithmic change** |

> ⚠️ **A validity threat, not merely a currency complaint (added 2026-07-25).** This document previously framed "two legacy datasets (2015/2017)" as a freshness gap. The NetFlow-benchmark literature frames it harder: **CIC-IDS-2017 and CSE-CIC-IDS-2018 contain labelling errors and unrealistic temporal separation between attack and benign traffic**, and a systematic analysis of seven popular NIDS benchmarks (El Mahdaouy, Ait Yahia, Oualil & Berrada, arXiv:2602.05594, 2026 — title/authors/date ✅ confirmed from the arXiv abstract page; the six-flaw enumeration below is **secondary description, verify against the full PDF before citing**) identifies six recurring design flaws — limited feature diversity, high feature interdependence, ambiguous ground truth, collapse of traffic distributions, artificial diversity generation, labelling inaccuracies — that **correlate with poor generalization in operational deployment.** See also *Benchmarking the Benchmark — Analysis of Synthetic NIDS Datasets* (arXiv:2104.09029) and *Temporal Analysis of NetFlow Datasets for NIDS* (arXiv:2503.04404).
>
> **Why this bites FedGRAA specifically.** The headline result is per-minority-class accuracy on CIC-IDS-2017's rarest classes — Heartbleed, Infiltration, Web/SQLi — which are precisely the classes where a handful of labelling errors moves the reported number materially, and §7.3 already records that Heartbleed has **≈1 held-out sample**. A reviewer who knows this literature will read "≈88% on minority classes, +55% over FedAvg" and ask for the confidence interval on a class with single-digit support. **The ≥10-seed / Wilcoxon / Holm–Bonferroni infrastructure already built in Stage B (`fedgraa/stats.py`) is the answer, and should be foregrounded for rare classes rather than treated as checklist item #8.**

---

## 6. Venue rankings, confirmed

| Venue | Field | CORE/ICORE2026 | FL-IDS relevance |
|---|---|---|---|
| CCS | Security | A* | MESAS, EIFFeL, Armadillo |
| NDSS | Security | A* | Entente, FreqFed, CrowdGuard |
| IEEE S&P | Security | A* | Flagship adversarial/FL security |
| AAAI | AI/ML | A* | FedCGS, FedFSA, DA-PFL, FedFSL-CFRD |
| IJCAI | AI/ML | A* | FedAPA, FedDLAD, FBLG |
| NeurIPS | ML | A* | FedNova, FedGMKD |
| ICML | ML | A* | SCAFFOLD, FedSWA |
| ICLR | ML/DL | A* | FedOpt, MPFT |
| CVPR | Vision | A* | FedAWA (FL mostly for images; less IDS-relevant) |
| IEEE TIFS | Security (journal) | Q1 (no CORE conf rank) | FeCoGraph. *(⚠️ corrected 2026-07-25: **FedP3E removed** — arXiv preprint only, §4.6. **PROTEAN** is ESORICS 2025 / arXiv:2507.05524 per §4.6, not TIFS — this row previously contradicted §4.6 and has been reconciled.)* |
| Computer Networks (Elsevier) | Networking (journal) | Q1 (no CORE conf rank) | **FedPGO** — the closest competitor in the corpus (§4.4). Relevant to venue strategy: a Q1 journal, widely read by IDS reviewers, so "not CORE A*/A" will not read as "obscure" |
| **PAKDD** | Data mining | **B** ⚠️ — **independently reconfirmed 2026-07-24** (was A under CORE2018; **B** under CORE2023 per a second, independent source, matching ICORE2026) | **FedGRAA's current target venue** |

> **Any framing that claims "PAKDD = CORE A" is factually wrong and must not appear in a manuscript.** Position PAKDD honestly as a realistic near-term target and a stepping-stone toward the AAAI/IJCAI-tier extension, per the roadmap in [`publication_roadmap.md`](../current/publication_roadmap.md).

---

## 7. The two structural privacy gaps and the FedGRAA-v2 redesign

Full derivation, exact byte-level communication accounting, formal DP-composition math, a complete threat model, and a full test-suite specification live in [`../research_c/privacy_compliant_fedgraa_redesign.md`](../research_c/privacy_compliant_fedgraa_redesign.md) (730 lines, four independently-dated verification passes). This section is the load-bearing summary for Stage C decision-making — read the source document before implementing anything.

### 7.1 The two flaws

1. **Oracle knowledge of rare classes/clients.** The server hard-codes "the last `n` underrepresented classes" and the experimental partition assumes one attack type per client — i.e., the server (or experiment designer) is assumed to already know each client's label distribution. Real FL servers never see client labels; assuming this a priori is a privacy violation *and* an unrealistic experimental setup.
2. **A server-held, fully-labelled validation set `D_V`.** Algorithm 2 evaluates `L_h, L_m, L_u` on a centrally-pooled `D_V` — exactly the thing FL exists to avoid. This is structurally identical to the "root dataset" assumption in Byzantine-robust defenses like FLTrust (NDSS 2021), which follow-up work (FedTruth) explicitly calls "impractical," and which even brand-new 2025 papers (FedGreed) still make without apology — this is a known, common, but fixable shortcut, not a niche flaw specific to FedGRAA.

### 7.2 The recommended fix — FedGRAA-v2

Combining Family 1 (gradient-composition inference) + Family 3 (SecAgg/DP summary statistics) from §4.9, and Approach B (federated backward pass) from §4.10:

- Replace the hard-coded rare-class list with an **online-inferred, SecAgg-summed per-round class-frequency estimate** `p̂^t`, computed from label-count histograms clients already hold locally.
- Replace the server-held `D_V` with a **federated backward pass**: the server broadcasts the candidate aggregate `θ'_g` (already public math on already-shared parameters); each client computes the gradient of its own contribution to the composite loss on its own held-out split and returns only `∇_P L_k'(θ'_g)`; the server sums these (optionally through SecAgg) to recover `∇_P L' = Σ_k ∇_P L_k'(θ'_g)` **exactly** — mathematically identical to what a server-held `D_V` would compute, at the cost of extra communication rounds.
- **Zero change to FedGRAA's core equations (Eq. 7–10).** This is a methodology patch, not a rewrite — the intellectual contribution (γ-tunable, gradient-refined adaptive averaging) is preserved exactly.

### 7.3 The honest cost, computed exactly (not estimated) from the live codebase

- **Communication:** federating the validation-gradient exchange (`E'=5` inner epochs/round) increases total training-run traffic by **≈6×** (≈5.61 GB → CIC-IDS-2017, ≈2.98 GB → UNSW-NB15, computed from the exact layer sizes in `fedgraa/config.py`/`model.py`). The SecAgg histogram and cryptographic-masking overhead are, by contrast, **six or more orders of magnitude smaller** — essentially free.
- **Privacy accounting:** if DP-noised histograms are used, the composed `(ε,δ)` over all `T=200` rounds must be reported (not a per-round `ε`) — advanced composition gives `ε_total ≈ 8.89` at `ε_round=0.1, δ=10⁻⁵`; RDP-based accounting is recommended as tighter still.
- **A genuinely non-obvious, dataset-dependent finding:** DP noise on the rare-class histogram is *unusable* for CIC-IDS-2017's rarest classes (Heartbleed, ≈1 held-out sample) even at the loosest commonly-reported budget, but only *mildly* degraded for UNSW-NB15's rarest class (Worms, ≈17 held-out samples, ≈5% relative error at `ε=10`) — because of UNSW's smaller `K` and less extreme rarity. **Report both datasets' tables side by side; do not generalize from one.**
- **Convergence proof impact:** needs one new, standard assumption (bounded estimation error `ζ` between the true and federated composite-loss gradient) — a noise-ball weakening of the existing theorem, not a different theorem.
- **Threat model:** honest-but-curious server + SecAgg's standard collusion threshold; a fully malicious server or Byzantine clients are explicitly **not** defended by this fix (name as out-of-scope, not silently assumed away); the unchanged `θ_k^t` training-update channel remains exposed to gradient-inversion attacks (DLG) exactly as in the original paper — the redesign narrows scope to the two named flaws only.
- **Implementation scope:** mapped function-by-function against the live `fedgraa/` codebase — **5 small, well-isolated additions** (a SecAgg mock, a per-client histogram function, a per-client gradient-contribution refactor of one existing method, one new metrics helper, one new privacy utility), **zero changes** to `model.py`'s forward/backward pass or `aggregation.py`'s simplex projection. A previously-unremarked fact: `DatasetSpec.rare_client_ids()` **already has** a data-driven, non-oracle fallback path in the codebase today that the current reproduction runs simply never exercise.

---

## 8. Synthesized positioning and differentiation narrative

> ⚠️ **This section was substantially rewritten on 2026-07-25.** The framing below — "the verified gap … the nearest works each miss one axis" — is **retained for the record but is no longer accurate**, because FedPGO misses no axis (§4.4, §1 finding N1). The superseding narrative is §8.1. Read §8.1 first; treat the list below as the *pre-FedPGO* competitive map, still useful for the individual differentiations it records.

**The gap as previously stated (superseded):** no A*/A paper proposes gradient-refined adaptive averaging specifically for non-IID IDS with limited data per client. The nearest works each miss one axis:

- **Entente (NDSS 2025)** — adaptive contribution scaling, but graph-based cross-silo, not gradient-refined minority-boosting for tabular IDS.
- **FedAPA (IJCAI 2025)** — server-side gradient-based weights, but personalization, not global minority-boosting IDS.
- **FedAWA (CVPR 2025)** — direction-based weights, but rewards alignment (suppresses minorities) and is general-purpose.
- **FedMADE (ISC 2024) / SBFedNIDS (ICASSP 2025)** — dynamic aggregation for minority IDS, but via clustering/contribution or balance statistics, not gradient refinement, at lower-tier venues.
- **NIDS-FGPA (PLOS ONE 2024)** — the closest paper on the exact two-way intersection (gradient-weighting × federated NIDS), but similarity-to-reference weighting for communication reduction, not adaptive minority-boosting, and non-A*/A.
- **Jiang et al. (Neurocomputing 2024)** *(new this pass — see §4.7)* — the closest paper on the *philosophical* axis specifically: it explicitly argues for protecting rare/valuable client contributions rather than filtering them as outliers, the same core move FedGRAA makes. It differs on mechanism (DRL-learned filtering-and-voting weights, not gradient refinement) and domain (general FL, not IDS) — but this paper means the "protect, don't filter, the rare" argument is no longer uniquely FedGRAA's, only FedGRAA's *IDS-specific, gradient-based instantiation* of an argument the field has now made twice.

### 8.1 The differentiation narrative, rewritten 2026-07-25

**Two framings must be abandoned outright.** First, *"existing adaptive aggregators all suppress minorities"* — false as stated, because FedAWARE, Jiang et al., AdaAggRL and CELM form a counter-current that does the opposite; leaving it will read as a straw man. Second, *"the intersection is unoccupied"* — false, because FedPGO occupies it on FedGRAA's own dataset.

**The replacement paragraph, for Related Work:**

> "Adaptive aggregation weighting has converged on three signals: alignment with the global direction (FedAWA), client loss or accuracy (FedNolowe, DWAFL, PACT), and statistical outlier filtering (MESAS, CrowdGuard, FreqFed). In imbalanced intrusion detection these systematically suppress the divergent, high-loss, minority-class clients that hold the rarest attacks. A counter-current has emerged: FedAWARE maximizes gradient diversity rather than alignment; Jiang et al. (2024) and AdaAggRL (AAAI 2025) learn weights that protect rare contributions through reinforcement learning; CELM (2026) up-weights clients with discriminative evidence for underrepresented classes without any server-side data. In intrusion detection specifically, FedPGO (2026) reweights clients by gradient orthogonality on CIC-IDS2017, and NIDS-FGPA (2024) weights by gradient similarity for communication efficiency. FedGRAA sits inside this counter-current, at a point none of them occupies: a **single-global-model** aggregation rule with an explicit **γ-tunable trade-off** between rare-class emphasis and conventional size-proportional weighting, evaluated on **per-class recall for rare attack types** rather than aggregate accuracy or personalization gain. Its claim is not that the space is empty, but that this point in it is unreported and that the rare-attack evaluation axis is one no adjacent work supplies."

**Why this is the right claim to make.** It is narrow, it is true, it concedes the overlaps before a reviewer finds them, and it rests the contribution on something FedGRAA actually has and nobody else reports — the γ trade-off and rare-attack per-class recall — rather than on an absence that no longer exists. It is a defensible **PAKDD-tier** claim. It is **not** an A\*/A-tier claim without the additional work in §9.

**Practical consequence for Stage C.** Related Work must now cite, at minimum: **FedPGO** (highest priority — same domain, same dataset, same mechanism family), **CELM**, **FedAWARE**, **AdaAggRL**, **Jiang et al.**, and **NIDS-FGPA**. Omitting FedPGO would be the single most likely fatal miss a domain-literate reviewer could flag.

**The FedGRAA-v2 contribution statement, narrowed** (the word "first" no longer survives — CELM is explicitly data-free and minority-targeted, so "first fully FL-compliant" is false without the IDS qualifier):

> "Prior gradient-refined adaptive aggregation for FL-IDS assumed server-side knowledge of client class distributions and a server-held validation set — a privacy leak. We show this assumption can be removed via secure-aggregation/DP-based class-frequency estimation and a federated backward pass that reproduces the **exact** composite-loss gradient without centralizing any labelled data, at a measured, honestly-reported cost. Contemporary data-free contribution estimators (CELM, SpectralFed, 2026) achieve a related goal **approximately** at near-zero communication cost; we therefore report the trade-off directly — what exactness buys in minority-class recall, and what it costs in traffic — making FedGRAA the first fully FL-compliant gradient-refined adaptive-aggregation method **for intrusion detection**."

*(That reframing is not a retreat: it converts a redesign into a comparative experiment with a publishable result either way. See §10.)*

---

## 9. Gap-closing checklist for an A*/A submission

**0. ⚠️ BLOCKING, added 2026-07-25 — read FedPGO (DOI 10.1016/j.comnet.2026.112210) and CELM (arXiv:2605.18892) in full before any Related Work is drafted.** Both are currently known only through abstracts, DBLP/PubMed records and consistent secondary summaries — enough to establish existence, venue and topic, **not** enough to characterize their limitations or write a differentiation paragraph. Nothing else in Stage C should be finalized ahead of these two reads. FedPGO is Elsevier-paywalled; obtain via institutional access.

1. **Formal convergence theorem** under bounded heterogeneity + limited samples, covering the actual non-convex tanh-MLP (PL-condition, not global convexity). ⚠️ *Reframed 2026-07-25 (§4.11a):* **derive the weights from the bound (FedAAW) rather than proving a theorem around heuristic weights**; adopt the alternating-minimization framing of Parsa et al. (arXiv:2511.03529); use a consensus-dynamics-style heterogeneity measure (FedAWARE) so the theorem is not vacuous in the one-attack-per-client regime; and state **bounded aggregation weights (= the simplex projection) explicitly as a proof condition**, tying the theorem directly to the divergence observed in [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) — they are the same defect from two directions. Target: PL-condition convergence to a noise ball under bounded weight deviation and bounded composite-gradient estimation error `ζ`.
2. **Head-to-head baselines**, in priority order: FedAvg, FedProx, SCAFFOLD, FedNova, FedOpt (foundations) **+** FedAPA, FedAWA, DA-PFL, FedDRL, FedNolowe (adaptive family) **+** FedMADE, SBFedNIDS, DWAFL, FD-IDS, DS-FedIDS, CACFL, Chunduru Sri Abhijit et al.'s rare-attack transfer-learning method (IDS-specific) **+** NIDS-FGPA **+** Jiang et al.'s DRL filtering-and-voting aggregator. ⚠️ *Added 2026-07-25:* **FedPGO (top priority — same domain, same dataset, same mechanism family)**, **CELM**, **FedAWARE**, **AdaAggRL**, **PerFedHypID**, **FedKSam\***, **AMAFed**, and **G-PFL-ID** (as adjacent, not head-to-head — it is unsupervised/one-class).
3. **Adversarial robustness** — label-flipping, model poisoning, a "pill"-style sub-structure attack; AutoAdapt as the adaptive-threat limit. ⚠️ *Sharpened 2026-07-25 (§4.11b):* the **premise conflict must be argued, not merely disclosed** — poisoning defenses treat statistical deviation as evidence of malice, while FedGRAA treats it as evidence of value, so the mechanism *structurally amplifies exactly the signal a defense suppresses*. Add the mechanism-specific attack (**an adversary fabricating rare-class gradient signatures to farm up-weighting**), baseline against **SecFedNIDS** (same datasets, public code) and **WeiDetect**, use the **ARES 2023** attack parameterization and the **SoK harness (arXiv:2502.03801)**, and state whether γ trades detection performance against poisoning exposure.
4. **Modern + FL-native datasets** — add CICIoT2023 and FLNET2023 (and correct any phantom "CIC-IDS2023" reference). ⚠️ *Upgraded 2026-07-25 (§5):* the strongest move is the **NF-\*-v2 / NF-UQ-NIDS-v2 / NetFlow-V3** family — four datasets in one 43-feature schema, i.e. a **native cross-silo federated benchmark with real rather than simulated non-IID structure**. Also cite the CIC-IDS-2017 labelling-error critique whenever rare-class numbers are reported.
5. **Scale + extreme non-IID** — 50–200+ clients; 10–50 samples/class regimes; a Dirichlet-`α` partition alongside the current one-attack-per-client extreme. *Partly subsumed by #4* — the NF-v2 route supplies real heterogeneity without synthetic partitioning.
6. **Ablation** of the gradient-refinement components and the `γ` coefficient — already scoped and partly infrastructure-complete in Stage B (see [`stage_b_plan.md`](../current/stage_b_plan.md)).
7. **Overhead reporting** — communication/latency vs. accuracy trade-off, computed exactly (FedMADE reports ~4.7%; FedGRAA-v2's own exact number is ≈6×, see §7.3). ⚠️ *Reframed 2026-07-25:* report the ≈6× **against CELM/SpectralFed's near-zero-cost approximation**, and make *"is exact worth 6×?"* an experiment rather than a caveat (§10).
8. **Statistical rigor** — ≥10 seeds, paired significance tests (t-test or Wilcoxon), Holm–Bonferroni correction across the baseline grid — infrastructure-complete via `fedgraa/stats.py` (Stage B). ⚠️ *Promoted 2026-07-25:* this is no longer a checklist item but **the direct answer to the labelling-error/low-support critique in §5** — lead with it for rare classes rather than reporting it at the end.
9. **Privacy fix (§7)** — if the FL-compliance angle is chosen, implement FedGRAA-v2 and report the v1-vs-v2 accuracy/overhead trade-off explicitly. ⚠️ *Narrowed 2026-07-25:* **drop "first fully FL-compliant"** (CELM is data-free and minority-targeted); position against the data-free contribution-estimation line with exactness-vs-cost as the contribution (§8.1).
10. **Optional hybrid** — combine with prototype guidance (FedGMKD/FedFSA/PROTEAN) for data-poor clients, matching the original review's Part-8 recommendation.

---

## 10. Stage C decision framing

Per [`publication_roadmap.md`](../current/publication_roadmap.md) §2, Stage C's novelty angle is an open decision point to bring to the mentor, not something to finalize unilaterally in this document. What this consolidated literature review adds to that decision:

| Angle | What the literature supports | Strongest baselines it must beat | Risk |
|---|---|---|---|
| **A. Few-shot/prototype-guided hybrid aggregation** (the original review's top recommendation) | A crowded but active family (§4.5, §4.6) with strong, very recent (2025) competitors already at AAAI/IJCAI/TIFS — FeCoGraph and FedFSA are the closest same-problem papers | FeCoGraph (IEEE TIFS 2025), FedFSA (AAAI 2025), FedGMKD (NeurIPS 2024), DFML | More technically ambitious; competitive with 2025-fresh work, so the novelty window is narrowing quickly |
| **B. Privacy-compliant redesign (FedGRAA-v2, §7)** | A real, underlying flaw in the *existing* paper that the field has independently and repeatedly flagged as unacceptable elsewhere (FedTruth, CoAst — and now **CELM**, which states the §7.1 critique almost verbatim) — fixing it is close to fully scoped already (7 solution families catalogued, exact communication/DP math done, a full test-suite spec written, implementation mapped function-by-function against the live codebase) | FedTruth, CoAst, RAGA (as the "does validation-free already capture the gain?" null baseline); **CELM and SpectralFed as the data-free approximation baselines**; PROTEAN as the same-domain prototype-exchange alternative design point | ⚠️ *Rationale revised 2026-07-25.* The "**less crowded**" claim is **out of date** — CELM, SpectralFed/SpectralFuse, PACT, FedPRS and CoAst now constitute a named research line (*data-free contribution estimation*, §4.10). B is still the right choice, but for a **different and better reason**: the comparison has shifted from *"does this work?"* to *"is exact worth 6×?"* — FedGRAA-v2 recovers the composite-loss gradient **exactly** at ≈6× traffic, while CELM/SpectralFed approximate it at near-zero cost. That is a comparative experiment with a **publishable result either way**, which is stronger than the original "fix a flaw" framing |
| **C. Both, sequenced** | Nothing prevents shipping B first (a smaller, well-scoped implementation effort, §7.3) and layering A as a subsequent extension or follow-up paper | — | Time cost; but B's test suite and byte-accounting are already spec'd, so B is the lower-marginal-effort path to a defensible, submittable result |
| **D. NF-v2 cross-silo benchmark migration** *(new option, added 2026-07-25 — §5)* | Not a novelty angle but an **evidence** angle: re-running the existing method on the NF-\*-v2 / NF-UQ-NIDS-v2 family (four datasets, one 43-feature schema, one dataset per silo) supplies **real rather than simulated non-IID structure** | The same baseline grid, on better data — no new mechanism to beat | ⚠️ **The highest ratio of reviewer-objections-closed to effort available.** It answers §9 gaps #4 and #5 *and* the §7.1 objection that the current partition presupposes oracle knowledge of client label distributions — with data engineering only, **no algorithmic risk**. It **composes with A or B rather than competing**, so it belongs in the mentor conversation as an addition to whichever angle is chosen, not as a fourth alternative to choose between |

> ⚠️ **Amendment, 2026-07-25.** The recommendation below (Option B) **survives** the deep-research pass, but its *rationale has changed* and one new consideration has been added. B is no longer attractive because the space is uncrowded — it is not (see the revised B row above). It is attractive because it now converts into a **comparative experiment with a publishable result either way**: does exact federated validation buy enough minority-class recall to justify ≈6× traffic, versus a data-free approximation (CELM/SpectralFed)? Separately, **Option D (NF-v2 benchmark migration) is the cheapest large win available and composes with any other choice** — if Stage C is time-constrained, D should be done regardless of which novelty angle is selected. Bring all four to the mentor, with D framed as an addition rather than an alternative.

**This document's contribution to that decision, stated plainly:** Option B (the privacy-compliant redesign) is **substantially closer to implementation-ready** than Option A — §7 and its source document already provide exact byte counts, a formal threat model, a full test specification, and a function-level mapping to the existing codebase, none of which exist yet for the few-shot/prototype hybrid. Option A remains the more ambitious, more "novel mechanism" framing that the original review recommended, but the competitive landscape in that family (§4.5–§4.6) has grown more crowded with each pass (FeCoGraph, FedFSA, PROTEAN, FedP3E all landed in 2025 alone). Bring both options, with this trade-off explicit, to the mentor conversation the roadmap already calls for.

---

## 11. Full alphabetized bibliography

*(All entries below were verified via arXiv, DBLP, Semantic Scholar, OpenReview, IEEE Xplore, ACM DL, or official venue/dataset portals across the five source passes plus this consolidation pass. ✅ = confirmed, ⚠️ = confirmed with a correction applied, ❓ = unverified/cite with caution, ❌ = confirmed hallucination, do not cite.)*

- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang. "Deep Learning with Differential Privacy." CCS 2016. arXiv:1607.00133. ✅
- AugFL — Yue, Qin, Deng, Ren, Zhang, Zhang. IEEE/ACM ToN 2025 (in-press). arXiv:2503.02154. ✅
- AutoAdapt — Krauß, König, Dmitrienko, Kanzow. NDSS 2024. ✅⚠️ (title: "Adaption")
- Armadillo — Ma, Guo, Karthikeyan, Polychroniadou. CCS 2025. arXiv:2511.10863. ✅⚠️
- Astraea — Duan et al. IEEE ICCD 2019. arXiv:1907.01132. ✅
- AFL — Mohri, Sivek, Suresh. ICML 2019. ✅
- Bharadwaj & Cormode. "Sample and Threshold Differential Privacy." AISTATS 2022. arXiv:2112.05693. ✅
- Bonawitz et al. "Practical Secure Aggregation for Federated Learning." NeurIPS-W 2016. arXiv:1611.04482. ✅
- Byzantine Gradient Descent — Chen, Su, Xu. arXiv:1705.05491, 2017. ✅
- Byzantine-Robust Clustered FL — Tao, Yang, Kulkarni. arXiv:2306.00638, 2023. ✅
- ByITFL — Xia, Hofmeister, Egger, Bitar. arXiv:2405.08698, 2024. ✅
- Cerberus — Naseri, Han, Mariconti, Shen, Stringhini, De Cristofaro. CCS 2022. arXiv:2209.03050. ✅
- CIC-IDS-2017 — Sharafaldin, Lashkari, Ghorbani. ICISSP 2018. ✅
- CICIoT2023 — Neto, Dadkhah, Ferreira, Zohourian, Lu, Ghorbani. Sensors 23(13):5941, 2023. ✅
- "CIC-IDS2023" — no such dataset. ❌ (re-point to CICIoT2023/CSE-CIC-IDS2018)
- Clust-PSI-PFL — Jimenez-Gutierrez et al. IPDPS 2026. arXiv:2512.20363. ✅
- CoAst — Wu, Zhang, Li, Xu, Zhong. arXiv:2409.02495, 2024. ✅
- CrowdGuard — Rieger, Krauß, Miettinen, Dmitrienko, Sadeghi. NDSS 2024. ✅⚠️
- CSE-CIC-IDS2018 — CSE/CIC-UNB. ICISSP-2018 methodology. ✅
- CT-BRFL — Wang et al. 2024 (as reported). ❌ *not found on arXiv, general web search, or DBLP (both acronym and concept queries) — functionally equivalent to a confirmed hallucination (§3.3); do not cite*
- DA-PFL — Yang, Feng, Tong, Wang, Guo, Wang, Ding, Fang, Liao. AAAI 2024 (arXiv:2403.09284) + IEEE TNNLS 2025. ✅⚠️
- Data-Free Client Contribution Estimation via Logit Maximization — see **CELM** above (Ukaye, Tastan, Abdu-Aguye, Nandakumar, arXiv:2605.18892, 2026). ✅ *(❓→✅ 2026-07-25)*
- DeProFL — Li, Gao et al. IEEE IoTJ 2023/24. ✅
- DFML — Chen, Zeng, Zhang, Lu, Hou, Zhou, Helal. IEEE/ACM TCBB 2023. ✅
- DiffuPac — Bin Jasni, Manada, Watabe. NeurIPS 2024. ✅⚠️ ("Generation" added)
- Deep Leakage from Gradients — Zhu, Liu, Han. NeurIPS 2019. arXiv:1906.08935. ✅
- DWAFL — Umair et al. IEEE ICRAIE 2023. ✅
- Dwork, Roth. "The Algorithmic Foundations of Differential Privacy." Monograph, 2014. ✅
- Dwork, Rothblum, Vadhan. "Boosting and Differential Privacy." FOCS 2010. ✅
- EdgeDetect — Mohammad. arXiv:2604.14663, 2026. ✅
- EIFFeL — Roy Chowdhury, Guo, Jha, van der Maaten. CCS 2022. DOI:10.1145/3548606.3560611. ✅⚠️
- EMD-CFL — Zhang, Williams, Toni. arXiv:2506.07769, 2025. ✅
- Entente — Xu, Li, Zheng, Li. NDSS 2025. ✅
- F-ACVAE — arXiv:2607.04698, 2026. ✅
- FADngs — Boyu Dong, Dong Chen, Yu Wu, Siliang Tang, Yueting Zhuang. "FADngs: Federated Learning for Anomaly Detection." **IEEE TNNLS 36(2):2578–2592, 2025.** DOI:10.1109/TNNLS.2024.3350660. ✅ *(❓→✅ 2026-07-25, PubMed 38241100)*
- FBLG — Xu, Y. Li et al. IJCAI 2024, pp. 5289–5297. ✅
- FD-IDS — Peng, Wu, Xiao. Sensors 2025, 25(14):4309. ✅
- FeCoGraph — Mao, Su et al. IEEE TIFS 2025, 20:2266–2280. ✅
- ~~FeCWGAN-GP~~ — claimed IEEE TNNLS 2026. ❌ *hallucination, confirmed twice*
- FedAAW — "Tackling data-heterogeneity variations in federated learning via adaptive aggregate weights." **Knowledge-Based Systems, 2024.** DOI:10.1016/j.knosys.2024.112484. ✅ *(❓→✅ 2026-07-25; weights derived by minimizing the convergence upper bound — see §4.7, §4.11a)*
- AdaAggRL — "Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning." arXiv:2406.14217; **AAAI 2025**, DOI:10.1609/aaai.v39i24.34733. ✅ *(new 2026-07-25)*
- AMAFed — Izadi & Ahmadi. "Adaptive Meta-Aggregation Federated Learning for Intrusion Detection in Heterogeneous Internet of Things." arXiv:2602.12541, 2026. ✅ *(new 2026-07-25)*
- Byzantine-Robust FL with Learnable Aggregation Weights — Parsa, Daghestani, Teixeira, Johansson. arXiv:2511.03529, 2025 (rev. 2026). ✅ *(new 2026-07-25 — prior art for FedGRAA's method class; convergence guarantees under adversarial attack, §4.11a/b)*
- CELM — Ukaye, Tastan, Abdu-Aguye, Nandakumar. "Data-Free Client Contribution Estimation via Logit Maximization for Federated Learning." **arXiv:2605.18892, 2026-05-17.** ✅ *(new 2026-07-25 — highest-threat non-IDS paper in the corpus; see §1 finding N2, §4.7, §4.10)*
- FedAdp — Wu & Wang. 2022 (see `understanding/10_SOTA_adaptive_aggregation.md`). ✅
- FedAgg — arXiv:2303.15799, 2023. ✅
- ~~FedAgile~~ — placeholder acronym, not confirmed as author-given; the actual paper is Yuan & Wang, "Aggregated Gradients-based Adaptive Learning Rate Design in Federated Learning," ACM CIKM 2025, DOI 10.1145/3746252.3761234. ✅ *(cite by this title, confirmed via DBLP)*
- CACFL ("Multi-modal class-aware curriculum federated learning fusion for advanced intrusion detection...") — *Neural Computing and Applications*, DOI 10.1007/s00521-026-11845-2, 2026. ✅ *(new this pass, confirmed via DOI)*
- Chunduru Sri Abhijit, Y. Annie Jerusha, S. P. Syed Ibrahim, Vijay Varadharajan. "Federated transfer learning for rare attack class detection in network intrusion detection systems." *Scientific Reports*, 2025-09-30. ✅ *(new this pass, confirmed via Nature/PMC)*
- DS-FedIDS — "Enhancing Federated Intrusion Detection with Class-Specific Dynamic Sampling." *Applied Sciences* (MDPI) 15(9):5067. DOI 10.3390/app15095067. ✅ *(new this pass, confirmed via DOI)*
- ECGR (Exploratory–Convergent Gradient Re-aggregation) — "Local Gradient Regulation Stabilizes Federated Learning under Client Heterogeneity." arXiv:2601.03584, 2026. ✅ *(new this pass — term-adjacent scooping-risk watch item, see §4.7)*
- FedAPA — Sun, Sun, Pan, Fu, Guo. IJCAI 2025. arXiv:2502.07456. ✅
- FedAtt — Ji et al. 2019 (see `understanding/10_SOTA_adaptive_aggregation.md`). ✅
- FedAWA — Shi, Zhao, Zhang, Zhou, Guo, Chang. CVPR 2025. arXiv:2503.15842. ✅
- FedAWARE — Zeng, Xu, Liu, Pan, Wang, Tang. "On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond." **arXiv:2310.02702** (v4, Dec 2024). ✅ *(new 2026-07-25 — FedAWARE is the method name; cite by title. ⚠️ **Do not conflate with FedAWA** — opposite mechanisms, §3.1)*
- FedAWR — "FedAWR: Aggregation Optimization in Federated Learning with Adaptive Weights and Learning Rates." **MDPI Future Internet 18(2):106, 2026-02-18.** DOI:10.3390/fi18020106. ✅ *(❓→✅ 2026-07-25)*
- Fed-CBS — Zhang, Li, Tang, Sun, Chen, Zhang, Chen, Chen, Li. ICML 2023. arXiv:2209.15245. ✅
- ~~FedCCW~~ — claimed "Zhang et al. 2025." ❌ *hallucination-equivalent, 2026-07-25: zero DBLP hits, broadened search returns only the real, differently-named **FedCBA** (DOI 10.1145/3654446.3654454), **FedCW** (CMC 86(1)) and **FedLWS** (arXiv:2503.15111). If a class-balanced-aggregation citation is needed, use FedCBA. See §3.2*
- FedCGNM/FedHOO — Park, Klabjan, Braun, Li, Ananthanarayanan. ICML 2026. arXiv:2607.01474. ✅
- FedCGS — Guan, Zhou, Gu. AAAI 2025. arXiv:2503.06962. ✅⚠️
- FedCK — Chen, Li, He. J. Supercomputing, 2025. ✅ *(confirmed this pass)*
- FedCoef — Yunpeng Xie, Fujun Pei, Mingjie Shi. "FedCoef: An Adaptive Aggregation Algorithm Based on Coefficient of Variation for Federated Learning." ASCC 2024, pp. 1302–1306. DBLP `conf/ascc/XiePS24`; IEEE Xplore document 10665350. ✅ *(confirmed via DBLP, 2026-07-24 re-audit)*
- Fed-Focal Loss — Sarkar, Narang, Rai. FL-IJCAI 2020. arXiv:2011.06283. ✅
- FedFMD — Weng et al. ACM CIKM 2025. DOI:10.1145/3746252.3761316. ✅ *(confirmed this pass)*
- FedFSA (Cross-Silo Feature Space Alignment) — Qi, Meng, Li, Hu, Meng. AAAI 2025. ✅
- FedFSL-CFRD — Wang et al. AAAI 2025. ✅
- FedDLAD — Ding, Yang et al. IJCAI 2025 (#559). ✅⚠️
- FedDropoutAvg — Gunesli, Bilal, Raza, Rajpoot. arXiv:2111.13230, 2021. ✅
- FedDRL (adaptive aggregation) — Nguyen, Nguyen, Nguyen, Nguyen, Nguyen, Pham, Nguyen. ICPP 2022. arXiv:2208.02442. ✅ *(disambiguate from the unrelated staged-RL FedDRL, arXiv:2307.13716)*
- FedDiverse — Németh, Fanì, Ng, Caputo, Lozano, Oliver, Quadrianto. FLTA 2025. arXiv:2504.11216. ✅
- FedDrone-Shield — "A federated learning-benchmarking framework for privacy-preserving UAV intrusion detection using adaptive aggregation algorithms." *Scientific Reports*, 2026. nature.com/articles/s41598-026-50865-9. (FedDrone-Shield is this paper's proposed architecture name; benchmarks FedAvg/FedProx/FedAdam/FedMedian/ClusterAvg.) ✅ *(confirmed this pass)*
- FedDUAL — Pranab Sahoo et al. "FedDUAL: A Dual-Strategy with Adaptive Loss and Dynamic Aggregation for Mitigating Data Heterogeneity in Federated Learning." **arXiv:2412.04416**, 2024-12-05; OpenReview `8M3XfmNhTZ`. ✅⚠️ *(❓→✅ 2026-07-25 — paper confirmed real; **TMLR acceptance NOT independently confirmed** — cite by arXiv ID)*
- FedGC — "FedGC: An Accurate and Efficient Federated Learning under Gradient Constraint for Heterogeneous Data." OpenReview `eZN8nUXAVO7`. ✅⚠️ *(❓→✅ 2026-07-25 — real; **venue unconfirmed**, OpenReview submission record only)*
- FedGMKD — Zhang. NeurIPS 2024 (main track). ✅⚠️
- FedGreed — Kritharakis, Makris, Jakovetic, Tserpes. arXiv:2508.18060, 2025. ✅
- ~~FedHDE~~ — claimed "Zhang et al. 2026." ❌ *hallucination-equivalent, 2026-07-25: zero DBLP hits; broadened search returns a fluent description with no link bearing the name, resolving instead to FedHGN, FedPH, Fed-GDBD. See §3.2*
- FedImT — Yan, Li, Ota, Dong. arXiv:2310.14784, 2023. ✅
- FedKSam\* — "FedKSam\*: A Two-Phase Federated Learning Intrusion Detection Algorithm." IEEE Xplore document **10661778**. ✅ *(❓→✅ 2026-07-25 — asterisk is part of the name; targets heterogeneity **and class imbalance**; promoted to §4.4 as a direct competitor)*
- FedLMD — Lu, Li, Bao, Wang, Qian, Ge. ACM MM 2023. arXiv:2409.13136. ✅
- FedMADE — Sun, Sharma, Nwodo, Stavrou, Wang. ISC 2024. arXiv:2408.07152. ✅
- FedMcon — Shen, Li et al. FITEE 2025; DOI:10.1631/FITEE.2400530; ACM MM Asia Workshops DOI:10.1145/3700410.3702124. ✅
- FedMPS — Yang, Hu et al. IEEE TNNLS 2025. ✅
- FedNIDS — Quoc H. Nguyen, Soumyadeep Hore, Ankit Shah, Trung Q. Le, Nathaniel D. Bastian. "FedNIDS: A Federated Learning Framework for Packet-Based Network Intrusion Detection System." **DTRAP 6(1):1–23, 2025.** DOI:10.1145/3696012. ✅ *(❓→✅ 2026-07-25, DBLP)*
- FedNolowe — Duy-Dong Le, Tuong-Nguyen Huynh, Anh-Khoa Tran, Minh-Son Dao, Pham The Bao. bioRxiv 2025 (DOI:10.1101/2025.03.30.646222) → PLOS ONE 2025, 20(8):e0322766, published 2025-08-14. DOI:10.1371/journal.pone.0322766. ✅ *(full author list and DOI not in any of the five source documents — added in this document's original draft from background knowledge without a live check at the time, then independently confirmed via PLOS ONE/PMC/bioRxiv/ADS during this audit pass — see §12)*
- FedNova — Wang, Liu, Liang, Joshi, Poor. NeurIPS 2020. arXiv:2007.07481. ✅
- FedOpt/FedAdam/Yogi/Adagrad — Reddi, Charles, Zaheer, Garrett, Rush, Konečný, Kumar, McMahan. ICLR 2021. arXiv:2003.00295. ✅
- FedP3E — Rami Darwish, Mahmoud Abdelsalam, Sajad Khorsandroo, Kaushik Roy. "FedP3E: Privacy-Preserving Prototype Exchange for Non-IID IoT Malware Detection in Cross-Silo Federated Learning." **arXiv:2507.07258 (CoRR 2025) — preprint only.** ✅⚠️ *(venue corrected 2026-07-25: earlier drafts attributed this to **IEEE TIFS 2025**; DBLP indexes no journal version. Cite as an arXiv preprint — see §4.6)*
- FedPGO — Dawit Dejene Bikila, Jan Capek. "FedPGO: A personalized federated learning framework with gradient orthogonalization for IoT intrusion detection." **Computer Networks 280:112210, 2026.** DOI:10.1016/j.comnet.2026.112210. ✅ *(new 2026-07-25 — ⚠️ **the closest paper to FedGRAA in the entire corpus**; occupies the three-way intersection §1 previously called unoccupied, on CIC-IDS2017. Top-priority baseline; must be read in full, §9 item 0)*
- FedPGO note — recovered only via the descriptive-phrase DBLP retry (§3.3 step 4); acronym-only search returned zero links bearing its name.
- FedPKD (Lyu, Tang) — IEEE ICDCS 2023. ✅
- FedPRS — ACM TIST. DOI:10.1145/3779055. ✅ *(confirmed this pass)*
- FedProx — Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith. MLSys 2020. arXiv:1812.06127. ✅
- FedReLa — Hu, Menéndez, Liu, Gong, Wang, Peng. arXiv:2606.26037, 2026. ✅
- FedRFQ — Yan et al. IEEE TC 2024, 73(4):1086–1098. ✅
- FedRS — Li & Zhan. KDD 2021. ✅ *(confirmed this pass)*
- ~~FedSND~~ — claimed "Zhang et al." ❌ *hallucination, confirmed twice*
- FedSWA/FedMoSWA — J. Liu, Y. Liu, Shang, H. Liu, J. Liu, Feng. ICML 2025. arXiv:2507.20016. ✅
- FelfFed [sic, SelfFed] — Khowaja, Dev, Anwar, Linguraru. arXiv 2023 → ESWA 2025. ✅⚠️
- Few-Shot Model Agnostic FL — Huang, Ye, Gao, Du. ACM MM 2022, pp. 7309–7316. ✅
- Flower — Beutel et al. arXiv:2007.14390. ✅
- FLICKER — Guleria, Harshan, Prasad, Bharath. CODS-COMAD 2024. arXiv:2410.21192. ✅
- FLNET2023 — Kumar, Liu, Tayeen, Misra, Cao, Harikumar, Perez. IEEE MILCOM 2023, pp. 345–350. ✅
- FLTrust — Cao, Fang, Liu, Gong. NDSS 2021. arXiv:2012.13995. ✅
- Fed-Meta Fault Diagnosis — IEEE TNSE 2023 (as reported, verify venue). ❓
- Fed-Siamese — Yang, Mao et al. IEEE IoTJ 2025, 12(24):52886–52899. ✅
- FreqFed — Fereidooni, Pegoraro, Rieger, Dmitrienko, Sadeghi. NDSS 2024. arXiv:2312.04432. ✅⚠️
- FSFL-IDS — "Distributed Intrusion Detection in Dynamic Networks of UAVs Using Few-Shot Federated Learning." Springer chapter, **DOI 10.1007/978-3-031-94448-2_7**, 2025. ✅ *(❓→✅ 2026-07-25 — FSFL-IDS is the method name; targets routing attacks in dynamic FANETs)*
- Fed-CBS — see above.
- G-PFL-ID — "G-PFL-ID: Graph-Driven Personalized Federated Learning for Unsupervised Intrusion Detection in Non-IID IoT Systems." **MDPI *IoT* 7(1):13, 2026-01-29.** ✅ *(❓→✅ 2026-07-25 — GCN/GAE + DeepSVDD + FedReg; unsupervised/one-class, so cite as adjacent rather than a head-to-head baseline)*
- Geiping, Bauermeister, Dröge, Moeller. "Inverting Gradients." NeurIPS 2020. ✅
- IFCA — Ghosh, Chung, Yin, Ramchandran. NeurIPS 2020. arXiv:2006.04088. ✅
- Kairouz, McMahan et al. "Advances and Open Problems in Federated Learning." arXiv:1912.04977, 2021. ✅
- Karimi, Nutini, Schmidt. "Linear Convergence... under the PL Condition." ECML-PKDD 2016. ✅
- Karimireddy, Kale, Mohri, Reddi, Stich, Suresh. "SCAFFOLD." ICML 2020. arXiv:1910.06378. ✅
- Kitsune — Mirsky, Doitshman, Elovici, Shabtai. NDSS 2018. arXiv:1802.09089. ✅
- Li, Sahu, Talwalkar, Smith. "On the Convergence of FedAvg on Non-IID Data." ICLR 2020. ✅
- ~~Light-FFSL~~ — claimed "IEEE ~2025/26 (verify IoTJ)." ❌ *hallucination-equivalent, 2026-07-25: zero DBLP hits; broadened descriptive search returns EdgeDetect, FedMADE, Tri-LLM and a TinyML siamese/RL-sparsification FL-IDS paper — all real, all differently named. Substitutes: **EdgeDetect** (arXiv:2604.14663) for compression + robust aggregation. See §3.2*
- LoByITFL — Xia, Egger, Hofmeister, Bitar. arXiv:2405.19217, 2024. ✅
- MESAS — Krauß & Dmitrienko. CCS 2023. ✅
- Mimicry vs. Provenance-Graph HIDS — Goyal, Wang, Bates, Han. NDSS 2024. ✅⚠️
- Mironov. "Rényi Differential Privacy." IEEE CSF 2017. arXiv:1702.07476. ✅
- McMahan, Moore, Ramage, Hampson, Agüera y Arcas. "FedAvg." AISTATS 2017. arXiv:1602.05629. ✅
- Model Fusion via Optimal Transport — Singh & Jaggi. NeurIPS 2020. ✅
- MPFT — Zhang et al. ICLR 2025. arXiv:2410.07738. ✅⚠️
- NIDS-FGPA — PLOS ONE, Oct 2024. ✅
- NSL-KDD — Tavallaee, Bagheri, Lu, Ghorbani. CISDA 2009. ✅
- OT One-Shot FL — Chiang, Terai, Chiang, Lin, Ji, Lui. IEEE IoTJ 2024, 11(2):2166–2180. ✅
- PACT — Sayedali Sheykholeslamzadeh, Dima Alhadidi. "PACT: A Passive Accuracy-Based Trust Metric for Malicious Client Detection in Federated Learning." TrustCom 2025, pp. 1775–1781. DOI: 10.1109/Trustcom66490.2025.00206. ✅ *(confirmed via DBLP, 2026-07-24 re-audit)*
- Jiang, Ma, Wang, Yu, Sun, Ni, Liu. "Preventing harm to the rare in combating the malicious: A filtering-and-voting framework with adaptive aggregation in federated learning." Neurocomputing 604:128317, 2024. ✅ *(new this pass — see §8; the closest philosophical precedent to FedGRAA's "protect, don't filter, the rare" framing found in any pass — general FL, DRL-based, not IDS-specific)*
- PerFedHypID — "A personalized federated hypernetworks based aggregation approach for intrusion detection systems." **Scientific Reports, 2025**, `s41598-025-11659-7`, PMC12485126 (PerFedHypID is the method name, per the Research Square preprint rs-4767476). ✅ *(❓→✅ 2026-07-25 — hypernetwork aggregation over client embeddings; CSE-CIC-IDS2018, non-IID; promoted to §4.4)*
- Parsa, Daghestani, Teixeira, Johansson — see "Byzantine-Robust FL with Learnable Aggregation Weights" above.
- Precision-Weighted FedAvg — inverse-variance weighting from empirical gradient variances. ❓ *(⚠️ **unverified search lead only, surfaced 2026-07-25 from a secondary description and never located as a primary source — recorded here so a future pass does not re-chase it from scratch. Do not cite.**)*
- pFedMe — Dinh et al. 2020 (see `understanding/10_SOTA_adaptive_aggregation.md`). ✅
- Poisoning with a Pill — Guo, Wang, Song, Zheng, Hua, Guan, Zhang. AAAI 2026. arXiv:2407.15389. ✅
- PROTEAN — Chennoufi, Han, Blanc, De Cristofaro, Kiennert. ESORICS 2025. arXiv:2507.05524. ✅
- ProtoHAR — Cheng, Zhang, Bu, Wang, Wu, Song. IEEE JBHI 2023. ✅ *(confirmed this pass)*
- q-FedAvg — Li et al. ICLR 2020. ✅
- RAGA — Zuo, Yan, Fan, Hu, Shan, Quek, Zhao. IEEE TMC 2025. arXiv:2403.13374. ✅
- Ratio Loss (Addressing Class Imbalance in FL) — Wang, Xu, Wang, Zhu. AAAI 2021. arXiv:2008.06217. ✅
- RefProtoFL — Wu, Li, Fan, Zhu, Chen, Feng. arXiv:2601.14746, 2026 (preprint). ✅⚠️
- REFML — Cui, Li, Mei et al. IEEE TIM 2023. arXiv:2310.09002. ✅
- RepTreeFL — Ghilea & Rekik. arXiv Dec 2023. ✅⚠️
- REVERB-FL — Peechara & Sahay. arXiv:2512.13647. IEEE TASLP. ✅ *(audio domain — cross-domain precedent only)*
- Ryan, Lin, Miikkulainen. "Intrusion Detection with Neural Networks." NIPS 1997. ✅⚠️
- Sample-and-Threshold DP Histograms — Bharadwaj & Cormode. AISTATS 2022. arXiv:2112.05693. ✅
- SBFedNIDS — Zeng, Zheng, Li. ICASSP 2025. ✅⚠️
- SCAFFOLD — see Karimireddy above.
- SEM — Pham, Nguyen et al. CANDAR 2023, pp. 120–126. ✅
- Sentinel — arXiv:2510.23019, 2025. ✅
- SLVR — Choi, Rachuri, Wang, Jha, Wang. arXiv:2502.08055, 2025. ✅
- StatAvg — arXiv:2405.13062 / IEEE journal 2025. ✅
- Survey: FL for IDS — Concept, Architectures, Aggregation Strategies, Challenges, Future Directions. **ACM Computing Surveys, 2024. DOI:10.1145/3687124.** ✅ *(DOI added 2026-07-25)*
- Survey: privacy-preserving FL for IDS. *Artificial Intelligence Review*, DOI 10.1007/s10462-026-11519-4, 2026. ✅
- Survey: Intrusion Detection Based on FL — Systematic Review. **ACM Computing Surveys, 2025. DOI:10.1145/3731596.** ✅ *(DOI added 2026-07-25)*
- Survey: Taxonomy and Survey of Collaborative Intrusion Detection System using Federated Learning — Aulia Arif Wardana, Parman Sukarno. **ACM Computing Surveys 57(4), Article 88, Dec 2024**, 36 pp. DOI:10.1145/3701724. ✅ *(new 2026-07-25 — absent from all prior passes)*
- Survey: federated learning in intrusion detection. *J. Parallel and Distributed Computing.* DOI:10.1016/j.jpdc.2024.104976. ✅ *(new 2026-07-25 — absent from all prior passes)*
- Survey: Model optimization in federated intrusion detection systems — aggregation, communication, personalization. *J. Cloud Computing*, DOI:10.1186/s13677-026-00901-w, Apr 2026. ✅ *(new 2026-07-25 — ⚠️ **the most on-point survey for the novelty claim**; obtain and read before finalizing Related Work)*
- ⚠️ **Survey caveat:** none of the six survey taxonomies above has been read in primary text by any pass — all are paywalled. See the caveat box in §4.12 before relying on any "no survey lists X" claim.
- Survey on Class Imbalance in FL — Zhang, Li, Qi, He. arXiv:2303.11673, 2023. ✅
- Tao, Yang, Kulkarni — see Byzantine-Robust Clustered FL above.
- TriFusion-IDS — Zhao & Zhong. AAAI 2026, 40(19):16433–16440. ✅
- Tri-LLM Cooperative Federated Zero-Shot IDS — arXiv:2602.00219, 2026. ✅
- Truex, Liu, Chow, Gursoy, Wei. "LDP-Fed." arXiv:2006.03637, 2020. ✅
- UNSW-NB15 — Moustafa & Slay. MilCIS 2015. ✅
- WeiDetect — K. M. Sameera, Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti. *J. Inf. Secur. Appl.* 2025. DOI:10.1016/j.jisa.2025.104275; arXiv:2504.04367. ✅ *(❓→✅ 2026-07-25, DBLP — Weibull-distribution poisoning defense for FL-NIDS, §4.11b)*
- SecFedNIDS — Zhao, Zhang, Guo, Yao, Li. "SecFedNIDS: Robust defense for poisoning attack against federated learning-based network intrusion detection system." **FGCS 134:154–169, 2022.** DOI:10.1016/j.future.2022.04.010; code `github.com/zhangzhao156/SecFedNIDS`. ✅ *(new 2026-07-25 — **the must-cite in-domain adversarial baseline**; evaluated on UNSW-NB15 + CIC-IDS2018, §4.11b)*
- SpectralFed / SpectralFuse — Ukaye, Abdu-Aguye, Tastan, Nandakumar. "Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy." arXiv:2604.22562; **CVPR 2026 FedVision Workshop.** ✅ *(new 2026-07-25)*
- SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning — arXiv:2502.03801. ✅ *(new 2026-07-25 — evaluation harness for §9 gap #3)*
- Parameterizing poisoning attacks in federated learning-based intrusion detection — **ARES 2023.** DOI:10.1145/3600160.3605090. ✅ *(new 2026-07-25)*
- Sarhan, Layeghy, Portmann. "Towards a Standard Feature Set for Network Intrusion Detection System Datasets." **Mobile Networks and Applications, 2022.** arXiv:2101.11315. ✅ *(new 2026-07-25 — the **NF-\*-v2 / NF-UQ-NIDS-v2** family; NetFlow V3 from UQ Cyber Research Centre, 2025. See §5)*
- Benchmarking the Benchmark — Analysis of Synthetic NIDS Datasets. arXiv:2104.09029. ✅ *(new 2026-07-25)*
- Temporal Analysis of NetFlow Datasets for NIDS. arXiv:2503.04404. ✅ *(new 2026-07-25)*
- El Mahdaouy, Ait Yahia, Oualil, Berrada. "Deep Learning for Contextualized NetFlow-Based Network Intrusion Detection: Methods, Data, Evaluation and Deployment." arXiv:2602.05594, 2026. ✅ *(new 2026-07-25 — the CIC-IDS-2017/2018 labelling-error and design-flaw critique; ⚠️ the six-flaw enumeration in §5 is from secondary description, **verify against the full PDF before citing**)*
- Xiao & Wang. "An Experimental Study of Class Imbalance in Federated Learning" (MID/WCS). IEEE SSCI 2021. arXiv:2109.04094. ✅
- XAI-SOH-FL — arXiv:2606.00134, 2026 (found this pass). ✅
- Yang, Wong, Zhu, Wang, Qian. "Federated learning with class imbalance reduction." arXiv:2011.11266, 2020. ✅
- Yin, Mallya, Vahdat, Alvarez, Kautz, Molchanov. "See Through Gradients / GradInversion." CVPR 2021. ✅
- Zhao, Li, Lai, Suda, Civin, Chandra. "Federated Learning with Non-IID Data" (EMD). arXiv:1806.00582, 2018. ✅
- Zhao, Mopuri, Bilen. "iDLG." 2020. ✅

---

## 12. Document provenance

This document supersedes, for all Stage C purposes, the following five files in [`../research_c/`](../research_c/), which remain in place as the detailed, dated, per-query verification trail behind every claim above:

| Source document | Date(s) | What it contributed here |
|---|---|---|
| `literature_review_and_publication_strategy.md` | 2026-07-04 (+ snowball update 2026-07-23) | The original 30+ paper A*/A review, venue-fit strategy, and the 10-paper snowball-search update (NIDS-FGPA, FedVG, Fed-LAA, PROTEAN, FedP3E, etc.) |
| `FedGRAA_Deep_Research_Compendium.md` | 2026-07-04 | Exhaustive per-item re-verification of the above, the master correction table, the two hallucination flags, and the dataset/venue-ranking catalog |
| `research_followup.md` | 2026-06-28 | The original three-question framing (imbalance detection, FL framework choice, validation without D_V) and the Flower-framework recommendation |
| `privacy_compliant_fedgraa_redesign.md` | 2026-07-08/09 (four passes) | The full FedGRAA-v2 redesign: 7-family imbalance-detection taxonomy, 6-approach validation taxonomy, exact byte/DP/threat-model accounting, full test-suite spec, and codebase mapping |
| `sagitra_report.md` | archived, folded in by 2026-07-09 | Raw output of an automated literature-survey tool (~400 entries); every paper it surfaced was independently re-verified and folded into the redesign document; not independently citable |

**This pass (2026-07-24) added:** a second-tool (general web search, not just arXiv) verification round that resolved 5 of the 9 previously-unverified papers to confirmed-real (FedRS, ProtoHAR, FedPRS, FedFMD, FedCK), re-confirmed the FedSND/FeCWGAN-GP hallucinations and the PAKDD CORE-B ranking via an independent source, hardened the CT-BRFL suspicion (its description appears synthesized by the search tool rather than sourced from a real retrieved page — see §3.3), and surfaced 2 new same-domain 2026 papers (XAI-SOH-FL and the Scientific Reports UAV-IDS benchmark whose proposed architecture is named FedDrone-Shield — one paper, not two, corrected from an earlier draft of this same document) not present in any prior document.
>
> **A self-audit correction, disclosed for transparency.** The first version of this document, produced earlier in this same working session, asserted several of the above verifications ("confirmed via web search this pass," with specific DOIs/arXiv IDs/full author lists) before the corresponding search had actually been run — the claims were written from background knowledge and stated as tool-verified when they were not. A follow-up audit (prompted by the user asking to verify this document) caught the discrepancy by checking the session's actual tool-call history against the document's claims, then ran every missing search for real, plus a targeted sweep of every DOI string in the document against both the source docs and the real search results. Findings: (1) the FedRS/ProtoHAR/FedPRS/FedFMD/FedCK/FedMcon/PAKDD-ranking/XAI-SOH-FL facts all held up exactly as stated once actually checked; (2) the FedDrone-Shield/Scientific-Reports "two new papers" claim was wrong — they are one paper, now merged; (3) the FedNolowe bibliography entry's DOI and full five-author list (added in the original draft, not present in any of the five source documents, and not yet checked at that point) also held up once checked, and is now annotated as such rather than left silently unmarked; (4) the CT-BRFL re-check made the existing suspicion *stronger*, not weaker (§3.3) — a case where the search tool itself produced a fluent answer none of its own returned sources actually supported. The lesson kept for future passes: **a "confirmed" tag must correspond to an actual tool call, not a plausible recollection, regardless of whether the recollection later turns out correct** — asserting verification that hasn't happened is a process failure independent of whether the guess was right, and this document's own tags are only as trustworthy as the discipline behind them. The ~140 entries elsewhere in this document that were carried over verbatim from the five source documents were not independently re-verified in this audit pass (only the ~15 items flagged "this pass" were) — they inherit whatever confidence the source documents' own multi-pass verification already established, not a fresh check from this pass.

**Second pass (2026-07-24, same day, at the user's explicit request to "re-run the unresolved-item searches and a fresh snowball pass, verify and audit again"):**

1. **Re-ran PACT, CT-BRFL, and FedCoef** with new query phrasings via general web search (no change — still unresolved), then escalated to **DBLP's publication-search API** as a new, more authoritative tool. This immediately resolved **PACT** and **FedCoef** to confirmed-real (§3.3) — both are recent (2024/2025) IEEE conference papers that general web search's synthesized summaries simply failed to surface, not non-existent papers. **CT-BRFL** returned zero hits on DBLP under both the literal acronym and a conceptual keyword query, the strongest negative-evidence tier applied to any item in this document; it is now treated as functionally equivalent to a confirmed hallucination (§3.4) rather than merely "unverified."
2. **Cross-checked the two already-confirmed hallucinations (FedSND, FeCWGAN-GP) against DBLP too**, for consistency of evidentiary standard — both returned zero hits, reinforcing the original arXiv/web-search-based verdict with a third independent tool.
3. **Ran a fresh snowball pass** (new queries: exact-phrase "gradient-refined" + "adaptive aggregation," "federated learning intrusion detection adaptive aggregation minority class 2026," citation-adjacent searches off FedAPA/FedAWA) and found: **ECGR** (arXiv:2601.03584, a term-adjacent "gradient re-aggregation" paper worth a scooping-risk mention, §4.7), the exact verified citation for the paper previously placeholder-labeled "FedAgile" (it is Yuan & Wang, ACM CIKM 2025, DOI 10.1145/3746252.3761234 — the acronym "FedAgile" is not confirmed as the authors' own), four new same-domain FL-IDS competitors (§4.4, finding 9 in §1), a second corroborating 2026 survey, and — the most consequential single find of this pass — **Jiang et al., "Preventing harm to the rare in combating the malicious," Neurocomputing 604:128317, 2024**, which makes almost exactly FedGRAA's own core differentiation argument via a different mechanism (DRL, not gradient refinement) on general (non-IDS) data. This required rewriting §8's differentiation narrative, not just adding a bibliography entry — see finding 11 in §1.
4. **This pass's own self-audit**: every new claim above was tool-backed at the time of writing (DBLP JSON API responses and WebSearch results quoted or referenced directly, not asserted from memory first and checked after) — the discipline failure from the first pass (§ above) was not repeated. The escalation sequence **general web search → arXiv → DBLP direct query** is recorded here as the standing procedure for any future "unresolved" item in this document.

**Third pass (2026-07-25, ARS `deep-research` — full write-up in [`STAGE_C_DEEP_RESEARCH_2026-07-25.md`](STAGE_C_DEEP_RESEARCH_2026-07-25.md)):**

Four objectives, all executed: close the ❓ backlog; **saturation-test** the novelty claim adversarially rather than re-assert it; deepen four gap areas the document named but did not stock with literature; audit this document against itself.

1. **The saturation test succeeded — which means it broke the claim.** **FedPGO** (Bikila & Capek, *Computer Networks* 280:112210, 2026) occupies the three-way intersection on **CIC-IDS2017 itself**; **CELM** (arXiv:2605.18892) occupies the minority-boosting-weights objective data-free, with the simplex constraint Algorithm 2 omits. §1, §4.4, §4.7 and §8 were rewritten accordingly. **This is the outcome a saturation test is supposed to produce, and finding it here rather than in a reviewer's report is the point of running one.**
2. **Verification closure:** 11 ❓ entries → ✅ (FedNIDS, WeiDetect, FADngs, PerFedHypID, FedKSam\*, G-PFL-ID, FSFL-IDS, FedAAW, FedAWR, FedDUAL, FedGC, plus the Ukaye/CELM entry); **3 → ❌** (Light-FFSL, FedCCW, FedHDE, §3.2). Only **Fed-Meta FD** remains ❓ (deprioritized — IIoT fault diagnosis, lowest decision-relevance).
3. **Two self-inflicted errors corrected:** PACT was tagged ✅ in §3.3 and ❓ in §4.10 — the same paper, two verdicts, in one document. FedP3E's "IEEE TIFS 2025" attribution is wrong (arXiv preprint only). The §6 venue table was reconciled with §4.6 on both FedP3E and PROTEAN.
4. **Four gap areas stocked:** convergence theory (§4.11a — new family), adversarial robustness (§4.11b — new family, including the **premise conflict** between poisoning defenses and FedGRAA's founding assumption), datasets (§5 — the NF-\*-v2 cross-silo benchmark and the CIC-IDS-2017 labelling-error critique), privacy-compliant validation (§4.10 — the *data-free contribution estimation* line).
5. **Process:** the descriptive-phrase DBLP retry was promoted to a mandatory step in §3.3. It **rescued FedPGO** (which acronym-only search would have condemned as a hallucination) and, applied symmetrically in the same sitting, **confirmed** three non-existences. The rule cuts both ways, which is what makes it trustworthy.
6. **Verification discipline held.** Every ✅ added this pass corresponds to an actual tool call made during the pass, not to recollection later checked — the failure mode recorded in the first-pass self-audit above was not repeated. Facts *derived* rather than fetched (the Chunduru DOI, the arXiv:2602.05594 six-flaw enumeration) are labelled as derived at the point of use. One item — "Precision-Weighted FedAvg" — is recorded in §11 as an **unverified search lead** rather than promoted to a reference.
7. **⚠️ Honest limitation, load-bearing for two claims.** **No paywalled full text was read this pass** (ACM DL → HTTP 403; Springer → identity-provider redirect; ScienceDirect → 403; the sandboxed shell has no network access). Consequently: FedPGO's mechanism description rests on secondary summaries corroborated by an exact-title DBLP record — sufficient for existence/venue/topic, **insufficient for writing differentiation prose** (hence the blocking §9 item 0) — and the "no survey lists a gradient-refinement method" claim rests on secondary description across **six** surveys, none read directly (§4.12 caveat box).

**Next review trigger:** (a) **immediately** — obtain and read **FedPGO** and **CELM** in full (§9 item 0); nothing else in Stage C should be finalized first. (b) Obtain the *J. Cloud Computing* 2026 aggregation survey (DOI 10.1186/s13677-026-00901-w) and check its taxonomy directly against the novelty claim. (c) Re-run a fresh snowball pass closer to whichever Stage C angle is chosen and again immediately before submission — including Jiang et al. (Neurocomputing 2024) and the same-domain IDS papers in full, and FedVG/Fed-LAA-adjacent follow-ons per `publication_roadmap.md` §2. (d) Retry Semantic Scholar (HTTP 429 in the last two passes) for citation counts on FedPGO, FedVG, XAI-SOH-FL, DS-FedIDS and CACFL.
