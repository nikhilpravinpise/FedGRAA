# FedGRAA — Final Stage C Literature Entry Point

> **What this document is.** The single, final, deduplicated Stage C literature entry point for FedGRAA. It consolidates and **supersedes** [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) and its three addenda, together with the five documents the master doc itself consolidated. Nine source documents in total; all nine remain in place as the itemized, dated audit trail (§12).
>
> **What this pass was.** A consolidation + close-out pass run **2026-07-27**, not a fresh search from scratch. Its job was to verify, merge, and resolve every still-open item, then write one clean document. Nine carried-forward open items (a)–(i) were worked; **seven are now resolved, two remain access-blocked** (§9).
>
> **Verification tags**, unchanged across the document series for continuity: **✅ Confirmed** (primary source or authoritative index record fetched) · **⚠️ Corrected/Clarified** (real, but earlier metadata or attribution was wrong) · **❌ Hallucination-equivalent** (searched and not found across ≥3 independent tools; do not cite) · **❓ Unverified** (open).
>
> **Provenance discipline.** Every ✅ either corresponds to a tool call made **in this pass** — marked `[S10]` — or explicitly names the prior pass it carries forward. Facts *derived* rather than *fetched* are labelled inline as derived. This is the discipline the master doc's own §12 self-audit established after it failed to observe it once; that failure is the standard not to repeat.

**Compiled:** 2026-07-27. **Supersedes:** all prior compilations. FedGRAA's draft: [`../paper/PAKDD_Final_Version_Submit_v3.md`](../paper/PAKDD_Final_Version_Submit_v3.md).

---

## Table of contents

1. [Executive summary](#1-executive-summary)
2. [What FedGRAA claims to contribute](#2-what-fedgraa-claims-to-contribute)
3. [Master verification ledger](#3-master-verification-ledger)
4. [Complete reference catalog by family](#4-complete-reference-catalog-by-family)
5. [Intrusion-detection datasets](#5-intrusion-detection-datasets)
6. [Venue rankings](#6-venue-rankings)
7. [The two structural privacy gaps and FedGRAA-v2](#7-the-two-structural-privacy-gaps-and-fedgraa-v2)
8. [Synthesized positioning and differentiation narrative](#8-synthesized-positioning-and-differentiation-narrative)
9. [Gap-closing checklist, with items (a)–(i) marked](#9-gap-closing-checklist-with-items-ai-marked)
10. [Stage C decision framing](#10-stage-c-decision-framing)
11. [Full alphabetized bibliography](#11-full-alphabetized-bibliography)
12. [Document provenance](#12-document-provenance)
13. [Method, limitations, and AI disclosure](#13-method-limitations-and-ai-disclosure)

**Provenance keys** used throughout: `[S1]` original review · `[S2]` compendium · `[S3]` follow-up · `[S4]` sagitra dump · `[S5]` v2 redesign · `[S6]` master doc (2026-07-24) · `[S7]` re-audit (2026-07-25) · `[S8]` deep research (2026-07-25) · `[S9]` FastTrack pass (2026-07-26) · `[S10]` **this pass (2026-07-27)**.

---

## 1. Executive summary

### 1.1 The settled novelty claim

**This pass did not change the novelty claim.** The claim narrowed on 2026-07-25 (FedPGO) and sharpened on 2026-07-26 (CELM's mechanism); a duplication test and a saturation re-check run this pass on a structurally different index **corroborated it without further narrowing** (§1.3). It is stated here once, and should not be hedged further unless a future pass finds cause:

> Adaptive aggregation for federated intrusion detection is an active and, as of 2026, **occupied** space. The nearest work is **FedPGO** (*Computer Networks* 280:112210, 2026), which reweights clients by a gradient-orthogonality score for IoT IDS **on CIC-IDS2017, FedGRAA's own primary dataset** — but with a *personalization* objective and a parameter-split architecture. In general FL, **CELM** (2026) pursues FedGRAA's minority-class objective data-free via activation-maximization probing; **FedAWARE** reweights to maximize gradient diversity *with convergence guarantees*; **Jiang et al.** (2024) and **AdaAggRL** (AAAI 2025) reach related conclusions via reinforcement learning; **Precision-Weighted Federated Learning** (2021/2025) derives weights from empirical gradient variances. **FedGRAA's residual contribution is narrow and must be stated narrowly:** a *single-global-model*, *γ-tunable*, gradient-refined aggregation rule whose explicit objective is **per-class recall on rare attack types**, evaluated on rare-attack detection rather than aggregate accuracy or personalization gain. The claim to defend is **"a different point in an occupied design space, with an evaluation axis no adjacent work reports"** — *not* "an unoccupied intersection."

That is a defensible **PAKDD-tier (CORE B)** claim. It is **not** an A\*/A-tier claim without the additional work in §9.

**Two framings remain permanently retired** (first retired `[S8]`, re-affirmed `[S10]`): *"the intersection is unoccupied"* (false — FedPGO), and *"existing adaptive aggregators all suppress minorities"* (false — FedAWARE, Jiang et al., AdaAggRL, CELM and Precision-Weighted FL form a counter-current; leaving this in will read as a straw man).

### 1.2 No new paper changed the claim this pass

Stated as plainly as `[S8]` stated the bad news it found: **this pass found no paper closer to FedGRAA than FedPGO.** A three-query nearest-neighbour duplication test against the *final* worded claim (not the pre-saturation claim) returned only papers already catalogued here — FD-IDS, general FL-IDS surveys, and adjacent IoT-security work `[S10]`. That is a negative result, and it is the one worth having.

### 1.3 What this pass did change — eight findings

**N1 — The saturation curve is much steeper than the previous pass could see, strengthening the "space is closing fast" conclusion.** `[S9]` reported 2022:1, 2024:1, 2025:7, 2026:4 from FastTrack. A full-index OpenAlex publication-year grouping on the equivalent query returns `[S10]`:

| Year | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 (to July) |
|---|---|---|---|---|---|---|---|---|
| Broad query | 3 | 6 | 7 | 26 | 59 | 92 | **297** | **292** |
| Tight (`federated` ∧ `intrusion` ∧ `aggregation` in title/abstract) | — | 3 | 16 | 24 | 53 | 75 | **231** | **278** |

⚠️ **Methodological correction owed to `[S9]`:** its absolute counts were a *relevance-limited retrieval*, not a full count — its **shape** conclusion was right, its magnitudes were an order of magnitude low. On the tight query, **2026 has already exceeded all of 2025 by July**, i.e. roughly a 2× annualized rate. The gap is not merely closing; it is closing faster than the prior pass's numbers implied.

**N2 — A citation-attribution error in the dataset critique (item h).** `[S8]`/`[S6]` attribute a six-flaw enumeration of NIDS-benchmark design defects to El Mahdaouy et al. (arXiv:2602.05594). The full text was read this pass `[S10]`: the enumeration is **real and matches verbatim** — *"six recurring design issues: limited feature diversity, high feature interdependence, ambiguous ground truth, collapse of traffic distributions, artificial diversity generation, and labeling inaccuracies"* (§6.1) — **but it is not that paper's own finding.** El Mahdaouy et al. cite it from **Flood, Engelen, Aspinall & Desmet, *"Bad Design Smells in Benchmark NIDS Datasets,"* IEEE EuroS&P 2024**, DOI 10.1109/eurosp60621.2024.00042 (17 citations; green OA at KU Leuven LIRIAS) `[S10]`. **Cite Flood et al. as the primary source**; citing El Mahdaouy et al. for it is a secondary-citation error a reviewer in this subfield would catch. Note also arXiv:2602.05594 is now at **v3** (submitted 5 Feb 2026, revised 3 Mar 2026), not v1.

**N3 — "Precision-Weighted FedAvg" is real (item d): ❓ → ✅, with an invented acronym.** `[S8]` recorded it as an unverified lead with "do not cite." The primary source is **Reyes, Di Jorio, Low-Kam & Kersten-Oertel, *"Precision-Weighted Federated Learning,"* arXiv:2107.09627 (20 Jul 2021)**, journal version ***Computational Intelligence*, 2025-11-04, DOI 10.1111/coin.70150** (11 citations) `[S10]`. Abstract confirmed verbatim: it weights the average by *"the variance of the stochastic gradients."* **The authors never use the string "Precision-Weighted FedAvg"** — that is the search tool's invention, the same failure mode as Fed-LAA `[S7]` and "FedAgile" `[S6]`. It is **prior art in FedGRAA's method class from 2021**: aggregation weights derived from gradient statistics. Not minority-class-targeted, so adjacent rather than occupying — but it means gradient-statistic-derived weighting is five years old, not new.

**N4 — Fed-Meta FD is real (item c): ❓ → ✅, and the master doc's own guess was right.** The last ❓ in the corpus. A descriptive-phrase retry `[S10]` resolves it to **Chen, Tang & Li, *"Industrial Edge Intelligence: Federated-Meta Learning Framework for Few-Shot Fault Diagnosis,"* IEEE Transactions on Network Science and Engineering, 2023, DOI 10.1109/tnse.2023.3266942** (51 citations). The master doc's parenthetical "(verify TNSE)" was **correct**. A companion paper by the same group exists: Chen, Tang & Chen, GLOBECOM 2022, DOI 10.1109/globecom48099.2022.10001294 (8 citations). "Fed-Meta FD" is not the authors' acronym — cite by title. This also resolves a **self-contradiction**: the item was tagged ✅ in master-doc §4.5 but ❓ in master-doc §11 — the same error class as the PACT contradiction `[S8]` caught.

**N5 — NIDS-FGPA, called "the single closest paper on the exact two-way intersection," had no DOI or authors on file until now — and its dataset claim needs correcting.** Full record `[S10]`: **Wang, Yang & Li, *"NIDS-FGPA: A federated learning network intrusion detection algorithm based on secure aggregation of gradient similarity models,"* PLoS ONE, 2024-10-24, DOI 10.1371/journal.pone.0308639** (20 citations, gold OA). ⚠️ Abstract fetched verbatim: it is evaluated on **Edge-IIoTset and CIC IoT 2023 — not on CIC-IDS-2017.** The mechanism read carries: GSA *"dynamically selects and weights updates from different models **to reduce communication overhead**"* — confirming the objective is communication efficiency, not minority-boosting.

**N6 — The Sentinel venue-upgrade lead is confirmed and closed.** `[S9]` flagged a possible duplicate and explicitly declined to assert it without an author cross-check. Done `[S10]`: arXiv:2510.23019 and IEEE IoTJ DOI 10.1109/jiot.2026.3650848 have **identical titles and identical author lists** — Gurpreet Singh, Keshav Sood, P. Rajalakshmi, Yong Xiang. **One paper.** Cite the journal version (IoTJ, 2026-01-12). Curiosity worth noting: the arXiv comment says *"under review for possible publication in IEEE TDSC"* — it landed in IoTJ instead.

**N7 — The §6 venue argument now has sourced numbers instead of an assertion.** See §6. Computer Networks: **2-year mean citedness 4.413, h-index 198**, 9,799 works, 309,646 citations (OpenAlex source `S63392143`) `[S10]`. PAKDD's actual container, Springer LNCS: **2-year mean citedness 1.409** `[S10]`. PAKDD **CORE B re-confirmed this pass** directly from `portal.core.edu.au` (ICORE2026: B) `[S10]`.

**N8 — One new same-domain paper, adjacent not occupying.** **Sun, Zhou, Wang & Li, *"Robust intrusion detection based on personalized federated learning for IoT environment,"* Computers & Security, 2025-03-21, DOI 10.1016/j.cose.2025.104442** (14 citations) `[S10]` — absent from all nine source documents. Personalized FL-IDS; no gradient-refined minority-boosting. Optional Related-Work add.

### 1.4 Two problems more consequential than the novelty gap

Carried unchanged from `[S6]`/`[S5]`; nothing this pass contradicts them.

1. **FedGRAA's design has two privacy-structural flaws** — a server-side oracle for which classes are rare, and a server-held labelled validation set `D_V` (§7). A fully-worked redesign (**FedGRAA-v2**) removes both while leaving the core equations untouched, at a precisely computed ≈6× communication cost.
2. **FedGRAA's true competitor set is a specific small group**, not the field in general: FedPGO, FedAPA, FedAWA, FedMADE, SBFedNIDS, DA-PFL, CELM, FedAWARE. **None is currently a head-to-head baseline.**

And a third, orthogonal but compounding: **Algorithm 2 lacks a simplex projection**, causing real divergence in round 1 on both datasets at every γ ([`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md); cross-checked this pass, **no drift** `[S10]`). CELM treats that same projection as a basic design requirement — so the fix is **standard practice with a citation**, never a contribution.

---

## 2. What FedGRAA claims to contribute

*Carried from master-doc §2. Cross-checked against the live draft this pass; **no drift found** `[S10]`.*

**Title:** *Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems* (PAKDD submission draft).

**Core problem.** FedAvg aggregates in proportion to dataset size, `θ_g^{t+1} = Σ_{k∈S_t} (|D_k|/|D|) θ_k^{t+1}`. Under non-IID intrusion-detection data, clients holding **rare attack classes** (Heartbleed, SQL Injection, Infiltration on CIC-IDS-2017) usually also hold **little data**, so the size weight crushes exactly the contributions that matter most.

**Proposed mechanism.** An adaptive, gradient-based weight update governed by a personalization coefficient `γ ∈ [0,1]`: as `γ → 1` the aggregator prioritizes minority-class clients; as `γ → 0` it reverts toward conventional emphasis. Contrasted against a static reparameterization baseline (weights `∝ 1/|D_k|`) that over-corrects and cannot adapt. The composite objective (draft Eq. 9) is

`L'(w) = (1 − γ)·L_a(θ_g'(w)) + γ·L_u(θ_g'(w))`

with `L_a = min(L_h, L_m)` on high-/moderate-frequency validation splits and `L_u` the rare-class loss on `D_V^n`; Algorithm 2 updates `w ← w − η'∇_w L'(w)` for `E'=5` inner epochs.

**Reported results.** ~88% accuracy on minority classes; up to +55% over FedAvg as minority-holding clients grow; CIC-IDS-2017 and UNSW-NB15; `n = 3` underrepresented classes (Heartbleed, SQL Injection, Infiltration / Worms, Shellcode, Backdoor); `η' = 0.1`, `T = 200` rounds.

**Honest gaps.** No convergence theorem robust to the actual non-convex tanh-MLP; no adversarial evaluation; two legacy datasets (2015/2017); small client counts (`K=14`, `K=9`); gradient refinement never ablated component-by-component; a hallucinated reference in its own bibliography (`FedSND`, ref [12]); and the two structural privacy assumptions (§7).

---

## 3. Master verification ledger

**Deduplication rule applied:** each item appears **exactly once**, at its final/best verdict, with a provenance pointer to the pass that first surfaced it and the pass that last touched it.

### 3.1 Metadata corrections (real papers, wrong details in earlier drafts)

| # | Item | Was cited as | Verified reality | First → last |
|---|---|---|---|---|
| 1 | EIFFeL | CCS 2021 | **CCS 2022**, DOI 10.1145/3548606.3560611 | `[S1]`→`[S2]` |
| 2 | FreqFed | NDSS 2023 | **NDSS 2024**, DOI 10.14722/ndss.2024.23620 | `[S1]`→`[S2]` |
| 3 | CrowdGuard | NDSS 2022 | **NDSS 2024**, DBLP `conf/ndss/RiegerKMDS24` | `[S1]`→`[S2]` |
| 4 | Mimicry / Prov-HIDS | NDSS 2023 | **NDSS 2024** | `[S1]`→`[S2]` |
| 5 | AutoAdapt | "…Adversarial Adaptation…" | Official title uses **"Adaption"** | `[S1]`→`[S2]` |
| 6 | Armadillo | "…Secure Aggregation for FL" | Title ends **"…with Input Validation"**; CCS 2025 | `[S1]`→`[S2]` |
| 7 | MPFT | ICLR 2024 | **ICLR 2025**, arXiv:2410.07738 | `[S1]`→`[S2]` |
| 8 | "Intrusion Detection with Neural Networks" (1997) | attributed loosely | **Ryan, Lin, Miikkulainen**, NIPS 1997 (not Cannady) | `[S1]`→`[S2]` |
| 9 | SelfFed | "Expert Systems 2023", IoT | **ESWA 2025**; domain is **IoMT/medical imaging** | `[S1]`→`[S2]` |
| 10 | FedCGS | "Capturing Global Feature Stats" | Title verb is **"Capture"**; AAAI 2025 | `[S1]`→`[S2]` |
| 11 | RepTreeFL | "Neural Networks 2023" | **arXiv Dec 2023**; journal placement unconfirmed | `[S1]`→`[S2]` |
| 12 | "CIC-IDS2023" | dataset name | **No such dataset** → CICIoT2023 / CSE-CIC-IDS2018 | `[S1]`→`[S2]` |
| 13 | PAKDD rank | implied A/A\* | **CORE B** — re-confirmed `[S10]` at portal.core.edu.au | `[S1]`→**`[S10]`** |
| 14 | FedGMKD | "GMM-based clustering" | **Cluster Knowledge Fusion + Discrepancy-Aware Aggregation**; NeurIPS 2024 | `[S1]`→`[S2]` |
| 15 | SBFedNIDS | paper title | Real title *"A Federated Learning NIDS for Multiple Imbalances"*; ICASSP 2025 | `[S1]`→`[S2]` |
| 16 | FedDLAD | short title | *"…for Enhancing Resilience Against Backdoor Attacks"* — a backdoor defence | `[S1]`→`[S2]` |
| 17 | DiffuPac | short title | *"…Adversarial Packets **Generation** via Diffusion Model"* | `[S1]`→`[S2]` |
| 18 | DA-PFL | "Dynamic Affinity-**based**" | *"Dynamic Affinity **Aggregation**"*; AAAI 2024 **and** TNNLS 2025 | `[S1]`→`[S2]` |
| 19 | PACT | ❓ in §4.10, ✅ in §3.3 — self-contradiction | **✅ TrustCom 2025**, DOI 10.1109/Trustcom66490.2025.00206 | `[S4]`→`[S8]` |
| 20 | FedP3E | "IEEE TIFS 2025" | **arXiv:2507.07258 preprint only** — DBLP indexes no journal version | `[S1]`→`[S8]` |
| 21 | Fed-LAA | acronym treated as author-given | Real paper: **Khan et al., *Applied Soft Computing* 201:115676, 2026**, DOI 10.1016/J.ASOC.2026.115676. Acronym is **this series' shorthand** — cite by title/DOI | `[S1]`→`[S7]` |
| 22 | "FedAgile" | placeholder acronym | **Yuan & Wang**, *"Aggregated Gradients-based Adaptive Learning Rate Design in FL,"* ACM CIKM 2025, DOI 10.1145/3746252.3761234 | `[S6]`→`[S6]` |
| 23 | CELM mechanism | "gradient-refined" | ⚠️ **Not gradient-based at all** — server-side **activation/logit maximization on synthetic probe inputs**, weights **frozen after a 5% warm-up** | `[S8]`→`[S9]` |
| 24 | **Six-flaw NIDS enumeration** | attributed to El Mahdaouy et al. (arXiv:2602.05594) | ⚠️ **Primary source is Flood, Engelen, Aspinall & Desmet, IEEE EuroS&P 2024**, DOI 10.1109/eurosp60621.2024.00042. El Mahdaouy et al. cite it. arXiv:2602.05594 is now **v3** | `[S8]`→**`[S10]`** |
| 25 | **NIDS-FGPA** | no DOI, no authors, implied CIC-IDS-2017 | ⚠️ **Wang, Yang & Li, PLoS ONE 2024**, DOI 10.1371/journal.pone.0308639. Evaluated on **Edge-IIoTset + CIC IoT 2023, NOT CIC-IDS-2017** | `[S1]`→**`[S10]`** |
| 26 | **Sentinel** | listed as arXiv:2510.23019 only | ⚠️ Same paper as **IEEE IoTJ 2026**, DOI 10.1109/jiot.2026.3650848 (identical title + 4 authors). Cite the journal version | `[S6]`→**`[S10]`** |
| 27 | **"Precision-Weighted FedAvg"** | ❓ unverified lead, "do not cite" | **✅ Reyes, Di Jorio, Low-Kam & Kersten-Oertel, arXiv:2107.09627 (2021)** → *Computational Intelligence* 2025, DOI 10.1111/coin.70150. Authors never use that acronym | `[S8]`→**`[S10]`** |
| 28 | **Fed-Meta FD** | ❓ (✅ in §4.5, ❓ in §11 — self-contradiction) | **✅ Chen, Tang & Li, IEEE TNSE 2023**, DOI 10.1109/tnse.2023.3266942. Acronym not author-given | `[S1]`→**`[S10]`** |

**Name-collision warnings — do not conflate.** ⚠️ **FedAWA (CVPR 2025) ≠ FedAWARE (arXiv:2310.02702)**: near-homonyms with *opposite* mechanisms — FedAWA rewards **alignment**, FedAWARE rewards **divergence** (FedGRAA's own philosophy). Conflating them inverts §8's argument. Also: DWAFL (ICRAIE 2023, IDS) ≠ FedDWA (IJCAI 2023). FedDRL (ICPP 2022, Nguyen et al.) ≠ FedDRL (arXiv:2307.13716, Chen et al.). FedPKD (ICDCS 2023) ≠ FedPKD (*Computing* 2025) ≠ FedProtoKD (arXiv:2508.19009). TriFusion-IDS (AAAI 2026) ≠ an unrelated 2026 namesake. CT-BRFL ≠ the real ACSAC 2024 "Adversarially Guided Stateful Defense…" (arXiv:2410.11205).

### 3.2 Confirmed hallucination-equivalents — do not cite

**Seven items.** All entered via automated survey tooling or acronym-first search; **none via a primary source.**

| Item | Claimed as | Evidence | First → last |
|---|---|---|---|
| **FedSND** (FedGRAA's own ref [12]) | "Adaptive FL adjusting weights by data-distribution statistics, Zhang et al." | arXiv + web + **DBLP: zero hits**. Only real "FedSND" is an unrelated encryption method. **Replace with FedAWA or FedNova** | `[S2]`→`[S6]` |
| **FeCWGAN-GP** | "IEEE TNNLS 2026", CGAN augmentation for vertical FL | arXiv + web + **DBLP: zero hits**. Nearest real: **FedCG** (IJCAI 2022), **IFL-GAN** (TNNLS 2022) | `[S2]`→`[S6]` |
| **CT-BRFL** | "Client-guided Trust", 2024 | **4 independent negative DBLP results**, acronym *and* concept queries. Use **FedTruth** or **CoAst** for the same point | `[S4]`→`[S7]` |
| **pFedCross** | "personalized federated cross learning", α-fairness for imbalanced FL-IDS | Zero arXiv/DBLP hits across two query strategies. Likely a conflation with real, general-FL **FedCross** (arXiv:2210.08285) | `[S7]`→`[S7]` |
| **Light-FFSL** | "IEEE ~2025/26 (verify IoTJ)" | Broadened descriptive search returns EdgeDetect, FedMADE, Tri-LLM — all real, all differently named. **DBLP: zero hits**. Substitute **EdgeDetect** | `[S6]`→`[S8]` |
| **FedCCW** | "Zhang et al., 2025" | Resolves to **FedCBA**, **FedCW**, **FedLWS** — three real, differently-named papers. **DBLP: zero hits**. Use **FedCBA** | `[S6]`→`[S8]` |
| **FedHDE** | "Zhang et al., 2026" | Fluent description, **no link bearing the name**; resolves to FedHGN, FedPH, Fed-GDBD. **DBLP: zero hits** | `[S6]`→`[S8]` |

**The shared signature — the "no link, no citation" test.** A fluent, mechanically detailed description in a search summary whose result list contains **no link actually bearing the paper's name** is evidence *against* existence, not for it. Observed again this pass `[S10]`: a web search for FedPGO returned a confident, accurate-looking mechanism description whose cited ScienceDirect URL (`S0167404825001312`) is a **different paper** — Sun et al.'s *Computers & Security* 2025 article (N8). The description happened to be right; the citation was not. That is exactly why the test is on *links*, not on *plausibility*.

### 3.3 The escalation path (standing procedure)

1. General web search on the acronym/title.
2. arXiv API title search.
3. **DBLP direct query** on the literal acronym.
4. ⚠️ **MANDATORY before any non-existence verdict — DBLP/OpenAlex retry with a *descriptive phrase* rather than the acronym.**
5. **OpenAlex / Semantic Scholar / Crossref** as structurally different indices.
6. Repository sweep (Unpaywall, OpenAIRE, CORE, institutional repositories, Figshare) before declaring full text unobtainable.

**Step 4 has now changed a conclusion five times.** It rescued **Fed-LAA** `[S7]`, **FedPGO** — the single most consequential paper in this corpus `[S8]` — and, this pass, **Precision-Weighted Federated Learning** and **Fed-Meta FD** `[S10]`. Applied symmetrically in the same sittings, it **confirmed** the non-existence of Light-FFSL, FedCCW and FedHDE. **The rule cuts both ways, which is what makes it trustworthy.** Steps 5–6 are added `[S10]`: they are what resolved NIDS-FGPA's missing metadata, all five survey author lists, and eleven citation counts that two prior passes could not obtain.

**Running ledger.** ~185 distinct papers surveyed across ten passes. **Zero ❓ remain** — the last one (Fed-Meta FD) closed this pass. Seven ❌. All other entries ✅ or ✅⚠️.

---

## 4. Complete reference catalog by family

*Each paper appears **exactly once**, at its primary family. Cross-family relevance is noted by pointer, not by re-listing.*

### 4.1 Foundational FL aggregation

| Method | Citation | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **FedAvg** | McMahan et al., AISTATS 2017, arXiv:1602.05629 | Size-weighted averaging | **The direct baseline/namesake** | ✅ `[S1]` |
| **FedProx** | Li et al., MLSys 2020, arXiv:1812.06127 | Proximal term bounds client drift | Client-side regularization vs. server-side reweighting | ✅ `[S1]` |
| **SCAFFOLD** | Karimireddy et al., ICML 2020, arXiv:1910.06378 | Control variates cancel drift | Corrects *direction*; FedGRAA adjusts *weights* | ✅ `[S1]` |
| **FedNova** | Wang et al., NeurIPS 2020, arXiv:2007.07481 | Normalizes by local-step count | **Closest foundational analogue** — both normalize weights | ✅ `[S1]` |
| **FedOpt** | Reddi et al., ICLR 2021, arXiv:2003.00295 | Server-side adaptive optimizer | Orthogonal, combinable | ✅ `[S1]` |
| ~~FedSND~~ | — | — | **❌ Hallucination** (§3.2) | ❌ `[S2]` |

### 4.2 A\* security venues

| Method | Citation | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **Entente** | Xu, Li, Zheng, Li, NDSS 2025 | Cross-silo graph-NIDS; "adaptive contribution scaling" | Closest A\*-venue analogue; graph-structured vs. tabular minority-boosting | ✅ `[S1]` |
| **EIFFeL** | Roy Chowdhury, Guo, Jha, van der Maaten, CCS 2022 | Secure aggregation + integrity checks | Complementary crypto layer | ✅⚠️ `[S1]` |
| **Cerberus** | Naseri et al., CCS 2022, arXiv:2209.03050 | FL feasibility for security-event forecasting | Motivation, not an algorithm | ✅ `[S1]` |
| **MESAS** | Krauß & Dmitrienko, CCS 2023 | Metric-cascade poisoning defense | Filter-and-remove — risks penalizing the clients FedGRAA protects | ✅ `[S1]` |
| **FreqFed** | Fereidooni et al., NDSS 2024, arXiv:2312.04432 | DCT-domain clustering filters poisoned updates | Same tension as MESAS | ✅⚠️ `[S1]` |
| **CrowdGuard** | Rieger et al., NDSS 2024 | SGX-protected activation clustering | Same tension as MESAS | ✅⚠️ `[S1]` |
| **Kitsune** | Mirsky et al., NDSS 2018, arXiv:1802.09089 | Ensemble-of-autoencoders online NIDS | Foundational centralized NIDS | ✅ `[S1]` |
| **Mimicry vs. Prov-Graph HIDS** | Goyal, Wang, Bates, Han, NDSS 2024 | Mimicry evasion | Tangential, attack-side | ✅⚠️ `[S1]` |
| **AutoAdapt** | Krauß, König, Dmitrienko, Kanzow, NDSS 2024 | Constrained-optimization poisoning evasion | **A threat FedGRAA must acknowledge** — its signal is manipulable | ✅⚠️ `[S1]` |
| **Armadillo** | Ma, Guo, Karthikeyan, Polychroniadou, CCS 2025 | Single-server secure aggregation w/ input validation | Crypto infra; complementary | ✅⚠️ `[S1]` |

### 4.3 A\* AI venues

| Method | Citation | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **FedDLAD** | Ding, Yang et al., IJCAI 2025 (#559) | Dual-layer backdoor-update screening | Filters malicious vs. up-weighting disadvantaged benign | ✅⚠️ `[S1]` |
| **TriFusion-IDS** | Zhao & Zhong, AAAI 2026, 40(19):16433–16440 | Graph+tabular+text contrastive fusion | Centralized, multimodal — orthogonal axis | ✅ `[S1]` |
| **Poisoning with a Pill** | Guo et al., AAAI 2026, arXiv:2407.15389 | Model-redundancy stealthy poisoning | Robustness stress test | ✅ `[S1]` |
| **DiffuPac** | Bin Jasni, Manada, Watabe, NeurIPS 2024 | BERT+diffusion evasive-packet generator | Threat landscape | ✅⚠️ `[S1]` |
| **Intrusion Detection with Neural Networks** | Ryan, Lin, Miikkulainen, NIPS 1997 | Backprop NN over user command profiles | Historical origin | ✅⚠️ `[S1]` |
| **MPFT** | Zhang et al., ICLR 2025, arXiv:2410.07738 | Prototype generation + adapter init + few-shot FT | Representation-side | ✅⚠️ `[S1]` |

### 4.4 Direct FL-IDS competitors — the niche FedGRAA occupies

| Method | Citation | Mechanism | Relation to FedGRAA | Status |
|---|---|---|---|---|
| **FedPGO** ⚠️ **highest priority** | Bikila & Čapek (Univ. of Pardubice), *Computer Networks* 280:112210, **2026-03-10**, DOI 10.1016/j.comnet.2026.112210. **2 citations** `[S10]` | **AGO** decomposes local gradients into components ∥/⊥ to the global gradient and **adaptively reweights**; **PGF** scores clients on gradient directionality + orthogonality; parameter-split architecture. **CICIoT2023 + CIC-IDS2017**, non-IID | **The closest paper in the corpus; occupies the three-way intersection on FedGRAA's own dataset.** Differentiate on: (a) personalization via parameter-split vs. **single global model**; (b) **no γ counterpart**; (c) accuracy/robustness vs. **per-minority-class recall** | ✅ *(metadata triple-confirmed: DBLP, OpenAlex, Semantic Scholar. **Full text access-blocked across 6 channels** — §9 item a)* `[S8]`→`[S10]` |
| **SBFedNIDS** | Zeng, Zheng, Li (Sun Yat-sen), ICASSP 2025 | Self-balancing local rebalance + balance-aware aggregation | **Closest in-domain competitor after FedPGO** — balance statistics vs. gradient | ✅⚠️ `[S1]` |
| **FedMADE** | Sun, Sharma, Nwodo, Stavrou, Wang (Virginia Tech), ISC 2024, arXiv:2408.07152 | Clusters by traffic pattern; weights by measured contribution | Up to **71.07%** minority-attack improvement — must-beat | ✅ `[S1]` |
| **FD-IDS** | Peng, Wu, Xiao (Zhejiang), *Sensors* 25(14):4309, 2025. **20 citations** `[S10]` | KD for non-IID IoT-IDS | Distillation-side; combinable | ✅ `[S1]` |
| **DWAFL** | Umair et al., IEEE ICRAIE 2023 | Dynamic weights from client validation accuracy/loss | Performance signal, not gradient; no minority focus | ✅ `[S1]` |
| **PerFedHypID** | *Scientific Reports* 2025, `s41598-025-11659-7`, PMC12485126 | Hypernetwork aggregation over **client embedding vectors**; 0.62M vs 1.17M params/round | CSE-CIC-IDS2018, non-IID; embedding vs. gradient signal | ✅ `[S8]` |
| **FedKSam\*** | IEEE Xplore doc. **10661778** (asterisk is part of the name) | Phase 1: SAM + KD vs. non-IID drift. Phase 2: re-optimizes classifier decision boundaries | Targets heterogeneity **and class imbalance** — directly comparable minority claims | ✅ `[S8]` |
| **G-PFL-ID** | MDPI *IoT* 7(1):13, 2026-01-29 | GCN/GAE + DeepSVDD + FedReg; 99.46% AUROC IoT-23 | **Unsupervised/one-class** — cite as adjacent, *not* head-to-head | ✅ `[S8]` |
| **AMAFed** | Izadi & Ahmadi, arXiv:2602.12541, 2026-02-13 | Meta-learned dynamic weighting by data quality/contribution; ToN-IoT/N-BaIoT/BoT-IoT | Same family, **no class-imbalance framing**, no dataset overlap | ✅ `[S8]` |
| **Federated transfer learning for rare attack classes** | Chunduru Sri Abhijit, Annie Jerusha, Syed Ibrahim, Varadharajan, *Scientific Reports* 2025-09-30, `s41598-025-02068-x` *(DOI **derived** from article ID — verify once before use)* | Adaptive personalized client layers + transfer learning for zero-day/rare attacks | **Same objective.** ⚠️ CIC-IDS2018 + UNSW-NB15 + Edge-IIoT — **not** CIC-IDS-2017 | ✅ `[S6]` |
| **DS-FedIDS** | *Applied Sciences* 15(9):5067, 2025-05-02, DOI 10.3390/app15095067. **4 citations** `[S10]` | FedPer + dynamic minority up-/majority down-sampling | Client-side resampling vs. server-side reweighting; comparable minority-recall claims | ✅ `[S6]`→`[S7]` |
| **CACFL** | *Neural Computing and Applications*, 2026-03-24, DOI 10.1007/s00521-026-11845-2. **0 citations** `[S10]` | Curriculum learning + **geometric-median aggregation** + CVAE-GAN + Bat–Kestrel–Antlion optimizer | Robust-statistics aggregation — also a §4.10-D validation-free baseline | ✅ `[S6]` |
| **Sentinel** ⚠️ *(merged `[S10]`)* | Singh, Sood, Rajalakshmi, Xiang. **IEEE Internet of Things Journal, 2026-01-12**, DOI 10.1109/jiot.2026.3650848 — preprint arXiv:2510.23019 (27 Oct 2025) | Dynamic knowledge distillation for personalized federated IDS, heterogeneous IoT | Personalization + KD; adjacent. **Cite the journal version** | ✅ `[S6]`→`[S10]` |
| **HADA-FL** | Chandu, Thumula, Parag et al., *IEEE Access* 2025, DOI 10.1109/access.2025.3592481. **17 citations** `[S10]` | "Hybrid Adaptive-Weight Aggregation": SHAP feature stability + per-device DP budgets | Adjacent, not occupying. **Unusually fast uptake** — citation-trajectory watch item | ✅ `[S9]` |
| **TACMA Fed** | Aljuaid, *Symmetry* 2026-07-14, DOI 10.3390/sym18071188. **0 citations** `[S10]` | Extends **AMAFed**; trust scoring + contrastive representation + **rare-class-aware weighting**; ToN-IoT | Second AMAFed-family paper; rare-class framing but no gradient refinement | ✅ `[S9]` |
| **Robust IDS via personalized FL** *(new `[S10]`)* | Sun, Zhou, Wang, Li, *Computers & Security*, 2025-03-21, DOI 10.1016/j.cose.2025.104442. **14 citations** | Personalized FL for IoT IDS robustness | Adjacent; no gradient-refined minority-boosting. Optional add | ✅ `[S10]` |

### 4.5 Federated few-shot / meta-learning

| Method | Citation | Mechanism | Relation | Status |
|---|---|---|---|---|
| **FedFSL-CFRD** | Wang et al. (Xidian), AAAI 2025 | Client-wise + class-wise feature disentanglement | Representation-side | ✅ `[S1]` |
| **Few-Shot Model Agnostic FL** | Huang, Ye, Gao, Du, ACM MM 2022, 7309–7316 | Heterogeneous architectures + public-set distillation | Distant premise | ✅ `[S1]` |
| **Industrial Edge Intelligence** *(was "Fed-Meta FD", ❓→✅ `[S10]`)* | **Chen, Tang & Li, IEEE TNSE 2023**, DOI 10.1109/tnse.2023.3266942. **51 citations**. Companion: Chen, Tang & Chen, GLOBECOM 2022, DOI 10.1109/globecom48099.2022.10001294 | Federated-meta learning for few-shot fault diagnosis; fast adaptation to new fault categories | **IIoT fault diagnosis, not IDS** — complementary, low decision-relevance. Cite by title, not the acronym | ✅ `[S1]`→`[S10]` |
| **REFML** | Cui, Li, Mei et al., IEEE TIM 2023, arXiv:2310.09002 | Representation encoding + meta-learning for few-shot fault diagnosis | Heterogeneity-aware; **distinct from the Chen et al. line above** | ✅ `[S1]` |
| **DFML** | Chen, Zeng, Zhang, Lu, Hou, Zhou, Helal, IEEE/ACM TCBB 2023 | Dynamic attention over meta-tasks + **accuracy-based dynamic client weighting** | **Closest-in-spirit** in this family — accuracy, not gradient | ✅ `[S1]` |
| **FeCoGraph** | Mao, Su et al., IEEE TIFS 2025, 20:2266–2280 | Line-graph flow embeddings + label-aware contrastive loss | **Same domain (few-shot NIDS)** — must-cite complement | ✅ `[S1]` |
| **Fed-Siamese** | Yang, Mao et al., IEEE IoTJ 2025, 12(24):52886–52899 | Meta-learned Siamese embedding + personalized classifier | RFID-HAR domain | ✅ `[S1]` |
| **FSFL-IDS** | Springer chapter, DOI 10.1007/978-3-031-94448-2_7, 2025 | Few-shot FL-IDS for UAV/FANET routing attacks | Same domain, few-shot | ✅ `[S8]` |
| ~~Light-FFSL~~ | — | — | **❌ Hallucination-equivalent** (§3.2) | ❌ `[S6]` |

### 4.6 Prototype-based FL

| Method | Citation | Mechanism | Relation | Status |
|---|---|---|---|---|
| **PROTEAN** | Chennoufi, Han, Blanc, De Cristofaro, Kiennert, ESORICS 2025, arXiv:2507.05524 | Class-prototype exchange for federated IDS under highly non-IID attacks | **Strongest same-domain precedent** — solves FedGRAA's exact problem via prototypes | ✅ `[S1]` |
| **FedGMKD** | Zhang, NeurIPS 2024 (main) | Cluster Knowledge Fusion + **Discrepancy-Aware Aggregation** + KD | Closest prototype analogue — aggregation-side | ✅⚠️ `[S1]` |
| **FedFSA** | Qi, Meng, Li, Hu, Meng, AAAI 2025 | Implicit semantic-preserving space learning for imbalance | **Same problem** — fixes representation, FedGRAA fixes weights | ✅ `[S1]` |
| **FedRFQ** | Yan et al., IEEE TC 2024, 73(4):1086–1098 | SoftPool redundancy reduction + BFT-detect aggregation | Robust-aggregation angle | ✅ `[S1]` |
| **FedPKD** (Lyu, Tang) | IEEE ICDCS 2023 | Prototype-based ensemble distillation | Complementary | ✅ `[S1]` |
| **DeProFL** | Li, Gao et al., IEEE IoTJ 2023/24 | Server-free prototype exchange, time-varying topology | Decentralized — different setting | ✅ `[S1]` |
| **FedMPS** | Yang, Hu et al., IEEE TNNLS 2025 | Multi-level prototype contrastive learning + soft labels | Representation-side | ✅ `[S1]` |
| **RefProtoFL** | Wu, Li, Fan, Zhu, Chen, Feng (Tianjin), arXiv:2601.14746, 2026 | External-referenced prototype alignment | Requires server public data | ✅⚠️ `[S6]` |
| **ProtoHAR** | Cheng, Zhang, Bu, Wang, Wu, Song, IEEE JBHI 2023 | Global activity prototypes correct local representation | HAR domain; also a §4.9 Family-7 signal | ✅ `[S4]`→`[S6]` |
| **FedCK** | Chen, Li, He, *J. Supercomputing* 2025 | Fuzzy-C-Means clustering by label-distribution similarity + conditional-generator KD | Clustering-as-imbalance-proxy; also §4.9 Family 7 | ✅ `[S4]`→`[S6]` |
| **FedP3E** ⚠️ | Darwish, Abdelsalam, Khorsandroo, Roy, **arXiv:2507.07258 — preprint only** | GMM-based prototype exchange + SMOTE for non-IID IoT malware | ⚠️ **Not IEEE TIFS** — cite as preprint | ✅⚠️ `[S1]`→`[S8]` |

### 4.7 Adaptive weighted aggregation — FedGRAA's true competitor set

**The highest-priority family for baselines.**

| Method | Citation | Weight signal | Objective | Directness vs. FedGRAA |
|---|---|---|---|---|
| **Jiang et al.** — *"Preventing harm to the rare in combating the malicious"* | Jiang, Ma, Wang, Yu, Sun, Ni, Liu (UTS + Data61/CSIRO + Zhejiang Lab), *Neurocomputing* 604:128317, **2024-08-10**. **3 citations (OpenAlex) / 6 (S2)** `[S10]` | **DRL-learned** aggregation weights inside a **filtering-and-voting** framework, explicitly designed to protect rare/valuable contributions rather than exclude them as outliers | Robustness + fairness for rare contributions in general FL | **The closest *philosophical* precedent in the corpus.** Makes almost exactly FedGRAA's "protect, don't filter, the rare" argument. Distinguish on (a) IDS-specific vs. general, (b) gradient-refinement vs. DRL, (c) FedGRAA has **no filtering layer** — and per §4.11b that omission is a design decision needing defence, because Jiang et al.'s filtering layer is a *resolution* of the tension FedGRAA leaves open |
| **FedAWARE** | Zeng, Xu, Liu, Pan, Wang, Tang, *"On the Power of Adaptive Weighted Aggregation in Heterogeneous FL and Beyond,"* arXiv:2310.02702 (v4, Dec 2024). **6 citations** `[S10]` | Server-side reweighting that **maximizes gradient diversity** — the convex combination least conflicting in direction. Introduces **client consensus dynamics** | Robustness + generalization under heterogeneity | **High threat, philosophically inconvenient.** Does the **opposite of FedAWA** — rewards divergence, FedGRAA's own philosophy — in general FL, **with convergence theory attached**. Also the best template for §9 gap #1 |
| **CELM** ⚠️ *(mechanism corrected `[S9]`)* | Ukaye, Tastan, Abdu-Aguye, Nandakumar (MBZUAI), arXiv:2605.18892, **2026-05-17, still v1** `[S10]`, 22 pp. | **Not gradient-based.** Server runs **activation maximization on synthetic probe inputs** during a `T_w = 0.05T` warm-up to build a client-class evidence matrix; de-biased, simplex-projected, EMA-smoothed; **weights frozen thereafter**. No raw data, no metadata, no auxiliary set | Minority-class performance under label skew | **Highest-threat non-IDS paper.** Shares FedGRAA's motivating sentence *and* objective; has the simplex projection Algorithm 2 omits. **Clean differentiation:** *logit-probed and frozen* vs. *per-round, gradient-informed, γ-tunable*. Datasets: FashionMNIST, CIFAR-10, FedISIC, EMNIST — **no NIDS data** |
| **Byzantine-Robust FL with Learnable Aggregation Weights** | Parsa, Daghestani, Teixeira, Johansson, arXiv:2511.03529 (Nov 2025, rev. May 2026) | **Aggregation weights as learnable parameters jointly optimized with the model**; alternating minimization; **convergence guarantees under adversarial attack** | Byzantine robustness + convergence | **Prior art for FedGRAA's entire method class.** FedGRAA cannot claim the *idea* of gradient-optimized aggregation weights — only the rare-class objective and IDS instantiation. Best template for gaps #1 **and** #3 |
| **AdaAggRL** | *"Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in FL,"* arXiv:2406.14217; **AAAI 2025**, DOI 10.1609/aaai.v39i24.34733. **4 citations** `[S10]` | MMD similarity of local distributions → **policy learning sets aggregation weights** | Poisoning robustness | The Jiang et al. RL family **at an A\* venue**. ⚠️ Its premise (*benign clients show higher distribution stability*) is in **direct tension** with FedGRAA's (*the valuable clients are the divergent ones*) — §4.11b |
| **Precision-Weighted Federated Learning** *(❓→✅ `[S10]`)* | Reyes, Di Jorio, Low-Kam, Kersten-Oertel, **arXiv:2107.09627 (2021)** → ***Computational Intelligence*, 2025-11-04**, DOI 10.1111/coin.70150. **11 citations** | **Inverse-variance (precision) weights from empirical stochastic-gradient variances**; down-weights noisy updates | Convergence + reliability under non-IID | **Prior art from 2021 for gradient-statistic-derived weighting.** Not minority-targeted, so adjacent — but it establishes that "derive aggregation weights from gradient statistics" is five years old. ⚠️ **Cite by title; "Precision-Weighted FedAvg" is not the authors' term** |
| **FedAPA** | Sun, Sun, Pan, Fu, Guo, IJCAI 2025, arXiv:2502.07456 | Gradient of parameter-change w.r.t. weights | Personalization | **Most similar mechanism** — top-priority baseline |
| **FedAWA** | Shi, Zhao, Zhang, Zhou, Guo, Chang, CVPR 2025, arXiv:2503.15842 | Client-vector **alignment** with global direction | Global generalization | **Very direct — opposite philosophy.** ⚠️ Do not conflate with FedAWARE |
| **DA-PFL** | Yang et al., AAAI 2024 (arXiv:2403.09284) + IEEE TNNLS 2025 | Client affinity/complementarity graph | Class-imbalance-aware personalized FL | **Direct — leading imbalance-aware citation** |
| **FedNolowe** | Le, Huynh, Tran, Dao, Bao, bioRxiv 2025 → *PLOS ONE* 20(8):e0322766, DOI 10.1371/journal.pone.0322766 | Normalized training loss (favours **low-loss** clients) | Robust non-IID | Direct — **opposite for minorities**: high-loss minority clients get down-weighted |
| **FedAAW** | *"Tackling data-heterogeneity variations in FL via adaptive aggregate weights,"* **Knowledge-Based Systems 2024**, DOI 10.1016/j.knosys.2024.112484. **7 citations** `[S10]` | ⚠️ Weights **theoretically derived by minimizing the FL convergence upper bound** on non-IID data | General FL; aggregation-error drift | **The single highest-leverage theoretical direction found in any pass.** It inverts FedGRAA's approach: choose the weights that minimize the bound rather than proposing them heuristically and seeking a tolerant theorem. See §4.11a |
| **FedMcon** | Shen, Li et al., FITEE 2025, DOI 10.1631/FITEE.2400530; ACM MM Asia W. DOI 10.1145/3700410.3702124 | Learned meta-controller on a proxy dataset | Learned aggregation | Direct; FedGRAA needs no proxy dataset |
| **FBLG** | Xu, Y. Li et al., IJCAI 2024, 5289–5297 | Local loss + Jensen–Shannon-divergence graph | Dual-skew (label+size) non-IID | Direct |
| **FedDRL** | Nguyen et al., ICPP 2022, arXiv:2208.02442 | DRL-learned per-client "impact factor" | Cluster-skew non-IID | Direct (RL vs. gradient). *Disambiguate from arXiv:2307.13716* |
| **FedSWA/FedMoSWA** | J. Liu et al., ICML 2025, arXiv:2507.20016 | Weight-space SWA + momentum control variates | Flat-minima generalization | Orthogonal lever, not minority-aware |
| **NIDS-FGPA** ⚠️ *(metadata resolved `[S10]`)* | **Wang, Yang & Li, *PLoS ONE*, 2024-10-24**, DOI 10.1371/journal.pone.0308639. **20 citations**, gold OA | **Gradient Similarity Aggregation (GSA)** dynamically selects and weights model updates **to reduce communication overhead**; + Paillier HE; 2DCNN-BiGRU backbone for incomplete features/missing labels | ⚠️ **Edge-IIoTset + CIC IoT 2023 — NOT CIC-IDS-2017.** Still the closest paper on the (gradient-weighting × federated-NIDS) two-way intersection; objective is communication efficiency, not minority-boosting |
| **FedVG** | Devkota, Thrasher, Adjeroh, Bhattarai, Gyawali (West Virginia U.), arXiv:2602.21399, Feb 2026 | Validation-set layer-wise **gradient-norm** weighting | General FL | Structurally closest general-FL analogue; needs a public validation set; **no mention of IDS anywhere** `[S7]` |
| **Fed-LAA** ⚠️ | Khan, Rizwan, Ahmad, Khan, Li, Kim, *Applied Soft Computing* 201:115676, **2026-06-11**, DOI 10.1016/J.ASOC.2026.115676. **0 citations** `[S10]` | Layer-wise weights from **gradient angular similarity** to the global descent direction + data-volume weighting | Same "weights from gradient info" family; layer-level, not IDS | ✅⚠️ **Cite by title/DOI — the acronym is this series' shorthand, not the authors'** |
| **ECGR** | *"Local Gradient Regulation Stabilizes FL under Client Heterogeneity,"* arXiv:2601.03584, 2026-01 | **Client-side** decomposition of mini-batch gradients into aligned/misaligned components, re-aggregated | **Term-adjacent scooping-risk watch item** — its "gradient re-aggregation"/"refining contributions" language is close enough to FedGRAA's terminology to need explicit distinction |
| **FedAgg** | arXiv:2303.15799, 2023 | Adaptive per-client learning rate from aggregated-gradient mean-field term | Learning-rate, not weight, adaptation |
| **Aggregated Gradients-based Adaptive LR Design** *(was "FedAgile")* | Yuan & Wang, ACM CIKM 2025, DOI 10.1145/3746252.3761234 | Aggregated-gradient-driven adaptive per-round learning rate | Same family as FedAgg. **Cite by title** |
| **FedAWR** | *Future Internet* 18(2):106, 2026-02-18, DOI 10.3390/fi18020106 | Joint client-weight + per-client-LR optimization by **compute capability** | Signal is compute capability, not gradient |
| **FedFMD** | ACM CIKM 2025, DOI 10.1145/3746252.3761316 | Mahalanobis distance between client updates and global distribution | Validation-free robust-statistics alternative |
| **FedCoef** | Xie, Pei, Shi, ASCC 2024, 1302–1306, DBLP `conf/ascc/XiePS24` | Coefficient-of-variation weighting | Statistically-principled alternative |
| **FedDUAL** | Sahoo et al., arXiv:2412.04416, OpenReview `8M3XfmNhTZ` | Adaptive loss keyed to local-global distributional distance + dynamic aggregation | ✅⚠️ **TMLR acceptance not confirmed — cite by arXiv ID** |
| **FedGC** | OpenReview `eZN8nUXAVO7` | Client- and server-side gradient-constraint projection | ✅⚠️ **Venue unconfirmed** |
| **FedAdp**, **FedAtt**, **pFedMe** | Wu & Wang 2022; Ji et al. 2019; Dinh et al. 2020 | Earlier adaptive/attention/personalized weighting | Foundational context (see `understanding/10_SOTA_adaptive_aggregation.md`) |
| ~~FedCCW~~, ~~FedHDE~~ | — | — | **❌ Hallucination-equivalent** (§3.2) |

### 4.8 Augmentation, knowledge transfer, one-shot / communication-efficient FL

| Method | Citation | Mechanism | Relation | Status |
|---|---|---|---|---|
| **AugFL** | Yue, Qin, Deng, Ren, Zhang, Zhang, IEEE/ACM ToN 2025, arXiv:2503.02154 | Server-held pretrained model as meta-learning regularizer, inexact-ADMM | FedGRAA is PM-free — complementary | ✅ `[S1]` |
| **RepTreeFL** | Ghilea & Rekik, arXiv Dec 2023 | Replica clients + **discrepancy-driven dynamic weights** | **Strongest methodological neighbour in this family** | ✅⚠️ `[S1]` |
| **FedLMD** | Lu, Li, Bao, Wang, Qian, Ge, ACM MM 2023, arXiv:2409.13136 | Label-masking distillation; mask pattern proxies class presence | Also a §4.9 Family-7 signal | ✅ `[S1]` |
| **FedCGS** | Guan, Zhou, Gu (CAS), AAAI 2025, arXiv:2503.06962 | Global feature statistics for training-free one-shot FL | One-shot/statistics vs. iterative/gradient | ✅⚠️ `[S1]` |
| **SEM** | Pham, Nguyen et al., CANDAR 2023, 120–126 | Lightweight model-agnostic local-training add-on | Stackable | ✅ `[S1]` |
| **SelfFed** | Khowaja, Dev, Anwar, Linguraru, arXiv 2023 → ESWA 2025 | Swin-Transformer SSL pretraining + aggregation regularizer | IoMT domain | ✅⚠️ `[S1]` |
| **OT One-Shot FL** | Chiang, Terai, Chiang, Lin, Ji, Lui, IEEE IoTJ 2024, 11(2):2166–2180 | OT neuron alignment before single-round fusion | Parameter misalignment vs. contribution imbalance | ✅ `[S1]` |
| **Model Fusion via OT** | Singh & Jaggi, NeurIPS 2020 | OT model fusion | Foundational | ✅ `[S1]` |
| **FedDiverse** | Németh, Fanì, Ng, Caputo, Lozano, Oliver, Quadrianto, FLTA 2025, arXiv:2504.11216 | Diversity-aware client selection | Selection-side | ✅ `[S6]` |
| ~~FeCWGAN-GP~~ | — | — | **❌ Hallucination** (§3.2) | ❌ `[S2]` |

### 4.9 Detecting class imbalance without a server-side oracle (7 families)

*Full taxonomy and communication math: [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) §2; summarized §7.*

| Family | Method | Citation | Mechanism | Status |
|---|---|---|---|---|
| 1 — Gradient-composition inference | **Ratio Loss** | Wang, Xu, Wang, Zhu, AAAI 2021, arXiv:2008.06217 | Infers per-round class composition from output-layer gradients | ✅ `[S3]` |
| 1 | **FedImT** | Yan, Li, Ota, Dong, arXiv:2310.14784, 2023 | Online/recursive composition estimate | ✅ `[S3]` |
| 1 | Class-imbalance reduction via selection | Yang, Wong, Zhu, Wang, Qian, arXiv:2011.11266, 2020 | Inferred composition drives client *selection* | ✅ `[S3]` |
| 1 | **FedReLa** | Hu, Menéndez, Liu, Gong, Wang, Peng, arXiv:2606.26037, 2026 | Corrects biased boundaries without knowing global class distribution | ✅ `[S6]` |
| 1 | **FedCGNM/FedHOO** | Park, Klabjan, Braun, Li, Ananthanarayanan, ICML 2026, arXiv:2607.01474 | Client-side class-grouped variance-normalized momentum | ✅ `[S6]` |
| 2 — Homomorphic encryption | **Fed-CBS** | Zhang et al., ICML 2023, arXiv:2209.15245 | HE-computed imbalance measure drives class-balanced sampling | ✅ `[S5]` |
| 2 | **FLICKER** | Guleria, Harshan, Prasad, Bharath, CODS-COMAD 2024, arXiv:2410.21192 | CKKS-HE client-attribute summaries pre-balance before FL | ✅ `[S5]` |
| 3 — SecAgg / DP summary stats | **SecAgg** | Bonawitz et al. (Google), NeurIPS-W 2016, arXiv:1611.04482 | Server learns only `Σ_k x_k`; tolerates ~1/3 dropout | ✅ `[S5]` |
| 3 | Sample-and-Threshold DP histograms | Bharadwaj & Cormode, AISTATS 2022, arXiv:2112.05693 | Formal `(ε,δ)`-DP histogram, no noise on disclosed data | ✅ `[S5]` |
| 3 | **LDP-Fed** | Truex, Liu, Chow, Gursoy, Wei, arXiv:2006.03637, 2020 | Local-DP noise client-side | ✅ `[S5]` |
| 4 — Loss-only reweighting | **Fed-Focal Loss** | Sarkar, Narang, Rai, FL-IJCAI 2020, arXiv:2011.06283 | Focal-loss reshaping + tunable contribution sampling | ✅ `[S5]` |
| 4 | **q-FedAvg** | Li et al., ICLR 2020 | Scalar per-client-loss reweighting | ✅ `[S5]` |
| 4 | **AFL** | Mohri, Sivek, Suresh, ICML 2019 | Agnostic FL, scalar-loss reweighting | ✅ `[S5]` |
| 5 — Clustering | **IFCA** | Ghosh, Chung, Yin, Ramchandran, NeurIPS 2020, arXiv:2006.04088 | Cluster-membership by local loss + per-cluster GD | ✅ `[S5]` |
| 5 | Byzantine-Robust Clustered FL | Tao, Yang, Kulkarni, arXiv:2306.00638, 2023 | IFCA + trimmed-mean/median | ✅ `[S5]` |
| 5 | **EMD-CFL** | Zhang, Williams, Toni, arXiv:2506.07769, 2025 | One-shot clustering via Earth Mover's Distance | ✅ `[S5]` |
| 5 | **Clust-PSI-PFL** | Jimenez-Gutierrez et al., IPDPS 2026, arXiv:2512.20363 | Population Stability Index clustering | ✅ `[S6]` |
| 5 | **Astraea** | Duan et al., IEEE ICCD 2019, arXiv:1907.01132 | Self-balancing via mediator-based rescheduling | ✅ `[S3]` |
| 6 — Heterogeneity metrics | EMD non-IID metric | Zhao, Li, Lai, Suda, Civin, Chandra, arXiv:1806.00582, 2018 *(already ref [13] in the draft)* | EMD between local and population class distribution | ✅ `[S3]` |
| 6 | MID/WCS metrics | Xiao & Wang, IEEE SSCI 2021, arXiv:2109.04094 | Two imbalance-quantification metrics for distributed settings | ✅ `[S5]` |
| 7 — Restricted-softmax / masked-logit / prototype | **FedRS** | Li & Zhan, **KDD 2021** | Restricted softmax; restriction shape signals label presence | ✅ `[S4]`→`[S6]` |
| 7 | *(also here: FedLMD §4.8, FedCK §4.6, ProtoHAR §4.6, PROTEAN §4.6)* | — | Cross-referenced, not re-listed | — |
| — | Class-imbalance survey | Zhang, Li, Qi, He, arXiv:2303.11673, 2023 | Frames the subfield around "estimate imbalance without knowing the data" | ✅ `[S5]` |

### 4.10 Validating adaptive aggregation without a server-held labelled set (6 approaches)

| Approach | Method | Citation | Status |
|---|---|---|---|
| **A** — Federated/distributed evaluation | Original FedAvg evaluation protocol | McMahan et al., AISTATS 2017 | ✅ `[S5]` |
| A | **FedDropoutAvg** | Gunesli, Bilal, Raza, Rajpoot, arXiv:2111.13230, 2021 | ✅ `[S5]` |
| **B** — Federated gradient-of-validation-loss | *FedGRAA-v2's own construction (§7)* — **not prior art** | — | — |
| **C** — Secure MPC validation | **SLVR** | Choi, Rachuri, Wang, Jha, Wang, arXiv:2502.08055, 2025 | ✅ `[S5]` |
| C | **LoByITFL** | Xia, Egger, Hofmeister, Bitar, arXiv:2405.19217, 2024 | ✅ `[S5]` |
| C | **ByITFL** | Xia, Hofmeister, Egger, Bitar, arXiv:2405.08698, 2024 | ✅ `[S5]` |
| **D** — Validation-free robust aggregation | Geometric median / trimmed mean | Chen, Su, Xu, arXiv:1705.05491, 2017 | ✅ `[S5]` |
| D | **RAGA** | Zuo, Yan, Fan, Hu, Shan, Quek, Zhao, IEEE TMC 2025, arXiv:2403.13374 | ✅ `[S5]` |
| D | **FedTruth** | Ebron, Zhang, Yang, ACISP 2025, arXiv:2311.10248 | ✅ `[S5]` |
| D | **FLTrust** *(the critiqued structural precedent)* | Cao, Fang, Liu, Gong, NDSS 2021, arXiv:2012.13995 | ✅ `[S5]` |
| D | **FedGreed** *(still makes the same `D_V` assumption, 2025)* | Kritharakis, Makris, Jakovetic, Tserpes, arXiv:2508.18060 | ✅ `[S6]` |
| **E** — Imbalance-aware metrics | **EdgeDetect** | Mohammad, arXiv:2604.14663, 2026 | ✅ `[S7]` |
| **F** — Validation-free contribution/trust | **CoAst** | Wu, Zhang, Li, Xu, Zhong, arXiv:2409.02495, 2024 | ✅ `[S5]` |
| F | **FedPRS** | ACM TIST, DOI 10.1145/3779055 | ✅ `[S4]`→`[S6]` |
| F | **PACT** | Sheykholeslamzadeh & Alhadidi, TrustCom 2025, 1775–1781, DOI 10.1109/Trustcom66490.2025.00206 | ✅ `[S4]`→`[S8]` |
| F | **REVERB-FL** | Peechara & Sahay, arXiv:2512.13647, IEEE TASLP | ✅ *(audio domain — cross-domain precedent only)* `[S6]` |
| F — **data-free sub-line** | **CELM** *(→ §4.7)* | Ukaye et al., arXiv:2605.18892 | ✅ `[S8]` |
| F — data-free | **SpectralFed / SpectralFuse** | Ukaye, Abdu-Aguye, Tastan, Nandakumar, *"Data-Free Contribution Estimation in FL using Gradient von Neumann Entropy,"* arXiv:2604.22562; **CVPR 2026 FedVision Workshop** | ✅ `[S8]` |
| F | **CGSV** / **ShapFed** | NeurIPS 2021 / IJCAI 2024 — CELM's own nearest comparators | ✅ `[S9]` |
| F | ~~CT-BRFL~~, ~~pFedCross~~ | — | **❌** (§3.2) |

⚠️ **Structural note carried from `[S9]`:** this is **two loosely-related sub-clusters**, not one lineage. CELM's own Related Work cites **CGSV** and **ShapFed** (Shapley/gradient-alignment descent) and **not** PACT, FedPRS or CoAst (validation-avoidance). Do not present "data-free contribution estimation" as a single named research line.

### 4.11 Threat model, DP, and testing foundations

| Item | Citation | Role |
|---|---|---|
| Deep Leakage from Gradients (DLG) | Zhu, Liu, Han, NeurIPS 2019, arXiv:1906.08935 | Residual gradient-inversion risk on the unchanged `θ_k^t` channel |
| iDLG | Zhao, Mopuri, Bilen, 2020 | Exact single-example label recovery from output-layer gradient sign |
| Inverting Gradients | Geiping, Bauermeister, Dröge, Moeller, NeurIPS 2020 | Label-set/per-class-count recovery |
| GradInversion | Yin, Mallya, Vahdat, Alvarez, Kautz, Molchanov, CVPR 2021 | Same family, image domain |
| DP-SGD / moments accountant | Abadi et al., CCS 2016, arXiv:1607.00133 | Underlies RDP composition |
| Rényi Differential Privacy | Mironov, IEEE CSF 2017, arXiv:1702.07476 | Tighter composition |
| Boosting and Differential Privacy | Dwork, Rothblum, Vadhan, FOCS 2010 | Advanced-composition theorem |
| Algorithmic Foundations of DP | Dwork & Roth, 2014 | Canonical composition source |
| Advances and Open Problems in FL | Kairouz, McMahan et al., arXiv:1912.04977 | Standard FL threat-model taxonomy |
| PL-condition convergence | Karimi, Nutini, Schmidt, ECML-PKDD 2016 | Replaces global convexity for the non-convex tanh-MLP |
| FedAvg non-IID convergence | Li, Sahu, Talwalkar, Smith, ICLR 2020 | Partial/stochastic-participation framing |
| Flower framework | Beutel et al., arXiv:2007.14390 | Recommended implementation target `[S3]` |

### 4.11a Convergence theory for adaptive-weight FL

| Item | Citation | What it gives FedGRAA |
|---|---|---|
| **FedAAW** — derive weights *from* the bound | Knowledge-Based Systems 2024, DOI 10.1016/j.knosys.2024.112484 | **The best framing available.** Reframes the theorem from a bolt-on into a *derivation* |
| **Learnable aggregation weights, jointly optimized** | Parsa et al., arXiv:2511.03529 | A ready-made proof architecture: alternating minimization with guarantees **under attack**. Also prior art |
| **FedAWARE** — fix the heterogeneity measure | Zeng et al., arXiv:2310.02702 | **Client consensus dynamics.** Matters because FedGRAA's one-attack-per-client partition is an extreme-heterogeneity regime where standard bounded-dissimilarity assumptions are near-vacuous — a theorem under them risks being technically true but empty |

**The recurring assumption set:** *L*-smooth possibly-non-convex local objectives; bounded stochastic-gradient variance; bounded learning rates; partial participation; and — critically — **aggregation weights bounded away from zero and from above.**

⚠️ **That last condition *is* the simplex projection Algorithm 2 omits.** The empirical divergence in [`fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) and the missing theorem are **the same defect from two directions** — the projection is required for the proof, and its absence is observable in the runs (round-1 violation on **all 60 literal-arm cells**, 10 seeds × 3 γ × 2 datasets, zero exceptions). Target theorem: **PL-condition convergence to a noise ball under bounded weight deviation and bounded composite-gradient estimation error `ζ`.** All four ingredients have citable precedent.

### 4.11b Adversarial robustness of reweighting aggregators

| Item | Citation | Role |
|---|---|---|
| **SecFedNIDS** | Zhao, Zhang, Guo, Yao, Li, **FGCS 134:154–169, 2022**, DOI 10.1016/j.future.2022.04.010; code `github.com/zhangzhao156/SecFedNIDS`. **100 citations** `[S10]` | **The must-cite in-domain baseline.** Model-level poisoned-model detection + data-level poisoned-traffic filtering. **+48% accuracy under attack on UNSW-NB15, +36% on CIC-IDS2018** — FedGRAA's own datasets, with public code |
| **WeiDetect** | Sameera, Vinod P., Rocha, Rehiman, Conti, *J. Inf. Secur. Appl.* 2025, DOI 10.1016/j.jisa.2025.104275; arXiv:2504.04367 | Weibull-based poisoning defense for FL-NIDS |
| **Parameterizing poisoning attacks in FL-based IDS** | **ARES 2023**, DOI 10.1145/3600160.3605090 | Attack-side parameterization to evaluate against |
| **AdaAggRL** *(→ §4.7)* | AAAI 2025 | RL-learned weights *as a poisoning defense* — the adversarial mirror-image of FedGRAA |
| **SoK: Benchmarking Poisoning Attacks and Defenses in FL** | arXiv:2502.03801 | Standard evaluation harness |

⚠️ **The structural tension FedGRAA must argue, not merely disclose.** Poisoning defenses identify malicious clients by **statistical deviation** — MESAS, CrowdGuard, FreqFed and AdaAggRL all rest on it (AdaAggRL states it outright: *benign clients exhibit significantly higher data-distribution stability than malicious clients*). FedGRAA's founding premise is that **the most valuable clients are precisely the divergent, high-loss, statistically anomalous ones.** These premises conflict directly. FedGRAA is therefore not merely *undefended* — it **structurally amplifies exactly the signal a poisoning defense suppresses**, and an adversary who makes an update look like a rare-class contribution gets up-weighted **by construction**.

This recasts **Jiang et al.**: their filtering-and-voting layer is a proposed **resolution** of this tension, not merely a parallel argument. FedGRAA's omission of any such layer is a design decision that needs defending. **Minimum credible robustness section:** label-flipping and model-poisoning under the SecFedNIDS protocol; **an adaptive attacker that fabricates rare-class gradient signatures to farm up-weighting** (the mechanism-specific attack nobody else can run for you); AutoAdapt as the adaptive-threat limit; and an explicit statement of whether **γ trades detection performance against poisoning exposure**.

### 4.12 Broader FL-IDS context and surveys

> ⚠️ **Survey-evidence caveat — read before relying on any row.** The novelty claim once leaned on *"no FL-IDS aggregation survey lists a gradient-refinement method."* That must now hold against **six** surveys. **No survey taxonomy has been read in primary text by any pass, including this one** — all six remain blocked (ACM DL 403 including its Figshare mirror; Springer IdP redirect even for gold-OA content) `[S10]`. Given that FedPGO exists and postdates most of them, **the absolute framing is not defensible.** Use the comparative framing: *"surveys of FL-IDS aggregation organize the field around size-, loss-, similarity- and cluster-based weighting; gradient-refinement approaches appear only in the most recent work (FedPGO 2026, NIDS-FGPA 2024)."*
>
> **What `[S10]` did add:** complete author lists, exact dates, and citation counts for all six — previously three had no authors on file at all — plus the abstract of the most on-point one.

| Survey | Citation | Cites `[S10]` | Relevance |
|---|---|---|---|
| **CSUR 2024 — Aggregation Strategies** | **Khraisat, Alazab, Singh, Jan, Gomez.** *"Survey on Federated Learning for Intrusion Detection System: Concept, Architectures, Aggregation Strategies, Challenges, and Future Directions."* ACM Computing Surveys, **2024-08-06**, DOI 10.1145/3687124 | **88** | The original taxonomy-gap citation. Bronze OA — but both the ACM PDF and its Figshare mirror returned 403 `[S10]` |
| **CSUR 2025 — Systematic Review** | **Hernández-Ramos, Karopoulos, Chatzoglou, Kouliaridis, Mármol Campos, González-Vidal, Kambourakis.** *"Intrusion Detection Based on Federated Learning: A Systematic Review."* ACM Computing Surveys, **2025-04-23**, DOI 10.1145/3731596 | **49** | Corroborating systematic review. Closed access |
| **CSUR 2024 — Collaborative IDS taxonomy** | **Wardana & Sukarno.** *"Taxonomy and Survey of Collaborative Intrusion Detection System using Federated Learning."* ACM CSUR **57(4) Art. 88**, 2024-10-25, 36 pp., DOI 10.1145/3701724 | **23** | Hybrid OA; ACM PDF 403 `[S10]` |
| **JPDC 2024** | **Zhang, Ye, Huang, Liu, Gu.** *"Survey of federated learning in intrusion detection."* *J. Parallel and Distributed Computing*, **2024-09-19**, DOI 10.1016/j.jpdc.2024.104976 | **46** | Fifth survey |
| **AI Review 2026** | **Bunko, Johnstone, Yang, Scott.** *"A survey of privacy-preserving federated learning for intrusion detection systems."* *Artificial Intelligence Review*, **2026-02-28**, DOI 10.1007/s10462-026-11519-4 | **3** | Also the likely source of the **pFedCross** conflation (§3.2) |
| **J. Cloud Computing 2026** ⚠️ **most on-point** | **Ma, Yang, Wang, Chen.** *"Model optimization in federated intrusion detection systems: a survey of aggregation, communication, and personalization."* *J. Cloud Computing*, **2026-04-30**, DOI 10.1186/s13677-026-00901-w | **0** | Gold OA, but Springer's IdP wall blocks even the OA PDF `[S10]`. **Abstract obtained via DOAJ** — see below |

**⚠️ Load-bearing finding on the J. Cloud Computing survey (item f, partial resolution `[S10]`).** The taxonomy could not be read, but the abstract was obtained verbatim from DOAJ and it is directly relevant. The survey organizes FL-IDS into *"three tightly coupled dimensions: aggregation, communication-efficient transmission, and personalization"* and states that these dimensions *"jointly shape not only detection accuracy, communication efficiency, and local adaptability, but also **rare-attack recall**, robustness to unreliable updates, concept-drift handling, and deployment latency."*

**Read this carefully, because it cuts both ways.** The most on-point survey in existence **explicitly names rare-attack recall as an outcome shaped by aggregation choice** — which (i) *validates* FedGRAA's framing as a recognized axis the field cares about, and (ii) *weakens* any suggestion that rare-attack recall is an evaluation axis nobody has thought about. The defensible §8 wording is that no adjacent **method paper** *reports* per-minority-class recall as its headline metric — not that the axis is unrecognized. **§8 is worded accordingly.**

| Other same-domain context | Citation | Note |
|---|---|---|
| **StatAvg** | arXiv:2405.13062 / IEEE journal 2025 | Per-feature statistics sharing corrects non-IID feature skew |
| **FedNIDS** | Nguyen, Hore, Shah, Le, Bastian, **DTRAP 6(1):1–23, 2025**, DOI 10.1145/3696012 | Packet-based FL-IDS framework |
| **FADngs** | Dong, Chen, Wu, Tang, Zhuang, **IEEE TNNLS 36(2):2578–2592, 2025**, DOI 10.1109/TNNLS.2024.3350660 | Federated anomaly detection |
| **XAI-SOH-FL** | Aslam, Hassan, Zahra, Shahzad (NUST), arXiv:2606.00134, 2026-05-28 | ⚠️ **See the γ reviewer-risk note below** |
| **FedDrone-Shield** | *Scientific Reports* 2026, `s41598-026-50865-9` | Benchmarks FedAvg/FedProx/FedAdam/FedMedian/ClusterAvg for UAV IDS. *(One paper, not two — corrected `[S6]`)* |
| **F-ACVAE** | arXiv:2607.04698, 2026 | Federated conditional VAE, constrained-momentum Gaussian aggregation |
| **Tri-LLM Cooperative Federated Zero-Shot IDS** | arXiv:2602.00219, 2026 | LLM zero-shot IDS, trust-aware aggregation |
| **Microsecond Federated SVD on Grassmann Manifold** *(new `[S10]`)* | arXiv:2510.18501, 2025 | Real-time IoT IDS via manifold SVD. Adjacent; not an aggregation-weighting method |

⚠️ **The XAI-SOH-FL γ collision — a near-verbatim, cheap-to-pre-empt reviewer question `[S9]`.** XAI-SOH-FL's own stated motivation is that its base method suffers from *"reliance on a manually tuned aggregation parameter **γ**"* — **the same Greek letter FedGRAA uses for its own manually-set trade-off coefficient** — and its fix is to make γ dynamic via similarity-thresholding and Bayesian optimization. A reviewer who has read it (same sub-field, same base dataset CICIDS2017, published shortly before a FedGRAA submission would land) has a ready-made question: *"the closest same-domain paper found manual γ tuning worth automating; why does FedGRAA not?"* **Cheap to pre-empt** (one sentence in Limitations/Future Work citing XAI-SOH-FL as the precedent), **expensive to be caught flat-footed by.** Checklist item §9.11.

---

## 5. Intrusion-detection datasets

| Dataset | Creator / Paper | Size / Features | Classes | FL relevance |
|---|---|---|---|---|
| **CIC-IDS-2017** ✅ *(primary)* | CIC (UNB); Sharafaldin, Lashkari, Ghorbani, ICISSP 2018 | ~50 GB PCAP, 80+ CICFlowMeter features, ~2.8M flows | Brute Force, DoS, Heartbleed, Web/SQLi, Infiltration, Botnet, PortScan, DDoS | De-facto benchmark; days/attack-types shard naturally into non-IID clients |
| **UNSW-NB15** ✅ *(secondary)* | Moustafa & Slay, MilCIS 2015 (+ IS Journal 2016) | 2,540,044 records, 49 features | Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode, Worms | Train/test mismatch and "Generic" dominance are known issues |
| **NSL-KDD** (+ KDD99 / DARPA98) ✅ | Tavallaee, Bagheri, Lu, Ghorbani, CISDA 2009 | 41 features, ~125,973/22,544 | DoS, Probe, R2L, U2R + Normal | Very dated; legacy comparability only |
| **CSE-CIC-IDS2018** ✅ | CSE + CIC/UNB, AWS Open Data | 80 features, 10M+ records | Brute-force, Heartbleed, Botnet, DoS, DDoS, Web, Infiltration | Larger successor to CIC-IDS-2017 |
| **"CIC-IDS2023"** ❌ | — | — | — | **Does not exist.** Re-point to CICIoT2023 or CSE-CIC-IDS2018 |
| **CICIoT2023** ✅ | CIC/UNB; Neto et al., *Sensors* 23(13):5941, 2023 | 105 real IoT devices, 46+ features | 33 attacks in 7 categories incl. Mirai | Modern IoT benchmark; **FedPGO's other dataset** |
| **Edge-IIoTset** ✅ *(added `[S10]`)* | Ferrag et al. | IIoT testbed | 14 attacks | **NIDS-FGPA's actual primary dataset** (N5) |
| **FLNET2023** ✅ *(FL-native)* | Kumar, Liu, Tayeen, Misra, Cao, Harikumar, Perez, IEEE MILCOM 2023, 345–350 | Naturally distributed across nodes | — | Most on-point for the FL claim; niche/low-citation |
| **NF-\*-v2 family → NF-UQ-NIDS-v2** ✅ ⚠️ **the recommended upgrade** | Sarhan, Layeghy & Portmann, *"Towards a Standard Feature Set for NIDS Datasets,"* **Mobile Networks and Applications 2022**, arXiv:2101.11315. **NetFlow V3** (UQ Cyber, 2025) adds 10 temporal features | Four independently-collected NIDS datasets in **one shared 43-feature schema** (53 under V3) | Union of four attack taxonomies | ⚠️ **The single best experimental upgrade available.** One dataset per silo makes the non-IID structure **real rather than synthetically partitioned** — no Dirichlet-α simulation, no one-attack-per-client artifice. Answers §9 gaps #4, #5 **and** the §7.1 oracle-partition objection at once. Cost: a feature-extraction pass, **no algorithmic change** |

> ⚠️ **A validity threat, not a currency complaint — with the citation corrected `[S10]`.** **CIC-IDS-2017 and CSE-CIC-IDS-2018 contain labelling errors and unrealistic temporal separation between attack and benign traffic.** The six recurring design flaws — *limited feature diversity, high feature interdependence, ambiguous ground truth, collapse of traffic distributions, artificial diversity generation, labelling inaccuracies* — are established by **Flood, Engelen, Aspinall & Desmet, *"Bad Design Smells in Benchmark NIDS Datasets,"* IEEE EuroS&P 2024**, DOI 10.1109/eurosp60621.2024.00042 ✅ `[S10]`. **Cite Flood et al., not El Mahdaouy et al.** — the latter (arXiv:2602.05594, now v3) is a survey that *reports* the finding and is the right citation for the deployment/evaluation framing (temporal leakage, data splitting, weak cross-dataset generalization), not for the six flaws. Corroborating primary studies named there: **Engelen et al. (2021)**, **Lanvin et al. (2022)**, **Liu et al. (2022)**. See also *Benchmarking the Benchmark* (arXiv:2104.09029) and *Temporal Analysis of NetFlow Datasets for NIDS* (arXiv:2503.04404).
>
> **Why this bites FedGRAA specifically.** The headline result is per-minority-class accuracy on CIC-IDS-2017's rarest classes — Heartbleed, Infiltration, Web/SQLi — precisely where a handful of labelling errors moves the number materially, and §7.3 records that **Heartbleed has ≈1 held-out sample**. A reviewer who knows this literature will read "≈88% on minority classes, +55% over FedAvg" and ask for the confidence interval on a class with single-digit support. **The ≥10-seed / Wilcoxon / Holm–Bonferroni infrastructure already built in Stage B (`fedgraa/stats.py`) is the answer, and should lead for rare classes rather than appear as checklist item #8.**

---

## 6. Venue rankings

*§6 previously argued the venue point by assertion. It now has sourced numbers `[S10]`.*

| Venue | Field | Rank | FL-IDS relevance |
|---|---|---|---|
| CCS | Security | A\* | MESAS, EIFFeL, Armadillo |
| NDSS | Security | A\* | Entente, FreqFed, CrowdGuard, AutoAdapt |
| IEEE S&P | Security | A\* | Flagship adversarial/FL security |
| AAAI | AI/ML | A\* | FedCGS, FedFSA, DA-PFL, FedFSL-CFRD, AdaAggRL |
| IJCAI | AI/ML | A\* | FedAPA, FedDLAD, FBLG |
| NeurIPS | ML | A\* | FedNova, FedGMKD |
| ICML | ML | A\* | SCAFFOLD, FedSWA, Fed-CBS |
| ICLR | ML/DL | A\* | FedOpt, MPFT |
| CVPR | Vision | A\* | FedAWA, SpectralFed (FedVision W.) |
| ESORICS | Security | A | PROTEAN |
| IEEE TIFS | Security (journal) | Q1, no CORE conf rank | FeCoGraph *(⚠️ FedP3E and PROTEAN removed — §3.1 #20)* |
| **Computer Networks** (Elsevier) | Networking (journal) | Q1, no CORE conf rank | **FedPGO** — the closest competitor |
| **PAKDD** | Data mining | **B** ✅ **re-confirmed `[S10]`** | **FedGRAA's target venue** |

### 6.1 The sourced comparison — closing the previously-assertion-only argument

The `[S8]` argument was: *"the literal novelty claim survives only through the A\*/A qualifier, and that is a weak position to argue from a CORE B venue, because Computer Networks is a well-cited Q1 venue IDS reviewers read — 'not CORE A\*/A' will not read as 'obscure'."* That was asserted. Here is the evidence `[S10]`, from OpenAlex source records:

| Metric | **Computer Networks** (`S63392143`) | **Springer LNCS** (PAKDD's container) |
|---|---|---|
| 2-year mean citedness | **4.413** | **1.409** |
| h-index | 198 | 714 *(inflated — series-wide across all of computer science)* |
| Works indexed | 9,799 | 605,838 |
| Total citations | 309,646 | — |
| Publisher | Elsevier BV | Springer |
| Open access | Hybrid (FedPGO itself: **closed**) | Mixed |

**Reading (mine, not the tool's).** Computer Networks' 2-year mean citedness is **≈3.1× that of LNCS**, the series in which PAKDD proceedings appear. The venue-qualifier defence is therefore **weaker than a bare rank comparison suggests**: FedPGO is not published somewhere obscure relative to PAKDD — by per-paper citation impact it is published somewhere *more* visible. **Do not rest any part of the novelty claim on "FedPGO isn't at an A\*/A venue."** It is a true statement that will not persuade a domain-literate reviewer, and §8's wording avoids relying on it.

⚠️ **Honest methodological limitation.** OpenAlex has **no usable standalone source record for PAKDD** — the conference entity (`S4306420658`) indexes only 8 works with h-index 3, because PAKDD proceedings are attributed to LNCS. LNCS is therefore the best available proxy, and it is an imperfect one: it aggregates every LNCS-published conference across computer science, so it neither isolates PAKDD nor controls for field. **This is a proxy comparison, not a like-for-like venue metric, and should be presented that way** if it is used in a rebuttal. The CORE rank (B, ICORE2026, verified directly at portal.core.edu.au `[S10]`) remains the authoritative venue-standing statement.

> **Any framing that claims "PAKDD = CORE A" is factually wrong and must not appear in a manuscript.** PAKDD was A under CORE2018 and is **B** under CORE2023/ICORE2026. Position it honestly as a realistic near-term target and a stepping-stone toward an AAAI/IJCAI-tier extension, per [`publication_roadmap.md`](../current/publication_roadmap.md).

---

## 7. The two structural privacy gaps and FedGRAA-v2

*Condensed to what Related Work needs. Full derivation, byte-level accounting, DP composition math, threat model, and test spec: [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) (four dated passes). Read that before implementing anything.*

### 7.1 The two flaws

1. **Oracle knowledge of rare classes/clients.** The server hard-codes "the last `n` underrepresented classes," and the experimental partition assumes one attack type per client — i.e. the server (or experiment designer) is assumed to already know each client's label distribution. Real FL servers never see client labels. This is a privacy violation **and** an unrealistic experimental setup.
2. **A server-held, fully-labelled validation set `D_V`.** Algorithm 2 evaluates `L_h, L_m, L_u` on a centrally-pooled `D_V` — exactly the thing FL exists to avoid. Structurally identical to the "root dataset" in FLTrust (NDSS 2021), which **FedTruth** explicitly calls impractical, which **CoAst** describes as needing data *"difficult to obtain in practical FL scenarios,"* which **CELM**'s abstract restates independently (*"validation-based scoring may be unavailable, biased, or costly to maintain"*), and which even 2025 work (**FedGreed**) still assumes without apology. **A known, common, fixable shortcut — not a niche flaw specific to FedGRAA.** Frame it that way.

### 7.2 The recommended fix — FedGRAA-v2

Family 1 (gradient-composition inference) + Family 3 (SecAgg/DP summary statistics) from §4.9, plus Approach B (federated backward pass) from §4.10:

- Replace the hard-coded rare-class list with an **online-inferred, SecAgg-summed per-round class-frequency estimate** `p̂^t`, from label-count histograms clients already hold locally.
- Replace `D_V` with a **federated backward pass**: the server broadcasts the candidate aggregate `θ'_g` (already-public math on already-shared parameters); each client returns only `∇_P L_k'(θ'_g)` computed on its own held-out split; the server sums these (optionally through SecAgg) to recover `∇_P L' = Σ_k ∇_P L_k'(θ'_g)` **exactly** — mathematically identical to what a server-held `D_V` would compute.
- **Zero change to Eq. 7–10.** A methodology patch, not a rewrite; the γ-tunable gradient-refined contribution is preserved exactly.

### 7.3 The honest cost, computed exactly from the live codebase

- **Communication:** ≈**6×** total training-run traffic (≈5.606 GB CIC-IDS-2017, ≈**2.908 GB** UNSW-NB15 — corrected 2026-08-03; the ≈2.98 GB figure this line previously carried used a stale `n_features=43` assumption for UNSW-NB15, live `fedgraa/config.py` uses 39, a 2.4% overstatement caught and fixed in `FEDGRAA_V2_MASTER_PLAN.md` F5 but never propagated back to this summary — see that document's exact-cost table for the authoritative figures, from exact layer sizes in `fedgraa/config.py`/`model.py`, `E'=5`). SecAgg histogram + masking overhead is **six or more orders of magnitude smaller** — essentially free.
- **Privacy accounting:** report the **composed** `(ε,δ)` over all `T=200` rounds, not a per-round `ε`. Advanced composition gives `ε_total ≈ 8.89` at `ε_round=0.1, δ=10⁻⁵`; RDP is tighter and recommended.
- **A genuinely non-obvious, dataset-dependent finding:** DP noise on the rare-class histogram is **unusable** for CIC-IDS-2017's rarest classes (Heartbleed, ≈1 held-out sample) even at the loosest common budget, but only **mildly** degraded for UNSW-NB15's rarest (Worms, ≈17 samples, ≈5% relative error at `ε=10`). **Report both datasets side by side; do not generalize from one.**
- **Convergence proof impact:** one new standard assumption — bounded estimation error `ζ` between the true and federated composite-loss gradient. A noise-ball weakening, not a different theorem.
- **Threat model:** honest-but-curious server + SecAgg's standard collusion threshold. A **fully malicious server and Byzantine clients are explicitly out of scope** (name this; do not silently assume it away). The unchanged `θ_k^t` channel remains exposed to gradient inversion (DLG) exactly as in the original paper.
- **Implementation scope:** **5 small, well-isolated additions**; **zero changes** to `model.py`'s forward/backward or `aggregation.py`'s simplex projection. A previously-unremarked fact: `DatasetSpec.rare_client_ids()` **already has** a data-driven, non-oracle fallback path that current reproduction runs simply never exercise.

### 7.4 What Related Work must concede

**The word "first" does not survive.** CELM is explicitly data-free and minority-targeted. The narrowed statement:

> "Prior gradient-refined adaptive aggregation for FL-IDS assumed server-side knowledge of client class distributions and a server-held validation set. We remove both via secure-aggregation/DP class-frequency estimation and a federated backward pass that reproduces the **exact** composite-loss gradient without centralizing any labelled data, at a measured, honestly-reported cost. Contemporary data-free contribution estimators (CELM, SpectralFed, 2026) achieve a related goal **approximately** at near-zero communication cost; we therefore report the trade-off directly — what exactness buys in minority-class recall, and what it costs in traffic — making FedGRAA the first fully FL-compliant gradient-refined adaptive-aggregation method **for intrusion detection**."

*That is not a retreat: it converts a redesign into a comparative experiment with a publishable result either way.*

---

## 8. Synthesized positioning and differentiation narrative

### 8.1 The final Related Work paragraph

*Wording finalized `[S10]`. The only change from `[S9]`'s version is the closing clause, tightened in light of the J. Cloud Computing survey's explicit naming of rare-attack recall as a recognized axis (§4.12) — "an axis nobody reports" would have been an overclaim.*

> "Adaptive aggregation weighting has converged on three signals: alignment with the global direction (FedAWA), client loss or accuracy (FedNolowe, DWAFL, PACT), and statistical outlier filtering (MESAS, CrowdGuard, FreqFed). In imbalanced intrusion detection these systematically suppress the divergent, high-loss, minority-class clients that hold the rarest attacks. A counter-current has emerged: FedAWARE maximizes gradient diversity rather than alignment; Jiang et al. (2024) and AdaAggRL (AAAI 2025) learn weights that protect rare contributions through reinforcement learning; CELM (2026) up-weights clients with discriminative evidence for underrepresented classes via data-free logit probing; and precision-weighted aggregation (Reyes et al., 2021) derives weights from gradient variance. In intrusion detection specifically, FedPGO (2026) reweights clients by gradient orthogonality on CIC-IDS2017, and NIDS-FGPA (2024) weights by gradient similarity for communication efficiency. FedGRAA sits inside this counter-current, at a point none of them occupies: a **single-global-model** aggregation rule with an explicit **γ-tunable trade-off** between rare-class emphasis and conventional size-proportional weighting. Recent surveys identify rare-attack recall as a key outcome shaped by aggregation choice, yet the adjacent methods above report aggregate accuracy, robustness, or personalization gain; FedGRAA is evaluated directly on **per-class recall for rare attack types**. Its claim is not that the space is empty, but that this point in it is unreported and that its evaluation axis is one no adjacent method paper supplies."

**Why this is the right claim.** It is narrow, true, concedes the overlaps before a reviewer finds them, and rests the contribution on something FedGRAA actually has — the γ trade-off and rare-attack per-class recall — rather than on an absence that no longer exists. It also **does not lean on the A\*/A venue qualifier**, which §6.1 shows is a losing argument.

### 8.2 Mandatory citations

Omitting **FedPGO** would be the single most likely fatal miss a domain-literate reviewer could flag. Related Work must cite, at minimum:

**FedPGO** (highest priority — same domain, same dataset, same mechanism family) · **CELM** · **FedAWARE** · **AdaAggRL** · **Jiang et al.** · **NIDS-FGPA** · **Parsa et al.** (prior art for the method class) · **SBFedNIDS**, **FedMADE**, **FedAPA**, **FedAWA**, **DA-PFL** (competitor set) · **PROTEAN** (same-domain alternative design point) · **Flood et al.** (dataset validity) · **SecFedNIDS** (adversarial baseline) · **XAI-SOH-FL** (γ precedent).

### 8.3 Three things that must not be written

1. ❌ *"The intersection is unoccupied."* — FedPGO occupies it on CIC-IDS2017.
2. ❌ *"Existing adaptive aggregators all suppress minorities."* — a straw man; the counter-current is now five papers deep.
3. ❌ *"We are the first to add a simplex projection / stabilize the weights."* — CELM treats it as a basic design requirement. **Cite it as standard practice.**

---

## 9. Gap-closing checklist, with items (a)–(i) marked

### 9.0 Carried-forward open items — final disposition

| # | Item | Disposition `[S10]` |
|---|---|---|
| **(a)** | **FedPGO full text** | ⛔ **STILL BLOCKED — now across six independent channels.** Elsevier direct `[S8]`; FastTrack/OpenAlex `[S9]`; SSRN preprint mirror (403) `[S9]`; and this pass: **OpenAlex OA-location enumeration** (`locations_count: 1`, `best_oa_location: null`, `any_repository_has_fulltext: false`), **Unpaywall**, **arXiv** (author search `au:Bikila` → 0 entries; phrase search `"gradient orthogonalization"` → 0), **OpenAIRE**, **CORE**, and the **University of Pardubice institutional repository**. **No preprint twin exists.** This is definitively an **access-logistics** problem, not a search-effort one. ➡️ **Actionable escalation:** institutional library ILL, or direct author contact — Dawit Dejene Bikila, ORCID **0000-0002-3159-1531**, Institute of System Engineering and Informatics, Faculty of Economics and Administration, University of Pardubice, Studentská 95, Pardubice 53009, Czech Republic (corresponding author, confirmed `[S10]`). **Do not spend further search effort.** |
| **(b)** | **CELM newer revision** | ✅ **RESOLVED — no v2.** arXiv:2605.18892 is still **v1**, published and last-updated 2026-05-17T05:49:48Z, 22 pp. The `[S9]` full-text read remains current. |
| **(c)** | **Fed-Meta FD** | ✅ **RESOLVED ❓→✅.** Chen, Tang & Li, IEEE TNSE 2023, DOI 10.1109/tnse.2023.3266942, 51 citations. The master doc's "(verify TNSE)" was correct. Acronym not author-given. **Also resolves a §4.5-vs-§11 self-contradiction.** *(This was the corpus's last ❓.)* |
| **(d)** | **"Precision-Weighted FedAvg"** | ✅ **RESOLVED ❓→✅.** Reyes, Di Jorio, Low-Kam & Kersten-Oertel, arXiv:2107.09627 (2021) → *Computational Intelligence* 2025, DOI 10.1111/coin.70150, 11 citations. **Promoted from "do not cite" to a cited reference** (§4.7). Acronym is the search tool's invention. |
| **(e)** | **Citation counts** | ✅ **MOSTLY RESOLVED.** Semantic Scholar 429'd again mid-pass, but **OpenAlex supplied 11 counts** the previous three passes could not get: FedPGO **2**, DS-FedIDS **4**, CACFL **0**, Fed-LAA/Khan **0**, HADA-FL **17**, TACMA **0**, SecFedNIDS **100**, FedAAW **7**, Jiang et al. **3**, AdaAggRL **4**, NIDS-FGPA **20**, plus all six surveys. ⚠️ **Still uncounted: FedVG and XAI-SOH-FL** — both arXiv-only with no DOI, so OpenAlex has no record and S2 was rate-limited. Low decision-relevance. |
| **(f)** | **J. Cloud Computing survey** | ⚠️ **PARTIALLY RESOLVED.** Full metadata + verbatim abstract obtained via DOAJ (Ma, Yang, Wang, Chen, 2026-04-30, 0 citations). **Taxonomy still unread** — Springer's IdP wall blocks even this gold-OA article's PDF. **The abstract alone changed §8's wording** (§4.12): it explicitly names *"rare-attack recall"* as an aggregation-shaped outcome, so "an axis nobody reports" was downgraded to "no adjacent *method paper* reports." Net: the check was worth running and produced a real correction, but the taxonomy question stays open. |
| **(g)** | **Jiang et al. full read** | ⚠️ **PARTIALLY RESOLVED.** Verbatim abstract obtained via OpenAlex; mechanism confirmed (filtering layer + voting layer + DRL dynamic weight adjustment; explicit goal of preserving *"diversity of shared information"* that outlier-exclusion destroys). **Full text still blocked** — ScienceDirect 403 despite the paper being **CC-BY hybrid OA**, which is a platform restriction, not a licensing one. Sufficient for the §4.7/§4.11b characterization; insufficient to characterize its own limitations. |
| **(h)** | **arXiv:2602.05594 six-flaw list** | ✅ **RESOLVED — with a correction.** Full text read. The enumeration is real and matches verbatim, **but is cited from Flood, Engelen, Aspinall & Desmet, IEEE EuroS&P 2024** (DOI 10.1109/eurosp60621.2024.00042, 17 citations, green OA). **Cite Flood et al.** Also: the paper is at **v3**, and its *own* abstract-level contribution is a different five-item list (temporal leakage, data splitting, dataset design flaws, limited dataset diversity, weak cross-dataset generalization). §5 updated. |
| **(i)** | **Six FL-IDS survey taxonomies** | ⚠️ **PARTIALLY RESOLVED.** All six now have complete author lists, exact dates and citation counts `[S10]` — three previously had **no authors on file at all**. **No taxonomy read in primary text.** The two ACM CSUR surveys are flagged OA by OpenAlex, but `dl.acm.org` returns 403 to both WebFetch and `curl`, and the **Figshare mirror** of the 2024 survey also 403s. ➡️ **The §4.12 comparative-framing caveat therefore stands unchanged and remains load-bearing.** This is the one substantive evidence gap that survives all ten passes. |

### 9.1 The A\*/A checklist

**0. ⚠️ BLOCKING (partially discharged).** Read **FedPGO** and **CELM** in full before drafting Related Work. **CELM: DONE** `[S9]` — full 22-page read, mechanism corrected, differentiation now writable at primary-source confidence. **FedPGO: STILL BLOCKED** (item a) — known only through abstract-level secondary summaries corroborated by three independent index records. That is **enough for existence, venue, dataset, topic and mechanism-in-outline; not enough to characterize FedPGO's own limitations.** ➡️ **Revised guidance:** do **not** gate drafting on it. Draft Related Work now, cite FedPGO at abstract-level confidence, avoid any sentence asserting what FedPGO *fails* to do, and pursue the full text in parallel by ILL or author email.

1. **Formal convergence theorem** under bounded heterogeneity, covering the actual non-convex tanh-MLP (PL-condition, not global convexity). **Derive the weights from the bound (FedAAW)** rather than proving a theorem around heuristic weights; adopt Parsa et al.'s alternating-minimization framing; use a consensus-dynamics heterogeneity measure (FedAWARE) so the theorem is not vacuous in the one-attack-per-client regime; state **bounded aggregation weights (= the simplex projection) explicitly as a proof condition**, tying it to the observed divergence — same defect, two directions.
2. **Head-to-head baselines**, in priority order: FedAvg, FedProx, SCAFFOLD, FedNova, FedOpt **+** **FedPGO (top priority)**, FedAPA, FedAWA, DA-PFL, FedDRL, FedNolowe, CELM, FedAWARE, AdaAggRL **+** FedMADE, SBFedNIDS, DWAFL, FD-IDS, DS-FedIDS, CACFL, PerFedHypID, FedKSam\*, AMAFed, Chunduru et al. **+** NIDS-FGPA **+** Jiang et al. *(G-PFL-ID as adjacent only — it is unsupervised/one-class.)*
3. **Adversarial robustness** — the **premise conflict must be argued, not merely disclosed** (§4.11b). Add the mechanism-specific attack (**an adversary fabricating rare-class gradient signatures to farm up-weighting**); baseline against **SecFedNIDS** (same datasets, public code) and **WeiDetect**; use the **ARES 2023** parameterization and the **SoK harness**; state whether **γ trades detection performance against poisoning exposure**.
4. **Modern + FL-native datasets** — the strongest move is the **NF-\*-v2 / NF-UQ-NIDS-v2 / NetFlow-V3** family: a native cross-silo benchmark with **real** non-IID structure. Also add CICIoT2023/FLNET2023, correct any phantom "CIC-IDS2023," and cite **Flood et al.** whenever rare-class numbers are reported.
5. **Scale + extreme non-IID** — 50–200+ clients; 10–50 samples/class; a Dirichlet-`α` partition alongside the current extreme. *Largely subsumed by #4.*
6. **Ablation** of the gradient-refinement components and `γ` — scoped and partly infrastructure-complete in Stage B.
7. **Overhead reporting** — report the ≈6× **against CELM/SpectralFed's near-zero-cost approximation**, making *"is exact worth 6×?"* an experiment rather than a caveat.
8. **Statistical rigor** — ≥10 seeds, paired Wilcoxon, Holm–Bonferroni across the baseline grid — infrastructure-complete via `fedgraa/stats.py`. **Lead with this for rare classes**; it is the direct answer to the labelling-error/low-support critique (§5).
9. **Privacy fix (§7)** — implement FedGRAA-v2 and report the v1-vs-v2 accuracy/overhead trade-off. **Drop "first fully FL-compliant"** except with the IDS qualifier.
10. **Optional hybrid** — combine with prototype guidance (FedGMKD/FedFSA/PROTEAN) for data-poor clients.
11. ⚠️ **Pre-empt the manual-γ question** in Limitations/Future Work, citing **XAI-SOH-FL** as the direct same-domain precedent for automating it (§4.12). One sentence; high value.
12. ⚠️ **Fix the bibliography before resubmission** — remove **FedSND** (the draft's own ref [12], a confirmed hallucination) → replace with FedAWA or FedNova; remove any "CIC-IDS2023."

---

## 10. Stage C decision framing

| Angle | What the literature supports | Strongest baselines it must beat | Risk |
|---|---|---|---|
| **A. Few-shot/prototype-guided hybrid** | Crowded but active (§4.5–4.6), with strong 2025 competitors already at AAAI/IJCAI/TIFS | FeCoGraph (TIFS 2025), FedFSA (AAAI 2025), FedGMKD (NeurIPS 2024), PROTEAN (ESORICS 2025), DFML | More ambitious; competitive with 2025-fresh work, so the novelty window is narrowing fast |
| **B. Privacy-compliant redesign (FedGRAA-v2, §7)** | A real flaw in the *existing* paper that the field has repeatedly flagged elsewhere (FedTruth, CoAst, CELM) — and it is close to fully scoped: 7 solution families, exact byte/DP math, full test spec, function-level codebase mapping | FedTruth, CoAst, RAGA (null baseline); **CELM and SpectralFed as the data-free approximation baselines**; PROTEAN as the same-domain alternative | ⚠️ The "less crowded" rationale is **dead**. B is still right, for a **better** reason: the comparison shifted from *"does this work?"* to ***"is exact worth 6×?"*** — FedGRAA-v2 recovers the composite-loss gradient **exactly** at ≈6× traffic; CELM/SpectralFed approximate it at near-zero cost. **A comparative experiment with a publishable result either way** |
| **C. Both, sequenced** | Nothing prevents shipping B first and layering A later | — | Time cost only; B's test suite and byte-accounting are already spec'd |
| **D. NF-v2 cross-silo benchmark migration** | Not a novelty angle but an **evidence** angle: real rather than simulated non-IID structure | The same baseline grid, on better data | ⚠️ **Highest ratio of reviewer-objections-closed to effort available.** Closes §9 gaps #4 and #5 *and* the §7.1 oracle-partition objection, with **data engineering only, no algorithmic risk**. It **composes with A or B rather than competing** — bring it as an addition, not a fourth alternative |

**Recommendation, unchanged and re-affirmed `[S10]`:** **Option B + Option D.** B is substantially closer to implementation-ready than A (§7.3 gives exact byte counts, a formal threat model, a full test specification, and a function-level mapping to the live codebase — none of which exists for A). D should be done **regardless of which novelty angle is selected**, and is the cheapest large win available. Bring all four to the mentor with this trade-off explicit, per the roadmap's own call for that conversation.

> **✅ Author decision, 2026-07-28 — this section's own Option C, chosen over the B + D recommendation above.** The mentor conversation this section calls for did not happen; the author decided directly, choosing **Option C — "both, sequenced": B + D now, A afterward as a follow-on, not simultaneously.** This is not a correction to the analysis above — the trade-offs this table and the B+D recommendation describe are still accurate as of when they were written, and A's higher risk/lower readiness relative to B stands, which is exactly why it's sequenced after B+D rather than built alongside it. It is the author choosing to accept A's eventual cost later rather than trade it away permanently. Tracking: `../current/STAGE_DIRECTION_ASSESSMENT.md`'s resolution note (top of that document) and `REMAINING_COMPUTE_WORK.md` §3a, which scopes Option A as deferred, not-yet-started work for that later phase — no design doc, code, or tests for it exist yet, and none are scheduled to start until B+D's active work (§2–§7 of that document) is further along.

**One thing the mentor conversation should now include that it could not before:** §6.1's finding that the venue-qualifier defence is quantitatively weak. If the Stage C goal is an A\*/A extension rather than a PAKDD submission, FedPGO's existence in a Q1 venue with 3.1× LNCS's per-paper citation impact is a material input to that decision.

---

## 11. Full alphabetized bibliography

*✅ confirmed · ⚠️ confirmed with a correction · ❌ hallucination-equivalent, do not cite. Entries carried from the nine source documents inherit their multi-pass verification unless marked `[S10]`, which denotes a fresh tool call this pass.*

- Abadi, Chu, Goodfellow, McMahan, Mironov, Talwar, Zhang. "Deep Learning with Differential Privacy." CCS 2016. arXiv:1607.00133. ✅
- AdaAggRL — "Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning." arXiv:2406.14217; **AAAI 2025**, DOI 10.1609/aaai.v39i24.34733. **4 citations** ✅ `[S10]`
- AFL — Mohri, Sivek, Suresh. ICML 2019. ✅
- AMAFed — Izadi & Ahmadi. "Adaptive Meta-Aggregation Federated Learning for Intrusion Detection in Heterogeneous Internet of Things." arXiv:2602.12541, 2026. ✅
- Armadillo — Ma, Guo, Karthikeyan, Polychroniadou. CCS 2025. arXiv:2511.10863. ✅⚠️
- Astraea — Duan et al. IEEE ICCD 2019. arXiv:1907.01132. ✅
- AugFL — Yue, Qin, Deng, Ren, Zhang, Zhang. IEEE/ACM ToN 2025. arXiv:2503.02154. ✅
- AutoAdapt — Krauß, König, Dmitrienko, Kanzow. NDSS 2024. ✅⚠️ *(title: "Adaption")*
- Benchmarking the Benchmark — Analysis of Synthetic NIDS Datasets. arXiv:2104.09029. ✅
- Bharadwaj & Cormode. "Sample and Threshold Differential Privacy." AISTATS 2022. arXiv:2112.05693. ✅
- Bonawitz et al. "Practical Secure Aggregation for Federated Learning." NeurIPS-W 2016. arXiv:1611.04482. ✅
- Bunko, Johnstone, Yang, Scott. "A survey of privacy-preserving federated learning for intrusion detection systems." *Artificial Intelligence Review*, 2026-02-28. DOI 10.1007/s10462-026-11519-4. **3 citations** ✅ `[S10]`
- Byzantine Gradient Descent — Chen, Su, Xu. arXiv:1705.05491, 2017. ✅
- Byzantine-Robust Clustered FL — Tao, Yang, Kulkarni. arXiv:2306.00638, 2023. ✅
- Byzantine-Robust FL with Learnable Aggregation Weights — Parsa, Daghestani, Teixeira, Johansson. arXiv:2511.03529, 2025 (rev. 2026). ✅ *(prior art for FedGRAA's method class)*
- ByITFL — Xia, Hofmeister, Egger, Bitar. arXiv:2405.08698, 2024. ✅
- CACFL — "Multi-modal class-aware curriculum federated learning fusion for advanced intrusion detection…" *Neural Computing and Applications*, 2026-03-24. DOI 10.1007/s00521-026-11845-2. **0 citations** ✅ `[S10]`
- CELM — Ukaye, Tastan, Abdu-Aguye, Nandakumar. "Data-Free Client Contribution Estimation via Logit Maximization for Federated Learning." **arXiv:2605.18892, 2026-05-17, v1** (no newer revision as of 2026-07-27 `[S10]`), 22 pp. ✅ ⚠️ *(mechanism is activation-maximization probing with post-warm-up frozen weights — **not** gradient-based)*
- Cerberus — Naseri, Han, Mariconti, Shen, Stringhini, De Cristofaro. CCS 2022. arXiv:2209.03050. ✅
- CGSV — NeurIPS 2021. ✅ *(CELM's own nearest comparator)*
- Chen, Tang, Chen. "Federated Meta-Learning Framework for Few-shot Fault Diagnosis in Industrial IoT." GLOBECOM 2022. DOI 10.1109/globecom48099.2022.10001294. **8 citations** ✅ `[S10]`
- Chen, Tang, Li. "Industrial Edge Intelligence: Federated-Meta Learning Framework for Few-Shot Fault Diagnosis." **IEEE Transactions on Network Science and Engineering, 2023.** DOI 10.1109/tnse.2023.3266942. **51 citations** ✅ `[S10]` *(this is the paper the corpus tracked as "Fed-Meta FD"; the acronym is not the authors' own — cite by title)*
- Chunduru Sri Abhijit, Annie Jerusha, Syed Ibrahim, Varadharajan. "Federated transfer learning for rare attack class detection in network intrusion detection systems." *Scientific Reports*, 2025-09-30, `s41598-025-02068-x`, PMC12484838. ✅ *(DOI **derived** from article ID — verify once before use. Evaluated on CIC-IDS2018 + UNSW-NB15 + Edge-IIoT, **not** CIC-IDS-2017)*
- CIC-IDS-2017 — Sharafaldin, Lashkari, Ghorbani. ICISSP 2018. ✅
- "CIC-IDS2023" — no such dataset. ❌ *(re-point to CICIoT2023 / CSE-CIC-IDS2018)*
- CICIoT2023 — Neto, Dadkhah, Ferreira, Zohourian, Lu, Ghorbani. *Sensors* 23(13):5941, 2023. ✅
- Clust-PSI-PFL — Jimenez-Gutierrez et al. IPDPS 2026. arXiv:2512.20363. ✅
- CoAst — Wu, Zhang, Li, Xu, Zhong. arXiv:2409.02495, 2024. ✅
- CrowdGuard — Rieger, Krauß, Miettinen, Dmitrienko, Sadeghi. NDSS 2024. ✅⚠️
- CSE-CIC-IDS2018 — CSE/CIC-UNB. ICISSP-2018 methodology. ✅
- ~~CT-BRFL~~ — claimed Wang et al. 2024. ❌ *4 independent negative DBLP results; do not cite. Use FedTruth or CoAst*
- DA-PFL — Yang, Feng, Tong, Wang, Guo, Wang, Ding, Fang, Liao. AAAI 2024 (arXiv:2403.09284) + IEEE TNNLS 2025. ✅⚠️
- DeProFL — Li, Gao et al. IEEE IoTJ 2023/24. ✅
- DFML — Chen, Zeng, Zhang, Lu, Hou, Zhou, Helal. IEEE/ACM TCBB 2023. ✅
- DiffuPac — Bin Jasni, Manada, Watabe. NeurIPS 2024. ✅⚠️ *("Generation" added)*
- DS-FedIDS — "Enhancing Federated Intrusion Detection with Class-Specific Dynamic Sampling." *Applied Sciences* 15(9):5067, 2025-05-02. DOI 10.3390/app15095067. **4 citations** ✅ `[S10]`
- DWAFL — Umair et al. IEEE ICRAIE 2023. ✅
- Dwork, Roth. "The Algorithmic Foundations of Differential Privacy." Monograph, 2014. ✅
- Dwork, Rothblum, Vadhan. "Boosting and Differential Privacy." FOCS 2010. ✅
- ECGR — "Local Gradient Regulation Stabilizes Federated Learning under Client Heterogeneity." arXiv:2601.03584, 2026. ✅ *(term-adjacent scooping-risk watch item)*
- EdgeDetect — Mohammad. arXiv:2604.14663, 2026. ✅
- Edge-IIoTset — Ferrag et al. ✅ `[S10]` *(NIDS-FGPA's primary dataset)*
- EIFFeL — Roy Chowdhury, Guo, Jha, van der Maaten. CCS 2022. DOI 10.1145/3548606.3560611. ✅⚠️
- El Mahdaouy, Ait Yahia, Oualil, Berrada. "Deep Learning for Contextualized NetFlow-Based Network Intrusion Detection: Methods, Data, Evaluation and Deployment." **arXiv:2602.05594, v1 5 Feb 2026, v3 3 Mar 2026.** ✅ `[S10]` ⚠️ *(cite for the evaluation-practice critique — temporal leakage, data splitting, weak cross-dataset generalization. **Do NOT cite for the six design flaws** — those are Flood et al. 2024, which this paper cites)*
- EMD-CFL — Zhang, Williams, Toni. arXiv:2506.07769, 2025. ✅
- Engelen et al. "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study." 2021. ✅ `[S10]`
- Entente — Xu, Li, Zheng, Li. NDSS 2025. ✅
- F-ACVAE — arXiv:2607.04698, 2026. ✅
- FADngs — Dong, Chen, Wu, Tang, Zhuang. "FADngs: Federated Learning for Anomaly Detection." **IEEE TNNLS 36(2):2578–2592, 2025.** DOI 10.1109/TNNLS.2024.3350660. ✅
- FBLG — Xu, Y. Li et al. IJCAI 2024, 5289–5297. ✅
- FD-IDS — Peng, Wu, Xiao. *Sensors* 25(14):4309, 2025. **20 citations** ✅ `[S10]`
- FeCoGraph — Mao, Su et al. IEEE TIFS 2025, 20:2266–2280. ✅
- ~~FeCWGAN-GP~~ — claimed IEEE TNNLS 2026. ❌ *Nearest real: FedCG (IJCAI 2022), IFL-GAN (TNNLS 2022)*
- FedAAW — "Tackling data-heterogeneity variations in federated learning via adaptive aggregate weights." **Knowledge-Based Systems, 2024-09-06.** DOI 10.1016/j.knosys.2024.112484. **7 citations** ✅ `[S10]` *(weights derived by minimizing the convergence upper bound — §4.11a)*
- FedAdp — Wu & Wang, 2022. ✅
- FedAgg — arXiv:2303.15799, 2023. ✅
- ~~FedAgile~~ — placeholder acronym, not author-given. The actual paper is **Yuan & Wang, "Aggregated Gradients-based Adaptive Learning Rate Design in Federated Learning," ACM CIKM 2025, DOI 10.1145/3746252.3761234.** ✅
- FedAPA — Sun, Sun, Pan, Fu, Guo. IJCAI 2025. arXiv:2502.07456. ✅
- FedAtt — Ji et al., 2019. ✅
- FedAvg — McMahan, Moore, Ramage, Hampson, Agüera y Arcas. AISTATS 2017. arXiv:1602.05629. ✅
- FedAWA — Shi, Zhao, Zhang, Zhou, Guo, Chang. CVPR 2025. arXiv:2503.15842. ✅ ⚠️ *(**do not conflate with FedAWARE** — opposite mechanisms)*
- FedAWARE — Zeng, Xu, Liu, Pan, Wang, Tang. "On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond." **arXiv:2310.02702** (v4, Dec 2024). **6 citations** ✅ `[S10]` *(FedAWARE is the method name; cite by title)*
- FedAWR — "FedAWR: Aggregation Optimization in Federated Learning with Adaptive Weights and Learning Rates." **Future Internet 18(2):106, 2026-02-18.** DOI 10.3390/fi18020106. ✅
- Fed-CBS — Zhang, Li, Tang, Sun, Chen, Zhang, Chen, Chen, Li. ICML 2023. arXiv:2209.15245. ✅
- ~~FedCCW~~ — claimed "Zhang et al. 2025." ❌ *Use **FedCBA** (DOI 10.1145/3654446.3654454) if a class-balanced-aggregation citation is needed*
- FedCGNM/FedHOO — Park, Klabjan, Braun, Li, Ananthanarayanan. ICML 2026. arXiv:2607.01474. ✅
- FedCGS — Guan, Zhou, Gu. AAAI 2025. arXiv:2503.06962. ✅⚠️
- FedCK — Chen, Li, He. *J. Supercomputing*, 2025. ✅
- FedCoef — Xie, Pei, Shi. "FedCoef: An Adaptive Aggregation Algorithm Based on Coefficient of Variation for Federated Learning." ASCC 2024, 1302–1306. DBLP `conf/ascc/XiePS24`. ✅
- FedDiverse — Németh, Fanì, Ng, Caputo, Lozano, Oliver, Quadrianto. FLTA 2025. arXiv:2504.11216. ✅
- FedDLAD — Ding, Yang et al. IJCAI 2025 (#559). ✅⚠️
- FedDrone-Shield — "A federated learning-benchmarking framework for privacy-preserving UAV intrusion detection using adaptive aggregation algorithms." *Scientific Reports* 2026, `s41598-026-50865-9`. ✅ *(one paper, not two)*
- FedDropoutAvg — Gunesli, Bilal, Raza, Rajpoot. arXiv:2111.13230, 2021. ✅
- FedDRL (adaptive aggregation) — Nguyen et al. ICPP 2022. arXiv:2208.02442. ✅ *(disambiguate from arXiv:2307.13716)*
- FedDUAL — Sahoo et al. **arXiv:2412.04416**, 2024-12-05; OpenReview `8M3XfmNhTZ`. ✅⚠️ *(**TMLR acceptance not confirmed** — cite by arXiv ID)*
- FedFMD — ACM CIKM 2025. DOI 10.1145/3746252.3761316. ✅
- FedFSA — Qi, Meng, Li, Hu, Meng. AAAI 2025. ✅
- FedFSL-CFRD — Wang et al. AAAI 2025. ✅
- FedGC — "FedGC: An Accurate and Efficient Federated Learning under Gradient Constraint for Heterogeneous Data." OpenReview `eZN8nUXAVO7`. ✅⚠️ *(**venue unconfirmed**)*
- FedGMKD — Zhang. NeurIPS 2024 (main track). ✅⚠️
- FedGreed — Kritharakis, Makris, Jakovetic, Tserpes. arXiv:2508.18060, 2025. ✅
- ~~FedHDE~~ — claimed "Zhang et al. 2026." ❌
- FedImT — Yan, Li, Ota, Dong. arXiv:2310.14784, 2023. ✅
- FedKSam\* — "FedKSam\*: A Two-Phase Federated Learning Intrusion Detection Algorithm." IEEE Xplore document **10661778**. ✅ *(asterisk is part of the name)*
- Fed-Focal Loss — Sarkar, Narang, Rai. FL-IJCAI 2020. arXiv:2011.06283. ✅
- Fed-LAA *(this series' shorthand)* — Khan, Rizwan, Ahmad, Khan, Li, Kim. "Accelerating model convergence in federated learning with layer-wise adaptive weight aggregation." ***Applied Soft Computing* 201:115676, 2026-06-11.** DOI 10.1016/J.ASOC.2026.115676. **0 citations** ✅⚠️ `[S10]` *(**cite by title/DOI — the acronym is not confirmed as the authors' own**)*
- FedLMD — Lu, Li, Bao, Wang, Qian, Ge. ACM MM 2023. arXiv:2409.13136. ✅
- FedMADE — Sun, Sharma, Nwodo, Stavrou, Wang. ISC 2024. arXiv:2408.07152. ✅
- FedMcon — Shen, Li et al. FITEE 2025, DOI 10.1631/FITEE.2400530; ACM MM Asia W., DOI 10.1145/3700410.3702124. ✅
- FedMPS — Yang, Hu et al. IEEE TNNLS 2025. ✅
- FedNIDS — Nguyen, Hore, Shah, Le, Bastian. "FedNIDS: A Federated Learning Framework for Packet-Based Network Intrusion Detection System." **DTRAP 6(1):1–23, 2025.** DOI 10.1145/3696012. ✅
- FedNolowe — Le, Huynh, Tran, Dao, Bao. bioRxiv 2025 (DOI 10.1101/2025.03.30.646222) → ***PLOS ONE* 20(8):e0322766, 2025-08-14.** DOI 10.1371/journal.pone.0322766. ✅
- FedNova — Wang, Liu, Liang, Joshi, Poor. NeurIPS 2020. arXiv:2007.07481. ✅
- FedOpt/FedAdam/Yogi/Adagrad — Reddi, Charles, Zaheer, Garrett, Rush, Konečný, Kumar, McMahan. ICLR 2021. arXiv:2003.00295. ✅
- FedP3E — Darwish, Abdelsalam, Khorsandroo, Roy. **arXiv:2507.07258 (CoRR 2025) — preprint only.** ✅⚠️ *(**not IEEE TIFS**)*
- FedPGO — **Dawit Dejene Bikila, Jan Čapek** (Univ. of Pardubice). "FedPGO: A personalized federated learning framework with gradient orthogonalization for IoT intrusion detection." ***Computer Networks* 280:112210, published 2026-03-10.** DOI 10.1016/j.comnet.2026.112210. **2 citations.** ✅ `[S10]` ⚠️ ***The closest paper to FedGRAA in the corpus.*** *Full text access-blocked across six independent channels; metadata triple-confirmed (DBLP `journals/cn/BikilaC26`, OpenAlex `W7134986038`, Semantic Scholar `CorpusId:286471251`). Note the diacritic in **Čapek**. Recovered originally only via the descriptive-phrase DBLP retry.*
- FedPGO predecessor *(SSRN)* — Bikila & Čapek. "A Personalized Federated Learning Framework for IoT Intrusion Detection." SSRN Electronic Journal, 2025. DOI 10.2139/ssrn.5637328. ⚠️ *Likely earlier draft; also inaccessible (403)*
- FedPKD (Lyu, Tang) — IEEE ICDCS 2023. ✅
- FedPRS — ACM TIST. DOI 10.1145/3779055. ✅
- FedProx — Li, Sahu, Zaheer, Sanjabi, Talwalkar, Smith. MLSys 2020. arXiv:1812.06127. ✅
- FedReLa — Hu, Menéndez, Liu, Gong, Wang, Peng. arXiv:2606.26037, 2026. ✅
- FedRFQ — Yan et al. IEEE TC 2024, 73(4):1086–1098. ✅
- FedRS — Li & Zhan. "FedRS: Federated Learning with Restricted Softmax for Label Distribution Non-IID Data." **KDD 2021.** ✅
- FedSiamese — Yang, Mao et al. IEEE IoTJ 2025, 12(24):52886–52899. ✅
- ~~FedSND~~ — claimed "Zhang et al." ❌ **This is FedGRAA's own reference [12] and must be removed before resubmission.** *Replace with FedAWA or FedNova*
- FedSWA/FedMoSWA — J. Liu, Y. Liu, Shang, H. Liu, J. Liu, Feng. ICML 2025. arXiv:2507.20016. ✅
- FedTruth — Ebron, Zhang, Yang. ACISP 2025. arXiv:2311.10248. ✅
- FLICKER — Guleria, Harshan, Prasad, Bharath. CODS-COMAD 2024. arXiv:2410.21192. ✅
- Flood, Engelen, Aspinall, Desmet. **"Bad Design Smells in Benchmark NIDS Datasets." IEEE EuroS&P 2024.** DOI 10.1109/eurosp60621.2024.00042. **17 citations**, green OA (KU Leuven LIRIAS). ✅ `[S10]` ⚠️ ***The primary source for the six NIDS-dataset design flaws*** *— cite this, not El Mahdaouy et al.*
- Flower — Beutel et al. arXiv:2007.14390. ✅
- FLNET2023 — Kumar, Liu, Tayeen, Misra, Cao, Harikumar, Perez. IEEE MILCOM 2023, 345–350. ✅
- FLTrust — Cao, Fang, Liu, Gong. NDSS 2021. arXiv:2012.13995. ✅
- FSFL-IDS — "Distributed Intrusion Detection in Dynamic Networks of UAVs Using Few-Shot Federated Learning." Springer, DOI 10.1007/978-3-031-94448-2_7, 2025. ✅
- FreqFed — Fereidooni, Pegoraro, Rieger, Dmitrienko, Sadeghi. NDSS 2024. arXiv:2312.04432. ✅⚠️
- G-PFL-ID — "G-PFL-ID: Graph-Driven Personalized Federated Learning for Unsupervised Intrusion Detection in Non-IID IoT Systems." **MDPI *IoT* 7(1):13, 2026-01-29.** ✅ *(unsupervised/one-class — adjacent, not head-to-head)*
- Geiping, Bauermeister, Dröge, Moeller. "Inverting Gradients." NeurIPS 2020. ✅
- HADA-FL — Chandu, Thumula, Parag et al. "Federated Learning for Distributed IoT Security: A Privacy-Preserving Approach to Intrusion Detection." ***IEEE Access* 2025.** DOI 10.1109/access.2025.3592481. **17 citations** ✅ `[S10]`
- Hernández-Ramos, Karopoulos, Chatzoglou, Kouliaridis, Mármol Campos, González-Vidal, Kambourakis. "Intrusion Detection Based on Federated Learning: A Systematic Review." **ACM Computing Surveys, 2025-04-23.** DOI 10.1145/3731596. **49 citations** ✅ `[S10]` *(taxonomy unread — §4.12 caveat)*
- IFCA — Ghosh, Chung, Yin, Ramchandran. NeurIPS 2020. arXiv:2006.04088. ✅
- Jiang, Ma, Wang, Yu, Sun, Ni, Liu. "Preventing harm to the rare in combating the malicious: A filtering-and-voting framework with adaptive aggregation in federated learning." ***Neurocomputing* 604:128317, 2024-08-10.** DOI 10.1016/j.neucom.2024.128317. **3 citations (OpenAlex) / 6 (S2).** ✅ `[S10]` *(abstract obtained verbatim; **full text blocked** despite CC-BY hybrid OA. The closest philosophical precedent to FedGRAA's "protect, don't filter, the rare" framing)*
- Kairouz, McMahan et al. "Advances and Open Problems in Federated Learning." arXiv:1912.04977, 2021. ✅
- Karimi, Nutini, Schmidt. "Linear Convergence… under the PL Condition." ECML-PKDD 2016. ✅
- Karimireddy, Kale, Mohri, Reddi, Stich, Suresh. "SCAFFOLD." ICML 2020. arXiv:1910.06378. ✅
- Khraisat, Alazab, Singh, Jan, Gomez. "Survey on Federated Learning for Intrusion Detection System: Concept, Architectures, Aggregation Strategies, Challenges, and Future Directions." **ACM Computing Surveys, 2024-08-06.** DOI 10.1145/3687124. **88 citations** ✅ `[S10]` *(taxonomy unread — ACM DL and its Figshare mirror both 403)*
- Kitsune — Mirsky, Doitshman, Elovici, Shabtai. NDSS 2018. arXiv:1802.09089. ✅
- Lanvin et al. "Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes." 2022. ✅ `[S10]`
- LDP-Fed — Truex, Liu, Chow, Gursoy, Wei. arXiv:2006.03637, 2020. ✅
- Li, Sahu, Talwalkar, Smith. "On the Convergence of FedAvg on Non-IID Data." ICLR 2020. ✅
- ~~Light-FFSL~~ — claimed "IEEE ~2025/26." ❌ *Substitute **EdgeDetect** (arXiv:2604.14663)*
- Liu et al. "Error Prevalence in NIDS Datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018." 2022. ✅ `[S10]`
- LoByITFL — Xia, Egger, Hofmeister, Bitar. arXiv:2405.19217, 2024. ✅
- Ma, Yang, Wang, Chen. "Model optimization in federated intrusion detection systems: a survey of aggregation, communication, and personalization." ***Journal of Cloud Computing*, 2026-04-30.** DOI 10.1186/s13677-026-00901-w. **0 citations.** ✅ `[S10]` ⚠️ ***The most on-point survey for the novelty claim.*** *Abstract obtained (DOAJ); taxonomy unread — Springer IdP blocks even this gold-OA article. Its abstract explicitly names "rare-attack recall" as an aggregation-shaped outcome (§4.12)*
- MESAS — Krauß & Dmitrienko. CCS 2023. ✅
- Microsecond Federated SVD on Grassmann Manifold for Real-time IoT Intrusion Detection — arXiv:2510.18501, 2025. ✅ `[S10]`
- MID/WCS metrics — Xiao & Wang. "An Experimental Study of Class Imbalance in Federated Learning." IEEE SSCI 2021. arXiv:2109.04094. ✅
- Mimicry vs. Provenance-Graph HIDS — Goyal, Wang, Bates, Han. NDSS 2024. ✅⚠️
- Mironov. "Rényi Differential Privacy." IEEE CSF 2017. arXiv:1702.07476. ✅
- Model Fusion via Optimal Transport — Singh & Jaggi. NeurIPS 2020. ✅
- MPFT — Zhang et al. ICLR 2025. arXiv:2410.07738. ✅⚠️
- NIDS-FGPA — **Jiaming Wang, Kai Yang, MinJing Li.** "NIDS-FGPA: A federated learning network intrusion detection algorithm based on secure aggregation of gradient similarity models." ***PLoS ONE*, 2024-10-24.** DOI 10.1371/journal.pone.0308639. **20 citations**, gold OA. ✅ `[S10]` ⚠️ *Evaluated on **Edge-IIoTset and CIC IoT 2023 — NOT CIC-IDS-2017**. GSA's objective is **communication reduction**, not minority-boosting*
- NSL-KDD — Tavallaee, Bagheri, Lu, Ghorbani. CISDA 2009. ✅
- OT One-Shot FL — Chiang, Terai, Chiang, Lin, Ji, Lui. IEEE IoTJ 2024, 11(2):2166–2180. ✅
- PACT — Sheykholeslamzadeh & Alhadidi. "PACT: A Passive Accuracy-Based Trust Metric for Malicious Client Detection in Federated Learning." **TrustCom 2025**, 1775–1781. DOI 10.1109/Trustcom66490.2025.00206. ✅
- Parameterizing poisoning attacks in federated learning-based intrusion detection — **ARES 2023.** DOI 10.1145/3600160.3605090. ✅
- PerFedHypID — "A personalized federated hypernetworks based aggregation approach for intrusion detection systems." ***Scientific Reports* 2025**, `s41598-025-11659-7`, PMC12485126. ✅
- ~~pFedCross~~ — claimed "personalized federated cross learning." ❌ *Likely a conflation with the real, general-FL **FedCross** (arXiv:2210.08285)*
- pFedMe — Dinh et al., 2020. ✅
- Poisoning with a Pill — Guo, Wang, Song, Zheng, Hua, Guan, Zhang. AAAI 2026. arXiv:2407.15389. ✅
- **Precision-Weighted Federated Learning** — **Reyes, Di Jorio, Low-Kam, Kersten-Oertel.** **arXiv:2107.09627, 2021-07-20** (10 pp., 11 figures) → ***Computational Intelligence*, 2025-11-04**, DOI 10.1111/coin.70150. **11 citations.** ✅ `[S10]` ⚠️ *(❓→✅ this pass. Inverse-variance/precision weighting from empirical stochastic-gradient variances. **The authors never use "Precision-Weighted FedAvg"** — cite by title. Prior art from 2021 for gradient-statistic-derived aggregation weights)*
- PROTEAN — Chennoufi, Han, Blanc, De Cristofaro, Kiennert. ESORICS 2025. arXiv:2507.05524. ✅
- ProtoHAR — Cheng, Zhang, Bu, Wang, Wu, Song. IEEE JBHI 2023. ✅
- q-FedAvg — Li et al. ICLR 2020. ✅
- RAGA — Zuo, Yan, Fan, Hu, Shan, Quek, Zhao. IEEE TMC 2025. arXiv:2403.13374. ✅
- Ratio Loss — Wang, Xu, Wang, Zhu. AAAI 2021. arXiv:2008.06217. ✅
- RefProtoFL — Wu, Li, Fan, Zhu, Chen, Feng. arXiv:2601.14746, 2026. ✅⚠️
- REFML — Cui, Li, Mei et al. IEEE TIM 2023. arXiv:2310.09002. ✅
- RepTreeFL — Ghilea & Rekik. arXiv Dec 2023. ✅⚠️
- REVERB-FL — Peechara & Sahay. arXiv:2512.13647. IEEE TASLP. ✅ *(audio domain)*
- Ryan, Lin, Miikkulainen. "Intrusion Detection with Neural Networks." NIPS 1997. ✅⚠️
- Sarhan, Layeghy, Portmann. "Towards a Standard Feature Set for Network Intrusion Detection System Datasets." **Mobile Networks and Applications, 2022.** arXiv:2101.11315. ✅ *(the NF-\*-v2 / NF-UQ-NIDS-v2 family; NetFlow V3, UQ Cyber, 2025)*
- SBFedNIDS — Zeng, Zheng, Li. ICASSP 2025. ✅⚠️
- SecFedNIDS — Zhao, Zhang, Guo, Yao, Li. "SecFedNIDS: Robust defense for poisoning attack against federated learning-based network intrusion detection system." ***FGCS* 134:154–169, 2022-04-18.** DOI 10.1016/j.future.2022.04.010; code `github.com/zhangzhao156/SecFedNIDS`. **100 citations** ✅ `[S10]` *(**the must-cite in-domain adversarial baseline** — FedGRAA's own datasets, public code)*
- SelfFed — Khowaja, Dev, Anwar, Linguraru. arXiv 2023 → ESWA 2025. ✅⚠️
- SEM — Pham, Nguyen et al. CANDAR 2023, 120–126. ✅
- Sentinel — **Gurpreet Singh, Keshav Sood, P. Rajalakshmi, Yong Xiang.** "Sentinel: Dynamic Knowledge Distillation for Personalized Federated Intrusion Detection in Heterogeneous IoT Networks." **IEEE Internet of Things Journal, 2026-01-12**, DOI 10.1109/jiot.2026.3650848; preprint **arXiv:2510.23019** (2025-10-27). ✅ `[S10]` ⚠️ *(one paper, two records — merged this pass; **cite the journal version**)*
- ShapFed — IJCAI 2024. ✅ *(CELM's own nearest comparator)*
- SLVR — Choi, Rachuri, Wang, Jha, Wang. arXiv:2502.08055, 2025. ✅
- SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning — arXiv:2502.03801. ✅
- SpectralFed / SpectralFuse — Ukaye, Abdu-Aguye, Tastan, Nandakumar. "Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy." arXiv:2604.22562; **CVPR 2026 FedVision Workshop.** ✅
- StatAvg — arXiv:2405.13062 / IEEE journal 2025. ✅
- Sun, Zhou, Wang, Li. "Robust intrusion detection based on personalized federated learning for IoT environment." ***Computers & Security*, 2025-03-21.** DOI 10.1016/j.cose.2025.104442. **14 citations** ✅ `[S10]` *(new this pass; absent from all nine source documents)*
- Survey on Class Imbalance in FL — Zhang, Li, Qi, He. arXiv:2303.11673, 2023. ✅
- Temporal Analysis of NetFlow Datasets for NIDS — arXiv:2503.04404. ✅
- TACMA Fed — Aljuaid. "Trust-Aware Contrastive Meta-Aggregation Federated Learning for Intrusion Detection in the Internet of Things." ***Symmetry* 2026-07-14.** DOI 10.3390/sym18071188. **0 citations** ✅ `[S10]` *(extends AMAFed; adds rare-class-aware weighting)*
- TriFusion-IDS — Zhao & Zhong. AAAI 2026, 40(19):16433–16440. ✅
- Tri-LLM Cooperative Federated Zero-Shot IDS — arXiv:2602.00219, 2026. ✅
- UNSW-NB15 — Moustafa & Slay. MilCIS 2015. ✅
- Wardana & Sukarno. "Taxonomy and Survey of Collaborative Intrusion Detection System using Federated Learning." **ACM Computing Surveys 57(4), Article 88, 2024-10-25**, 36 pp. DOI 10.1145/3701724. **23 citations** ✅ `[S10]` *(taxonomy unread — ACM DL 403)*
- WeiDetect — Sameera, Vinod P., Rocha, Rehiman, Conti. *J. Inf. Secur. Appl.* 2025. DOI 10.1016/j.jisa.2025.104275; arXiv:2504.04367. ✅
- Yang, Wong, Zhu, Wang, Qian. "Federated learning with class imbalance reduction." arXiv:2011.11266, 2020. ✅
- Yin, Mallya, Vahdat, Alvarez, Kautz, Molchanov. "GradInversion." CVPR 2021. ✅
- Zhang, Ye, Huang, Liu, Gu. "Survey of federated learning in intrusion detection." ***J. Parallel and Distributed Computing*, 2024-09-19.** DOI 10.1016/j.jpdc.2024.104976. **46 citations** ✅ `[S10]` *(taxonomy unread)*
- Zhao, Li, Lai, Suda, Civin, Chandra. "Federated Learning with Non-IID Data" (EMD). arXiv:1806.00582, 2018. ✅
- Zhao, Mopuri, Bilen. "iDLG." 2020. ✅
- Zhu, Liu, Han. "Deep Leakage from Gradients." NeurIPS 2019. arXiv:1906.08935. ✅

---

## 12. Document provenance

**All nine source documents remain in place and unedited.** They are the itemized, dated, per-query verification trail behind every claim above. This document supersedes them **as the entry point**; it does not replace them as evidence.

| # | Source document | Date(s) | What it contributed | Final disposition |
|---|---|---|---|---|
| 1 | [`literature_review_and_publication_strategy.md`](literature_review_and_publication_strategy.md) | 2026-07-04 (+ snowball 2026-07-23) | The original 30+ paper A\*/A review; venue-fit strategy; the 10-paper snowball update that first surfaced NIDS-FGPA, FedVG, Fed-LAA, PROTEAN, FedP3E | **Superseded by this doc.** Kept as audit trail |
| 2 | [`FedGRAA_Deep_Research_Compendium.md`](FedGRAA_Deep_Research_Compendium.md) | 2026-07-04 | Exhaustive per-item re-verification of #1; the master correction table (§3.1 rows 1–18); the first two hallucination flags (FedSND, FeCWGAN-GP); the dataset and venue catalogs | **Superseded.** Kept as audit trail |
| 3 | [`research_followup.md`](research_followup.md) | 2026-06-28 | The original three-question framing (imbalance detection without an oracle; FL framework choice; validation without `D_V`); the Flower recommendation; Families 1/6 seed references | **Superseded.** Kept as audit trail |
| 4 | [`sagitra_report.md`](sagitra_report.md) | archived, folded in by 2026-07-09 | Raw output of an automated survey tool (~400 entries). Every paper it surfaced was independently re-verified before use — and it is **the entry vector for most of the seven hallucinations** (§3.2) | **Archived. Not independently citable** under any circumstances |
| 5 | [`privacy_compliant_fedgraa_redesign.md`](privacy_compliant_fedgraa_redesign.md) | 2026-07-08/09 (four passes) | The full FedGRAA-v2 redesign: 7-family imbalance-detection taxonomy, 6-approach validation taxonomy, exact byte/DP/threat-model accounting, full test-suite spec, function-level codebase mapping | **Still authoritative for §7 detail.** This doc condenses it; read the source before implementing |
| 6 | [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) | 2026-07-24, upd. 2026-07-25 | The prior consolidation of #1–5; the escalation path; the DBLP-API standard; the §12 self-audit that set this series' verification discipline | **Superseded by this doc as the entry point.** Kept as audit trail |
| 7 | [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md) | 2026-07-25 | Rescued **Fed-LAA**; killed **pFedCross**; reconfirmed CT-BRFL; **established the mandatory descriptive-phrase retry** — the single most valuable process contribution in the series | **Folded in.** Kept as audit trail |
| 8 | [`STAGE_C_DEEP_RESEARCH_2026-07-25.md`](STAGE_C_DEEP_RESEARCH_2026-07-25.md) | 2026-07-25 | **Found FedPGO and CELM** — broke the novelty claim; closed 11 ❓; added 3 ❌; stocked §4.11a/§4.11b/§5/§4.10; caught the PACT and FedP3E self-contradictions | **Folded in.** Kept as audit trail |
| 9 | [`STAGE_C_FASTTRACK_FINAL_PASS_2026-07-26.md`](STAGE_C_FASTTRACK_FINAL_PASS_2026-07-26.md) | 2026-07-26 | **Read CELM in full** and corrected its mechanism (not gradient-based); cross-index saturation confirmation; HADA-FL and TACMA Fed; the XAI-SOH-FL γ reviewer risk; the Sentinel lead | **Folded in.** Kept as audit trail. ⚠️ Its saturation *counts* are corrected by §1.3 N1 |

**Cross-checked, not superseded** (both confirmed consistent this pass — **no drift found** `[S10]`):
[`../current/fedgraa_convergence_defect.md`](../current/fedgraa_convergence_defect.md) — the missing-simplex-projection finding, which CELM's §3 bears on (§4.11a) — and [`../paper/PAKDD_Final_Version_Submit_v3.md`](../paper/PAKDD_Final_Version_Submit_v3.md), the draft this literature work serves (§2 verified against Eq. 7–10, Algorithm 2, and the experimental setup).

---

## 13. Method, limitations, and AI disclosure

### 13.1 What this pass actually was

An ARS `deep-research` fact-check + synthesis pass, run 2026-07-27, in consolidation/close-out mode: read all nine source documents in order, cross-check two external documents, resolve nine carried-forward open items, re-run the duplication and saturation instruments against the *final* novelty claim, and write one deduplicated entry point.

### 13.2 ⚠️ Tooling deviation from the brief — disclosed in full

**The brief specified three MCP servers as mandatory: `FastTrack_Literature_Open`, `alphaXiv`, and `Consensus`. None was connected in this session.** The deferred-tool registry was checked three ways (`+FastTrack`, `+alphaXiv`, and a keyword search over the full registry); only WebSearch, WebFetch, Bash, and the file tools were available.

**Mitigation used, and why it is close to equivalent for most purposes.** FastTrack is a wrapper over **OpenAlex and Semantic Scholar**; both expose free public REST APIs, and `curl` reaches them from this environment. This pass therefore hit those indices *directly*:

| Brief's requested tool | Substitute actually used | Adequate? |
|---|---|---|
| `search_papers` / `get_paper` | OpenAlex `/works` (search + DOI lookup), Semantic Scholar Graph API, Crossref, DBLP API | **Yes** — same underlying records, more fields |
| `run_duplication_test` | Three-query OpenAlex nearest-neighbour relevance search against the final claim (§1.2) | **Approximately** — no learned similarity ranking, so recall may be lower |
| `check_gap_saturation` | OpenAlex `group_by=publication_year` over two query formulations (§1.3) | **Better** — full-index counts rather than a relevance-limited retrieval; this is how the `[S9]` magnitude error was caught |
| `get_journal_profile` | OpenAlex `/sources` summary statistics (§6.1) | **Yes** — 2-yr mean citedness, h-index, works/citation counts |
| `map_topic_debate` | ⚠️ **Not substituted.** No equivalent citation-graph clustering was run | **No** — `[S9]`'s result (no unmapped citation cluster) is carried forward, not re-verified |
| `alphaXiv` full-text tools | WebFetch on `arxiv.org/abs/*` and `arxiv.org/html/*` | **Yes for arXiv content** — this is how item (h) was resolved |
| `Consensus` third-opinion index | ⚠️ **Not substituted** | **No** — third-index triangulation for items (a)/(f) was not performed |

**Every ✅ in this document is labelled with the pass that produced it.** No verdict was written from memory.

### 13.3 Tools that worked, and tools that failed

**Worked:** OpenAlex REST (the workhorse — resolved NIDS-FGPA, all six surveys, 11 citation counts, both venue profiles, the saturation curves, Jiang et al.'s abstract, Flood et al.); Semantic Scholar Graph API (until rate-limited); DBLP API (intermittently); WebFetch on arXiv, DOAJ, and `portal.core.edu.au`; WebSearch.

**Failed, with the failure mode named:**

| Target | Failure | Consequence |
|---|---|---|
| `export.arxiv.org` via `curl` | **HTTP 000** on every request including a known-good sanity query — host unreachable from this environment, not a query error | Routed around via WebFetch, which reaches arXiv fine. Diagnosed by running a control query first rather than assuming zero results meant "not found" |
| `dl.acm.org` | **403** to both WebFetch and `curl` with a browser UA — *including for articles OpenAlex flags as OA* | **Item (i) only partially resolved.** §4.12's caveat stands |
| Figshare mirror of CSUR 2024 | **403** to WebFetch; **HTTP 000** to the Figshare v2 API | The one promising route around ACM's wall closed |
| Springer / `link.springer.com` | **303 redirect to `idp.springer.com`** — including for the **gold-OA** J. Cloud Computing PDF | **Item (f) only partially resolved.** Abstract recovered via DOAJ instead |
| ScienceDirect | **403** — including for Jiang et al., which is **CC-BY**. A platform restriction, not a licensing one | **Item (g) only partially resolved** |
| Semantic Scholar | **HTTP 429**, as in all three prior passes | FedVG and XAI-SOH-FL citation counts remain unobtained (item e) |
| DBLP API | Intermittent **500/503** on multi-term queries | Worked around via OpenAlex descriptive search — which is what resolved items (c) and (d) |
| Unpaywall, OpenAIRE, CORE, `dk.upce.cz` | 000 / 429 / no matching record | Confirmed no FedPGO open copy exists anywhere (item a) |

### 13.4 What remains unverifiable, and why

1. **FedPGO's full text** — the only *substantive* blocker. Six independent channels exhausted; no preprint twin exists; OpenAlex reports `locations_count: 1` and `any_repository_has_fulltext: false`. **This is a paywall, not a search failure.** Resolvable only by institutional access or author contact (§9.0a), which are outside this pass's reach.
2. **All six FL-IDS survey taxonomies** — the most consequential *evidentiary* gap, because the novelty claim's survey-based half depends on them. Publisher access control, not availability: two are flagged open access and are still unreadable. §4.12's comparative framing is the honest workaround and should not be upgraded to an absolute claim without a real read.
3. **Jiang et al.'s and the J. Cloud Computing survey's full texts** — both partially recovered (verbatim abstracts), both sufficient for how they are used here, neither sufficient to characterize the papers' own limitations.
4. **`map_topic_debate` and a third-index (Consensus) cross-check** — not run at all (§13.2). `[S9]`'s citation-cluster result is carried forward on its authority, not re-verified on mine.
5. **Two citation counts** (FedVG, XAI-SOH-FL) — arXiv-only papers with no DOI; OpenAlex has no record and S2 was rate-limited. Low decision-relevance.
6. **Derived-not-fetched facts, labelled at point of use:** the Chunduru et al. DOI (mechanically derived from a *Scientific Reports* article ID) and the Engelen/Lanvin/Liu dataset-error citations (read from within El Mahdaouy et al.'s reference list, not fetched independently).

### 13.5 Verification discipline

Every ✅ in this document corresponds either to a tool call made in this pass (marked `[S10]`) or to an explicitly named prior pass. The master doc's documented process failure — asserting "confirmed via search this pass" for checks that had not been run — was the standard not to repeat, and was not repeated. Where this pass could not verify something, it says so rather than inheriting a comfortable verdict: items (f), (g) and (i) are marked **partially** resolved, not resolved, and item (a) is marked blocked rather than quietly dropped.

Two of this pass's own findings are corrections *to the prior passes in this series* — `[S9]`'s saturation magnitudes (§1.3 N1) and `[S8]`'s six-flaw attribution (§1.3 N2). Both are stated plainly, in the same register the series has used for bad news about FedGRAA itself.

### 13.6 AI-usage disclosure

This document was compiled by an AI assistant (Claude Opus 5) operating under the ARS `deep-research` skill in fact-check and synthesis mode, at the direction of the repository author. Its role was: reading and consolidating nine human-directed prior research documents; executing bibliographic API queries and web retrievals; cross-checking metadata against OpenAlex, Semantic Scholar, Crossref, DBLP, arXiv, DOAJ and CORE Portal; and drafting this synthesis. **All verification verdicts trace to named tool calls in a named pass**; none is asserted from model knowledge. **Interpretive readings are labelled as such** — in particular the saturation-curve reading (§1.3 N1) and the venue-profile comparison (§6.1) are the author's inferences from tool output, not tool conclusions, and §6.1 states its proxy limitation explicitly. Substantive judgement calls — the Stage C angle recommendation, the final Related Work wording, and what counts as a blocking gap — are offered as recommendations for the author and the mentor conversation the roadmap calls for, not as decisions taken. **No text in this document should be pasted into a manuscript without the author's own read of the underlying sources**, and the ~140 entries carried forward from source documents without a fresh `[S10]` check inherit those documents' verification, not this pass's.

---

## Verdict: is Stage C literature work complete enough to start drafting Related Work?

# ✅ YES — with one named, non-blocking exception.

**Ten passes, ~185 papers, six independent index/tool stacks. The corpus now has zero ❓ entries** — the last one closed this pass. The novelty claim has survived an adversarial saturation test that *broke* it once (FedPGO, `[S8]`), a cross-index re-test that corroborated the narrowed version (`[S9]`), and a third re-test on a structurally different, full-index basis this pass that **found nothing closer and no new occupant** (§1.2). A claim that has been attacked three times from three directions and narrowed only once is a claim worth defending.

**Start drafting.** Specifically:

- **§8.1 is the final wording.** Use it. It is narrow, true, concedes overlaps pre-emptively, and — unlike earlier versions — does not lean on the A\*/A venue qualifier that §6.1 shows is quantitatively weak.
- **§8.2 is the mandatory citation list.** Omitting FedPGO is the single most likely fatal miss.
- **§8.3 is the do-not-write list.** Three sentences that must not appear.

**The one named exception — and why it does not block.** **FedPGO's full text is unobtainable** (§9.0a: six channels, no preprint, definitively a paywall). Prior passes made reading it a blocking prerequisite. **That guidance is revised here:** the block was justified when it was unclear whether the paper was merely unfound. It is now established that it is unfindable-for-free, and further search effort is pure waste. What is *known* about FedPGO — existence, venue, authors, date, both datasets, mechanism in outline, and its personalization-vs-single-global-model distinction from FedGRAA — is triple-confirmed across DBLP, OpenAlex and Semantic Scholar, and is **sufficient to cite it, position against it, and differentiate from it.** It is **not** sufficient to assert what FedPGO *fails* to do.

➡️ **Therefore: draft Related Work now, with one discipline — no sentence may claim a FedPGO limitation.** Pursue the full text in parallel by ILL or a direct email to Dawit Dejene Bikila (ORCID 0000-0002-3159-1531, University of Pardubice), and revisit only the differentiation paragraph if it arrives.

**Two secondary gaps, neither blocking:** the six survey taxonomies remain unread (§9.0i) — handled by the comparative framing in §4.12, which must not be upgraded to an absolute claim; and `map_topic_debate`/Consensus cross-checks were not run (§13.2) — carried forward on `[S9]`'s authority.

**What genuinely gates the *paper*, as opposed to the *literature review*, is no longer literature work at all.** It is §9's items 1–3: the convergence theorem, the head-to-head baseline grid, and the adversarial evaluation. All three now have citable templates (§4.11a, §4.11b) and a concrete highest-value data move (§5, Option D). **Stage C's literature phase is closed. The next bottleneck is experimental.**

---

## 14. FedPGO full-text accessibility — resolved (dated correction, 2026-07-31)

This dated correction supersedes the FedPGO accessibility statements below,
**in the accessibility sense only**; every prior statement remains historically
true of the pass that wrote it and is retained in place. The full text
(FedPGO/FedPGO.md, 103,210 bytes) was obtained on 2026-07-30 via the
University of Pardubice institutional repository (`dk.upce.cz`) and committed
in `9d20ac9`; the mechanism was then reimplemented faithfully (commit
`25611ce`) and the 20 FedPGO cells rerun (commit `96adb05`). See
[`STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md`](STAGE_C_FEDPGO_FAITHFUL_PLAN_2026-07-31.md)
and [`STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md`](STAGE_C_COMPUTE_SESSION_UPDATE_2026-07-30.md) §7.

| Site | Statement | Corrected status (2026-07-31) |
|---|---|---|
| §4.4 FedPGO row | "Full text access-blocked across 6 channels" | Full text local; mechanism reimplemented from primary text (Eqs. 6–10, Algorithms 1–3), eight judgment calls JC1–JC8 disclosed in `fedgraa/baselines_t5_fedpgo.py`. |
| §9.0 item (a) | ⛔ STILL BLOCKED — six channels exhausted | ✅ RESOLVED 2026-07-30: primary text obtained via `dk.upce.cz`, committed `9d20ac9`. The SSRN preprint mirror (item in §11) remains 403. |
| §9.1 item 0 | "FedPGO: STILL BLOCKED — known only through abstract-level secondary summaries" | DONE: full primary-text read; FedPGO's mechanism, personalization scheme, and its own claims are now writable at primary-source confidence, and the "no sentence asserting what FedPGO fails to do" discipline is lifted subject to the faithful implementation's judgment-call disclosures. |
| §11 FedPGO entry | "Full text access-blocked across six independent channels" | Full text local since 2026-07-30 (committed `9d20ac9`); provenance note retained. |
| §13.4 item 1 | "FedPGO's full text — the only substantive blocker" | Resolved 2026-07-30 via the institutional repository; no longer unverifiable. |
| Verdict (§13.5–Verdict, lines 913–915) | "FedPGO's full text is unobtainable" / "not sufficient to assert what FedPGO fails to do" | The full text has been read and the mechanism reimplemented; the differentiation paragraph is now grounded in the primary source rather than the abstract. |

The historical saturation narrative at §13.5 ("the novelty claim survived an
adversarial saturation test that broke it once (FedPGO, `[S8]`)") is untouched:
the saturation tests' findings are independent of text accessibility.

*Correction appended 2026-07-31, after commit `96adb05`.*

---

*Compiled 2026-07-27 via ARS `deep-research` (fact-check + synthesis mode). Indices queried this pass: OpenAlex REST, Semantic Scholar Graph API, Crossref, DBLP publication-search API, arXiv API (via WebFetch), DOAJ, Unpaywall, OpenAIRE, CORE, Figshare, CORE Portal (portal.core.edu.au), plus general web search and direct document retrieval. ⚠️ The FastTrack Literature Open, alphaXiv, and Consensus MCP servers named in this pass's brief were **not available in this session** — see §13.2 for the substitutions used and the two instruments that were consequently not run. Supersedes [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) and its three addenda as the Stage C entry point; all nine source documents remain in place as the audit trail.*
