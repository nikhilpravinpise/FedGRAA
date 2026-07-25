# Stage C — Deep Research Pass: Verification Closure, Saturation Test, and Four Deepened Gap Areas

> **What this is.** A standalone deep-research pass over [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) (compiled 2026-07-24) and its addendum [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md), run via the ARS `deep-research` skill on **2026-07-25**. Four objectives, all executed: (1) close the ~18 entries the master doc still carried as **❓ Unverified**; (2) adversarially *saturation-test* the central novelty claim rather than re-assert it; (3) deepen four areas the master doc names as gaps but does not stock with literature — convergence theory, adversarial robustness, datasets, privacy-compliant validation; (4) audit the master doc for internal inconsistency.
>
> **Relationship to the master doc.** This is an **addendum and a proposed-revision list**, not a replacement. It does not modify the master doc. §11 lists the exact edits to apply there. Where this document and the master doc disagree, this document is the later evidence — but the master doc remains the single entry point once §11 is applied.
>
> **Headline result, stated plainly up front.** The central novelty claim as currently worded **no longer survives contact with the 2026 literature in the form the master doc gives it.** A previously-unseen paper — **FedPGO** (Bikila & Capek, *Computer Networks* 280:112210, 2026, DOI 10.1016/j.comnet.2026.112210) — occupies the exact three-way intersection the master doc calls unoccupied: gradient-derived adaptive aggregation reweighting × federated network intrusion detection × non-IID, evaluated on **CIC-IDS2017 itself**. A second paper, **CELM** (arXiv:2605.18892, May 2026), independently occupies the *minority-class-boosting adaptive-aggregation-weights* objective in a data-free formulation that also removes the server validation set — i.e. it partly pre-empts the FedGRAA-v2 redesign in §7 of the master doc. The claim is still defensible, but only after a real narrowing, spelled out in §6.4. This is the outcome a saturation test is *supposed* to produce, and it is much better to find it now than in a reviewer's report.

**Verification tags** follow the master doc's own convention: **✅ Confirmed** (primary or authoritative bibliographic source fetched this pass), **⚠️ Corrected** (real paper, master doc has wrong metadata), **❌ Hallucination-equivalent** (searched across ≥2 independent tools including a broadened descriptive-phrase retry, zero trace), **❓ Unverified** (still open).

**Escalation path used throughout**, per the master doc's standing procedure plus the 2026-07-25 re-audit's amendment: general web search → arXiv API (`export.arxiv.org/api/query`) → DBLP publication-search API (`dblp.org/search/publ/api`) → **broadened descriptive-phrase DBLP retry before any non-existence verdict**. That last sub-step is what rescued Fed-LAA on 2026-07-25; **this pass it rescued FedPGO, the single most consequential find in the document.** See §10.2.

---

## Table of contents

1. [Six findings that change Stage C decisions](#1-six-findings-that-change-stage-c-decisions)
2. [Verification closure — the ❓ backlog](#2-verification-closure--the--backlog)
3. [Three new hallucination-equivalents](#3-three-new-hallucination-equivalents)
4. [Corrections owed to the master doc](#4-corrections-owed-to-the-master-doc)
5. [New competitors surfaced this pass](#5-new-competitors-surfaced-this-pass)
6. [Saturation test — the novelty claim, re-assessed](#6-saturation-test--the-novelty-claim-re-assessed)
7. [Deepened gap area A — convergence theory for adaptive-weight FL](#7-deepened-gap-area-a--convergence-theory-for-adaptive-weight-fl)
8. [Deepened gap area B — adversarial robustness of reweighting aggregators](#8-deepened-gap-area-b--adversarial-robustness-of-reweighting-aggregators)
9. [Deepened gap area C — datasets and benchmarks](#9-deepened-gap-area-c--datasets-and-benchmarks)
10. [Deepened gap area D — privacy-compliant validation and data-free contribution estimation](#10-deepened-gap-area-d--privacy-compliant-validation-and-data-free-contribution-estimation)
11. [Devil's advocate checkpoint](#11-devils-advocate-checkpoint)
12. [Revised positioning and updated gap-closing checklist](#12-revised-positioning-and-updated-gap-closing-checklist)
13. [Edits to apply to the master doc](#13-edits-to-apply-to-the-master-doc)
14. [Method, limitations, and disclosure](#14-method-limitations-and-disclosure)

---

## 1. Six findings that change Stage C decisions

**F1 — The three-way intersection is occupied. FedPGO is the paper that occupies it.**
Bikila & Capek, *"FedPGO: A personalized federated learning framework with gradient orthogonalization for IoT intrusion detection,"* **Computer Networks 280:112210 (2026)**, DOI 10.1016/j.comnet.2026.112210 (✅ confirmed via DBLP exact-title hit). Its Adaptive Gradient Orthogonalization (AGO) decomposes each local gradient into components parallel and orthogonal to the global gradient and **adaptively reweights them**; its Personalized Gradient Fusion (PGF) aggregates client updates through a scoring function over gradient directionality and orthogonality. Evaluated on **CICIoT2023 and CIC-IDS2017** under non-IID conditions. This is gradient-derived adaptive aggregation weighting, in FL-IDS, framed on non-IID — on FedGRAA's own primary dataset. See §6.1.

**F2 — The FedGRAA-v2 privacy redesign has been partly pre-empted, by a paper that also already has the stabilization FedGRAA is missing.**
Ukaye, Tastan, Abdu-Aguye & Nandakumar, *"Data-Free Client Contribution Estimation via Logit Maximization for Federated Learning"* (**CELM**), arXiv:2605.18892, 17 May 2026 (✅ abstract fetched verbatim). Its opening premise is FedGRAA's: *"practical FL deployments often exhibit severe class imbalance and label skew, causing standard aggregation protocols to overfit dominant clients and degrade minority-class performance."* Its mechanism is FedGRAA's objective: *"contribution weights that upweight clients providing strong, discriminative evidence for underrepresented classes."* It requires **no raw data, no client metadata, and no auxiliary public dataset** — which is exactly what master-doc §7's FedGRAA-v2 sets out to achieve — and its aggregation is *"stable due to **simplex constraints** and momentum smoothing,"* which is precisely the projection [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) found **missing** from FedGRAA's Algorithm 2. Domain is vision benchmarks, not IDS. See §6.2 and §10.1.

**F3 — Both remaining "we have no theory / no robustness analysis" gaps now have a template in the literature, for exactly FedGRAA's mechanism class.**
Parsa, Daghestani, Teixeira & Johansson, *"Byzantine-Robust Federated Learning with Learnable Aggregation Weights,"* arXiv:2511.03529 (submitted Nov 2025, revised May 2026) (✅ abstract fetched verbatim) treats **aggregation weights as learnable parameters jointly optimized with the model**, solves it by alternating minimization with *"strong convergence guarantees under adversarial attack,"* and analyses Byzantine resilience of the objective. This is one paper that closes, for the general case, both gap #1 (convergence proof) and gap #3 (adversarial evaluation) of master-doc §9 — meaning FedGRAA can no longer present either as an open problem it is solving, only as an IDS-specific instantiation it is supplying. See §7.2 and §8.1.

**F4 — FedGRAA's dataset problem is a validity threat, not just a currency complaint.**
The master doc frames "two legacy datasets (2015/2017)" as a freshness gap. The NetFlow-benchmark literature frames it harder: CIC-IDS-2017 and CSE-CIC-IDS-2018 carry **labelling errors and unrealistic temporal separation between attack and benign traffic**, and a systematic analysis of seven popular NIDS benchmarks identifies six recurring design flaws (limited feature diversity, high feature interdependence, ambiguous ground truth, collapsed traffic distributions, artificial diversity generation, labelling inaccuracies) that **correlate with poor operational generalization**. There is also a clean fix available: the **NF-*-v2 / NetFlow-V3** family (Sarhan, Layeghy & Portmann) re-expresses UNSW-NB15, BoT-IoT, ToN-IoT and CSE-CIC-IDS2018 in one shared 43-feature space (53 with V3 temporal features) — a *natural cross-silo federated benchmark* in which each dataset is a client with an identical feature schema. See §9.

**F5 — Six independent FL-IDS surveys now exist, not three; the master doc has DOIs for two of them.**
The novelty claim leans on "no FL-IDS aggregation survey lists a gradient-refinement method." That claim now has to hold against six surveys (§6.3), and this pass could not read any of their taxonomies in primary text — all are paywalled. This is an honest weakening of the evidence base for the claim, recorded as a limitation, not papered over.

**F6 — The master doc has one flat internal contradiction and one probably-wrong venue.**
PACT is ✅ confirmed in §3.3 but still tagged ❓ "(unresolved)" in §4.10 — the same paper, two verdicts, in one document. And FedP3E is attributed to *IEEE TIFS 2025* in §4.6 and §11, but DBLP indexes **only a CoRR/arXiv preprint** (arXiv:2507.07258); no journal version is listed. See §4.

---

## 2. Verification closure — the ❓ backlog

Fourteen of the master doc's ❓ entries were run this pass. **Ten resolve to ✅ Confirmed**, one resolves ✅ with a venue caveat, and three resolve to ❌ (§3).

| Item | Master doc said | Verified this pass | Verdict |
|---|---|---|---|
| **FedNIDS** | "Nguyen et al., DTRAP 2025 (as reported)" | Quoc H. Nguyen, Soumyadeep Hore, Ankit Shah, Trung Q. Le, **Nathaniel D. Bastian**. *"FedNIDS: A Federated Learning Framework for Packet-Based Network Intrusion Detection System."* **Digital Threats: Research and Practice 6(1):1–23, 2025**, DOI 10.1145/3696012 | ✅ (DBLP) — description was accurate |
| **WeiDetect** | "Sameera et al., JISA 2025 (as reported)" | K. M. Sameera, Vinod P., **Anderson Rocha**, Rafidha Rehiman K. A., **Mauro Conti**. *J. Inf. Secur. Appl.* 2025, DOI 10.1016/j.jisa.2025.104275; preprint arXiv:2504.04367 | ✅ (DBLP, two records) |
| **FedP3E** | "IEEE TIFS 2025 (as reported)" | Rami Darwish, Mahmoud Abdelsalam, Sajad Khorsandroo, Kaushik Roy. *"FedP3E: Privacy-Preserving Prototype Exchange for Non-IID IoT Malware Detection in Cross-Silo Federated Learning."* **CoRR/arXiv:2507.07258, 2025 — preprint only** | ✅⚠️ **venue wrong** — see §4 |
| **FADngs** | "Dong et al., IEEE TNNLS 2025 (as reported)" | Boyu Dong, Dong Chen, Yu Wu, Siliang Tang, Yueting Zhuang. *"FADngs: Federated Learning for Anomaly Detection."* **IEEE TNNLS 36(2):2578–2592, 2025**, DOI 10.1109/TNNLS.2024.3350660 | ✅ (PubMed 38241100) |
| **PerFedHypID** | "Scientific Reports, 2025 (as reported)" | Published title is *"A personalized federated hypernetworks based aggregation approach for intrusion detection systems,"* **Scientific Reports 2025**, `s41598-025-11659-7`, PMC12485126; PerFedHypID is the method name (used in the Research Square preprint, rs-4767476). Embedding-vector rather than weight-based aggregation; evaluated on **CSE-CIC-IDS2018** under non-IID; 0.62 M params/round vs 1.17 M baseline | ✅ — and a **relevant competitor**, see §5 |
| **FedKSam\*** | "Zhang et al., CCC 2024 (as reported)" | *"FedKSam\*: A Two-Phase Federated Learning Intrusion Detection Algorithm."* IEEE Xplore document **10661778**. Phase 1: SAM optimizer + knowledge distillation against non-IID drift; Phase 2: re-optimizes classifier decision boundaries. Addresses **data heterogeneity and class imbalance** in IDS | ✅ (IEEE Xplore record) — asterisk is part of the name |
| **G-PFL-ID** | "IoT (MDPI), 2026 (as reported)" | *"G-PFL-ID: Graph-Driven Personalized Federated Learning for Unsupervised Intrusion Detection in Non-IID IoT Systems,"* **MDPI *IoT* 7(1):13, published 2026-01-29**, mdpi.com/2624-831X/7/1/13. GCN/GAE + DeepSVDD + FedReg regularization; 99.46% AUROC on IoT-23, 97.74% on N-BaIoT | ✅ (MDPI listing) |
| **FSFL-IDS** | "Springer chapter, ~2025 (as reported)" | *"Distributed Intrusion Detection in Dynamic Networks of UAVs Using Few-Shot Federated Learning,"* Springer chapter **DOI 10.1007/978-3-031-94448-2_7**, 2025. FSFL-IDS is the method name; targets routing attacks in dynamic FANETs | ✅ (Springer DOI; landing page itself is behind an IdP redirect) |
| **FedAAW** | "Knowledge-Based Systems, 2024 (as reported)" | *"Tackling data-heterogeneity variations in federated learning via adaptive aggregate weights,"* **Knowledge-Based Systems 2024**, DOI 10.1016/j.knosys.2024.112484. **Aggregation weights are derived by minimizing the convergence upper bound of FL on non-IID data** | ✅ — and **promoted**, see §7.1 |
| **FedAWR** | "MDPI Future Internet, 2026 (as reported)" | *"FedAWR: Aggregation Optimization in Federated Learning with Adaptive Weights and Learning Rates,"* **Future Internet 18(2):106, 2026-02-18**, DOI 10.3390/fi18020106. Weight signal is client compute capability; evaluated on Steel Rail Defect + CIFAR-10 | ✅ — master doc's description was accurate |
| **FedDUAL** | "Sahoo et al., TMLR 2024 (as reported)" | *"FedDUAL: A Dual-Strategy with Adaptive Loss and Dynamic Aggregation for Mitigating Data Heterogeneity in Federated Learning,"* Pranab Sahoo et al., **arXiv:2412.04416** (5 Dec 2024), OpenReview `8M3XfmNhTZ`. Adaptive loss keyed to statistical distance between local and global distributions + dynamic aggregation | ✅⚠️ paper real; **TMLR acceptance not independently confirmed** — cite by arXiv ID |
| **FedGC** | "Xi et al., 2025 (as reported)" | *"FedGC: An Accurate and Efficient Federated Learning under Gradient Constraint for Heterogeneous Data,"* OpenReview `eZN8nUXAVO7`. Client- and server-side gradient-constraint projection (CGC/SGC) | ✅⚠️ real; **venue unconfirmed** (OpenReview submission record only) |
| **Data-Free Client Contribution Estimation via Logit Maximization** | "Ukaye et al., 2026 (as reported)" | **CELM** — Asim Ukaye, Nurbek Tastan, Mubarak Abdu-Aguye, Karthik Nandakumar, arXiv:2605.18892, 17 May 2026 | ✅ — and **the second-most consequential find of the pass**, see §6.2 / §10.1 |
| **Fed-Meta FD** | "IEEE 2023 (verify TNSE)" | Not run this pass — deprioritized (IIoT fault diagnosis, not IDS; lowest decision-relevance of the backlog) | ❓ still open |

**Bonus resolution not in the backlog.** The master doc cites *"Federated transfer learning for rare attack class detection in NIDS"* (Chunduru Sri Abhijit et al., *Scientific Reports* 2025) **without a DOI**. It is `nature.com/articles/s41598-025-02068-x`, PMC12484838 (DOI therefore **10.1038/s41598-025-02068-x**, derived mechanically from the Scientific Reports article ID rather than independently fetched — verify once before it goes into a manuscript). Reported: 98.90% on CIC-IDS2018, 98.70% on UNSW-NB15, 97.92% on Edge-IIoT — note it does **not** evaluate on CIC-IDS-2017.

---

## 3. Three new hallucination-equivalents

Each was searched on general web search with a broadened descriptive phrase **and** on DBLP; each returned zero primary trace. This is the same evidentiary tier the master doc applies to FedSND, FeCWGAN-GP, CT-BRFL and pFedCross — and, importantly, the *same* test that **FedPGO passed** (§10.2), so the standard is being applied symmetrically rather than selectively.

| Item | Master doc claim | Evidence against | Verdict |
|---|---|---|---|
| **Light-FFSL** | "IEEE ~2025/26 (verify IoTJ)" — compression + trust-based selection + robust aggregation + FSL | Literal-name web search: zero results titled Light-FFSL. Broadened descriptive search ("federated few-shot IDS + compression + trust-based client selection + robust aggregation"): returned FedMADE, EdgeDetect, Tri-LLM, a TinyML siamese/RL-sparsification paper — **all real, all differently named**. DBLP combined query: **ZERO HITS** | ❌ Do not cite. Nearest real substitutes with the same components: **EdgeDetect** (arXiv:2604.14663) for compression + robust aggregation; the CICIoT/siamese TinyML few-shot FL-IDS line for FSL |
| **FedCCW** | "Zhang et al., 2025 (as reported)" | Literal search returns **FedCBA** (sample-level class-balancing aggregation, DOI 10.1145/3654446.3654454), **FedCW** (client selection with adaptive weight, CMC 86(1)), **FedLWS** (arXiv:2503.15111) — three real, differently-named papers, the classic conflation signature. DBLP: **ZERO HITS** | ❌ Do not cite. If a class-balanced-aggregation citation is needed, **FedCBA** is the real paper that fits the description |
| **FedHDE** | "Zhang et al., 2026 (as reported)" | Broadened search returns a fluent description (hyper-knowledge distillation + tensor optimization + dynamic client selection, teacher–assistant–student hierarchy) with **no link actually titled FedHDE**; results resolve to FedHGN, FedPH, Fed-GDBD. DBLP: **ZERO HITS** | ❌ Do not cite |

**Pattern worth recording.** All three failures share the master doc's diagnosed signature: *a confident, mechanically detailed description synthesized by a summarizing search tool, with a result list in which every link points to a different, real, differently-named paper.* The count of confirmed-or-equivalent hallucinations in this corpus now stands at **seven**: FedSND, FeCWGAN-GP, CT-BRFL, pFedCross, Light-FFSL, FedCCW, FedHDE. All seven entered the corpus through automated survey tooling or acronym-first search, never through a primary source — which is the actual lesson: **acronym-first retrieval is the failure mode, and descriptive-phrase retrieval is the fix in both directions** (it kills false positives *and* rescues real papers, per §10.2).

---

## 4. Corrections owed to the master doc

| # | Location | Current text | Correction | Severity |
|---|---|---|---|---|
| C1 | §4.10 approach F, PACT row | ❓ *"(unresolved)"* | **✅ Confirmed** — §3.3 of the same document already resolves it (Sheykholeslamzadeh & Alhadidi, TrustCom 2025, pp. 1775–1781, DOI 10.1109/Trustcom66490.2025.00206). One paper, two contradictory verdicts in one document | **High** — a self-contradiction a careful reader will catch |
| C2 | §4.6 row, §11 bibliography | "FedP3E — IEEE TIFS 2025" | DBLP indexes **only** arXiv:2507.07258 (CoRR 2025); **no TIFS version is listed**. Authors are Darwish, Abdelsalam, Khorsandroo & Roy. Cite as an arXiv preprint | **High** — same error class as the EIFFeL/FreqFed year corrections the master doc already fixed |
| C3 | §6 venue table, IEEE TIFS row | Lists "FeCoGraph, PROTEAN, FedP3E" as TIFS papers | Remove FedP3E (per C2). PROTEAN is ESORICS 2025 / arXiv:2507.05524 in §4.6 — reconcile with the TIFS listing here; the two sections disagree | Medium |
| C4 | §1 finding 6 | "8 of 9 previously-unverified papers now confirmed real" | Arithmetic superseded: this pass closed 11 more (§2), reclassified 3 as ❌ (§3), and leaves Fed-Meta FD open. Restate as a running count | Low (bookkeeping) |
| C5 | §4.4, Chunduru et al. row; §11 | No DOI | Add `s41598-025-02068-x` / PMC12484838; note the evaluation is CIC-IDS2018 + UNSW-NB15 + Edge-IIoT, **not** CIC-IDS-2017 | Low |
| C6 | §4.12 + §11, three survey rows | "ACM Computing Surveys, 2024" / "2025" with no DOIs | Real identifiers now in hand: **10.1145/3687124** (*Survey on FL for IDS: Concept, Architectures, Aggregation Strategies…*), **10.1145/3731596** (*Intrusion Detection Based on FL: A Systematic Review*), plus **10.1145/3701724** (Wardana & Sukarno, *Taxonomy and Survey of Collaborative IDS using FL*, CSUR 57(4) Art. 88, Dec 2024) which is **not currently in the master doc at all** | Medium — these are the citations the novelty claim rests on |

---

## 5. New competitors surfaced this pass

Beyond FedPGO and CELM (§6), the following are real, verified, and absent from the master doc.

| Method | Citation | Mechanism | Threat level to FedGRAA |
|---|---|---|---|
| **FedAWARE** | Zeng, Xu, Liu, Pan, Wang, Tang. *"On the Power of Adaptive Weighted Aggregation in Heterogeneous Federated Learning and Beyond."* arXiv:2310.02702 (v4, Dec 2024) | Server-side adaptive reweighting that **maximizes gradient diversity** — the convex combination of client updates that least conflicts in direction; introduces a *client consensus dynamics* heterogeneity measure and proves FedAvg handles heterogeneity under an appropriate aggregation strategy; plug-in module | **High, and philosophically inconvenient.** The master doc's core differentiation sentence contrasts FedGRAA against aggregators that "reward alignment with the global direction (FedAWA)." FedAWARE does the **opposite of FedAWA** — it rewards *divergence* — which is FedGRAA's own stated philosophy, in general FL, with convergence theory attached. The FedAWA/FedAWARE near-homonym makes this doubly easy for a reviewer to raise |
| **SecFedNIDS** | Zhao, Zhang, Guo, Yao, Li. *"SecFedNIDS: Robust defense for poisoning attack against federated learning-based NIDS."* **FGCS 134:154–169, 2022**, DOI 10.1016/j.future.2022.04.010; code at `github.com/zhangzhao156/SecFedNIDS` | Model-level poisoned-model detection + data-level poisoned-traffic filtering by class-path similarity. **+48% accuracy under attack on UNSW-NB15, +36% on CIC-IDS2018** | Medium — not a competitor mechanism, but the **must-cite in-domain baseline** for master-doc gap #3 (adversarial evaluation), on FedGRAA's own datasets, with public code |
| **AdaAggRL** | *"Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning."* arXiv:2406.14217; **AAAI 2025**, DOI 10.1609/aaai.v39i24.34733 | MMD-based pairwise similarity of local data distributions → policy learning determines aggregation weights adaptively | Medium-high — an **A\* venue** paper doing RL-learned adaptive aggregation weights, i.e. the Jiang et al. (Neurocomputing 2024) family with better venue standing. Strengthens the master doc's finding 11 rather than contradicting it |
| **AMAFed** | Izadi & Ahmadi. *"Adaptive Meta-Aggregation Federated Learning for Intrusion Detection in Heterogeneous Internet of Things."* arXiv:2602.12541, 13 Feb 2026 | Meta-learning dynamic weighting: adaptive importance to local models by data quality and contribution. ToN-IoT / N-BaIoT / BoT-IoT | Medium — same family (adaptive aggregation × IDS × heterogeneity) but **no class-imbalance framing** and no overlap with FedGRAA's datasets |
| **Byzantine-Robust FL with Learnable Aggregation Weights** | Parsa, Daghestani, Teixeira, Johansson. arXiv:2511.03529 | Aggregation weights as learnable parameters jointly optimized with the model; alternating minimization; convergence guarantees **under adversarial attack** | High as *prior art for the method class*; see §7.2 and §8.1 |
| **SpectralFed / SpectralFuse** | Ukaye, Abdu-Aguye, Tastan, Nandakumar. *"Data-Free Contribution Estimation in Federated Learning using Gradient von Neumann Entropy."* arXiv:2604.22562; **CVPR 2026 FedVision Workshop** | Matrix von Neumann (spectral) entropy of final-layer updates as a data-free contribution signal; SpectralFed uses normalized entropy directly as aggregation weights | Medium — same group as CELM, same data-free premise; general FL, no minority framing |
| **Precision-Weighted FedAvg** | Surfaced in secondary description only this pass — inverse-variance weights from empirical gradient variances | Gradient-statistics-derived weighting, down-weights noisy updates | ❓ **Not verified — do not cite until located.** Recorded here as a search lead, explicitly not as a confirmed reference |

---

## 6. Saturation test — the novelty claim, re-assessed

The master doc's claim: *"no A\*/A-venue paper combines gradient-refined adaptive aggregation weighting, applied specifically to intrusion detection, framed around non-IID/limited-data clients."*

### 6.1 FedPGO occupies the intersection

| Criterion FedGRAA claims | FedPGO |
|---|---|
| Gradient-derived aggregation weighting | ✅ AGO decomposes local gradients into parallel/orthogonal components w.r.t. the global gradient and adaptively reweights; PGF scores clients on gradient directionality + orthogonality |
| Server-side, aggregation-stage | ✅ PGF is the aggregation rule |
| Network intrusion detection | ✅ IoT IDS |
| Non-IID framing | ✅ explicit |
| FedGRAA's own dataset | ✅ **CIC-IDS2017**, plus CICIoT2023 |
| A\*/A venue | ❌ *Computer Networks* is a Q1 journal, not a CORE-ranked A\*/A conference |

So the literal claim survives **only** through the venue qualifier. That is a materially weaker position than the master doc conveys, for three reasons. First, PAKDD is itself CORE **B** (master doc §6), so a novelty argument that turns on "no *A\*/A* paper has done this" is being made from a B venue — a reviewer can fairly ask why the A\*/A restriction is the relevant frame. Second, *Computer Networks* is a well-cited Q1 venue that IDS reviewers read; "not CORE A\*/A" will not read as "obscure." Third, the mechanism overlap is not thematic but structural — gradient decomposition, then adaptive reweighting, evaluated on the same dataset.

**What genuinely still separates FedGRAA from FedPGO**, and must be stated explicitly rather than assumed:

- **Objective.** FedPGO is *personalization*-oriented (parameter-split architecture, per-client personalized layers, "mitigating gradient dominance"). FedGRAA produces a **single global model** and targets **minority-class detection** specifically. These are different goals that happen to share a mechanism family.
- **The γ coefficient.** FedGRAA's tunable `γ ∈ [0,1]` interpolating between minority-boosting and conventional size-proportional emphasis has no counterpart in FedPGO's formulation.
- **Rare-attack evaluation.** FedPGO reports accuracy/robustness; FedGRAA's claim is per-minority-class recall. The evaluation axes differ.

None of these is nothing. But they are *refinements within an occupied space*, not the occupation of an empty one — and the paper must be written that way.

### 6.2 CELM occupies the objective, and pre-empts part of the redesign

CELM (§1/F2) is not IDS work, so it does not threaten the domain claim. It threatens two other things:

1. **The minority-boosting-via-aggregation-weights objective** is now published, with the same motivating sentence FedGRAA uses. FedGRAA cannot present "up-weight clients holding underrepresented classes" as a new idea; it can present the gradient-refinement route to it as a distinct mechanism.
2. **The FedGRAA-v2 selling point is partly spent.** Master-doc §8 proposes the contribution statement *"the first fully FL-compliant gradient-refined adaptive-aggregation method…"* — "first fully FL-compliant" now has to contend with a 2026 paper that is explicitly data-free (no raw data, no client metadata, **no auxiliary public dataset**) and minority-targeted. The word "first" should be dropped or narrowed to "first *for intrusion detection*"; the gradient-refinement qualifier still holds.

There is also a sharper, more uncomfortable point. CELM's aggregation is stable *because of* **simplex constraints and momentum smoothing** — and the missing simplex projection is exactly the defect [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) diagnosed as causing real divergence in FedGRAA's Algorithm 2. A 2026 paper in the same objective space treats that constraint as a basic design requirement. When FedGRAA fixes it, the fix should be framed as standard practice with a citation, not as a novel contribution.

### 6.3 The survey-based half of the claim is weaker than stated

The master doc supports the gap with three surveys. Six now exist:

| Survey | Identifier | In master doc? |
|---|---|---|
| *Survey on FL for IDS: Concept, Architectures, **Aggregation Strategies**, Challenges, Future Directions* | ACM CSUR, DOI **10.1145/3687124** | Yes, without DOI |
| *Intrusion Detection Based on FL: A Systematic Review* | ACM CSUR, DOI **10.1145/3731596** | Yes, without DOI |
| *A survey of privacy-preserving FL for IDS* | *AI Review*, DOI 10.1007/s10462-026-11519-4 (2026) | Yes |
| *Taxonomy and Survey of Collaborative IDS using FL* | Wardana & Sukarno, ACM CSUR **57(4) Art. 88**, Dec 2024, DOI **10.1145/3701724** | **No** |
| *Survey of federated learning in intrusion detection* | JPDC, DOI 10.1016/j.jpdc.2024.104976 | **No** |
| *Model optimization in federated IDS: a survey of **aggregation**, communication, and personalization* | *J. Cloud Computing*, DOI **10.1186/s13677-026-00901-w**, Apr 2026 | **No** |

The last one is the most on-point survey in existence for this claim — its entire first axis is aggregation in FL-IDS — and it postdates the master doc's compilation. **This pass could not read any of the six taxonomies in primary text** (ACM DL returned HTTP 403; Springer redirected to an identity provider). So the "no survey lists a gradient-refinement method" assertion is currently supported by secondary descriptions only. Given that FedPGO exists and postdates most of them, the more defensible framing is comparative — *"surveys of FL-IDS aggregation organize the field around size-, loss-, similarity- and cluster-based weighting; gradient-refinement approaches appear only in the most recent work (FedPGO 2026, NIDS-FGPA 2024)"* — rather than absolute.

### 6.4 The claim, restated honestly

> Adaptive aggregation for federated intrusion detection is an active and, as of 2026, **occupied** space. The closest work is FedPGO (Computer Networks 2026), which reweights clients using a gradient-orthogonality score for IoT IDS on CIC-IDS2017 — but with a personalization objective and a parameter-split architecture. In general federated learning, CELM (2026) pursues the same minority-class objective as FedGRAA through data-free logit-maximization probing, FedAWARE reweights to maximize gradient diversity with convergence guarantees, and Jiang et al. (2024) and AdaAggRL (AAAI 2025) reach related conclusions via reinforcement learning. **FedGRAA's residual contribution is narrow and should be stated narrowly:** a single-global-model, γ-tunable, gradient-refined aggregation rule whose explicit objective is per-class recall on rare attack types, evaluated on rare-attack detection rather than aggregate accuracy. Every one of the above must appear in Related Work; the claim to defend is *"a different point in an occupied design space, with an evaluation axis nobody else reports,"* not *"an unoccupied intersection."*

That is a defensible PAKDD-tier claim. It is not an A\*/A-tier claim without the additional work in §12.

---

## 7. Deepened gap area A — convergence theory for adaptive-weight FL

Master-doc gap #1 asks for a convergence theorem covering the non-convex tanh-MLP. The literature now supplies three usable templates, in increasing order of fit.

### 7.1 Derive the weights *from* the bound (FedAAW)

**FedAAW** (Knowledge-Based Systems 2024, DOI 10.1016/j.knosys.2024.112484) — ✅ confirmed this pass, upgraded from ❓ — does the thing FedGRAA should arguably do: its **aggregation weights are theoretically derived by minimizing the convergence upper bound of FL on non-IID data across nodes**, yielding a tighter bound. It also reports ~3× faster time-to-target-accuracy than adaptive-local-aggregation FL and negligible added communication.

This reframes the theory gap productively. FedGRAA currently proposes weights heuristically and then needs a theorem that tolerates them. FedAAW inverts that: choose the weights that minimize the bound. If FedGRAA's γ-tuned minority-boosting weights can be shown to *approximate* a bound-minimizing choice under a rare-class-weighted objective, the theorem stops being a bolt-on and becomes the derivation. This is the single highest-leverage theoretical direction found this pass.

### 7.2 Optimize the weights jointly, with guarantees (Parsa et al.)

**arXiv:2511.03529** treats aggregation weights as learnable parameters jointly optimized with model parameters, solved by **alternating minimization with strong convergence guarantees under adversarial attack**. Structurally this is FedGRAA's mechanism (server-side weights updated by gradient information) given a formal optimization framing. Two consequences: (a) it is a ready-made proof architecture to adapt; (b) it is prior art — FedGRAA cannot claim the *idea* of gradient-optimized aggregation weights, only the rare-class objective and IDS instantiation.

### 7.3 Fix the heterogeneity measure (FedAWARE)

**arXiv:2310.02702** argues that the FL theory/practice inconsistency comes from the *heterogeneity measures* used in analysis, introduces **client consensus dynamics**, and proves FedAvg handles heterogeneity when the aggregation strategy is appropriate. Relevant because FedGRAA's setting (one attack type per client) is an extreme-heterogeneity regime where standard bounded-dissimilarity assumptions are close to vacuous. Adopting a consensus-dynamics-style measure may be what makes any FedGRAA theorem non-trivial rather than technically-true-but-empty.

### 7.4 Practical assumption set

Across FedAAW, FedAWR, FedDUAL and the asynchronous-compression convergence literature, the recurring assumption set for non-convex adaptive-weight FL is: **L-smooth, possibly non-convex local objectives; bounded stochastic-gradient variance; bounded learning rates; partial participation; and — critically — aggregation weights bounded away from zero and from above.** That last condition is the theoretical statement of the simplex projection FedGRAA's Algorithm 2 omits. The empirical divergence in [`fedgraa_convergence_defect.md`](fedgraa_convergence_defect.md) and the missing theorem are therefore **the same defect seen from two directions**, which is a clean, honest framing for the paper: the projection is required for the proof, and its absence is observable in the runs.

Combined with master-doc §7.3's note that FedGRAA-v2 needs one new bounded-estimation-error assumption `ζ`, the target theorem is: *PL-condition convergence to a noise ball under bounded weight deviation and bounded composite-gradient estimation error.* All four ingredients now have citable precedent.

---

## 8. Deepened gap area B — adversarial robustness of reweighting aggregators

Master-doc gap #3 asks for label-flipping / model-poisoning / pill-style evaluation. The specific concern the master doc raises in §4.2 — *"gradient-driven reweighting is a manipulable signal"* — is now directly addressed in the literature.

### 8.1 The mechanism-class analysis exists

Parsa et al. (arXiv:2511.03529, §7.2) analyse **Byzantine resilience of a learnable-aggregation-weight objective** and show robustness in settings with highly heterogeneous data and a large proportion of malicious clients. This is the closest thing to a security analysis of FedGRAA's mechanism class. FedGRAA should either adopt its threat model or explain why a different one applies.

### 8.2 The in-domain baseline exists, on FedGRAA's own datasets

**SecFedNIDS** (FGCS 2022, §5) is the reference point: FL-NIDS poisoning defense, evaluated on **UNSW-NB15 and CIC-IDS2018**, public code. Alongside it, **WeiDetect** (JISA 2025, Weibull-distribution poisoning defense for FL-NIDS — resolved from ❓ in §2) and *"Parameterizing poisoning attacks in federated learning-based intrusion detection"* (**ARES 2023**, DOI 10.1145/3600160.3605090) give attack-side and defense-side in-domain anchors.

### 8.3 The structural tension FedGRAA must address head-on

This is the sharpest reviewer risk in the whole corpus, and it deserves stating precisely.

Poisoning defenses identify malicious clients by **statistical deviation** — MESAS, CrowdGuard, FreqFed and AdaAggRL all operate on the premise that anomalous updates are suspect; AdaAggRL states it directly (*"benign clients exhibit significantly higher data distribution stability than malicious clients"*). FedGRAA's founding premise is that **the most valuable clients are precisely the divergent, high-loss, statistically anomalous ones**. These two premises are in direct conflict.

FedGRAA therefore has an obligation the master doc understates: it is not merely *undefended* against poisoning, it is a mechanism that **structurally amplifies exactly the signal a poisoning defense would suppress**. An adversary who can make their update look like a rare-class contribution gets up-weighted by construction. Jiang et al. (Neurocomputing 2024) is important here for a reason beyond the philosophical overlap the master doc notes: their *filtering-and-voting* layer is a proposed **resolution** of this tension, not just a parallel argument. FedGRAA's omission of any such layer is a design decision that now needs defending, not just disclosing.

A minimum credible robustness section: label-flipping and model-poisoning under the SecFedNIDS protocol; an adaptive attacker that fabricates rare-class gradient signatures to farm up-weighting (the mechanism-specific attack, which nobody else can run for you); AutoAdapt cited as the adaptive-threat limit; and an explicit statement of whether the γ coefficient trades detection performance against poisoning exposure. **SoK: Benchmarking Poisoning Attacks and Defenses in Federated Learning** (arXiv:2502.03801) supplies a standard harness.

---

## 9. Deepened gap area C — datasets and benchmarks

### 9.1 The critique is stronger than "legacy"

Per the NetFlow-benchmark literature (El Mahdaouy, Ait Yahia, Oualil & Berrada, arXiv:2602.05594, Feb–Mar 2026 — ✅ title/authors/date confirmed from the arXiv abstract page; the six-flaw enumeration below is from secondary description of the paper's dataset analysis and **should be checked against the full PDF before citing**): CIC-IDS-2017 and CSE-CIC-IDS-2018 contain **labelling errors** and **unrealistic temporal separation between attack and benign traffic**, and seven popular benchmarks share six design flaws — limited feature diversity, high feature interdependence, ambiguous ground truth, collapse of traffic distributions, artificial diversity generation, labelling inaccuracies — which **correlate with poor generalization in operational deployments**.

This matters beyond citation hygiene. FedGRAA's headline result is per-minority-class accuracy on CIC-IDS-2017's rarest classes — Heartbleed, Infiltration, Web/SQLi. Those are precisely the classes where a handful of labelling errors moves the reported number materially, and master-doc §7.3 already observes that Heartbleed has ≈1 held-out sample. A reviewer who knows this literature will read "≈88% on minority classes, +55% over FedAvg" and ask what the confidence interval is on a class with single-digit support. **The ≥10-seed / Wilcoxon / Holm–Bonferroni infrastructure already built in Stage B (`fedgraa/stats.py`) is the answer, and it should be foregrounded rather than treated as a checklist item.**

### 9.2 The concrete fix: NetFlow v2/v3 as a native federated benchmark

Sarhan, Layeghy & Portmann, *"Towards a Standard Feature Set for Network Intrusion Detection System Datasets,"* **Mobile Networks and Applications 2022**, arXiv:2101.11315, re-express four major NIDS datasets in one shared **43-feature** NetFlow schema: **NF-UNSW-NB15-v2, NF-BoT-IoT-v2, NF-ToN-IoT-v2, NF-CSE-CIC-IDS2018-v2**, plus the merged **NF-UQ-NIDS-v2**. The UQ Cyber Research Centre published **NetFlow V3** in 2025, adding 10 temporal features (53 total). Related: *Benchmarking the Benchmark — Analysis of Synthetic NIDS Datasets* (arXiv:2104.09029) and *Temporal Analysis of NetFlow Datasets for NIDS* (arXiv:2503.04404).

**Why this is the single best experimental upgrade available to FedGRAA.** A shared feature schema across four independently-collected datasets *is* a cross-silo federated setting: assign one dataset per silo and the non-IID structure is real rather than synthetically partitioned — no Dirichlet-α simulation, no one-attack-per-client artifice, and directly responsive to master-doc §7.1's objection that the current partition presupposes server-side knowledge of client label distributions. It simultaneously answers gap #4 (modern datasets), gap #5 (scale and realistic non-IID), and the deepest structural criticism in §7.1. Cost: a feature-extraction pass, no algorithmic change.

FLNET2023 (already in the master doc) remains the FL-native option, but is low-citation and narrower; the NF-v2 route is better supported and gives four attack landscapes rather than one.

---

## 10. Deepened gap area D — privacy-compliant validation and data-free contribution estimation

### 10.1 A named research line, converging on FedGRAA-v2's premise

Master-doc §4.10 catalogues six approaches to validating aggregation without a server-held labelled set. This pass shows the area has consolidated into a **named research line — data-free contribution estimation** — whose motivating statement is the master doc's own §7.1 critique, arrived at independently. From CELM's abstract: contribution must be estimated *"without relying on self-reported metadata or server-side validation data,"* because *"metadata-driven weighting can be unreliable or strategically manipulated, while validation-based scoring may be unavailable, biased, or costly to maintain."*

That is FedTruth's and CoAst's argument, sharpened, in 2026, from a group publishing at CVPR. Three consequences for Stage C:

1. **Option B in master-doc §10 is validated as a real problem** — the field independently reached the same critique. Good news for the framing.
2. **Option B is no longer uncontested** — the master doc's §10 table says Option B is attractive partly because it is "less crowded." CELM, SpectralFed/SpectralFuse and PACT/FedPRS/CoAst make that assessment out of date. It is not crowded like the few-shot/prototype family, but it is no longer empty.
3. **The comparison shifts from "does this work?" to "is exact worth 6×?"** FedGRAA-v2's federated backward pass recovers the composite-loss gradient **exactly**, at ≈6× communication (master-doc §7.3). CELM/SpectralFed estimate contribution **approximately** at essentially zero extra communication. That is now the experiment that matters, and it is the strongest empirical result available on this angle: *does exactness buy enough minority-class recall to justify 6× traffic, versus a data-free approximation?* Either answer is publishable, and running it converts a redesign into a finding.

### 10.2 A methodological result: the descriptive-phrase retry cuts both ways

The 2026-07-25 re-audit added a sub-step to the escalation path — *retry DBLP with a descriptive phrase before concluding non-existence* — after a literal-acronym search nearly condemned the real Fed-LAA. This pass is the clean validation of that rule, in both directions from a single sitting:

- **FedPGO** was described consistently across two independent searches with **zero links actually titled FedPGO** — textbook pFedCross/CT-BRFL signature. A descriptive-phrase DBLP query (`gradient orthogonalization personalized federated intrusion detection`) returned it immediately as an exact-title hit in *Computer Networks*. Under acronym-only search it would have been filed as a hallucination — **and it is the most consequential paper in this pass.**
- **Light-FFSL, FedCCW and FedHDE** were given the identical broadened treatment and returned nothing but adjacent real papers. They fail on stronger evidence than any earlier ❌ verdict in this corpus.

The rule should be promoted from a footnote in the re-audit doc to the master doc's standing §3.3 procedure, with this pass cited as the case where it changed a strategic conclusion rather than a bibliography line.

---

## 11. Devil's advocate checkpoint

Mandatory adversarial pass over this document's own conclusions.

**Challenge 1 — "You are overstating FedPGO. Personalized FL is a different problem from single-global-model minority-boosting."**
Partly fair. FedPGO's parameter-split personalization genuinely differs from FedGRAA's single global model, and its reported objective is accuracy/robustness, not per-class rare-attack recall. But the overlap is in the aggregation rule, which is FedGRAA's contribution, and on FedGRAA's own dataset. A reviewer does not need the objectives to match to ask "how does your gradient-based reweighting differ from theirs?" — and right now the paper has no answer, because it does not know the paper exists. **Verdict: the finding stands; the framing in §6.4 already concedes the distinction.**

**Challenge 2 — "The A\*/A qualifier is legitimate. Venue tiers are how the field measures novelty."**
Weak, and weaker here than usual. The submission target is CORE **B** (master-doc §6, twice-confirmed). An argument that restricts the prior-art frame to venues above the target venue's own tier invites exactly the rebuttal it deserves. **Verdict: drop the venue qualifier from the novelty claim; keep it only in the positioning discussion where it belongs.**

**Challenge 3 — "You did not read FedPGO or CELM in full. You are reasoning from abstracts and DBLP records."**
Correct, and a real limitation. FedPGO is paywalled (Elsevier); CELM's abstract was fetched verbatim but the full PDF was not analysed. Mechanism descriptions for FedPGO come from consistent secondary summaries corroborated by an exact-title DBLP record — enough to establish *existence, venue and topic* with confidence, and enough to establish *mechanism* provisionally. It is **not** enough to characterize their limitations or to write a differentiation paragraph. **Both papers must be read in full before Related Work is finalized** — flagged in §12 as a blocking item, not a nice-to-have.

**Challenge 4 — "Six surveys and you read none of them. The taxonomy-gap claim is unsupported in both directions."**
Correct, and stated as such in §6.3. This cuts against the master doc's claim more than against this document's, since this document downgrades the claim rather than relying on it. **Verdict: acknowledged limitation; the comparative reframing in §6.3 is the mitigation.**

**Challenge 5 — "Three new hallucination verdicts on one afternoon's searching is a high false-positive risk, given FedPGO nearly got the same verdict."**
The strongest challenge here, and the reason §10.2 exists. Mitigation: every ❌ this pass got the same broadened descriptive-phrase treatment that rescued FedPGO and Fed-LAA, plus a DBLP query, and in each case the broadened search returned *specific, real, differently-named papers* that plausibly explain the conflation (FedCBA/FedCW/FedLWS for FedCCW; FedHGN/FedPH for FedHDE; EdgeDetect/FedMADE for Light-FFSL). That is a positive explanation of the error, not merely absence of evidence. **Residual risk acknowledged: all three came from automated survey tooling, and none is load-bearing for any claim — the correct action is "do not cite," not "conclusively does not exist."**

**Challenge 6 — "Recommending a whole new dataset family (NF-v2) is scope creep for a Stage C decision."**
Fair on timing, not on substance. It is a data-preparation task, not an algorithmic one, and it answers three separate master-doc gaps at once. **Verdict: correctly scoped as an option in §12, sequenced after the two blocking reads.**

**Checkpoint verdict: PASS with two conditions** — (i) §6.4's narrowed claim replaces the master doc's absolute claim; (ii) the full-text reads of FedPGO and CELM are treated as blocking before any Related Work is written.

---

## 12. Revised positioning and updated gap-closing checklist

### 12.1 The differentiation paragraph, rewritten

Replacing master-doc §8's sentence, which predates FedPGO, CELM, FedAWARE and AdaAggRL:

> Adaptive aggregation weighting has converged on three signals: alignment with the global direction (FedAWA), client loss or accuracy (FedNolowe, DWAFL, PACT), and statistical outlier filtering (MESAS, CrowdGuard, FreqFed). In imbalanced intrusion detection these systematically suppress the divergent, high-loss, minority-class clients that hold the rarest attacks. A counter-current has emerged: FedAWARE maximizes gradient diversity rather than alignment; Jiang et al. (2024) and AdaAggRL (AAAI 2025) learn weights that protect rare contributions through reinforcement learning; CELM (2026) up-weights clients with discriminative evidence for underrepresented classes without any server-side data. In intrusion detection specifically, FedPGO (2026) reweights clients by gradient orthogonality on CIC-IDS2017, and NIDS-FGPA (2024) weights by gradient similarity for communication efficiency. FedGRAA sits inside this counter-current, at a point none of them occupies: a **single-global-model** aggregation rule with an explicit **γ-tunable trade-off** between rare-class emphasis and conventional size-proportional weighting, evaluated on **per-class recall for rare attack types** rather than aggregate accuracy or personalization gain. Its claim is not that the space is empty, but that this point in it is unreported and that the rare-attack evaluation axis is one no adjacent work supplies.

### 12.2 Revised checklist (deltas against master-doc §9)

| # | Item | Change |
|---|---|---|
| **0 — new, blocking** | **Read FedPGO (DOI 10.1016/j.comnet.2026.112210) and CELM (arXiv:2605.18892) in full** before any Related Work is drafted | New. Nothing else in Stage C should be finalized first |
| 1 | Convergence theorem | **Reframed** — derive weights from the bound (FedAAW) rather than proving a theorem around heuristic weights; adopt the Parsa et al. alternating-minimization framing; use a consensus-dynamics-style heterogeneity measure (FedAWARE); state bounded-weight (= simplex) explicitly as a proof condition, tying it to the observed divergence defect (§7) |
| 2 | Head-to-head baselines | **+FedPGO** (highest priority — same domain, same dataset, same mechanism family), **+CELM**, **+FedAWARE**, **+AdaAggRL**, **+PerFedHypID**, **+FedKSam\***, **+AMAFed**, **+G-PFL-ID**. Retain the master doc's existing set |
| 3 | Adversarial robustness | **Sharpened** — the premise conflict in §8.3 must be argued, not just disclosed. Baselines: SecFedNIDS (same datasets, public code), WeiDetect, ARES-2023 attack parameterization, SoK harness (arXiv:2502.03801). Add the mechanism-specific attack: a fabricated rare-class gradient signature farming up-weighting |
| 4 | Modern datasets | **Upgraded** — NF-\*-v2 / NF-UQ-NIDS-v2 / NetFlow-V3 as a native cross-silo federated benchmark (§9.2), not just "add CICIoT2023." Also cite the CIC-IDS-2017 labelling-error critique when reporting rare-class numbers |
| 5 | Scale + extreme non-IID | **Partly subsumed by #4** — the NF-v2 route supplies real rather than simulated heterogeneity |
| 6 | Ablation of γ and the refinement components | Unchanged; Stage B infrastructure exists |
| 7 | Overhead reporting | **Reframed** — report ≈6× against CELM/SpectralFed's near-zero-cost approximation, and make "is exact worth 6×?" an experiment rather than a caveat (§10.1) |
| 8 | Statistical rigor | **Promoted** — no longer a checklist item but the direct answer to the labelling-error/low-support critique in §9.1. Lead with it for rare classes |
| 9 | Privacy fix (FedGRAA-v2) | **Narrowed** — drop "first fully FL-compliant"; position against the data-free contribution-estimation line, with exactness-vs-cost as the contribution |
| 10 | Prototype hybrid | Unchanged |

### 12.3 Effect on the Stage C option choice

Master-doc §10 recommends Option B (privacy redesign) as closer to implementation-ready than Option A (few-shot/prototype hybrid). **That recommendation survives this pass, with its rationale changed.** Option B is no longer attractive because the space is uncrowded — it is not (§10.1). It is attractive because §10.1 converts it into a *comparative experiment with a publishable result either way* (exact federated validation vs. data-free approximation), on top of an implementation that master-doc §7.3 has already mapped function-by-function against the live codebase.

The genuinely new consideration is that **F4/§9.2 opens a third, cheaper path**: the NF-v2 cross-silo benchmark answers three catalogued gaps with data engineering and no algorithmic risk. If Stage C is time-constrained, that is the highest ratio of reviewer-objections-closed to effort available — and it composes with either Option A or Option B rather than competing with them. Worth putting to the mentor alongside the existing three.

---

## 13. Edits to apply to the master doc

None of these have been applied — this document does not modify [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md).

1. **§1 executive summary** — replace the "unoccupied intersection" claim with §6.4's narrowed statement. This is the most important single edit in the list.
2. **§1 findings list** — add FedPGO as the new lead finding, ahead of the current finding 11 (Jiang et al.); add CELM as a second new finding.
3. **§3.3 / §4.10** — resolve the PACT contradiction (C1): tag ✅ in both places.
4. **§4.6, §6, §11** — correct FedP3E to a CoRR/arXiv preprint (C2, C3); add the author list.
5. **§4.4** — add rows for FedPGO, PerFedHypID, FedKSam\*, G-PFL-ID, AMAFed; add the DOI for the Chunduru et al. rare-attack paper (C5).
6. **§4.7** — add FedAWARE, CELM, AdaAggRL, and Parsa et al.'s learnable-aggregation-weights paper; note the FedAWA/FedAWARE homonym as a name-collision warning in §3.1.
7. **§4.10** — add CELM and SpectralFed/SpectralFuse under a new "data-free contribution estimation" heading; update the Ukaye et al. row from ❓ to ✅ with its arXiv ID.
8. **§4.11 / new subsection** — add the convergence-theory cluster (FedAAW's bound-minimizing derivation, Parsa et al., FedAWARE) and the robustness cluster (SecFedNIDS, WeiDetect, ARES 2023, SoK arXiv:2502.03801).
9. **§4.12** — add the three missing surveys and DOIs for all six (C6).
10. **§5 datasets** — add the NF-\*-v2 / NF-UQ-NIDS-v2 / NetFlow-V3 family with the Sarhan et al. citation, and a note on the CIC-IDS-2017 labelling-error critique.
11. **§8** — replace the differentiation paragraph with §12.1.
12. **§9** — apply the §12.2 deltas, including the new blocking item 0.
13. **§10** — update the Option B "less crowded" rationale per §12.3 and add the NF-v2 benchmark path as a fourth option.
14. **§11 bibliography** — add all ✅ items from §2 and §5; move Light-FFSL, FedCCW and FedHDE to the ❌ list with the §3 evidence; upgrade the eleven ❓ entries closed in §2.
15. **§3.3 procedure box** — promote the descriptive-phrase retry to a numbered step, citing §10.2 as the case where it changed a strategic conclusion.

---

## 14. Method, limitations, and disclosure

**Search strategy.** Iterative, four-track: (a) targeted acronym-and-metadata verification of every ❓ entry; (b) adversarial saturation queries constructed to *falsify* rather than confirm the novelty claim, varying each of the three axes independently; (c) thematic sweeps on the four thin areas; (d) internal-consistency audit of the master doc against itself. Tools: general web search, arXiv API (`export.arxiv.org/api/query`), DBLP publication-search API (`dblp.org/search/publ/api`), PubMed, and direct publisher/DOI fetches.

**Tools that failed, and how that constrains conclusions.** ACM Digital Library returned **HTTP 403** on full-text fetches; SpringerLink **303-redirected to an identity provider** on both the FSFL-IDS chapter and the two 2026 surveys; ScienceDirect returned **403**; the sandboxed shell has **no network access**, so all retrieval went through WebFetch/WebSearch; DBLP intermittently returned **429/ECONNRESET** under parallel load, requiring serialized queries. Consequence: **no paywalled full text was read this pass.** Every mechanism description for a paywalled paper (FedPGO, the six surveys, FedAAW, the arXiv:2602.05594 six-flaw enumeration) rests on abstracts, DBLP/PubMed records, or consistent secondary summaries — sufficient for existence, venue and topic; **insufficient for characterizing limitations or writing differentiation prose.** This is why §12.2 item 0 is blocking.

**Verification discipline.** Following the master doc's own recorded process failure (§12 of that document): **every ✅ in this document corresponds to an actual tool call made during this pass**, not to recollection later checked. Where a fact is derived rather than fetched — the Chunduru DOI in §2, the six-flaw enumeration in §9.1 — it is labelled as derived at the point of use. Where a claim rests on secondary description — FedPGO's mechanism, the survey taxonomies — that is stated inline rather than in a footnote. One item, "Precision-Weighted FedAvg" (§5), is recorded explicitly as an *unverified search lead* rather than promoted to a reference, per the gray-zone-equals-fail rule.

**Coverage limitations.** Fed-Meta FD remains ❓ (deprioritized). Citation counts were not verified for any 2025–2026 paper — Semantic Scholar was rate-limited (HTTP 429) in the previous pass and was not retried here. Non-English literature was not searched. Papers published after ~2026-07-25 are out of scope by construction. The saturation test cannot prove absence — it establishes only that a targeted adversarial search across three tools failed to find further occupants of the intersection *beyond the two it did find*, which is a materially weaker guarantee now that the intersection is known to be occupied.

**AI disclosure.** This document was produced with AI-assisted research tooling (Claude Code, ARS `deep-research` skill) for literature retrieval, cross-source verification and synthesis. All bibliographic claims were checked against primary or authoritative bibliographic sources as described above; all judgments about strategic implications for FedGRAA are interpretations, offered as such, and should be reviewed by the author before acting on them. No text was copied from any source; abstracts quoted verbatim are marked with quotation marks and attributed.

---

*Compiled 2026-07-25. Companion to [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) and [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md). Next trigger: after the blocking full-text reads of FedPGO and CELM, and again immediately before submission.*
