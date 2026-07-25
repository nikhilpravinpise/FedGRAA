# FedGRAA Follow-up Research

> Research notes answering the three concerns raised about the FedGRAA paper
> (*"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection
> Systems"*), plus a framework survey and an exhaustive catalog of the
> `mtuann/federated-learning-updated-papers` collection.
>
> Date: 2026-06-28

---

## 0. Why these questions matter (recap of the FedGRAA assumptions)

FedGRAA (as reproduced in this repo) makes **two assumptions that conflict with
standard FL privacy principles**:

1. **It assumes the server knows which classes are rare on which clients.** The
   partition is "one attack type per client" and the server *protects the last
   `n` rare classes* by name (Heartbleed, SQL Injection, Infiltration, ...). In
   real FL the server cannot see a client's label distribution, so hard-coding
   "these classes are rare" violates the FL assumption that raw data / label
   statistics stay on the device.
2. **It holds a server-side validation set `D_V`** (10% from every client,
   further split into `D_V^h`, `D_V^m`, `D_V^n` by class frequency). Collecting
   labelled validation data centrally is exactly what FL is supposed to avoid;
   it also requires the server to *know the class of every validation sample*.

The three tasks below research how the literature handles each of these without
breaking FL privacy, and how the FedGRAA design could be made compliant.

---

## 1. Detecting client class imbalance *without* violating FL privacy

**The core problem.** The server must decide *which classes are under-represented*
to drive the adaptive weighting — but it is not allowed to look at client data or
even at explicit per-client label counts.

### 1.1 Key paper — inferring composition from gradients (the cleanest answer)

**Wang, Xu, Wang, Zhu — "Addressing Class Imbalance in Federated Learning",
AAAI 2021** (arXiv:2008.06217).
- Proposes a **monitoring scheme** that *infers the composition (class
  distribution) of the training data used in each FL round directly from the
  model updates / gradients* — the server never sees raw data or explicit label
  counts, so **client privacy is preserved**.
- The intuition it exploits: the **gradients of the output (softmax) layer encode
  label frequency**. For cross-entropy loss, the gradient w.r.t. the logits is
  `(softmax(z) − y)`; summed over a batch, the per-class gradient of the output
  bias is proportional to `(expected_count − actual_count)` of each class. The
  server can therefore solve for the class proportions behind an update without
  ever receiving labels.
- It then introduces a **Ratio Loss** that uses the inferred composition to
  down-weight over-represented classes and mitigate imbalance.
- Headline result: significantly outperforms prior imbalance methods **"while
  maintaining client privacy."**

> **This is the single most relevant reference for FedGRAA.** It shows you can
> *estimate* which classes are rare from the updates the server already receives,
> replacing FedGRAA's hard-coded "last `n` rare classes" with an
> **inferred** rare-class set — fully FL-compliant.

### 1.2 The same trick from the gradient-leakage literature (use it constructively)

A parallel line of work shows label/label-distribution recovery from gradients:
- **Zhao, Mopuri, Bilen — "iDLG: Improved Deep Leakage from Gradients" (2020):**
  proves the **ground-truth label of a single example is exactly recoverable from
  the sign of the output-layer gradient.**
- **Geiping et al. — "Inverting Gradients" (NeurIPS 2020)** and **Yin et al. —
  "See through Gradients / GradInversion" (CVPR 2021):** recover label *sets* and
  per-class counts from aggregated gradients.

These are usually framed as *attacks*, but the same math is what lets a **server
legitimately estimate the global / per-update class histogram** for adaptive
aggregation — without the client ever transmitting labels. (Caveat: under secure
aggregation or DP noise the estimate degrades, which is the privacy/utility
trade-off to discuss in the paper.)

### 1.3 Self-balancing without central data — Astraea

**Duan et al. — "Astraea: Self-balancing Federated Learning", IEEE ICCD 2019**
(arXiv:1907.01132).
- Introduces a **mediator** that reschedules clients to balance training based on
  the **KL divergence (KLD) of their local class distributions**, plus global
  data-distribution-aware augmentation.
- Relevance & caveat: Astraea has clients **report their local label
  distribution** to the mediator — a *summary statistic*, not raw data. This is a
  weaker privacy posture than the gradient-inference approach (§1.1) but stronger
  than a server-side validation set. It is a good "middle-ground" baseline to
  contrast against.

### 1.4 Other relevant directions (from the mtuann collection, §4)

- **FedRW / FedRW-style reweighting** ("Efficient Privacy-Preserving Data
  Reweighting") — reweight classes/samples under privacy constraints.
- **DFedReweighting — "A Unified Framework for Objective-Oriented Reweighting in
  Decentralized FL"** — objective-driven reweighting without a central oracle.
- **FedNolowe / FedNola — normalized-loss-based weighted aggregation** — uses
  per-client *loss* (not data) as the signal for aggregation weight, which the
  server already observes.
- **Clustered FL / FedGPS (statistical rectification)** — infer heterogeneity
  structure from updates rather than from declared distributions.

### 1.5 What this means for FedGRAA (actionable)

To make FedGRAA privacy-compliant, **replace the two oracle inputs**:

| FedGRAA oracle (privacy-violating) | FL-compliant replacement |
|---|---|
| Hard-coded "last `n` rare classes" | **Infer rare classes** from output-layer gradient histogram each round (Wang et al. 2021) |
| Per-client labelled `D_V` on server | Estimate `L_h / L_m / L_u` proxies from the **inferred composition** + **client-reported scalar losses**, or move evaluation to clients (see §3) |

This keeps the *gradient-refined adaptive weighting* idea intact while removing
the need to "know" the imbalance a priori.

---

## 2. FL frameworks — Flower and the alternatives

Source: Apheris, *"Top 7 Open-Source Frameworks for Federated Learning"*
(GitHub star/contributor counts as of April 2024), plus the Flower paper
(Beutel et al., arXiv:2007.14390).

| Framework | Maintainer | Best for | Notes |
|---|---|---|---|
| **Flower (`flwr`)** | Adap / Flower Labs | **Research + prototyping; framework-agnostic** | PyTorch, TF, scikit-learn, **"even raw NumPy for users who enjoy computing gradients by hand"** — directly fits this repo's from-scratch NumPy model. Scales to ~15M simulated clients; same code runs in simulation and on real devices. **Strongest fit for FedGRAA.** |
| **NVIDIA FLARE** | NVIDIA | Production, security/governance | DP, homomorphic encryption, FL simulator, MONAI/HF integration, security-hardened. |
| **TensorFlow Federated (TFF)** | Google | Research on TF models | Federated Core (FC) + FL API; simulation-focused. |
| **PySyft / PyGrid** | OpenMined | Privacy research (DP, encrypted compute) | Computation over data "you cannot see"; needs PyGrid for networking. |
| **FATE** | WeBank | Enterprise cross-silo | Standalone + cluster; uses protocol buffers. |
| **OpenFL** | Intel / Linux Foundation | Cross-silo, certificate-secured | mTLS, FL Plan (YAML), Docker-isolated. |
| **Substra** | Owkin / Linux Foundation | Healthcare, traceability | Trusted execution enclaves, immutable ledger; heavy Kubernetes. |

### 2.1 Why Flower is the recommended choice here

- **Framework-agnostic**, including **raw NumPy** — the FedGRAA model in this repo
  is hand-written NumPy, so it can plug into a Flower `NumPyClient` with no
  TensorFlow dependency (which the repo avoided due to Python 3.13).
- **Custom `Strategy` API**: FedGRAA's server-side adaptive weight learner
  (Algorithm 2) maps cleanly onto a custom `Strategy.aggregate_fit()` — you
  override aggregation, which is exactly what FedGRAA changes vs. FedAvg.
- **Federated evaluation** built in (`Strategy.aggregate_evaluate()` /
  `evaluate_fn`) — this is the mechanism needed for §3 (validate on clients, not
  on a server-side set).
- **Simulation Engine** lets you run all `K=14`/`K=9` clients on one machine,
  then migrate to real devices unchanged.

> Recommendation: keep the NumPy core, wrap clients in `flwr.client.NumPyClient`,
> and implement FedGRAA as a custom `flwr` `Strategy`. This turns the repo's
> "from-scratch loop" into a standards-compliant FL system and makes the privacy
> fixes from §1 and §3 natural to express.

---

## 3. Validating under class imbalance *without* a server-side validation set

**The problem.** FedGRAA scores the global model on a central `D_V` split into
`D_V^h/D_V^m/D_V^n`. Holding labelled validation data centrally is not FL-compliant.

### 3.1 Federated (distributed) evaluation — the standard answer

The original FL paper (**McMahan et al., "Communication-Efficient Learning of
Deep Networks from Decentralized Data", AISTATS 2017**) already evaluates by
**broadcasting the global model and having each client compute metrics on its own
local held-out data**, then aggregating the metrics on the server. No raw data and
no labels leave the device.

- The server receives only **scalar metrics** (loss, per-class accuracy/recall,
  confusion-matrix counts), not data.
- Frameworks that implement this directly: **Flower** (`aggregate_evaluate`),
  **TFF**, **FLARE**, **LEAF** benchmark's distributed evaluation.

### 3.2 Metrics that are valid under imbalance (and computable locally)

Accuracy is misleading under imbalance; report instead, aggregated across clients:
- **Per-class recall / precision / F1** and **macro-F1** (treats rare classes
  equally to frequent ones).
- **Balanced accuracy** (mean per-class recall).
- **Matthews Correlation Coefficient (MCC)** and **G-mean** — common in IDS papers.
- **Confusion-matrix counts** summed across clients → exact global per-class
  metrics without centralizing data.

For FedGRAA specifically, the **rare-class recall** (its "last-`n` accuracy") can
be computed **per client locally** and **summed** on the server — giving the same
quantity the paper reports, but FL-compliant.

### 3.3 How imbalance-aware FL papers validate (from the collection, §4)

- **FedCAR, FedRE, FedNolowe, Balanced coarse-to-fine FL** — evaluate with
  **distributed/federated evaluation** and report **per-class / macro metrics**
  rather than a centralized validation accuracy.
- **Long-tailed / non-IID FL papers** (e.g. "An Adaptive FL Algorithm with
  Attenuated Memory on Non-IID and Long-Tail Data", "Fine-Tuning Impairs the
  Balancedness of Foundation Models in Long-tailed Personalized FL") use
  **class-balanced test sets evaluated locally** and report **tail-class accuracy**.
- **FedGPS / UniVarFL / Geometric Knowledge-Guided alignment** — validate
  generalization on **held-out clients** (leave-client-out), another way to avoid
  a central validation set.

### 3.4 What this means for FedGRAA (actionable)

Two compliant options to replace `D_V`:

1. **Move the FedGRAA validation losses onto the clients.** Each client keeps its
   own 10% split, computes `L_h/L_m/L_u`-style losses **locally** on the
   broadcast candidate model, and returns **scalars**. The server's weight-learner
   (Algorithm 2) then differentiates a *surrogate* assembled from these scalars.
   (Note: the exact gradient `∂L'/∂w_k = ⟨∇_P L', P_k⟩` needs the gradient w.r.t.
   parameters; a federated variant has each client return that inner product or a
   sketch of it.)
2. **Replace `D_V` evaluation with inferred composition (§1.1) + reported losses**,
   so no labelled central set is ever assembled.

Either way, **report rare-class recall via summed local confusion matrices**, not
via a central `D_V^n`.

---

## 4. The `mtuann/federated-learning-updated-papers` collection

### 4.1 Scope note (please read)

The interactive site [`mtuann.github.io/papers`](https://mtuann.github.io/papers/)
indexes **25,664 "Federated Learning" papers** (and 2,264 in the curated
"FL – Awesome (by Yuwen Yang)" topic shown in your screenshots). **Listing every
one of ~25k entries verbatim is neither feasible in this document nor useful** —
the site itself exists to search/filter that volume, and the data is sourced from
IEEE Xplore, ACM, ScienceDirect, Springer, OpenReview, arXiv, DBLP, OpenAlex and
Google Scholar.

What *is* tractable and valuable is the repository's curated **"Federated Learning
Papers with Code"** list (the README), which is the set most relevant to
reproduction work like this project. The complete retrieved set (200 most-recent
entries, May 2026 snapshot) is cataloged below, grouped by the themes that matter
for the three questions above. Each row: *title — first author(s) — date — venue*.

> If you genuinely need the full 25k as data, the right move is to **scrape the
> site's backing JSON** (it loads papers client-side) rather than hand-transcribe;
> I can write that scraper on request.

### 4.2 Papers directly relevant to the three questions

**Class imbalance / non-IID / long-tail (Q1 & Q3):**
- Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized FL — Hou et al. — 2026-05-04 — arXiv
- An Adaptive FL Algorithm with Attenuated Memory on Non-IID and Long-Tail Data — Wu et al. — 2026-03-02 — INFORMS J. Computing
- FEDCADS: Robust FL via Dual Distillation and Participation-Aware Optimization Under non-IID Data — Lai et al. — 2026-04-21 — OpenAlex
- RFL-NLCP: Robust FL with non-IID Data and Limited Client Participation — Lai et al. — 2026-04-21 — OpenAlex
- Domain-Skewed FL with Feature Decoupling and Calibration — Wang et al. — 2026-03-15 — arXiv
- FedGPS: Statistical Rectification Against Data Heterogeneity in FL — Yang et al. — 2025-10-23 — arXiv
- Geometric Knowledge-Guided Localized Global Distribution Alignment for FL — Ma et al. — 2025-03-01 — CVPR
- UniVarFL: Uniformity and Variance Regularized FL for Heterogeneous Data — Gupta et al. — 2025-06-01 — AAAI
- Balanced coarse-to-fine FL for noisy heterogeneous clients — Han et al. — 2025-01-07 — Complex & Intelligent Systems
- Redefining non-IID Data in FL: Labels to Embeddings — Borazjani et al. — 2025-03-01 — arXiv
- ACSA: Adaptive Client-Specific Augmentation for Mitigating Data Heterogeneity — Milan K et al. — 2026-04-13 — J. Comp. & Cognitive Eng.
- RAFed: Responsive Augmentation and Approximate Update for FL with Non-IID Data — Di & Zhang — 2026-01-01 — OpenAlex

**Adaptive / reweighted aggregation (core FedGRAA theme):**
- FedVG: Gradient-Guided Aggregation for Enhanced FL — Devkota et al. — 2026-02-24 — arXiv
- DFedReweighting: Unified Objective-Oriented Reweighting in Decentralized FL — Zhang et al. — 2025-12-12 — arXiv
- FedNolowe: Normalized Loss-Based Weighted Aggregation for Robust FL — Le et al. — 2025-04-04 — bioRxiv
- FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation — Sun et al. — 2025-02-11 — OpenAlex
- Intrinsic Training Signals for FL Aggregation — Fiorini et al. — 2025-01-01 — arXiv
- FedPAW: FL with Personalized Aggregation Weights (vehicle speed) — He et al. — 2024-12-01 — IEEE TCC
- FedCAR: Cross-client Adaptive Re-weighting for Generative Models — Kim et al. — 2024-12-01 — arXiv
- FedRW: Efficient Privacy-Preserving Data Reweighting for FL of LMs — Ye et al. — 2025-11-01 — arXiv
- Beyond Aggregation: Guiding Clients in Heterogeneous FL — Wang et al. — 2025-01-01 — arXiv
- A Fast and Flat FL Method via Weighted Momentum and SAM — Li et al. — 2025-11-27 — NeurIPS 2025
- FedSWA: Momentum-Based Stochastic Controlled Weight Averaging — Liu et al. — 2025-01-01 — arXiv

**Intrusion detection / anomaly detection in FL (the FedGRAA application domain):**
- Stones From Other Hills: Intrusion Detection in Statistical Heterogeneous IoT by Self-Labeled Personalized FL — Lu et al. — 2025-01-01 — IEEE IoT Journal
- FADngs: Federated Learning for Anomaly Detection — Dong et al. — 2025-01-01 — IEEE TNNLS
- FedDLAD: FL Dual-Layer Anomaly Detection vs Backdoor Attacks — Ding et al. — 2025-01-01 — OpenAlex

**Privacy / fairness / contribution estimation (relevant to "don't peek at clients"):**
- Towards Explainable Privacy Preservation in FL via Shapley Value-Guided Noise Injection — Li et al. — 2026-04-21 — OpenAlex
- DSA: Direction and Sign Alignment for Contribution Evaluation in FL — Wang et al. — 2026-04-21 — OpenAlex
- Owen Sampling Accelerates Contribution Estimation in FL — KhademSohi et al. — 2025-01-01 — Frontiers AI
- Achieving Distributive Justice in FL via Uncertainty Quantification — Carey & Wu — 2025-04-01 — arXiv
- The Cost of Local and Global Fairness in FL — Duan et al. — 2025-01-01
- Accurate Target Privacy Preserving FL Balancing Fairness and Utility — Sun et al. — 2025-10-30 — INFOCOM 2026

**Frameworks / benchmarks / simulation (relevant to Q2):**
- BlazeFL: Fast and Deterministic FL Simulation — Azuma & Nishio — 2026-01-01 — arXiv
- FLEET: A FL Emulation and Evaluation Testbed — Abu Hamdan et al. — 2026-01-09 — OpenAlex
- OmniFed: A Modular Framework for Configurable FL from Edge to HPC — Tyagi et al. — 2025-01-01 — SC Workshops
- HtFLlib: Comprehensive Heterogeneous FL Library and Benchmark — Zhang et al. — 2025-06-01 — OpenAlex
- ByzFL: Research Framework for Robust FL — González et al. — 2025-01-01 — arXiv
- UltraFlwr: Efficient Federated Medical/Surgical Object Detection (Flower-based) — Li et al. — 2025-03-01 — arXiv
- FedFitTech: A Baseline in FL for Fitness Tracking — Oz et al. — 2025-01-01 — arXiv
- SmartFLow: Communication-Efficient SDN Framework for Cross-Silo FL — Abu Hamdan et al. — 2026-01-09 — OpenAlex
- Standardized Methods and Recommendations for Green FL — Tapp et al. — 2026-01-30 — arXiv

### 4.3 Full retrieved "Papers with Code" list (May 2026 snapshot, 200 entries)

<details>
<summary>Click to expand the complete list (title — authors — date — venue)</summary>

1. Federated Learning for Early Prediction of EV Charging Demand — Perifanis et al. — 2026-05-06 — arXiv
2. Fine-Tuning Impairs the Balancedness of Foundation Models in Long-tailed Personalized FL — Hou et al. — 2026-05-04 — arXiv
3. FedSLoP: Memory-Efficient FL with Low-Rank Gradient Projection — He et al. — 2026-04-27 — arXiv
4. Conditional Imputation for Within-Modality Missingness in Multi-Modal FL — Zheng et al. — 2026-04-25 — arXiv
5. FedSIR: Spectral Client Identification and Relabeling for FL with Noisy Labels — Gholami et al. — 2026-04-22 — arXiv
6. DCFL: Dual End Constraint FL with an Adaptive Analytic Anchor — Liu et al. — 2026-04-21 — OpenAlex
7. DSA: Direction and Sign Alignment for Contribution Evaluation in FL — Wang et al. — 2026-04-21 — OpenAlex
8. FEDCADS: Robust FL via Dual Distillation and Participation-Aware Optimization Under non-IID Data — Lai et al. — 2026-04-21 — OpenAlex
9. FedGPAI: Personalized FL Based on Parameter Sensitivity Adaptive Interpolation — Li et al. — 2026-04-21 — OpenAlex
10. RFL-NLCP: Robust FL with non-IID Data and Limited Client Participation — Lai et al. — 2026-04-21 — OpenAlex
11. Towards Explainable Privacy Preservation in FL via Shapley Value-Guided Noise Injection — Li et al. — 2026-04-21 — OpenAlex
12. A Novel Personalized FL Method for Privacy-Preserving Smart Mobile Health Monitoring — Chai et al. — 2026-04-20 — INFORMS J. Computing
13. DLR Federated Learning Logging Base — Heinrich & Franke — 2026-04-20 — Zenodo
14. Adaptive Client-Specific Augmentation (ACSA) for Mitigating Data Heterogeneity in FL — Milan K et al. — 2026-04-13 — J. Comp. & Cognitive Eng.
15. Unveiling the Security Risks of FL in the Wild: From Research to Practice — Chen et al. — 2026-03-21 — arXiv
16. Domain-Skewed FL with Feature Decoupling and Calibration — Wang et al. — 2026-03-15 — arXiv
17. FedBCD: Communication-Efficient Accelerated Block Coordinate Gradient Descent for FL — Liu et al. — 2026-03-05 — arXiv
18. An Adaptive FL Algorithm with Attenuated Memory on Non-IID and Long-Tail Data — Wu et al. — 2026-03-02 — INFORMS J. Computing
19. FedNSAM: Consistency of Local and Global Flatness for FL — Liu et al. — 2026-02-27 — arXiv
20. FedVG: Gradient-Guided Aggregation for Enhanced FL — Devkota et al. — 2026-02-24 — arXiv
21. SAGE: Sustainable and green energy-aware client selection for FL in IoT — Savoia et al. — 2026-02-23 — Applied Energy
22. SelfDistillCore: a FL framework with sparse updates and historical knowledge integration — Le et al. — 2026-02-13 — PeerJ CS
23. Dynamic FL: coping with label noise in real-time network data — Ouyang et al. — 2026-02-11 — J. King Saud Univ. CIS
24. TaxFL: FL and Federated Graph Intelligence for Cross-Border Tax Compliance — Frantz — 2026-02-10 — Zenodo
25. Standardized Methods and Recommendations for Green FL — Tapp et al. — 2026-01-30 — arXiv
26. FLEET: A FL Emulation and Evaluation Testbed for Holistic Research — Abu Hamdan et al. — 2026-01-09 — OpenAlex
27. SmartFLow: A Communication-Efficient SDN Framework for Cross-Silo FL — Abu Hamdan et al. — 2026-01-09 — OpenAlex
28. FedKDX: FL with Negative Knowledge Distillation for Enhanced Healthcare AI Systems — Pham et al. — 2026-01-08 — arXiv
29. FedHENet: A Frugal FL Framework for Heterogeneous Environments — Dopico-Castro et al. — 2026-01-01 — arXiv
30. TIP: Resisting Gradient Inversion via Targeted Interpretable Perturbation in FL — Wang & Su — 2026-01-01 — arXiv
31. RAFed: Responsive Augmentation and Approximate Update Method for FL with Non-IID Data — Di & Zhang — 2026-01-01 — OpenAlex
32. Quantization Impact on Accuracy/Communication Trade-off in FL for Aerospace Predictive Maintenance — Loukili — 2026-01-01 — arXiv
33. Med-MMFL: A Multimodal FL Benchmark in Healthcare — Chhetri et al. — 2026-01-01 — arXiv
34. HEART-PFL: Stable Personalized FL under Heterogeneity — Kim & Kim — 2026-01-01 — arXiv
35. Few-for-Many Personalized FL — Guo et al. — 2026-01-01 — arXiv
36. PTOPOFL: Privacy-Preserving Personalised FL via Persistent Homology — Vomo-Donfack et al. — 2026-01-01 — arXiv
37. FedDAP: Domain-Aware Prototype Learning for FL under Domain Shift — Le et al. — 2026-01-01 — arXiv
38. DriftGuard: Mitigating Asynchronous Data Drift in FL — Han et al. — 2026-01-01 — arXiv
39. Dawid-Skene-model-based label-noise mitigation for FL — Dong et al. — 2026-01-01 — Information Sciences
40. A secure cross-domain FL scheme based on blockchain fair payment — Li et al. — 2026-01-01 — PeerJ CS
41. DCInject: Persistent Backdoor Attacks via Frequency Manipulation in Personal FL — Birhan et al. — 2026-01-01 — arXiv
42. BlazeFL: Fast and Deterministic FL Simulation — Azuma & Nishio — 2026-01-01 — arXiv
43. Beyond Denial-of-Service: The Puppeteer's Attack in Ranking-Based FL — Chen et al. — 2026-01-01 — arXiv
44. FL-MedSegBench: A Comprehensive Benchmark for FL on Medical Image Segmentation — Zhu et al. — 2026-01-01 — arXiv
45. DFedReweighting: A Unified Framework for Objective-Oriented Reweighting in Decentralized FL — Zhang et al. — 2025-12-12 — arXiv
46. Decentralized Trust for Space AI: Blockchain-Based FL Across Multi-Vendor LEO Satellite Networks — Elmahallawy & Akbarfam — 2025-12-09 — arXiv
47. The MICCAI Federated Tumor Segmentation (FeTS) Challenge 2024 — Linardos et al. — 2025-12-01 — MELBA
48. Improving Generalization in FL for SSVEP Classification (Soft Gripper) — Wei et al. — 2025-12-01 — IEEE TNSRE
49. A Fast and Flat FL Method via Weighted Momentum and Sharpness-Aware Minimization — Li et al. — 2025-11-27 — NeurIPS 2025
50. FedRE: A Representation Entanglement Framework for Model-Heterogeneous FL — Yao et al. — 2025-11-27 — arXiv
51. FLARE: Adaptive Multi-Dimensional Reputation for Robust Client Reliability in FL — Younesi et al. — 2025-11-18 — arXiv
52. Optimizing Quantum FL: non-IID via global data sharing in weighted model averaging and clustering — Guo et al. — 2025-11-14 — Physica Scripta
53. Foam Segmentation in Wastewater Treatment Plants: A FL Approach with SAM 2 — Duman et al. — 2025-11-11 — arXiv
54. Curriculum Guided Personalized Subgraph FL — Kang & Park — 2025-11-08 — OpenAlex
55. FLARE: Adaptive Multi-Dimensional Reputation for Robust Client Reliability in FL — Younesi et al. — 2025-11-01 — arXiv
56. FedMGP: Personalized FL with Multi-Group Text-Visual Prompts — Wang et al. — 2025-11-01 — arXiv
57. FedRW: Efficient Privacy-Preserving Data Reweighting for FL of Language Models — Ye et al. — 2025-11-01 — arXiv
58. DP-FedPGN: Finding Global Flat Minima for Differentially Private FL via Penalizing Gradient Norm — Liu et al. — 2025-10-31 — arXiv
59. Accurate Target Privacy Preserving FL Balancing Fairness and Utility — Sun et al. — 2025-10-30 — INFOCOM 2026
60. FedDPGu: Adaptive Prompt-tuning with Built-in Unlearning for FL — Yang et al. — 2025-10-30 — OpenAlex
61. FedGPS: Statistical Rectification Against Data Heterogeneity in FL — Yang et al. — 2025-10-23 — arXiv
62. CEPerFed: Communication-Efficient Personalized FL for Multi-Pulse MRI Classification — Li et al. — 2025-10-20 — arXiv
63. RAIM: three-stage Stackelberg game for hierarchical FL with reputation-aware incentive — Zuo et al. — 2025-10-02 — Research Square
64. FedPF: Accurate Target Privacy Preserving FL Balancing Fairness and Utility — Sun et al. — 2025-10-01 — ICDCS 2026
65. Validation of Various Normalization Methods for Brain Tumor Segmentation (FL) — Fiszer et al. — 2025-09-24 — LNCS
66. GradCAM-AE: A New Shield Defense against Poisoning Attacks on FL — Zheng et al. — 2025-09-12 — ACM TOPS
67. PPFL: A Personalized FL Framework for Heterogeneous Population — Hao et al. — 2025-08-25 — INFORMS J. Computing
68. Privacy-Preserving Crowd Counting via Quantum-Enhanced FL — Zhang et al. — 2025-07-28 — Expert Systems
69. Privacy Protection and Statistical Efficiency Trade-Off for FL — Qi et al. — 2025-07-15 — INFORMS J. Computing
70. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in FL — Dao et al. — 2025-07-01 — arXiv
71. Geo-ORBIT: A Federated Digital Twin Framework for Scene-Adaptive Lane Geometry Detection — Tamaru et al. — 2025-07-01 — arXiv
72. Gradients as an Action: Communication-Efficient Federated Recommender Systems — Lu et al. — 2025-07-01 — arXiv
73. S2FGL: Spatial Spectral Federated Graph Learning — Tan et al. — 2025-07-01 — arXiv
74. pFedMMA: Personalized Federated Fine-Tuning with Multi-Modal Adapter for VLMs — Ghiasvand et al. — 2025-07-01 — arXiv
75. HtFLlib: A Comprehensive Heterogeneous FL Library and Benchmark — Zhang et al. — 2025-06-01 — OpenAlex
76. UniVarFL: Uniformity and Variance Regularized FL for Heterogeneous Data — Gupta et al. — 2025-06-01 — AAAI
77. Secure Multi-Key Homomorphic Encryption for Privacy-Preserving FL — Wu et al. — 2025-06-01 — arXiv
78. Latency Optimization for Wireless FL in Multihop Networks — Shaon et al. — 2025-06-01 — IEEE TVT
79. Federated ADMM from Bayesian Duality — Möllenhoff et al. — 2025-06-01 — arXiv
80. FL Assisted Edge Caching Scheme Based on Lightweight Architecture DDPM — Li et al. — 2025-06-01 — IEEE Networking Letters
81. FedShield-LLM: A Secure and Scalable Federated Fine-Tuned LLM — Mia & Amini — 2025-06-01 — arXiv
82. FedCLAM: Client Adaptive Momentum with Foreground Intensity Matching for Medical Segmentation — Siomos et al. — 2025-06-01 — arXiv
83. Addressing the Collaboration Dilemma in Low-Data FL via Transient Sparsity — Xiao et al. — 2025-06-01 — arXiv
84. Personalized Subgraph FL with Differentiable Auxiliary Projections — Zhuo et al. — 2025-05-01 — arXiv
85. Voronoi-grid-based Pareto Front Learning and Collaborative FL — Chen et al. — 2025-05-01 — arXiv
86. Unlearning for Federated Online Learning to Rank: A Reproducibility Study — Tao et al. — 2025-05-01 — arXiv
87. The Panaceas for Improving Low-Rank Decomposition in Communication-Efficient FL — Li et al. — 2025-05-01 — arXiv
88. Performance Guaranteed Poisoning Attacks in FL: A Sliding Mode Approach — Pan et al. — 2025-05-01 — OpenAlex
89. Multimodal FL With Missing Modalities through Feature Imputation Network — Poudel et al. — 2025-05-01 — LNCS
90. Mosaic: Data-Free Knowledge Distillation via MoE for Heterogeneous Distributed Environments — Liu et al. — 2025-05-01 — arXiv
91. Defending the Edge: Representative-Attention for Mitigating Backdoor Attacks in FL — Obioma et al. — 2025-05-01 — arXiv
92. DP-RTFL: Differentially Private Resilient Temporal FL — Talluri — 2025-05-01 — arXiv
93. A Federated Random Forest Solution for Secure Distributed Machine Learning — Cotorobai et al. — 2025-05-01 — arXiv
94. FedNolowe: A Normalized Loss-Based Weighted Aggregation Strategy for Robust FL — Le et al. — 2025-04-04 — bioRxiv
95. FL for the pathogenicity annotation of genetic variants in multi-site clinical settings — Montalvo et al. — 2025-04-04 — medRxiv
96. The More is not the Merrier: Investigating the Effect of Client Size on FL — Wallach et al. — 2025-04-01 — arXiv
97. mixEEG: Enhancing EEG FL for Cross-subject Classification with Tailored mixup — Liu et al. — 2025-04-01 — arXiv
98. Token-Level Prompt Mixture with Parameter-Free Routing for Federated Domain Generalization — Gong et al. — 2025-04-01 — arXiv
99. Achieving Distributive Justice in FL via Uncertainty Quantification — Carey & Wu — 2025-04-01 — arXiv
100. Soft-Label Caching and Sharpening for Communication-Efficient Federated Distillation — Azuma et al. — 2025-04-01 — arXiv
101. POPri: Private FL using Preference-Optimized Synthetic Data — Hou et al. — 2025-04-01 — arXiv
102. Federated Spectral Graph Transformers Meet Neural ODEs for Non-IID Graphs — Gurumurthy et al. — 2025-04-01 — arXiv
103. A Study on the Efficiency of Combined Reconstruction and Poisoning Attacks in FL — Becker et al. — 2025-03-20 — JDSIS
104. FedVSR: Towards Model-Agnostic FL in Video Super-Resolution — Dehaghi et al. — 2025-03-01 — OpenAlex
105. UltraFlwr — An Efficient Federated Medical and Surgical Object Detection Framework — Li et al. — 2025-03-01 — arXiv
106. TS-Inverse: A Gradient Inversion Attack for Federated Time Series Forecasting — Meijer et al. — 2025-03-01 — arXiv
107. Robust Asymmetric Heterogeneous FL with Corrupted Clients — Fang et al. — 2025-03-01
108. Mind the Gap: Confidence Discrepancy Guides Federated Semi-Supervised Learning — Liu et al. — 2025-03-01 — arXiv
109. Geometric Knowledge-Guided Localized Global Distribution Alignment for FL — Ma et al. — 2025-03-01 — CVPR
110. Federated nnU-Net for Privacy-Preserving Medical Image Segmentation — Skorupko et al. — 2025-03-01 — arXiv
111. Federated Semantic Learning for Privacy-preserving Cross-domain Recommendation — Lu et al. — 2025-03-01 — arXiv
112. Redefining non-IID Data in FL: Migrating from Labels to Embeddings — Borazjani et al. — 2025-03-01 — arXiv
113. Fair Federated Medical Image Classification Against Quality Shift — Wu et al. — 2025-03-01 — arXiv
114. A Survey on Federated Fine-tuning of Large Language Models — Wu et al. — 2025-03-01 — arXiv
115. FAA-CLIP: Federated Adversarial Adaptation of CLIP — Wu et al. — 2025-03-01 — arXiv
116. Dynamic Allocation Hypernetwork with Adaptive Model Recalibration for Federated Continual Learning — Qi et al. — 2025-03-01 — IPMI
117. FLIP: Towards Comprehensive and Reliable Evaluation of Federated Prompt Learning — Liao et al. — 2025-03-01 — arXiv
118. Detecting Backdoor Attacks in FL via Direction Alignment Inspection — Xu et al. — 2025-03-01 — CVPR
119. Decentralized Personalization for Federated Medical Image Segmentation via Gossip Contrastive Mutual Learning — Chen & Yuan — 2025-03-01 — arXiv
120. BTFL: A Bayesian-based Test-Time Generalization Method for FL — Zhou & Liu — 2025-03-01 — OpenAlex
121. FedAPA: Server-side Gradient-Based Adaptive Personalized Aggregation for FL on Heterogeneous Data — Sun et al. — 2025-02-11 — OpenAlex
122. Fed-SB: A Silver Bullet for Extreme Communication Efficiency in (Private) Federated LoRA — Singhal et al. — 2025-02-01 — arXiv
123. FedBM: Stealing Knowledge from Pre-trained Language Models for Heterogeneous FL — Zhu et al. — 2025-02-01 — Medical Image Analysis
124. Forgetting Any Data at Any Time: Certified Unlearning for Vertical FL — Wang & Wang — 2025-02-01 — arXiv
125. PM-MOE: Mixture of Experts on Private Model Parameters for Personalized FL — Feng et al. — 2025-02-01 — OpenAlex
126. A New Perspective on Privacy Protection in FL with Granular-Ball Computing — Lai et al. — 2025-01-08 — arXiv
127. Balanced coarse-to-fine FL for noisy heterogeneous clients — Han et al. — 2025-01-07 — Complex & Intelligent Systems
128. Intrinsic Training Signals for FL Aggregation — Fiorini et al. — 2025-01-01 — arXiv
129. Latte: Collaborative Test-Time Adaptation of VLMs in FL — Bao et al. — 2025-01-01 — arXiv
130. Private FL using Preference-Optimized Synthetic Data — Hou et al. — 2025-01-01 — arXiv
131. Learning Reconfigurable Representations for Multimodal FL with Missing Data — Nguyen et al. — 2025-01-01 — arXiv
132. MMiC: Mitigating Modality Incompleteness in Clustered FL — Yang et al. — 2025-01-01 — OpenAlex
133. Gradient Compression and Correlation Driven FL for Wireless Traffic Prediction — Zhang et al. — 2025-01-01 — IEEE TCCN
134. OmniFed: A Modular Framework for Configurable FL from Edge to HPC — Tyagi et al. — 2025-01-01 — SC Workshops
135. On Evaluating the Poisoning Robustness of FL under Local Differential Privacy — Wang et al. — 2025-01-01 — arXiv
136. On the Out-of-Distribution Backdoor Attack for FL — Xu et al. — 2025-01-01 — OpenAlex
137. Optimized Local Updates in FL via Reinforcement Learning — Murad et al. — 2025-01-01 — IJCNN
138. Optimizing Communication Efficiency through Training Potential in Multi-Modal FL — Zhang et al. — 2025-01-01 — ACM TOIT
139. Owen Sampling Accelerates Contribution Estimation in FL — KhademSohi et al. — 2025-01-01 — Frontiers AI
140. Traceable Black-box Watermarks for FL — Xu et al. — 2025-01-01 — arXiv
141. Robust FL against Noisy Clients via Masked Optimization — Jiang et al. — 2025-01-01 — arXiv
142. SoK: Benchmarking Poisoning Attacks and Defenses in FL — Zhang et al. — 2025-01-01 — arXiv
143. Stones From Other Hills: Intrusion Detection in Statistical Heterogeneous IoT by Self-Labeled Personalized FL — Lu et al. — 2025-01-01 — IEEE IoT Journal
144. Subgraph FL for Local Generalization — Kim et al. — 2025-01-01 — ICLR
145. TAP: Two-Stage Adaptive Personalization of Multi-task and Multi-Modal Foundation Models in FL — Lee et al. — 2025-01-01 — arXiv
146. The Cost of Local and Global Fairness in FL — Duan et al. — 2025-01-01
147. Towards Straggler-Resilient Split FL: An Unbalanced Update Approach — Liang et al. — 2025-01-01 — arXiv
148. UFGraphFR: Graph Federation Recommendation System based on User Text features — Wang et al. — 2025-01-01 — arXiv
149. Uncertainty-Aware Label Refinement on Hypergraphs for Personalized Federated FER — Ding et al. — 2025-01-01
150. kMoL: an open-source machine and FL library for drug discovery — Cozac et al. — 2025-01-01 — J. Cheminformatics
151. sat-QFL: Secure Quantum FL for Low Orbit Satellites — Gurung & Pokhrel — 2025-01-01 — arXiv
152. Generalizable FL using Client Adaptive Focal Modulation — Ashraf & Gillani — 2025-01-01 — arXiv
153. Generalizable Reconstruction for Accelerating MR Imaging via FL With NAS — Wu et al. — 2025-01-01 — IEEE TMI
154. Zero-Shot Decentralized FL — Masano et al. — 2025-01-01 — IJCNN
155. GPT-FL: Generative Pre-trained Model-Assisted FL — Zhang et al. — 2025-01-01 — CVPR Workshops
156. FedBiF: Communication-Efficient FL via Bits Freezing — Li et al. — 2025-01-01 — IEEE TPDS
157. From continuous pre-training to alignment: toolkit for LLMs in FL — Zhang et al. — 2025-01-01 — Neurocomputing
158. A New One-Shot FL Framework for Medical Imaging with Rectified Flow and KD — Ma et al. — 2025-01-01 — Frontiers AI
159. Allosteric Feature Collaboration for Model-Heterogeneous FL — Yang et al. — 2025-01-01 — IEEE TNNLS
160. Beyond Aggregation: Guiding Clients in Heterogeneous FL — Wang et al. — 2025-01-01 — arXiv
161. ByzFL: Research Framework for Robust FL — González et al. — 2025-01-01 — arXiv
162. CFL-SparseMed: Communication-Efficient FL for Medical Imaging with Top-k Sparse Updates — Habib et al. — 2025-01-01 — arXiv
163. Capture Global Feature Statistics for One-Shot FL — Guan et al. — 2025-01-01 — AAAI
164. Choice Outweighs Effort: Complementary Knowledge Fusion in FL — Yang et al. — 2025-01-01 — Frontiers AI
165. Cyst-X: AI-Powered Pancreatic Cancer Risk Prediction (Centralized and FL) — Pan et al. — 2025-01-01 — OpenAlex
166. DictPFL: Efficient and Private FL on Encrypted Gradients — Xue et al. — 2025-01-01 — arXiv
167. FADngs: Federated Learning for Anomaly Detection — Dong et al. — 2025-01-01 — IEEE TNNLS
168. FNBench: Benchmarking Robust FL against Noisy Labels — Jiang et al. — 2025-01-01 — OpenAlex
169. UniTrans: A Unified Vertical Federated Knowledge Transfer Framework — Huang et al. — 2025-01-01 — arXiv
170. FedDLAD: A FL Dual-Layer Anomaly Detection Framework vs Backdoor Attacks — Ding et al. — 2025-01-01 — OpenAlex
171. FedRTS: Federated Robust Pruning via Combinatorial Thompson Sampling — Huang et al. — 2025-01-01 — arXiv
172. Fedgac: optimizing generalization in personalized FL via adaptive initialization and client selection — Yu et al. — 2025-01-01 — Research Square
173. Federated Discrete Denoising Diffusion Model for Molecular Generation with OpenFL — Ta et al. — 2025-01-01 — arXiv
174. FedSWA: Improving Generalization in FL via Momentum-Based Stochastic Controlled Weight Averaging — Liu et al. — 2025-01-01 — arXiv
175. FedFetch: Faster FL with Adaptive Downstream Prefetching — Yan et al. — 2025-01-01 — INFOCOM
176. FL for digital twin applications: a privacy-preserving and low-latency approach — Li & Wang — 2025-01-01 — PeerJ CS
177. FedRIR: Rethinking Information Representation in FL — Huang et al. — 2025-01-01 — OpenAlex
178. FedQS: Optimizing Gradient and Model Aggregation for Semi-Asynchronous FL — Li et al. — 2025-01-01 — arXiv
179. FedKDC: Consensus-Driven KD for Personalized FL in EEG-Based Emotion Recognition — Qiu et al. — 2025-01-01 — IEEE JBHI
180. FedKD-hybrid: Federated Hybrid Knowledge Distillation for Lithography Hotspot Detection — Li et al. — 2025-01-01 — arXiv
181. FedFitTech: A Baseline in FL for Fitness Tracking — Oz et al. — 2025-01-01 — arXiv
182. Calibre: Towards Fair and Accurate Personalized FL with Self-Supervised Learning — Chen et al. — 2024-12-27 — OpenAlex
183. SplitFedZip: Learned Compression for Data Transfer Reduction in Split-FL — Shiranthika et al. — 2024-12-18 — arXiv
184. Task Diversity in Bayesian FL: Simultaneous Classification and Regression — Lyu et al. — 2024-12-14 — OpenAlex
185. Benchmarking FL for Semantic Datasets: Federated Scene Graph Generation — Ha et al. — 2024-12-11 — Pattern Recognition Letters
186. One-shot FL via Synthetic Distiller-Distillate Communication — Zhang et al. — 2024-12-06 — NeurIPS
187. FedDW: Distilling Weights through Consistency Optimization in Heterogeneous FL — Liu et al. — 2024-12-05 — arXiv
188. FedAH: Aggregated Head for Personalized FL — Zhou et al. — 2024-12-02 — OpenAlex
189. FedCAR: Cross-client Adaptive Re-weighting for Generative Models in FL — Kim et al. — 2024-12-01 — arXiv
190. Vertical Federated Unlearning via Backdoor Certification — Han et al. — 2024-12-01 — arXiv
191. Optimizing Personalized FL through Adaptive Layer-Wise Learning — Chen et al. — 2024-12-01 — OpenAlex
192. FedPAW: FL with Personalized Aggregation Weights for Urban Vehicle Speed Prediction — He et al. — 2024-12-01 — IEEE TCC
193. Generalising Battery Control in Net-Zero Buildings via Personalised Federated RL — Cuadrado Avila et al. — 2024-12-01 — arXiv
194. DapperFL: Domain Adaptive FL with Model Fusion Pruning for Edge Devices — Jia et al. — 2024-12-01 — NeurIPS
195. BEFL: Balancing Energy Consumption in FL for Mobile Edge IoT — Ju et al. — 2024-12-01 — arXiv
196. Covariances for Free: Exploiting Mean Distributions for Training-free FL — Goswami et al. — 2024-12-01 — arXiv
197. Covariances for Free: Exploiting Mean Distributions for FL with Pre-Trained Models — Goswami et al. — 2024-12-01 — arXiv
198. FedReMa: Improving Personalized FL via Leveraging the Most Relevant Clients — Liang et al. — 2024-11-04 — Frontiers AI
199. Adaptive Client Selection with Personalization for Communication Efficient FL — de Souza et al. — 2024-11-01 — Ad Hoc Networks
200. Energy-efficient FL with Dynamic Model Size Allocation — Kumar et al. — 2024-11-01 — IEEE Big Data

</details>

The repo also links sibling collections worth searching:
[LLM papers](https://github.com/mtuann/llm-updated-papers),
[Backdoor Learning](https://github.com/mtuann/backdoor-ai-resources),
[Machine Unlearning](https://github.com/mtuann/machine-unlearning-papers).

---

## 5. Bottom line for the FedGRAA project

1. **Q1 (imbalance detection):** Hard-coding rare classes and using declared label
   counts is not FL-compliant. **Infer the per-round class composition from the
   output-layer gradients** (Wang et al., AAAI 2021, *Ratio Loss*) — the server
   already receives the updates, so no privacy is lost. Astraea (KLD mediator) is
   a weaker-privacy fallback.
2. **Q2 (framework):** Use **Flower** — it is framework-agnostic (works with this
   repo's raw NumPy model), its custom `Strategy` API maps directly to FedGRAA's
   server-side adaptive weight learner, and it provides federated evaluation for Q3.
3. **Q3 (validation):** Drop the server-side `D_V`. Use **federated evaluation** —
   clients evaluate the broadcast model on their own held-out data and return
   **scalar per-class metrics / confusion-matrix counts**; the server aggregates
   them. Report **macro-F1, balanced accuracy, per-class (rare-class) recall**, not
   plain accuracy.
4. **Paper corpus:** The full 25k-paper site cannot be (and should not be) listed
   by hand; the curated **200 "Papers with Code"** above is the actionable set,
   with the most FedGRAA-relevant entries grouped in §4.2. Ask if you want a
   scraper to pull the complete site JSON.

### Key references
- Wang, Xu, Wang, Zhu. *Addressing Class Imbalance in Federated Learning.* AAAI 2021. arXiv:2008.06217.
- Duan et al. *Astraea: Self-balancing Federated Learning.* IEEE ICCD 2019. arXiv:1907.01132.
- Beutel et al. *Flower: A Friendly Federated Learning Research Framework.* arXiv:2007.14390.
- McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
- Zhao et al. *iDLG: Improved Deep Leakage from Gradients.* 2020. / Geiping et al. *Inverting Gradients.* NeurIPS 2020.
- Apheris. *Top 7 Open-Source Frameworks for Federated Learning.* 2024–2026.
- mtuann. *federated-learning-updated-papers* (May 2026 snapshot) — https://mtuann.github.io/papers/
