# Literature Review & Publication Strategy for FedGRAA
## Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems

---

## Executive Summary

This document provides a comprehensive literature review of federated learning and intrusion detection systems research published at A* and A conferences (CCS, NDSS, AAAI, IJCAI, NeurIPS, ICML, ICLR), identifies how existing work differs from the proposed FedGRAA approach, and outlines a strategic roadmap for publication at top-tier venues.

**Key Finding:** No prior work directly combines gradient-refined adaptive averaging specifically tailored to the unique heterogeneity patterns found in distributed intrusion detection systems with limited data per client.

---

## Part 1: Summary of Papers Found from Top Venues

### Overview
Relevant papers from A* and A conferences related to federated learning for intrusion detection have been organized by research focus and venue tier.

---

## Part 2: Key Papers from A* Security Conferences (CCS & NDSS)

| Paper | Venue | Year | Key Focus | How It Differs from FedGRAA |
|-------|-------|------|-----------|----------------------------|
| Entente: Cross-silo Intrusion Detection on Network Log Graphs with FL | NDSS | 2025 | Graph-based NIDS (GNIDS) with FL; uses graph neural networks across organizations | **Graph-based approach** — FedGRAA uses adaptive averaging on model weights/updates, while Entente focuses on graph-structured network log data and graph neural networks for APT detection |
| EIFFeL: Ensuring Integrity for Federated Learning | CCS | 2021 (117 cites) | Formal framework for update integrity in FL via zero-knowledge proofs; prevents poisoning | **Integrity/security focus** — FedGRAA focuses on improving convergence under heterogeneous data via adaptive averaging; EIFFeL addresses the orthogonal problem of ensuring clients submit well-formed updates |
| Cerberus: Exploring Federated Prediction of Security Events | CCS | 2022 | Predicting future security events using FL across organizations | **Predictive security** — Cerberus predicts future attacks rather than detecting ongoing intrusions; different objective entirely |
| MESAS: Poisoning Defense for FL Resilient against Adaptive Attackers | CCS | 2023 (57 cites) | Poisoning defense using multi-layered filtering against adaptive adversaries | **Adversarial robustness** — MESAS focuses on defending against malicious clients, not on improving aggregation for heterogeneous data |
| FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks in FL | NDSS | 2023 (59 cites) | Frequency-domain analysis of model updates to detect/remove poisoned updates | **Frequency-based defense** — FedGRAA adaptively averages gradients based on refinement; FreqFed transforms updates to frequency domain to filter malicious ones |
| CrowdGuard: Federated Backdoor Detection in FL | NDSS | 2022 (51 cites) | Backdoor detection using a trusted audit server and reputation system | **Backdoor-specific defense** — CrowdGuard is a defense framework rather than an aggregation algorithm improvement |
| Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection | NDSS | 2018 (975 cites) | Lightweight ensemble of autoencoders for online NIDS at network gateways | **Not federated** — Kitsune is a centralized, lightweight neural NIDS; no FL or privacy-preserving aspect |
| Sometimes, You Aren't What You Do: Mimicry Attacks against Provenance Graph Host IDS | NDSS | 2023 (73 cites) | Mimicry attacks against provenance-graph-based HIDS | **Attack analysis** — studies adversarial evasion of host-based IDS, not FL aggregation |
| Automatic Adversarial Adaption for Stealthy Poisoning Attacks in FL | NDSS | 2024 (33 cites) | Poisoning attacks that adapt to evade FL defenses | **Attack methodology** — proposes new poisoning attacks, not a defense or aggregation method |
| Armadillo: Robust Single-Server Secure Aggregation for FL | CCS | 2025 | Secure aggregation with disruption resistance against malicious clients | **Cryptographic focus** — Armadillo ensures correct aggregation via cryptographic protocols; FedGRAA focuses on heterogeneous data adaptation |

---

## Part 3: Key Papers from A* AI Conferences (AAAI, IJCAI, NeurIPS, ICLR)

| Paper | Venue | Year | Key Focus | How It Differs from FedGRAA |
|-------|-------|------|-----------|----------------------------|
| FedDLAD: A Federated Learning Dual-Layer Anomaly Detection Framework | IJCAI | 2025 | Dual-layer anomaly detection for backdoor defense in FL | **Backdoor defense via anomaly detection** — identifies malicious clients through a two-layer detection mechanism; FedGRAA focuses on data heterogeneity, not adversary detection |
| TriFusion-IDS: Multimodal Graph-Tabular-Text Contrastive Framework for IDS | AAAI | 2026 | Multimodal (graph+tabular+text) cross-dataset IDS using contrastive learning | **Multimodal representation learning** — TriFusion-IDS uses three complementary views of network traffic; not FL-based |
| Poisoning with a Pill: Circumventing Detection in FL | AAAI | 2026 | Stealthy poisoning attack exploiting model redundancy | **Attack paper** — proposes a method to bypass FL defenses by manipulating only non-redundant parameters |
| DiffuPac: Contextual Mimicry in Adversarial Packets via Diffusion Model | NeurIPS | 2024 | Adversarial packet generation using diffusion models to evade NIDS | **Generative adversarial attacks** — not FL; generates adversarial network packets |
| Intrusion Detection with Neural Networks | NeurIPS | 1997 (555 cites) | Early work on neural networks for intrusion detection | **Pioneering work** — pre-FL era, centralized neural NIDS |
| Enhancing Federated Domain Adaptation with Multi-Domain Prototype-Based Federated Fine-Tuning | ICLR | 2024 | Domain adaptation in FL using prototype-based fine-tuning | **Domain adaptation** — addresses data heterogeneity via prototypes and fine-tuning, not via gradient-refined averaging |

---

## Part 4: The Most Direct Competitors

The papers closest to FedGRAA (FL + improved aggregation for IDS under non-IID/heterogeneous data) are:

1. **SBFedNIDS (ICASSP 2025)** — "Self-Balancing Federated learning NIDS"
   - Tackles size imbalance, heterogeneous data, and global imbalance using a self-balancing mechanism in aggregation
   - Your approach of gradient-refined adaptive averaging is a different mechanism but addresses the same core problem

2. **FD-IDS (Sensors, 2025)** — "Federated Learning with Knowledge Distillation for IDS in Non-IID IoT Environments"
   - Uses knowledge distillation rather than adaptive averaging to handle non-IID data

3. **Dynamic Weighted Aggregation FL for NIDS (ICRAIE 2023)**
   - Uses dynamic weighted aggregation, but at a lower-tier conference

4. **FedMADE (ISC 2024)** — "Robust FL for IDS using Dynamic Aggregation Method"
   - Closest in spirit, using dynamic aggregation for IDS in IoT

**Key Gap:** None of the A*/A venue papers directly propose a gradient-refined adaptive averaging mechanism specifically designed for IDS under heterogeneous data distributions. This is FedGRAA's unique angle.

---

## Part 5: How to Strengthen Your Paper for A*/A Conference Publication

### 1. Strengthen the Technical Novelty

**Current Gap:** "Adaptive averaging" alone is well-explored. What makes yours gradient-refined?

Papers like EIFFeL and Armadillo succeed because they formalize a precise security property with formal guarantees.

**Action:**
- Formalize your convergence guarantees under heterogeneous data (like FedProx did for non-IID)
- Prove that your gradient refinement mechanism converges faster or achieves a better stationary point than FedAvg and other adaptive methods

### 2. Add Robustness Evaluations

**What A*/A papers do:** Every top security paper evaluates against adversarial scenarios. MESAS (CCS) and FreqFed (NDSS) test against adaptive attackers. Cerberus (CCS) evaluates realistic multi-organization settings.

**Action:** Evaluate your method under:
- Byzantine/poisoning attacks (label flipping, model poisoning)
- Data poisoning scenarios (backdoor attacks)
- Differential privacy integration

Show that your adaptive averaging is naturally robust against certain attacks, or integrate a defense mechanism.

### 3. Use Real-World, Large-Scale Datasets

**What top papers do:** Entente (NDSS 2025) evaluates on real-world enterprise network logs. Kitsune (NDSS 2018, 975 cites) created its own dataset from real network captures.

**Action:** Go beyond UNSW-NB15, NSL-KDD, or CIC-IDS2017. Use:
- CIC-IDS2023 or CICIoT2023
- FLNET2023 — a dataset specifically designed for FL-based IDS evaluation
- Real-world traffic captures or collaborate with industry partners

### 4. Address Heterogeneity Distinctively

**What A*/A papers do:** FedDLAD (IJCAI 2025) addresses both data heterogeneity and backdoor attacks simultaneously. Entente (NDSS 2025) handles cross-silo heterogeneity across organizations.

**Action:** Define precisely what kind of heterogeneity your method solves:
- Label distribution skew
- Feature distribution skew
- Quantity skew

Compare systematically against FedAvg, FedProx, SCAFFOLD, FedNova on IDS-specific heterogeneity metrics.

### 5. Add Theoretical Analysis

**What A*/A papers do:** EIFFeL (CCS) provides formal security proofs. Armadillo (CCS) provides formal disruption-resistance guarantees.

**Action:** Include:
- Convergence analysis of your adaptive averaging under non-IID data
- Upper bounds on gradient divergence after refinement
- Communication efficiency analysis (rounds to convergence)

### 6. Target the Right Venue

| Venue | Fit | Requirements |
|-------|-----|--------------|
| **CCS (A*)** | Strong if you add security guarantees (defense against poisoning) + formal analysis | Need security formalism, adversarial evaluation, real datasets |
| **NDSS (A*)** | Strong if you target practical deployment (cross-organization, real-world logs) | Need realistic deployment study, system implementation |
| **AAAI (A*)** | Good if you emphasize ML novelty (new aggregation theory) | Need strong theoretical ML contributions, comparisons on ML benchmarks |
| **IJCAI (A*)** | Good for algorithm + application combo | Need algorithmic novelty + thorough empirical validation |
| **IEEE S&P (A*)** | Only if you make privacy/security guarantees the central focus | Heavy formalism required |

### 7. Concrete Experiments to Add

- **Scalability:** Show performance with 50–200+ clients (most FL-IDS papers use 5–20)
- **Extreme non-IID:** Simulate realistic IDS heterogeneity (different organizations have different attack profiles)
- **Ablation study:** Exactly which component of your "gradient refinement" contributes to improvement?
- **Latency/overhead:** Measure communication cost and training time vs. accuracy trade-off
- **Privacy-utility trade-off:** Integrate differential privacy and show the trade-off curve

### 8. Positioning Strategy

Position your paper as addressing the intersection gap that no A*/A paper currently fills:

> "While existing top-venue works address FL security (poisoning, backdoors) or centralized IDS, and while adaptive FL aggregation has been studied in general ML (FedProx, SCAFFOLD), no prior work combines gradient-refined adaptive averaging specifically tailored to the unique heterogeneity patterns found in distributed intrusion detection systems."

---

## Part 6: Algorithm Families for Handling Limited Data

### Family 1: Federated Few-Shot Learning / Meta-Learning

These combine meta-learning's rapid adaptation with FL's privacy preservation — ideal when each client has very few samples.

| Algorithm | Venue | Year | Key Idea for Handling Limited Data |
|-----------|-------|------|-----------------------------------|
| FedFSL-CFRD | AAAI | 2025 | Collaborative feature representation disentanglement to address domain bias and feature coupling under sample scarcity |
| Few-Shot Model Agnostic FL | ACM Multimedia | 2022 | Enables participants with limited private data to learn from others without sharing data or model structures |
| Fed-Meta for Fault Diagnosis | IEEE TNSE | 2023 | Federated-meta learning framework for few-shot fault diagnosis; leverages shared knowledge across clients |
| Representation-Encoding Fed-Meta | IEEE TIM | 2023 | Representation encoding to tackle domain discrepancy and data scarcity simultaneously |
| DFML | IEEE/ACM TCBB | 2023 | Dynamic Federated Meta-Learning for rare disease prediction; handles extreme data scarcity across hospitals |
| FeCoGraph | IEEE TIFS | 2025 | Label-aware federated graph contrastive learning for few-shot NIDS; addresses annotation scarcity |
| Lightweight Fed-FSL-NIDS | IEEE IoTJ | 2026 | Lightweight few-shot FL for resource-constrained IoT devices with limited labeled data |
| Fed-Siamese | IEEE IoTJ | 2025 | Federated meta-learning with siamese networks for HAR under data scarcity |

**How they differ from FedGRAA:** All of these rely on meta-learning (learning to learn) or siamese/contrastive architectures to generalize from extremely few samples (e.g., 1-20 per class). FedGRAA's gradient-refined adaptive averaging is an aggregation-side solution; these are model-side solutions operating at the local training level.

### Family 2: Prototype-Based Federated Learning

Instead of sharing full model parameters, clients share class prototypes (centroids of feature representations) — much more efficient when data is scarce.

| Algorithm | Venue | Year | Key Idea for Handling Limited Data |
|-----------|-------|------|-----------------------------------|
| FedGMKD | NeurIPS | 2024 | Combines prototype-based knowledge distillation with GMM-based clustering; handles data heterogeneity with limited samples |
| FedRFQ | IEEE TC | 2024 | Reduces prototype redundancy and failure; improves quality under limited and heterogeneous data |
| FedPKD | ICDCS | 2023 | Prototype-based knowledge distillation for heterogeneous FL; reduces communication while handling non-IID data |
| DeProFL | IEEE IoTJ | 2024 | Decentralized prototype FL with time-varying topology; handles heterogeneity with limited communication |
| Cross-Silo Feature Space Alignment | AAAI | 2025 | Aligns feature spaces across clients with imbalanced data; addresses limited samples per class |
| FedMPS | IEEE TNNLS | 2025 | Multi-level prototype contrastive learning; extracts robust features from limited data |
| RefProtoFL | arXiv | 2026 | External-referenced prototype alignment for communication-efficient FL with limited data |

**How they differ from FedGRAA:** Prototype methods fundamentally change what is communicated (prototypes vs. model weights). They are particularly effective when each client has very few samples because prototypes act as compact, informative summaries. FedGRAA focuses on how to weight/average model updates — orthogonal and potentially complementary to prototype-based approaches.

### Family 3: Adaptive Weighted Aggregation (Closest to FedGRAA)

These are most similar to FedGRAA — they improve aggregation weights beyond simple data-size-based averaging.

| Algorithm | Venue | Year | Key Idea for Handling Limited Data |
|-----------|-------|------|-----------------------------------|
| FedAPA | IJCAI | 2025 | Server-side gradient-based adaptive personalized aggregation; updates weights based on gradients of client-personalized models |
| FedAWA | CVPR | 2025 | Learns optimal aggregation weights using client representation vectors — outperforms dataset-size-based weighting |
| FedSWA | ICML | 2025 | Stochastic controlled weight averaging; improves generalization under highly heterogeneous data |
| DA-PFL | IEEE TNNLS | 2025 | Dynamic affinity-based aggregation for personalized FL under class imbalance |
| FedNolowe | bioRxiv | 2025 | Normalized loss-based weighted aggregation; weights clients by training loss to favor those with better local models |
| FedMcon | Front. ITEE | 2025 | Meta-controller learns aggregation strategy adaptively |
| FBLG | IJCAI | 2024 | Local graph-based approach for dual-skewed non-IID data (both label and size skew) |
| FedDRL | ICPP | 2022 | Deep RL-based adaptive aggregation for non-IID data |

**How they differ from FedGRAA:** These are your direct competitors. FedAPA (IJCAI 2025) and FedAWA (CVPR 2025) are particularly relevant — they both propose adaptive weighting schemes but use different mechanisms (gradient-based vs. representation-based vs. loss-based). FedGRAA's novelty would be in how you refine the gradient information specifically for IDS workloads with limited data per client.

### Family 4: Data Augmentation & Knowledge Transfer

These tackle limited data by generating more data or transferring knowledge from pretrained models.

| Algorithm | Venue | Year | Key Idea for Handling Limited Data |
|-----------|-------|------|-----------------------------------|
| AugFL | IEEE TON | 2025 | Augments FL with pretrained models to alleviate data requirements |
| FeCWGAN-GP | IEEE TNNLS | 2026 | Conditional GAN-based data augmentation for vertical FL under heterogeneity |
| SEM | CANDAR | 2023 | Simple model-agnostic local training mechanism to tackle data sparsity and scarcity |
| SelfFed | Expert Systems | 2023 | Self-supervised learning for label scarcity and data heterogeneity |
| Capture Global Feature Stats | AAAI | 2025 | One-shot FL that captures global feature statistics in a single communication round |

### Family 5: One-Shot & Communication-Efficient FL

For extreme data scarcity, these reduce FL to a single round.

| Algorithm | Venue | Year | Key Idea |
|-----------|-------|------|----------|
| Capture Global Feature Statistics | AAAI | 2025 | Single-round FL using global feature statistics |
| Optimal Transport One-Shot FL | IEEE IoTJ | 2024 | OT-based model fusion in one communication round |
| Replica Tree FL | Neural Networks | 2023 | Limited data FL using replica tree structures |

---

## Part 7: Comparison Table - Algorithm Families vs. FedGRAA

| Aspect | FedGRAA | FedAPA (IJCAI'25) | FedAWA (CVPR'25) | FeCoGraph (TIFS'25) | FedGMKD (NeurIPS'24) | FedFSL-CFRD (AAAI'25) |
|--------|---------|------------------|------------------|---------------------|----------------------|----------------------|
| **Core Mechanism** | Gradient-refined adaptive averaging | Gradient-based adaptive aggregation weights | Learned aggregation weights from client vectors | Graph contrastive + few-shot | Prototype + GMM + KD | Feature disentanglement for few-shot |
| **Handles Limited Data** | Partially (via better weighting) | Indirectly | Indirectly | ✅ Yes (few-shot) | ✅ Yes (prototypes) | ✅ Yes (few-shot) |
| **Target Domain** | IDS | General | General | NIDS | General | General |
| **Top Venue** | — | ✅ IJCAI | ✅ CVPR | ✅ TIFS | ✅ NeurIPS | ✅ AAAI |
| **Non-IID Handling** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Novelty Gap** | — | Aggregation via gradients of personalized models | Learns weight function via representation vectors | Graph-based few-shot NIDS | Prototype fusion with GMM | Collaborative representation disentanglement |

---

## Part 8: Strategic Recommendations for A/A* Publication

### Your Current Position

FedGRAA proposes gradient-refined adaptive averaging — an aggregation-side improvement. This is closest to FedAPA (IJCAI 2025), FedAWA (CVPR 2025), and FedNolowe. However, these papers don't specifically target limited data scenarios — they focus on general non-IID heterogeneity.

### Recommended Enhancements

#### 1. Add Explicit "Limited Data" Handling

Your current approach addresses heterogeneity, but top venues want to see how you specifically handle the case where some clients have very few samples. Consider combining your adaptive averaging with:

- **Prototype-based refinement** (like FedGMKD): Use prototypes from data-rich clients to guide the gradient refinement for data-poor clients
- **Few-shot meta-learning** (like FedFSL-CFRD): Add a meta-learning outer loop that trains clients to quickly adapt from limited data

#### 2. Benchmark on Small-Data FL Scenarios

Most IDS papers use large datasets like CIC-IDS2017/2019. Create experimental setups where each client has only 10-50 samples per class. Show that your aggregation matters more when data is scarce.

#### 3. Add Theoretical Convergence Guarantees

Papers at top venues (ICML, NeurIPS, AAAI) typically include convergence analysis. Prove that your gradient refinement converges under bounded heterogeneity and limited sample sizes.

#### 4. Consider a Hybrid Approach

Integrate your adaptive averaging with prototype sharing or knowledge distillation across clients. For example:

- **Data-poor clients:** Use your gradient refinement + knowledge distillation from data-rich clients' prototypes
- **Data-rich clients:** Standard local training + contribution to global prototype bank

#### 5. Address the "Cold Start" Problem

What happens when a new client joins with extremely limited data? Show how your method handles zero-shot or one-shot adaptation for new participants.

#### 6. Strengthen IDS-Specific Motivation

FeCoGraph (IEEE TIFS 2025) and Lightweight Fed-FSL-NIDS (IEEE IoTJ 2026) are doing few-shot NIDS at top venues. Position your work as: **"adaptive aggregation is the missing piece that makes few-shot FL work for IDS"**.

### Suggested Title for A/A* Version

**"Gradient-Refined Adaptive Aggregation with Prototype-Guided Knowledge Transfer for Few-Shot Federated Intrusion Detection"**

### Target Venues (in Order of Fit)

1. **AAAI** — Strong on both security+AI and few-shot learning
   - FedFSL-CFRD, Capture Global Feature Stats, Cross-Silo Alignment all published here
   
2. **IJCAI** — Good for algorithm + application combo
   - FedAPA and FedDLAD both appear here
   
3. **CCS/NDSS** — If you keep the IDS focus strong
   - Entente and CrowdGuard at NDSS show FL+IDS acceptance
   
4. **IEEE TIFS** — Shows FL+IDS acceptance
   - FeCoGraph demonstrates this venue's interest
   
5. **ICML** — If you emphasize theoretical contributions
   - FedSWA appears here; strong ML theory focus

---

## Part 9: Conclusion

### Key Takeaways

1. **Your Unique Position:** Gradient-refined adaptive averaging for IDS is a genuine gap in the literature
2. **Immediate Competitors:** FedAPA, FedAWA, and FedNolowe use similar techniques but not IDS-specific
3. **Enhancement Path:** Combine your approach with few-shot learning and prototype-based methods
4. **Publication Strategy:** Target AAAI or IJCAI with strong theoretical analysis and comprehensive limited-data experiments
5. **Positioning:** Emphasize the "missing piece" for few-shot federated IDS

### Next Steps

1. Add convergence analysis with formal theorems
2. Implement small-data experiments (10-50 samples per client)
3. Integrate prototype-based refinement for data-poor clients
4. Evaluate on real-world datasets (CIC-IDS2023, FLNET2023)
5. Add adversarial robustness evaluation
6. Target AAAI or IJCAI as primary venues

---

**Document prepared:** 2026-07-04

For questions or to dive deeper into specific algorithm families or evaluation methodologies, refer to the cited papers or conduct targeted searches in the identified venues.

---

## Part 10: Snowball Citation Search Update (2026-07-23)

Backward + forward citation search from the 10 anchor papers (FedAPA, FedAWA, FedNolowe, FeCoGraph, FedGMKD, FedFSL-CFRD, FedMADE, SBFedNIDS, FD-IDS, FedSWA), plus targeted phrase and venue searches. Goal was saturation, not exhaustiveness. Papers already listed in Parts 2–8 are not repeated here.

### Scooping Risk Assessment

**Verdict: No near-duplicate of FedGRAA's exact contribution found.** Nothing in the search combines (a) a gradient-based *refinement* mechanism, (b) applied to IDS specifically, and (c) framed around heterogeneous/limited-data clients, under matching or near-matching terminology.

That said, three papers are close enough on individual axes that they must be read in full and explicitly distinguished in the related-work section, because a reviewer familiar with this space will raise them unprompted:

1. **NIDS-FGPA (PLOS ONE, Oct 2024)** — "Gradient Similarity Model Aggregation" (GSA) for federated NIDS. This is the only paper found that combines *gradient-based aggregation weighting* with *federated network intrusion detection* — the same two-way intersection FedGRAA sits in. It differs mechanically (similarity-to-reference weighting for communication reduction, paired with Paillier homomorphic encryption, motivated by incomplete features/missing labels — not adaptive averaging for heterogeneity/limited data) and sits in a non-A*/A venue. Still, this is the single closest paper found in the entire search and should be cited and distinguished explicitly, not just bucketed into "gradient-based FL" generally.
2. **FedVG (arXiv 2602.21399, Feb 2026)** — "Gradient-Guided Aggregation": weights clients by the norm of validation-set gradients per layer (flatter landscape → more influence). Structurally the closest *general-FL* analogue to "gradient-refined adaptive averaging" found — but it is not IDS-specific, targets vision/medical-imaging benchmarks, and requires a public validation set (a meaningfully different assumption from FedGRAA). Very recent (posted after this review's original literature pass); worth a citation-alert going forward since it is unreviewed arXiv-only as of this date.
3. **Fed-LAA (journal, ScienceDirect, June 2026)** — layer-wise aggregation weights derived from gradient *angular similarity* between local updates and the global descent direction. Same "aggregation weight from gradient information" family as FedGRAA, but operates at the layer level, is not IDS-specific, and is evaluated on vision/structured benchmarks, not network traffic.

None of these three use the phrase "gradient-refined" or an equivalent, and a direct search for that phrase plus variants ("gradient refinement," "gradient-refinement") returned no matches to FedGRAA's mechanism. Recommended action: read NIDS-FGPA, FedVG, and Fed-LAA in full before the next draft and add each to related work with an explicit one-sentence differentiation (mirroring the treatment already given to FedAPA/FedAWA in Part 7), and re-run this specific check again closer to submission since FedVG and Fed-LAA are only weeks old and each could spawn IDS-focused follow-ups quickly.

### New Papers Found

#### New Competitors (adaptive/gradient-based aggregation, or few-shot/prototype methods specifically for IDS)

| Paper | Venue | Year | Key Focus | How It Differs from FedGRAA | Found via |
|-------|-------|------|-----------|------------------------------|-----------|
| NIDS-FGPA | PLOS ONE | 2024 | Gradient Similarity Aggregation (GSA) — dynamically weights client updates by similarity, paired with Paillier encryption, for federated NIDS with incomplete features/missing labels | Similarity-based weighting for communication/security, not adaptive averaging for heterogeneity or limited data; non-A*/A venue | Search: "adaptive gradient averaging federated learning IDS" |
| StatAvg | arXiv 2405.13062 / IEEE journal 2025 | 2024/2025 | Clients share per-feature statistics (not gradients); server builds global statistics to correct non-IID feature skew before/alongside any aggregation strategy, for FL-IDS | Statistics-sharing preprocessing step, not a gradient-based aggregation-weight mechanism; can be composed with FedGRAA rather than compete with it | Search: "federated learning intrusion detection non-IID limited data" |
| FedVG | arXiv 2602.21399 | 2026 | Gradient-Guided Aggregation — validation-set layer-wise gradient norms set client weights (flatter landscape = more influence) | General FL (vision/medical imaging), not IDS; requires a public validation set; weighting signal is gradient *norm*, not a refinement step | Snowball from "gradient refinement federated learning aggregation" search |
| Fed-LAA | ScienceDirect journal | 2026 | Layer-wise aggregation weights from gradient angular similarity to the global descent direction | General FL, not IDS; layer-level granularity; alignment-based, not refinement-based | Snowball from "adaptive gradient averaging federated learning IDS" search |
| FedAAW | Knowledge-Based Systems | 2024 | Adaptive aggregate weights recomputed each round to track changing data-heterogeneity/optimization direction | General FL, not IDS-specific; addresses aggregation-error drift over rounds rather than a gradient-refinement step | Search: "adaptive gradient averaging federated learning IDS heterogeneous data" |
| FedAgg | arXiv 2303.15799 | 2023 | Adaptive per-client learning rate derived from an aggregated-gradient mean-field term | General FL, learning-rate adaptation rather than aggregation-weight adaptation; not IDS | Search: "gradient refinement federated learning aggregation" |
| FedAgile (CIKM 2025) | CIKM | 2025 | Aggregated-gradient term used to design adaptive per-round learning rate | General FL, same family as FedAgg; not IDS; addresses learning-rate not aggregation-weight design | Snowball from FedAgg |
| PROTEAN | ESORICS | 2025 | Prototype-based knowledge sharing (class prototypes) for federated IDS under highly non-IID attack distributions | Prototype-exchange approach (model-free knowledge sharing), not aggregation-weight refinement; adds an IDS-specific prototype method missing from the existing "Family 2" table | Search: "federated learning intrusion detection non-IID limited data 2026" |
| PerFedHypID | Scientific Reports | 2025 | Hypernetwork-based personalized aggregation using client embedding vectors instead of model weights, for IDS | Embedding/hypernetwork mechanism, not gradient-based; personalization-focused; IDS-specific competitor missing from Part 6/Part 7 tables | Search: "PAKDD federated learning intrusion detection aggregation" |
| FedP3E | IEEE TIFS | 2025 | GMM-based class prototype exchange with SMOTE augmentation for non-IID IoT malware detection, cross-silo FL | Prototype-exchange, not aggregation-weight mechanism; malware detection (IDS-adjacent) rather than NIDS | Snowball from PROTEAN |
| FedAWR | MDPI Future Internet | 2026 | Aggregation optimization jointly adapting client weights and per-client learning rates based on compute capability | General FL, not IDS; weighting signal is compute capability, not gradient information | Snowball from "PAKDD adaptive aggregation federated gradient" search |

#### New Context (tangentially relevant — cite for motivation/framing, not direct competition)

| Paper | Venue | Year | Key Focus | Relevance | Found via |
|-------|-------|------|-----------|-----------|-----------|
| EdgeDetect | arXiv 2604.14663 | 2026 | Gradient compression (binarization) + Paillier homomorphic aggregation for bandwidth-constrained federated IDS in 6G-IoT | Communication/security-efficiency angle on FL-IDS, not aggregation-weight design; useful for "related FL-IDS systems" framing | Search: "gradient refined federated averaging aggregation intrusion detection" |
| FSFL-IDS | Springer chapter | ~2025 | Few-shot FL for intrusion detection in dynamic UAV/FANET networks; noted as sensitive to non-IID drift | Confirms the field's own literature identifies non-IID sensitivity in few-shot FL-IDS as an open problem — good motivating citation | Search: "federated learning intrusion detection non-IID limited data" |
| G-PFL-ID | IoT (MDPI) | 2026 | Graph-driven personalized FL for unsupervised intrusion detection in non-IID IoT systems | Graph + personalization angle, unsupervised setting; adjacent to FeCoGraph/graph family already in doc | Search: "federated learning intrusion detection non-IID limited data 2026" |
| Survey: "Survey on Federated Learning for IDS: Concept, Architectures, Aggregation Strategies, Challenges, and Future Directions" | ACM Computing Surveys | 2024 | Comprehensive survey explicitly taxonomizing FL-IDS aggregation strategies | Strong citation for motivating the "no prior gradient-refined aggregation for IDS" gap claim in Part 8/9 — check whether it already lists a gradient-refinement method (it does not, as of this search) | Search: "Survey Federated Learning Intrusion Detection System Aggregation Strategies" |
| Survey: "Intrusion Detection Based on Federated Learning: A Systematic Review" | ACM Computing Surveys | 2025 | Systematic review of FL-based IDS literature | Same use as above — corroborating survey to cite alongside the 2024 one | Search: "Survey Federated Learning Intrusion Detection System Aggregation Strategies" |
| Tri-LLM Cooperative Federated Zero-Shot IDS | arXiv 2602.00219 | 2026 | LLM-based zero-shot IDS with trust-aware aggregation across cooperating LLMs | Different modeling paradigm (LLM-based, zero-shot) — shows the field is diversifying past classic weight-averaging FL-IDS; not a competitor | Search: ""PAKDD" intrusion detection federated 2024 2025 2026" |

### Note on Search Saturation

This pass took roughly three rounds and ~25 distinct queries (10 anchor-paper lookups, ~10 targeted phrase/topic searches, and a second-level snowball off the five most promising first-round finds — NIDS-FGPA, StatAvg, FedVG, PROTEAN, and Fed-LAA). By the third round, new queries were returning largely the same handful of papers already surfaced (FedAgg, the CIKM aggregated-gradient paper, the layer-wise aggregation family, FedAAW) rather than novel ones — a reasonable saturation signal for a snowball search of this scope, though not a guarantee against non-English-indexed or very recent (post-July-2026) preprints. PAKDD's own proceedings (2023–2026, checked via dblp) show federated learning papers most years but none on IDS or gradient/adaptive aggregation specifically — PAKDD has not previously published in FedGRAA's exact niche, which is a positive signal for novelty at the target venue.
