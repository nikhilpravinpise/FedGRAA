Computer Networks 280 (2026) 112210

Contents lists available at ScienceDirect

# Computer Networks

ELSEVIER logo

journal homepage: www.elsevier.com/locate/comnet

Computer Networks journal cover thumbnail

# FedPGO : A personalized federated learning framework with gradient orthogonalization for IoT intrusion detection

Check for updates icon

Dawit Dejene Bikila ORCID icon <sup>\*</sup>, Jan Čapek ORCID icon

Institute of System Engineering and Informatics, Faculty of Economics and Administration, University of Pardubice, Studentska 95, Pardubice, 53009, Czech Republic

## a r t i c l e i n f o

**Keywords:**
IoT
Personalized federated learning
Deep learning
Intrusion detection system
Personalized gradient fusion

## a b s t r a c t

In the era of the Internet of Things (IoT), distributed infrastructures, in which heterogeneous devices autonomously share data pose significant privacy and security challenges. Although Machine Learning (ML) techniques have improved attack detection, sharing data for centralized training introduces privacy risks. Traditional Federated Learning (FL) approaches mitigate these risks by training models locally; however, they often assume client homogeneity, leading to performance degradation under highly Non-Independent and Identically Distributed (Non-IID) data. Personalized Federated Learning (PFL) has emerged as a promising paradigm in addressing privacy concerns and statistical heterogeneity. However, global gradients tend to align with dominant local gradients, suppressing informative updates from less dominant directions. This gradient dominance reduces generalization, under-represents diverse client contributions, and increases the risk of overfitting to majority data distributions. To address these challenges, we propose a Personalized Federated learning with Gradient Orthogonalization (FedPGO) framework that mitigates gradient dominance and enhances personalization in Non-IID IoT settings. FedPGO decomposes local gradients into components parallel and orthogonal to the global gradient and adaptively reweights them using an Adaptive Gradient Orthogonalization (AGO) mechanism. In addition, a Personalized Gradient Fusion (PGF) strategy aggregates client updates through a personalization-aware scoring function that jointly considers gradient directionality and orthogonality. A parameter-split model structure further enables independent optimization of shared and personalized layers for fine-grained client adaptations. Extensive experiments on CICIoT2023 and CICIDS2017 demonstrate that FedPGO outperforms state-of-the-art FL and PFL methods with higher accuracy and robustness for IoT intrusion detection under Non-IID conditions.

## 1. Introduction

The adoption of the Internet of Things (IoT), in which heterogeneous devices connect autonomously to share data, poses significant privacy and security challenges. These challenges are further intensified by the inherent design of IoT devices, built to collect and generate sensitive data, which makes them vulnerable to cyber-attacks [1]. This reveals that the need to recognize malicious behaviors and develop a novel intrusion detection approach for IoT is crucial [2].

Machine Learning (ML) and Deep Learning (DL) methods have improved attack detection in IoT security and gained popularity among researchers. However, these methods require raw training data to be collected in a single agent or server for centralized model training [3]. Sharing large data with a central agent is resource-demanding, associated with data latency and data leakage that raises potential privacy concerns in IoT [4,5]. Thus, industries are faced with security breaches and are forced to restrict data sharing for analysis. An attack on one device can lead to unforeseen outcomes on other connected devices [6].

Consequently, Federated Learning (FL) is a cutting-edge alternative that has emerged as a privacy-preserving solution to centralized ML-based intrusion detection challenges and achieves promising results [7].

FL has a capacity to facilitate clients to collaboratively train a shared model and then aggregate into a global model without exposing raw sensitive data, unlike traditional methods. This approach enhances data privacy, mitigates a single-point of failure, and reduces high False Alarm Rates (FAR). Thus, FL is a potentially feasible solution for IoT attack detection frameworks [6,8].

However, FL effectiveness is highly dependent on the statistical characteristics of client data, severe class imbalance, and limited attack samples in IoT applications. In practice, client data are typically Non-Independent and Identically Distributed (Non-IID) due to device heterogeneity, which degrades model convergence and global

<sup>\*</sup> Corresponding author.
E-mail addresses: [dawitdejene.bikila@student.upce.cz](mailto:dawitdejene.bikila@student.upce.cz) (D.D. Bikila), [capek@upce.cz](mailto:capek@upce.cz) (J. Čapek).

[https://doi.org/10.1016/j.comnet.2026.112210](https://doi.org/10.1016/j.comnet.2026.112210)

Received 17 October 2025; Received in revised form 4 February 2026; Accepted 8 March 2026
Available online 10 March 2026

1389-1286/© 2026 Elsevier B.V. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

D.D. Bikila and J. Čapek

Computer Networks 280 (2026) 112210

performance [9]. Additionally, a higher number of parameter exchanges heavily burdens IoT devices, impose communication and privacy risks (e.g., membership inference, model extraction, inversion), while recent studies highlight the need for aggregation strategies tailored to resource-constrained IoT settings to enhance robustness with minimal rounds [2,10,11].

FL methods such as parameter averaging (FedAvg), regularization-based approaches (FedProx), gradient correction (SCAFFOLD), and model-splitting techniques (MOON) have been proposed. However, these methods employ uniform aggregation and are suboptimal for robust heterogeneous IoT applications. To mitigate the challenges posed by Non-IID data, Personalized Federated Learning (PFL) has emerged as a promising approach to develop client-specific models using private data alongside co-training techniques [2,12]. Notable PFL methods include FedALA [13], which adaptively adjusts the aggregation weights to form the global model; FedMCA [12], which aggregates multiple classifiers through a learnable personalization layer; and MiniPFL [14], which selectively prioritizes high-value clients via a two-part framework. However, they do not explicitly account for gradient dominance in the aggregation process.

Furthermore, we have observed that the global gradients often aligned with dominant local gradients, leading to a loss of valuable information in less dominant directions in Non-IID settings. This gradient dominance reduces model generalization, under-represents diverse client contributions, and increases overfitting to majority data distribution. Hence, ensuring security becomes a challenge in Non-IID FL settings [15]. This highlights the need for methods that effectively handle data heterogeneity, personalization, and robustness in Non-IID IoT infrastructures.

To address these, we propose FedPGO, a novel PFL framework that integrates gradient structure with client-level personalization. FedPGO decomposes the dominant and less dominant client’s gradient into components aligned and orthogonal to the global direction, dynamically reweights and fuses updates using a personalization-aware aggregation strategy. It combines Adaptive Gradient Orthogonalization (AGO) with Personalized Gradient Fusion (PGF), and employing a parameter-split model structure, FedPGO preserves minority information, enhances convergence in less rounds. It enables fine-grained client adaptation without compromising global generalization. Extensive experiments demonstrate that FedPGO outperforms baselines and state-of-the-art methods in both accuracy and personalization under Non-IID federated environments. Our key contributions are summarized as follows:

* We propose FedPGO, a personalized federated learning framework for IoT intrusion detection that mitigates gradient dominance through gradient orthogonalization and personalization-aware aggregation, improving robustness and personalization under Non-IID data without additional communication overhead.

* PGF, a personalization-aware aggregation strategy dynamically, reweights and fuses client updates based on gradient structure and personalization effectiveness, enabling fine-grained adaptation while preserving global generalization.

* AGO is incorporated to decompose local updates into parallel and orthogonal components relative to the global gradient direction. Balances integration of dominant and minority updates, to improve global convergence, and preserve local diversity.

* Extensive evaluations on CICIoT2023 and CICIDS2017 demonstrate that FedPGO outperforms state-of-the-art methods in accuracy and personalization under Non-IID settings.

The rest of the paper is structured as follows: Section 2 presents related works. Section 3 provides a detailed explanation on the proposed PFL framework. In Section 4, the experimental setup and evaluation metrics are explained. The discussion of experimental results and a comparison is given in Section 5. The future work and conclusion are expressed in Section 6.

# 2. Related work

In this section, we provide background related to ML and DL based anomaly detection methods for IoT. Prior FL and PFL-based frameworks proposed to address challenges related to Non-IID are also presented thematically with their limitations .

## 2.1. ML and DL-based anomaly detection methods

ML and DL methods have improved attack detection in IoT security and have gained popularity among researchers. These techniques achieved state-of-the-art performance in IoT security, outperforming traditional methods [20]. Early studies explored a wide range of centralized models, including Deep Neural Networks (DNN), Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory (LSTM), and Auto-Encoders (AE), demonstrating strong detection capabilities in centralized learning settings [21].

Several works proposed hybrid or specialized approaches to address challenges such as class imbalance and overfitting. For instance, Random Forest combined with Fuzzy Logic was introduced to mitigate imbalance-related risks, while gradient boosting models such as XGBoost were integrated with rough set theory to improve performance on independent and imbalanced datasets [22,23]. Other studies focused on one-dimensional CNNs with class-weighting techniques to enhance minority class detection [20]. On the other hand, hybrid temporal–spatial models, such as LSTM-CNN, further improved detection accuracy by capturing sequential and spatial dependencies [24]. However, LSTM-CNN is resource intensive for constrained IoT devices.

Collectively, centralized ML and DL approaches require raw training data to be collected on a central server, this raises privacy risks due to data leakage. Moreover, the Non-IID data distribution of IoT is not considered in these methods. These limitations motivated the adoption of decentralized learning approaches, particularly FL and PFL.

## 2.2. FL-based anomaly detection methods

FL has increasingly been adopted for IoT intrusion detection to tackle privacy risks in centralized ML and DL approaches without exchanging raw training data. In traditional FL, a single unified global model is collaboratively trained on distributed clients. Method like FedAvg [25] improves privacy and communication efficiency through multiple local updates, but struggles in Non-IID settings. To address this, a regularization-based extension FedProx [26] adds a proximal term, while a gradient correction method SCAFFOLD [27] uses control terms to enhance local update drift. However, these methods assume that a single global model can generalize across heterogeneous devices.

Recent FL-based IDS frameworks employ ensemble models, hybrid architectures, and secure aggregation mechanisms to explicitly tackle client drift and model bias in Non-IID data. For example, FL approaches such as a federated ensemble Intrusion Detection System (IDS) [16] that employs hybrid CNN-GRU and LSTM-GRU with wFedAvg and wFedProx aggregation methods was proposed and achieved high detection accuracy on CICIoT2023 and FLNET2023 while, a blockchain-based FL IDS [28] integrates security mechanisms into FL, incorporating Self-Sovereign Identity (SSI) and trimmed-mean aggregation shows higher accuracy in CICIoT2023. Another method [18] introduced a dual server-side correction with a momentum-like gradient update shows high accuracy on UNSW-NB15 and CICIDS2018. Other work, Yang et al. [29] proposed a server-side mechanism to filter out clients before global aggregation incorporating a scoring system with Manhattan similarity and K-Means clustering achieved significant accuracy boost on CICIDS2017. However, it may converge to a local minimum and perform poorly on unevenly distributed clusters. A single dataset is insufficient; broader real-world testing is needed for robustness. Additional approaches address heterogeneity using a distribution-aware or clustering-oriented strategies, such as [17] introduces a DNN trained under both IID and

<page_number>2</page_number>

*Computer Networks 280 (2026) 112210*
*D.D. Bikila and J. Čapek*

**Table 1** Summary of FL based works.

<table>
  <thead>
    <tr>
      <th>Work</th>
      <th>Year</th>
      <th>Method</th>
      <th>PFL</th>
      <th>Contributions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>[16]</td>
      <td>2024</td>
      <td>CNN-GRU/LSTM-GRU</td>
      <td>No</td>
      <td>Ensemble IDS IoT attacks</td>
    </tr>
    <tr>
      <td>[17]</td>
      <td>2025</td>
      <td>DNN</td>
      <td>No</td>
      <td>Blockchain-enabled FL-based IDS</td>
    </tr>
    <tr>
      <td>[12]</td>
      <td>2025</td>
      <td>CNN</td>
      <td>Yes</td>
      <td>PFL framework that uses multiple classifier aggregation</td>
    </tr>
    <tr>
      <td>[18]</td>
      <td>2025</td>
      <td>MLP</td>
      <td>No</td>
      <td>IDS dual server-side correction</td>
    </tr>
    <tr>
      <td>[19]</td>
      <td>2024</td>
      <td>CNN</td>
      <td>Yes</td>
      <td>Dual knowledge transfer to improve personalization and generalization of client models</td>
    </tr>
    <tr>
      <td>Ours</td>
      <td>2025</td>
      <td>1D-CNN</td>
      <td>Yes</td>
      <td>PFL based IDS for non-IID settings</td>
    </tr>
  </tbody>
</table>

Non-IID settings to secure model aggregation achieving strong performance in binary classification on CICIoT2023. However, multiclass tasks remain moderate, indicating the need for improvement in handling complex attacks in Non-IID settings [11]. A Federated clustering ensemble algorithm [30] further improves convergence stability and detection precision across the UNSW-NB15, CIC-IDS2017/2018, CIC-DDoS2019, and BCCC-DDoS2024 datasets using a distribution-based aggregated weights and adaptive cluster initialization via OneBiPartition, while FedWDP [15] a Wasserstein distance regularization and differential privacy reduce divergence. Overall, in Non-IID settings, poor cluster initialization and uniform

<table>
  <caption>System Architecture Diagram</caption>
  <tr>
    <td>This image is a schematic diagram of a system architecture and does not contain quantitative data for a table.</td>
  </tr>
</table>

vacy reduce divergence.

Overall, in Non-IID settings, poor cluster initialization and uniform aggregation may bias the global model toward dominant data distributions. These FL methods overlook the misalignment between global optimization objectives and the personalized needs of individual clients. These limitations reveal the need for PFL, which better aligns the global objective with the requirements of the individual client.

## *2.3. PFL-based anomaly detection methods*

PFL extends standard FL by enabling model components to adapt to client-specific data, to address the mismatch between global and local optimization objectives. Several PFL methods have recently been introduced for IoT intrusion detection designed to tackle challenges related to Non-IID and privacy-sensitive data. A multiple classifier aggregation PFL FedMCA [12] framework that incorporates knowledge distillation to improve generalization. While, it improves accuracy compared to existing FL and PFL methods, it complex and aggregated classifiers may encode divergent distributional biases, making the distilled representation unstable and sensitive to dominant clients. Other PFL framework pFedCross [31], employed collaborative cross-aggregation to minimize gradient divergence, outperforms FedAvg, FedProx, and FedCross on NSL-KDD, CICIDS2017, CICIoT2023, and BoT-IoT datasets. The approach improves fairness, unlike previous FL methods that use uniform aggregation. However, clients with larger data typically achieve higher validation accuracy, thus receiving a larger $\alpha$-fairness, which may still lead to dominance effects.

A Self-labeled PFL approach SOH-FL [32] introduced a cosine simi-

**Fig. 1.** Basic workflow of the proposed framework.

To this end, we propose FedPGO, a novel PFL framework that integrates gradient structure with client-level personalization. FedPGO decomposes each client’s gradient into components aligned and orthogonal to the global direction, and dynamically re-weights and fuses them via a personalization-aware aggregation strategy and parameter-split model design. This preserves minority information, accelerates convergence, and enables fine-grained client adaptation without compromising global generalization.

lead to dominance effects.

A Self-labeled PFL approach SOH-FL [32] introduced a cosine similarity based autoencoder (CT-AE) for feature extraction and an aggregation method (BS-Agg), outperforms baselines in CICIDS2017. However, it does not address the core issue of gradient dominance and the suppression of minority gradient directions. Other PFL frameworks [33] introduced Moreau envelope based model to capture knowledge from other clients while preventing overfitting to local data, while, pFedKT [19], which uses dual knowledge transfer to strengthen generalization, further demonstrated the advantages of PFL.

In summery, despite notable progress, existing FL- and PFL-based

# 3. Methodology

ther demonstrated the advantages of PFL.

In summery, despite notable progress, existing FL- and PFL-based IDS (see Table 1) remain limited under heterogeneous IoT data. FL relies on a single global model that degrades under Non-IID distributions, while many PFL methods incur additional overhead and employ fairness mechanisms that still favor dominant clients. Moreover, global gradients often align with majority updates, suppressing minority directions, which degrades generalization and increases overfitting. These limitations highlight the need for personalization mechanisms that explicitly address heterogeneity, convergence, and robustness in Non-IID federated settings.

In this section, a detailed discussion is presented on the proposed FedPGO framework, the workflow, dataset chosen and preprocessing. The data balancing, feature normalization and selection, and algorithms used are also described in detail.

The framework is designed to improve personalization and conver-

used are also described in detail.

The framework is designed to improve personalization and convergence in federated learning environments characterized by Non-IID data distributions and class imbalance. It integrates adaptive gradient decomposition, personalization-aware aggregation, and federated training using a parameter-split Conv1D model. The framework is tailored for Non-IID, multi-class IoT intrusion detection tasks.

<page_number>3</page_number>

*Computer Networks 280 (2026) 112210*
D.D. Bikila and J. Čapek

**Table 2** Sample distribution of CICIoT2023 dataset.

The input features and target labels were then separated, and the input space was standardized using a StandardScaler to ensure uniform scale across all features. This normalization step is critical for stable and accelerated convergence in gradient-based learning. The processed input features were directly fed into the proposed parameter-split Conv1D model, which integrates a globally shared convolutional extractor with locally personalized classification heads. This structure enables each client to learn meaningful representations from their local data while contributing to a robust global model under non-IID conditions. To replicate realistic FL conditions as in a real-world settings, we em-

<table>
<thead>
<tr>
<th>Label</th>
<th>Number of samples</th>
</tr>
</thead>
<tbody>
<tr>
<td>DDoS</td>
<td>1,994,255</td>
</tr>
<tr>
<td>DoS</td>
<td>473,862</td>
</tr>
<tr>
<td>Mirai</td>
<td>154,806</td>
</tr>
<tr>
<td>Benign</td>
<td>64,377</td>
</tr>
<tr>
<td>Spoofing</td>
<td>28,745</td>
</tr>
<tr>
<td>Recon</td>
<td>20,968</td>
</tr>
<tr>
<td>Web</td>
<td>1417</td>
</tr>
<tr>
<td>BruteForce</td>
<td>777</td>
</tr>
</tbody>
</table>

contributing to a robust global model under non-IID conditions.

To replicate realistic FL conditions as in a real-world settings, we employed a Dirichlet heterogeneous (Non-IID) data partitioning approach. In practical deployments, edge devices or clients typically encounter localized data patterns, which are rarely independent or identically distributed. To reflect this heterogeneity, we adopted a label-distribution-based Non-IID partitioning strategy. Specifically, each client is assigned a subset of attack categories, selected randomly from the available class set. To ensure a practical baseline, the benign class is enforced to be present in all client datasets. For each selected class, an unequal (random) number of samples is drawn to introduce both label distribution skew and data quantity skew. This results in clients having disjoint and imbalanced class distributions, while the global dataset remains complete when all clients are aggregated.

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be the full dataset with $C$ classes. The data in-

**Table 3** Sample distribution of CICIDS2017 dataset.
___

<table>
  <thead>
    <tr>
      <th>Label</th>
      <th>Number of samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><sup>BENIGN</sup></td>
      <td>1,594,540</td>
    </tr>
    <tr>
      <td>DoS</td>
      <td>167,775</td>
    </tr>
    <tr>
      <td>Portscan</td>
      <td>159,066</td>
    </tr>
    <tr>
      <td>DDoS</td>
      <td>95,144</td>
    </tr>
    <tr>
      <td>Infiltration - Portscan</td>
      <td>71,767</td>
    </tr>
    <tr>
      <td>Brute Force</td>
      <td>6933</td>
    </tr>
    <tr>
      <td>DoS Slowloris</td>
      <td>3859</td>
    </tr>
    <tr>
      <td>Botnet</td>
      <td>736</td>
    </tr>
    <tr>
      <td>Web Attack</td>
      <td>104</td>
    </tr>
    <tr>
      <td>Infiltration</td>
      <td>36</td>
    </tr>
    <tr>
      <td>Heartbleed</td>
      <td>11</td>
    </tr>
  </tbody>
</table>

plete when all clients are aggregated.

Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be the full dataset with $C$ classes. The data indices for each class $c \in \{1, 2, \dots, C\}$ are denoted by $\mathcal{I}_c = \{i \mid y_i = c\}$. For each client $k \in \{1, \dots, K\}$, a random subset $\mathcal{C}_k \subset \{1, \dots, C\}$ of classes is selected, such that $|\mathcal{C}_k| < C$. For each $c \in \mathcal{C}_k$, a random number of samples from $\mathcal{I}_c$ is allocated to client $k$, denoted as $\mathcal{D}_k$. This partitioning results in:

### *3.1. Dataset and pre-processing*

It is the preliminary stage of data preparation to render the dataset so that it fits for machine learning model training. This stage removes outliers, irrelevant data, and missing values from the dataset. A uniform scale of the whole features was generated initially using standard scalar normalization techniques.

$$\bigcup_{k=1}^{K} D_k = D, \quad D_i \cap D_j = \emptyset \text{ for } i \neq j \quad (1)$$

$k=1$

This approach creates statistical heterogeneity among clients and poses a personalization challenge for federated optimization algorithms. Prior to model training, all input features are standardized to ensure

a personalization challenge for federated optimization algorithms.

Prior to model training, all input features are standardized to ensure that they have zero mean and unit variance. This step is crucial to stabilize convergence and avoid dominance by high-magnitude features. We used the standard scalar method [2], which transforms each input feature $x_j$ <sup>as:</sup>

### 3.1.1. Dataset description
This study utilizes a re

### 3.1.1. Dataset description
This study utilizes a recently released, publicly accessible, and realistic IoT benchmark dataset CICIoT2023 that contains both benign and malicious traffic. This dataset <sup>serves</sup> as a benchmark for developing broad-spectrum intrusion detection models within IoT environments. It was sourced from the Canadian Institute for cyber security and is widely adopted in the research community for anomaly detection tasks [34]. The dataset originates from a real-world IoT environment and captures a comprehensive range of up-to-date, sophisticated, and rare cyberattacks, along with complex and typical network traffic patterns. It includes a total of 33 distinct attack types grouped into <sup>seven</sup> categories: Reconnaissance, DDoS, Brute Force, Recon, DoS, Mirai, Web-based, and Spoofing attacks, executed on an IoT setup comprising 105 devices. Additionally, this work used the CICIDS2017 dataset for further eval-

$$z_j = \frac{x_j - \mu_j}{\sigma_j} \quad \quad (2)$$

where $\mu_j$ and $\sigma_j$ are the mean and standard deviation of feature $x_j$ computed over the training set. This normalization ensures that all features contribute equally to the learning process and improves the numerical stability of the training.

## *3.2. Proposed FedPGO framework*

Spoofing attacks, executed on an IoT setup comprising 105 devices.

Additionally, this work used the CICIDS2017 dataset for further evaluation and benchmarking of the effectiveness and generalization of the proposed model in diverse attack scenarios. Also obtained from the Canadian Institute for Cybersecurity, this dataset contains 2,830,743 records with 79 features. Malicious network flows represent 19.70%, while the remaining 80.30% are benign. It encompasses a broad spectrum of real-world cyber threats, including six primary attack categories DoS, Botnet, Infiltration, Brute Force, Portscan, and Web Attacks [35]. Its comprehensive and high-fidelity features make it a standard benchmark for evaluating intrusion detection system performance. Tables 2 and 3 show the sample distribution of both dataset used in this work.

To enable effective training within the federated learning pipeline,

A central challenge under Non-IID conditions is gradient dominance, where clients with larger or more homogeneous data distributions generate gradients of higher magnitude, thereby disproportionately influencing global updates [36]. As a result, the aggregated model tends to overfit dominant distributions, suppress minority behaviors, and exhibit reduced detection robustness. Formally, in each communication round, the global update under standard federated averaging is:

$$w_{t+1} = w_t - \eta \sum_{i=1}^{K} p_i g_i, \quad \quad \quad (3)$$

where $g_i = \nabla \mathcal{L}_i(w_t)$ is the local gradient computed by client $i$, and $p_i$ is its aggregation weight determined by data size or participation ratio. Under Non-IID conditions, clients with larger or more homogeneous datasets produce gradients of higher magnitude, thereby dominating the aggregated update and degrading generalization.

To tackle the challenges of heterogeneous data and personalization

and 3 show the sample distribution of both dataset used in this work.

To enable effective training within the federated learning pipeline, all non-numeric attributes in both dataset were transformed into numeric form using label encoding, ensuring compatibility with deep learning models. The class labels were also converted into integer values for supervised learning. To reduce feature redundancy and enhance learning efficiency, Pearson correlation analysis was applied, and features exhibiting a correlation coefficient greater than 0.99 were removed.



To tackle the challenges of heterogeneous data and personalization in federated learning, FedPGO, a PFL framework that integrates gradient structure with client-level personalization. It decomposes each client’s

<page_number>4</page_number>

D.D. Bikila and J. Čapek
Computer Networks 280 (2026) 112210

gradient into components aligned and orthogonal to the global direction, and dynamically re-weights them through a personalization-aware aggregation scheme and a parameter-split model (Conv1D). This framework preserves minority information, accelerates convergence, and enables fine-grained client adaptation without sacrificing global generalization. The basic workflow is shown in Fig. 1.

The federated learning workflow follows these steps:

1. Shared Training: Clients jointly train shared feature layers and local classifiers using focal loss.

2. Personalization Phase: Clients fine-tune only the local classification using validation data.

3. Gradient Decomposition: Each client’s gradient is decomposed into global-aligned and orthogonal (local) components.

4. Fusion Scoring: Clients compute a personalized score to reflect both alignment and validation performance.

5. Score-Weighted Aggregation: The server aggregates client updates based on PGF scores to update the global model.

This workflow enables collaborative learning across clients while supporting personalization for local distributions, making it particularly effective for decentralized intrusion detection across IoT networks. FedPGO is designed to improve convergence stability under Non-IID data by explicitly controlling gradient interference during federated aggregation. By decomposing each client update into a component aligned with the global gradient direction and an orthogonal component capturing client-specific heterogeneity, FedPGO ensures that global optimization is primarily driven by consensus-preserving updates while personalized information is retained without dominating the aggregation process. The adaptive weighting mechanism further suppresses conflicting gradients that could otherwise induce oscillations or slow convergence. As a result, FedPGO preserves the convergence order of standard federated averaging under mild smoothness assumptions, while reducing gradient variance caused by data heterogeneity, leading to more stable and efficient convergence in practice without requiring nested optimization or auxiliary models.

## 3.3. Focal loss for imbalanced classification

To address the class imbalance present in IoT intrusion datasets, we used focal loss [[37]](https://doi.org/10.1109/TPAMI.2018.2856582), which modifies the standard cross-entropy to focus learning on hard-to-classify samples. Given the predicted probability $\hat{y}_t$ for the true class $t$, the focal loss is defined as:

$$ \mathcal{L}_{\text{focal}} = -\alpha_t (1 - \hat{y}_t)^\gamma \log(\hat{y}_t) \quad (4) $$

where, $\alpha_t \in [0, 1]$ class-specific weighting factor, $\gamma \geq 0$ focusing parameter and $\hat{y}_t$ predicted probability for the true class.

Here $\alpha_t \in [0, 1]$ denote the class-specific weighting factor, commonly employed to mitigate class imbalance by adjusting the relative importance of each class in the loss computation. The modulating factor $(1 - \hat{y}_t)^\gamma$ reduces the loss contribution from well-classified instances (i.e., those with $\hat{y}_t \approx 1$) and amplifies it for misclassified or ambiguous samples. This adaptive scaling mechanism ensures that the model focuses more on harder examples during training. The overall loss is subsequently aggregated either by averaging or summation across the batch, depending on the chosen reduction strategy. This formulation effectively suppresses the influence of easily classified samples while accentuating those that are misclassified, thereby enhancing the model’s capacity to learn from underrepresented or complex data distributions.

## 3.4. Parameter-split Conv1D architecture

Our architecture comprises two components: a globally trained shared feature extractor and a locally personalized classifier head.

This architecture enables global feature learning via $f_s$, while preserving local adaptability through $f_p$, crucial for personalization under Non-IID settings. For an input sample $x \in \mathbb{R}^{1 \times L}$, the prediction is obtained as:

$$ f(x) = f_p(f_s(x)) \quad (5) $$

where, $f_s(\cdot)$ is a shared convolutional feature extractor trained globally and $f_p(\cdot)$ is a personalized fully connected classifier head trained locally.

This architecture enables efficient feature learning while allowing clients to adapt their classification heads to their local data distributions.

## 3.5. Gradient decomposition: Adaptive gradient orthogonalization (AGO)

To preserve both global trends and local variations, we decompose each client’s local gradient as indicated in Algorithm 1. Let $g_i$ denote the flattened gradient of the local model at client $i$, and $g_G$ the previous round’s global gradient. FedPGO decomposes $g_i$ into two aligned (parallel) components as follows:

$$ g_{\parallel} = \frac{g_i \cdot g_G}{\|g_G\|^2 + \epsilon} \cdot g_G \quad (6) $$

Where, $g_{\parallel}$ is projection onto global direction, $g_{\perp}$ is orthogonal (personalization-relevant) component and $\epsilon$ is a small constant for numerical stability.

The orthogonal component is defined as:

$$ g_{\perp} = g_i - g_{\parallel} \quad (7) $$

To recombine the components, we apply a dynamic weighting strategy denoted as follows:

$$ g'_i = \alpha_i \cdot g_{\parallel} + (1 - \alpha_i) \cdot g_{\perp} \quad (8) $$

where $\alpha_i \in [0, 1]$ is a client-specific coefficient reflecting the importance of global vs. local features. The weighting coefficient $\alpha_i \in [0, 1]$ is client-specific and may be adapted based on data diversity or gradient alignment. This allows clients with highly diverse data to emphasize on their personalized component.

**Algorithm 1** Gradient decomposition via adaptive gradient orthogonalization (AGO).

1: **Input:** Local gradient $g_i$, global gradient $g_G$, personalization weight $\alpha_i$, constant $\epsilon$

2: **Output:** client gradient $g'_i$

3: **Initialization:** Compute projection scalar $c_i \leftarrow \frac{g_i \cdot g_G}{\|g_G\|^2 + \epsilon}$

4: Compute aligned component $g_{\parallel}^{(i)} \leftarrow c_i \cdot g_G$

5: Compute orthogonal component $g_{\perp}^{(i)} \leftarrow g_i - g_{\parallel}^{(i)}$

6: Recompose final gradient $g'_i \leftarrow \alpha_i \cdot g_{\parallel}^{(i)} + (1 - \alpha_i) \cdot g_{\perp}^{(i)}$

7: **return** $g'_i$

## 3.6. PGF: Personalized gradient fusion scoring

To guide the aggregation, we propose a Personalized Gradient Fusion (PGF) score $S_i$ that ranks client updates using gradient structure and personalization effectiveness.

$$ S_i = \lambda_1 \cdot \cos(g_i, g_G) + \lambda_2 \cdot \|g_{\perp}\| + \lambda_3 \cdot \Delta\mathcal{L}_{\text{val}} \quad (9) $$

where, $\cos(g_i, g_G) = \frac{g_i \cdot g_G}{\|g_i\| \cdot \|g_G\|}$ is the gradient alignment with global direction, $\|g_{\perp}\|$ is magnitude of personalized direction and $\Delta\mathcal{L}_{\text{val}} = \mathcal{L}_{\text{before}} - \mathcal{L}_{\text{after}}$ is the validation loss reduction after personalization.

Hyperparameters $\lambda_1, \lambda_2, \lambda_3$ balance the influence of each term and are tuned via validation performance.

## 3.7. Gradient aggregation with dynamic weights

To finalize the global model update, the server aggregates personalized updates from all participating clients using a score-based weighting

<page_number>5</page_number>

*Computer Networks 280 (2026) 112210*
D.D. Bikila and J. Čapek

<table>
  <thead>
    <tr>
      <th>Attack Type</th>
      <th>Client 1</th>
      <th>Client 2</th>
      <th>Client 3</th>
      <th>Client 4</th>
      <th>Client 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Web Attack</td>
      <td>16</td>
      <td>13</td>
      <td>9</td>
      <td>-</td>
      <td>4</td>
    </tr>
    <tr>
      <td>Portscan</td>
      <td>10,386</td>
      <td>1,092</td>
      <td>51,501</td>
      <td>-</td>
      <td>26,698</td>
    </tr>
    <tr>
      <td>Infiltration - Portscan</td>
      <td>1,295</td>
      <td>33,413</td>
      <td>13,354</td>
      <td>979</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Infiltration</td>
      <td>36</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Heartbleed</td>
      <td>11</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>DoS Slowloris</td>
      <td>-</td>
      <td>144</td>
      <td>18</td>
      <td>299</td>
      <td>1,857</td>
    </tr>
    <tr>
      <td>DoS</td>
      <td>-</td>
      <td>15,429</td>
      <td>2,119</td>
      <td>8,582</td>
      <td>39,291</td>
    </tr>
    <tr>
      <td>DDoS</td>
      <td>30,565</td>
      <td>15,712</td>
      <td>8,960</td>
      <td>401</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Brute Force</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>175</td>
      <td>-</td>
    </tr>
    <tr>
      <td>Botnet</td>
      <td>41</td>
      <td>83</td>
      <td>-</td>
      <td>-</td>
      <td>422</td>
    </tr>
    <tr>
      <td>BENIGN</td>
      <td>132,537</td>
      <td>121,805</td>
      <td>122,914</td>
      <td>299,976</td>
      <td>36,404</td>
    </tr>
  </tbody>
</table>

**Fig. 2.** Dirichlet distribution of labels in Non-IID split across clients (CICIDS2017).

<table>
<caption>Client 5</caption>
<thead>
<tr>
<th>Category</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Web</td>
<td>0</td>
</tr>
<tr>
<td>Spoofing</td>
<td>9841</td>
</tr>
<tr>
<td>Recon</td>
<td>3599</td>
</tr>
<tr>
<td>Mirai</td>
<td>0</td>
</tr>
<tr>
<td>DoS</td>
<td>0</td>
</tr>
<tr>
<td>DDoS</td>
<td>314298</td>
</tr>
<tr>
<td>BruteForce</td>
<td>114</td>
</tr>
<tr>
<td>Benign</td>
<td>14899</td>
</tr>
</tbody>
</table>

<table>
<caption>Client 1</caption>
<thead>
<tr>
<th>Category</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Web</td>
<td>286</td>
</tr>
<tr>
<td>Spoofing</td>
<td>0</td>
</tr>
<tr>
<td>Recon</td>
<td>2927</td>
</tr>
<tr>
<td>Mirai</td>
<td>0</td>
</tr>
<tr>
<td>DoS</td>
<td>106773</td>
</tr>
<tr>
<td>DDoS</td>
<td>668894</td>
</tr>
<tr>
<td>BruteForce</td>
<td>0</td>
</tr>
<tr>
<td>Benign</td>
<td>598</td>
</tr>
</tbody>
</table>

<table>
<caption>Client 3</caption>
<thead>
<tr>
<th>Category</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Web</td>
<td>633</td>
</tr>
<tr>
<td>Spoofing</td>
<td>6042</td>
</tr>
<tr>
<td>Recon</td>
<td>-</td>
</tr>
<tr>
<td>Mirai</td>
<td>33257</td>
</tr>
<tr>
<td>DoS</td>
<td>131687</td>
</tr>
<tr>
<td>DDoS</td>
<td>-</td>
</tr>
<tr>
<td>BruteForce</td>
<td>-</td>
</tr>
<tr>
<td>Benign</td>
<td>1191</td>
</tr>
</tbody>
</table>

<table>
<caption>Client 4</caption>
<thead>
<tr>
<th>Category</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Web</td>
<td>43</td>
</tr>
<tr>
<td>Spoofing</td>
<td>0</td>
</tr>
<tr>
<td>Recon</td>
<td>3715</td>
</tr>
<tr>
<td>Mirai</td>
<td>0</td>
</tr>
<tr>
<td>DoS</td>
<td>159315</td>
</tr>
<tr>
<td>DDoS</td>
<td>51144</td>
</tr>
<tr>
<td>BruteForce</td>
<td>0</td>
</tr>
<tr>
<td>Benign</td>
<td>4846</td>
</tr>
</tbody>
</table>

<table>
<caption>Dirichlet distribution of labels in Non-IID split across clients (CICIoT2023) - Client 2</caption>
<thead>
<tr>
<th>Label</th>
<th>Value</th>
</tr>
</thead>
<tbody>
<tr>
<td>Web</td>
<td>227</td>
</tr>
<tr>
<td>Spoofing</td>
<td>0</td>
</tr>
<tr>
<td>Recon</td>
<td>0</td>
</tr>
<tr>
<td>Mirai</td>
<td>31937</td>
</tr>
<tr>
<td>DoS</td>
<td>0</td>
</tr>
<tr>
<td>DDoS</td>
<td>604919</td>
</tr>
<tr>
<td>BruteForce</td>
<td>170</td>
</tr>
<tr>
<td>Benign</td>
<td>5189</td>
</tr>
</tbody>
</table>

**Fig. 3.** Dirichlet distribution of labels in Non-IID split across clients (CICIoT2023).

scheme. Unlike standard FedAvg, where each client’s update is equally or data-size weighted, FedPGO leverages the previously computed PGF scores $S_i$ to dynamically prioritize each client’s contribution. Hence, each client’s contribution is weighted and the global update is computed as:

First, the contribution weight $w_i$ for each client is computed as:

**Algorithm 2 Gradient aggregation with dynamic weights.** 1: **Input:** Personalized gradients $\{g'_k\}^K_{k=1}$, RGF scores $\{S_k\}^K_{k=1}$

****Algorithm 2** Gradient aggregation with dynamic weights.**
<hr/>
1: **Input:** Personalized gradients $\{g'_i\}_{i=1}^K$, PGF scores $\{S_i\}_{i=1}^K$, learning rate $\eta$, constant $\epsilon$ 2: **Output:** Updated global model $w_{t+1}$

rate $\eta$, constant $\epsilon$ 2: **Output:** Updated global model $w_{t+1}$ 3: Initialization: Compute total PGF scor

2: **Output:** Updated global model $w_{t+1}$ 3: **Initialization:** Compute total PGF score $Z \leftarrow \sum_{j=1}^{K} S_j + \epsilon$ 4: **for each client $i = 1, \dots, K$ do**

s: First, the contribution weight $w_i$ for each client is computed as:

<table>
  <thead>
    <tr>
      <th>Algorithm 2 Gradient aggregation with dynamic weights.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1: Input: Personalized gradients {g'<sub>i</sub>}<sup>K</sup><sub>i=1</sub>, PGF scores {S<sub>i</sub>}<sup>K</sup><sub>i=1</sub>, learning rate η, constant ε</td>
    </tr>
    <tr>
      <td>2: Output: Updated global model w<sub>t+1</sub></td>
    </tr>
    <tr>
      <td>3: Initialization: Compute total PGF score Z ← Σ<sup>K</sup><sub>j=1</sub> S<sub>j</sub> + ε</td>
    </tr>
    <tr>
      <td>4: for each client i = 1, ..., K do</td>
    </tr>
    <tr>
      <td>5:     Compute weight w<sub>i</sub> ← S<sub>i</sub> / Z</td>
    </tr>
    <tr>
      <td>6: end for</td>
    </tr>
    <tr>
      <td>7: Aggregate gradient g'<sub>G</sub> ← Σ<sup>K</sup><sub>i=1</sub> w<sub>i</sub> · g'<sub>i</sub></td>
    </tr>
    <tr>
      <td>8: Update global model: w<sub>t+1</sub> ← w<sub>t</sub> - η · g'<sub>G</sub></td>
    </tr>
    <tr>
      <td>9: return w<sub>t+1</sub></td>
    </tr>
  </tbody>
</table>

3: Initialization: Compute total PG 4: **for** each client $i = 1, \dots, K$ **do** 5: Compute weight $w_i \leftarrow \frac{S_i}{}$

4: **for** each client $i = 1, \dots, K$ **do** 5: $\quad$ Compute weight $w_i \leftarrow \frac{S_i}{Z}$ 6: **end for**

$$w_i = \frac{S_i}{\sum_{j=1}^{K} S_j + \epsilon} \quad (10)$$

5: Comp 6: **end for** 7: Aggregate

6: **end for** 7: Aggregate gradient $g'_G \leftarrow \sum_{i=1}^{K} w_i \cdot g'_i$ 8: Update global model: $w_{t+1} \leftarrow w_t - \eta \cdot g$

7: Aggregate gradient $g'_G \leftarrow \sum_{i=1} w_i \cdot g'_i$ 8: Update global model: $w_{t+1} \leftarrow w_t - \eta \cdot g'_G$ 9: **return** $w_{t+1}$

Where, $S_i$ is the PGF score of client $i$, reflecting personalization utility and gradient relevance, $\sum_{j=1}^{K} S_j$ is the normalization <sup>term</sup> that denotes the sum all client scores and $\epsilon$ is a small constant added for numerical stability to avoid division by zero. This normalization ensures that all weights $w_i \in [0, 1]$ and sum to

8: Update globa 9: **return** $w_{t+1}$



This normalization ensures that all weights ∈ [0, 1] and sum to approximately 1. Clients that demonstrate better alignment with the global direction, stronger orthogonal gradient magnitude that is person- alization potential, and greater improvement in validation performance receive higher weights.

Once weights are computed, the server performs a score-weighted aggregation of client gradients:

where, ′ is the recomposed update from client , combining aligned and orthogonal gradient components and ′ is the aggregated gradient used to update the global model.

This aggregation strategy is fundamentally personalization-aware. It allows the global model to be influenced more heavily by clients that contribute diverse yet meaningful gradients and demonstrate local im- provement. The integration of score weighting into aggregation ensures that clients with noisy, misaligned, or low-impact updates have reduced influence, thereby enhancing robustness and convergence across het- erogeneous client distributions. All the steps are clearly described in Algorithm 2.

In FL, the global objective is to minimize a loss function ༹( ), using distributed and disjoint datasets { 1 , 2 , … , } across clients. Let ∈ ℝ denote the global model parameters at communication round , and = | | the number of samples on client . The total number of samples is = ∑ =1 . The FL optimization objective can be written as:

where ༹ ( ) is the empirical risk over client ’s dataset can be describes as:

However, under Non-IID distributions, the gradients ∇༹ ( ) may di- verge significantly among clients. This gradient heterogeneity causes instability and hinders convergence of the global model.

To address this, FedPGO decomposes each client’s local gradient into orthogonal components with respect to the previous global gradient

<page_number>6</page_number>

D.D. Bikila and J. Čapek

Computer Networks 280 (2026) 112210

$g_G$. The decomposition is given by:

$$g_{||}^{(i)} = \frac{g_i \cdot g_G}{\|g_G\|^2 + \epsilon} \cdot g_G \tag{14}$$

$$g_{\perp}^{(i)} = g_i - g_{||}^{(i)} \tag{15}$$

The recomposed gradient update is a weighted combination of the two components:

$$g'_i = \alpha_i g_{||}^{(i)} + (1 - \alpha_i) g_{\perp}^{(i)} \tag{16}$$

where $\alpha_i \in [0, 1]$ is a client-specific coefficient balancing generalization and personalization.

To determine the importance of each client's contribution, FedPGO introduces the (PGF) score:

$$S_i = \lambda_1 \cos(g_i, g_G) + \lambda_2 \|g_{\perp}^{(i)}\| + \lambda_3 \Delta \mathcal{L}_{val}^{(i)} \tag{17}$$

where, $\cos(g_i, g_G) = \frac{g_i \cdot g_G}{\|g_i\| \cdot \|g_G\|}$ is the cosine similarity, $\|g_{\perp}^{(i)}\|$ is the norm of the client-specific gradient, $\Delta \mathcal{L}_{val}^{(i)}$ is the validation loss reduction after personalization and $\lambda_1, \lambda_2, \lambda_3$ are tunable hyperparameters.

The global model is updated using a score-weighted aggregation of personalized gradients:

$$w_{t+1} = w_t - \eta \sum_{i=1}^K \frac{S_i}{\sum_{j=1}^K S_j + \epsilon} \cdot g'_i \tag{18}$$

By isolating client-specific gradient components and dynamically scoring updates, FedPGO mitigates inter-client gradient conflicts, promotes stability, and enables effective personalization. Our approach addresses divergence in gradient directions caused by Non-IID data, which is the reality in IoT settings, thereby enhancing both convergence and generalization in heterogeneous federated settings like IoT. The Non-IID split of labels distribution of both datasets is shown in Figs. 2 and 3. These figures show the distribution of samples of classes for only the first 5 clients from the total of 10, used for the simulation on both datasets.

In summery, we have presented derivations of the gradient projection and orthogonal decomposition, showing how these components reduce directional variance and stabilize global model under Non-IID updates. AGO is designed for federated personalization and decomposes each client gradient with respect to the global gradient direction, explicitly separating globally consistent updates from client-specific components induced by Non-IID data. Importantly, the orthogonal component is not discarded but adaptively reused through Personalized Gradient Fusion, enabling controlled personalization rather than interference avoidance. This tight integration of orthogonalization and aggregation allows FedPGO to mitigate gradient dominance while preserving heterogeneity, with negligible computational and communication overhead. A unified parameter-split architecture that couples these elements into a coherent optimization framework. This formulation establishes a structured objective in which gradients, personalization signals, and aggregation dynamics are jointly modeled, thereby distinguishing FedPGO from earlier approaches.

## 3.9. Federated training with personalization

FedPGO enables personalization and stability in FL through gradient structure-aware decomposition AGO. It also employed Personalization-aware scoring and aggregation (PGF) for improved client adaptation. A random search is used for hyperparasite tuning, where each client samples multiple hyperparameter configurations uniformly at random from the search space and trains shortly on a local validation split. Then the server aggregates and selects the best parameters after receiving reports of validation losses from clients. The PFL mechanism is described in Algorithm 3. FedPGO provides a personalized and structurally adaptive federated learning framework that directly addresses the limitations of standard federated learning under heterogeneous data and Non-IID IoT settings. By leveraging full gradient decomposition, dynamic weighting,

and personalization-aware aggregation, FedPGO achieves enhanced personalization and robust convergence in diverse IoT settings. This makes the framework suitable for decentralized settings with data heterogeneity such as IoT.

**Algorithm 3** FedPGO federated training with random search hyperparameter tuning.

1: **Input:** Initial global model parameters $w_0$, number of rounds $R$, clients $C = \{1, \dots, K\}$, learning rate $\eta$

2: **Output:** Final global model $w_R$

3: Initialize global gradient history: $g_G \leftarrow \vec{0}$

4: **Random Search Initialization:**

5: Each client $i \in C$ samples multiple hyperparameter configurations uniformly at random from the predefined search space, trains shortly on a local validation split, and reports validation losses to the server

6: Server aggregates results across clients and selects the best-performing configuration $\theta^* = \{\eta, WD, E, B\}$

7: **for** each round $t = 1, \dots, R$ **do**

8: Server broadcasts $w_t$ and hyperparameters $\theta^*$ to all clients $i \in C$

9: **for** each client $i \in C$ **do**

10: Train local model $f_i$ on dataset $D_i$ for $E = \theta^*_{epochs}$ local epochs, batch size $B = \theta^*_{batch}$

11: Evaluate pre-personalization loss $\mathcal{L}_{before}^{(i)}$

12: Fine-tune personalized head using validation data

13: Evaluate post-personalization loss $\mathcal{L}_{after}^{(i)}$

14: Compute validation gain: $\Delta \mathcal{L}_{val}^{(i)} = \mathcal{L}_{before}^{(i)} - \mathcal{L}_{after}^{(i)}$

15: Compute local gradient: $g_i = \nabla \mathcal{L}_i(w_t)$

16: Decompose gradient: $g_i \rightarrow (g_{||}^{(i)}, g_{\perp}^{(i)})$

17: Recompose personalized gradient: $g'_i = \alpha_i \cdot g_{||}^{(i)} + (1 - \alpha_i) \cdot g_{\perp}^{(i)}$

18: Compute PGF score: $S_i$

19: **end for**

20: Normalize client weights: $w_i = \frac{S_i}{\sum_{j=1}^K S_j + \epsilon}, \quad \forall i \in C$

21: Aggregate global gradient: $g'_G = \sum_{i=1}^K w_i \cdot g'_i$

22: Update global model: $w_{t+1} = w_t - \eta \cdot g'_G$

23: Update gradient memory: $g_G \leftarrow g'_G$

24: **end for**

25: **return** $w_R$

# 4. Experiments

This section presents the experimental setup and the evaluation metrics that we have used in this work. we evaluated the proposed FedPGO framework under diverse federated intrusion detection scenarios. FedPGO is compared against recently relevant and widely adopted FL and PFL baselines across benchmark CICIoT2023 and CICIDS2017 datasets to assess its effectiveness Non-IID and imbalanced IoT traffic distributions. An ablation study is performed to quantify the contribution of AGO and PGF components. Additionally, evaluations were conducted to examine communication efficiency, convergence behavior, and robustness under client participation to reflect real-world deployment constraints in distributed IoT settings.

## 4.1. Experimental setup

**Datasets:** We have trained and experimentally validated the performance of our model on the recent CICIoT2023 and CICIDS2017 datasets. We have used the `train_test_split` function of scikit-learn and divided the dataset into training, testing, and validation sets for both datasets, with a proportion of 60%, 20%, and 20% respectively. To simulate realistic decentralized deployment, a Non-IID heterogeneity,we employed a Dirichlet distribution partitioning with ($\alpha = 0.3$). This introduces sufficient heterogeneity to highlight the benefits of personalization while maintaining training stability. The sample allocation per client remains

<page_number>7</page_number>

D.D. Bikila and J. Čapek

Computer Networks 280 (2026) 112210

adequate, even for minority classes, enabling robust local model updates on both datasets. These configurations represent practical factors such as rare attack occurrence, uneven device behavior, and divergent traffic modalities which are common in IoT environments. The choice of 10 clients also ensures that computation and communication overheads remain practical, allowing a larger number of global training rounds and faster convergence without excessive resource usage. This setup effectively captures statistical heterogeneity across clients, which is crucial for evaluating personalization in federated learning.

**Training setup:** FedPGO and all baseline methods were implemented within a federated architecture comprising 10 simulated IoT clients. A lightweight Conv1D-based intrusion detection was employed considering the limited memory and computing capacity of IoT devices. To find the best hyperparameters for our PFL method, we used Random Search, a commonly used approach for optimizing hyperparameters. It randomly selects configurations from set of ranges rather than checking every possible combination. This method is efficient and works well, especially in federated learning where training costs are high and all hyperparameters may not be equally important.

In our FedPGO framework, we first defined a search space for key hyperparameters like client learning rate, personalization coefficient, and number of local epochs. Each client then randomly sampled configurations, trained on a withheld validation set, and recorded the validation performance. The server collected these results and identified the configuration that performed best among all clients and then returned to all clients for further training. After tuning, the best parameter values found were: batch size of 64, weight decay of 0.00001, number of local epochs of 5, number of rounds of 25, and learning rate of 0.0005 for CICIoT2023; and batch size of 64, weight decay of 0.00001, number of rounds of 40, number of local epochs of 5, and learning rate of 0.0001 for CICIDS2017. Early stopping was set with a patience of 3 rounds and Adam as the optimizer for all methods to ensure fair comparison. Table 4 summarizes the parameter configurations specific to the data set and specific to baseline methods.

**Simulation Environment:** In this work, all models were simulated on a system configuration that utilizes an Intel (R) Core (TM) i5-12450HX processor, NVIDIA GTX 3050 6GB (GDDR6) GPU, CPU@ 2.40GHz, 24.0 GB Memory (RAM) and 1TB SSD run on a win-64-bit operating system. A Jupyter Notebook with Python 3.10 and its popular Python ML stack is used for model training and evaluation. To ensure reproducibility and minimize confounding hardware effects, all methods were trained under identical execution settings. Each experiment conducted follows FL baselines codes publicly available for consistency and parameter alignment.

**Compared methods:** The evaluation includes traditional FL and contemporary PFL methods chosen based on their relevance to Non-IID IoT constraints. The baselines we compared are: FedAvg [25] serves as the canonical FL baseline with global aggregation. FedProx [26] introduces a proximal term ($\mu = 0.1$) to reduce client drift. pFedMe [33] adopts a bi-level optimization strategy with Moreau envelope regularization ($\lambda = 15$, $K = 5$ inner steps). FedCross [38] enables collaborative feature-level sharing across clients. FedPAC [39] incorporates prototype-based knowledge distillation using feature alignment ($T = 2$, $\gamma = 1.0$). FedCP [40] separates global and personalized representations via conditional policies. These methods represent the major paradigms applied to address personalization, robustness, and resource limitations in the detection of decentralized IoT intrusions. Table 7 reports the comparative performance of FedPGO against these baselines under Non-IID IoT traffic distributions. A comparison of the experimental results with the state-of-the-art approaches is presented in the next section.

In our experiments, each client uses an identical Conv1D model, resulting in $\theta_k = 92,040$ trainable parameters for CICIoT2023 and $\theta_k = 174,347$ trainable parameters for CICIDS2017, which are a 32-bit floating-point representation (float32, 4 bytes per parameter). FedPGO incurs a communication cost per-round of $2 \sum_{k=1}^N \theta_k$, where N is the number of clients.

## 4.2. Result evaluation metrics

A machine learning model can be assessed using various metrics. In this work, we used f1-score, accuracy, recall, and precision. A confusion matrix is typically used to calculate these metrics. True Positive (TP) represents instances correctly predicted as attacks, while True Negative (TN) are correctly predicted as benign. False Negative (FN) indicates instances incorrectly predicted as benign, and False Positive (FP) indicates instances incorrectly predicted as attacks. The formulas for accuracy, precision, recall, and f1-score are as follows:

$$ Accuracy = \frac{(TP + TN)}{(TP + TN + FP + FN)} $$ (19)

$$ Precision = \frac{TP}{(TP + FP)} $$ (20)

$$ Recall = \frac{TP}{(TP + FN)} $$ (21)

$$ F1\text{-}score = \frac{2 \times (Precision \times Recall)}{Precision + Recall} $$ (22)

Precision is calculated as TP divided by the sum of TP and FP, as shown in Eq. (20). Recall is the ratio of TP to the sum of TP and FN, as given in Eq. (21). The F1-score is the harmonic mean of Precision and Recall, calculated as twice the product of Precision and Recall divided by their sum, as detailed in Eq. (22). The False Positive Rate (FPR) and True Positive Rate (TPR) were also employed to evaluate the effectiveness of the methods. These two rates can be visualized using the ROC curve. The equations for FPR and TPR are presented as follows: FPR is FP divided by the sum of FP and TN, as shown in Eq. (23), and TPR is TP divided by the sum of TP and FN, as shown in Eq. (24).

$$ FPR = \frac{FP}{(FP + TN)} $$ (23)

$$ TPR = \frac{TP}{(TP + FN)} $$ (24)

# 5. Result and discussion

This section outlines the experimental results of the proposed model and their implications for the overall attack classification task.

## 5.1. Performance of FedPGO

The model’s training and validation demonstrate a significant improvement in classification accuracy along with a reduction in loss. The validation loss and training loss lines decrease in a similar manner, indicating effective learning from unseen data over successive rounds, as illustrated in Fig. 4(a)). Furthermore, as depicted in Fig. 5(a)), the validation accuracy and training accuracy lines increase in a consistent pattern, showing effective learning with a strong fit and good generalization ability on unseen data across rounds.

In both datasets, the FedPGO framework has a minimal training and validation loss value less than 0.01. Both training and validation loss shows a decrease in the first few round which indicates a good learning. The stabilized losses at a low value also indicate convergence after round 5 and round 10 for the CICIoT2023 and CICIDS2017 datasets, respectively, as depicted in Fig. 4. This confirms the superior performance of FedPGO framework in a Non-IID FL settings. The confusion matrix in Fig. 6(a)) and (b) depicts the evaluation of the proposed model in classifying classes correctly and incorrectly predicted. Our model classified the majority of the classes with their actual respective categories correctly, including the minority attack classes. The minority classes, that models usually struggle to detect, are also detect well without using oversampling or balancing techniques. We have used the real data, as using synthetic data may lead to model poor performance and generalizability. A focal loss function that can handle the minority classes in an

<page_number>

8
</page_number>

*Computer Networks 280 (2026) 112210*
D.D. Bikila and J. Čapek

**Table 4** Parameter configuration of FL and PFL baseline methods on CICIoT2023 and CICIDS2017. All methods share the same Conv1D backbone, Adam optimizer, batch size of 64, weight decay of $1 \times 10^{-5}$, 5 local epochs, and early stopping with patience of 3 rounds.

**Table 4**

Parameter configuration of FL and PFL baseline methods on CICIoT2023 and CICIDS2017. All methods share the same Conv1D backbone, Adam optimizer, batch size of 64, weight decay of $1 \times 10^{-5}$, 5 local epochs, and early stopping with patience of 3 rounds.

<table>
  <thead>
    <tr>
      <th rowspan="2">Baseline</th>
      <th colspan="2">CICIoT2023</th>
      <th colspan="2">CICIDS2017</th>
      <th rowspan="2">Method specific Parameters</th>
    </tr>
    <tr>
      <th>LR</th>
      <th>Rounds</th>
      <th>LR</th>
      <th>Rounds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FedAvg</td>
      <td rowspan="6">5 × 10⁻⁴</td>
      <td rowspan="6">25</td>
      <td rowspan="6">1 × 10⁻⁴</td>
      <td rowspan="6">40</td>
      <td>Client fraction C = 1.0, sample-size weighted aggregation</td>
    </tr>
    <tr>
      <td><sup>FedProx</sup></td>
      <td>Proximal coefficient μ = 0.1</td>
    </tr>
    <tr>
      <td>pFedMe</td>
      <td>Regularization λ = 15, <sup>inner</sup> steps K = 5</td>
    </tr>
    <tr>
      <td><sup>FedCross</sup></td>
      <td>Feature exchange weight α = 0.5</td>
    </tr>
    <tr>
      <td><sup>FedPAC</sup></td>
      <td>Distillation temperature T = 2, prototype weight γ = 1.0</td>
    </tr>
    <tr>
      <td><sup>FedCP</sup></td>
      <td>Conditional policy learning rate η = 0.01</td>
    </tr>
  </tbody>
</table>

<table>
<caption>Loss over Rounds (b) CICIDS2017</caption>
<thead>
<tr>
<th>Rounds</th>
<th>Train Loss</th>
<th>Val Loss</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>0.052</td>
<td>0.056</td>
</tr>
<tr>
<td>1</td>
<td>0.025</td>
<td>0.026</td>
</tr>
<tr>
<td>2</td>
<td>0.021</td>
<td>0.022</td>
</tr>
<tr>
<td>3</td>
<td>0.018</td>
<td>0.019</td>
</tr>
<tr>
<td>4</td>
<td>0.015</td>
<td>0.016</td>
</tr>
<tr>
<td>5</td>
<td>0.013</td>
<td>0.015</td>
</tr>
<tr>
<td>6</td>
<td>0.012</td>
<td>0.014</td>
</tr>
<tr>
<td>7</td>
<td>0.011</td>
<td>0.013</td>
</tr>
<tr>
<td>8</td>
<td>0.010</td>
<td>0.012</td>
</tr>
<tr>
<td>9</td>
<td>0.009</td>
<td>0.012</td>
</tr>
<tr>
<td>10</td>
<td>0.008</td>
<td>0.010</td>
</tr>
<tr>
<td>11</td>
<td>0.009</td>
<td>0.010</td>
</tr>
<tr>
<td>12</td>
<td>0.008</td>
<td>0.011</td>
</tr>
<tr>
<td>13</td>
<td>0.008</td>
<td>0.010</td>
</tr>
<tr>
<td>14</td>
<td>0.008</td>
<td>0.010</td>
</tr>
<tr>
<td>15</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>16</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>17</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>18</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>19</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>20</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>21</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>22</td>
<td>0.006</td>
<td>0.009</td>
</tr>
<tr>
<td>23</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>24</td>
<td>0.008</td>
<td>0.010</td>
</tr>
<tr>
<td>25</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>26</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>27</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>28</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>29</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>30</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>31</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>32</td>
<td>0.007</td>
<td>0.009</td>
</tr>
<tr>
<td>33</td>
<td>0.007</td>
<td>0.010</td>
</tr>
<tr>
<td>34</td>
<td>0.007</td>
<td>0.009</td>
</tr>
</tbody>
</table>

<table>
<caption>Loss over Rounds (a) CICIoT2023</caption>
<thead>
<tr>
<th>Rounds</th>
<th>Train Loss</th>
<th>Val Loss</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>0.0365</td>
<td>0.0375</td>
</tr>
<tr>
<td>1</td>
<td>0.0092</td>
<td>0.0100</td>
</tr>
<tr>
<td>2</td>
<td>0.0080</td>
<td>0.0092</td>
</tr>
<tr>
<td>3</td>
<td>0.0082</td>
<td>0.0098</td>
</tr>
<tr>
<td>4</td>
<td>0.0075</td>
<td>0.0090</td>
</tr>
<tr>
<td>5</td>
<td>0.0078</td>
<td>0.0092</td>
</tr>
<tr>
<td>6</td>
<td>0.0074</td>
<td>0.0090</td>
</tr>
<tr>
<td>7</td>
<td>0.0073</td>
<td>0.0095</td>
</tr>
<tr>
<td>8</td>
<td>0.0074</td>
<td>0.0098</td>
</tr>
<tr>
<td>9</td>
<td>0.0073</td>
<td>0.0091</td>
</tr>
<tr>
<td>10</td>
<td>0.0075</td>
<td>0.0095</td>
</tr>
<tr>
<td>11</td>
<td>0.0071</td>
<td>0.0090</td>
</tr>
<tr>
<td>12</td>
<td>0.0070</td>
<td>0.0092</td>
</tr>
<tr>
<td>13</td>
<td>0.0073</td>
<td>0.0089</td>
</tr>
<tr>
<td>14</td>
<td>0.0073</td>
<td>0.0092</td>
</tr>
<tr>
<td>15</td>
<td>0.0072</td>
<td>0.0091</td>
</tr>
<tr>
<td>16</td>
<td>0.0073</td>
<td>0.0090</td>
</tr>
<tr>
<td>17</td>
<td>0.0074</td>
<td>0.0096</td>
</tr>
<tr>
<td>18</td>
<td>0.0069</td>
<td>0.0091</td>
</tr>
<tr>
<td>19</td>
<td>0.0070</td>
<td>0.0092</td>
</tr>
<tr>
<td>20</td>
<td>0.0069</td>
<td>0.0088</td>
</tr>
<tr>
<td>21</td>
<td>0.0068</td>
<td>0.0088</td>
</tr>
</tbody>
</table>

**Fig. 4.** FedPGO training-validation loss across datasets: (a) CICIoT2023 and (b) CICIDS2017.

<table>
<caption>Accuracy over Rounds</caption>
<thead>
<tr>
<th>Rounds</th>
<th>Train Accuracy</th>
<th>Val Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>0.936</td>
<td>0.936</td>
</tr>
<tr>
<td>1</td>
<td>0.990</td>
<td>0.989</td>
</tr>
<tr>
<td>2</td>
<td>0.991</td>
<td>0.990</td>
</tr>
<tr>
<td>3</td>
<td>0.991</td>
<td>0.9897</td>
</tr>
<tr>
<td>4</td>
<td>0.9915</td>
<td>0.9905</td>
</tr>
<tr>
<td>5</td>
<td>0.991</td>
<td>0.9902</td>
</tr>
<tr>
<td>6</td>
<td>0.9917</td>
<td>0.9904</td>
</tr>
<tr>
<td>7</td>
<td>0.9915</td>
<td>0.9903</td>
</tr>
<tr>
<td>8</td>
<td>0.9913</td>
<td>0.9901</td>
</tr>
<tr>
<td>9</td>
<td>0.9914</td>
<td>0.9902</td>
</tr>
<tr>
<td>10</td>
<td>0.9913</td>
<td>0.9900</td>
</tr>
<tr>
<td>11</td>
<td>0.9916</td>
<td>0.9904</td>
</tr>
<tr>
<td>12</td>
<td>0.9915</td>
<td>0.9903</td>
</tr>
<tr>
<td>13</td>
<td>0.9917</td>
<td>0.9905</td>
</tr>
<tr>
<td>14</td>
<td>0.9916</td>
<td>0.9904</td>
</tr>
<tr>
<td>15</td>
<td>0.9914</td>
<td>0.9902</td>
</tr>
<tr>
<td>16</td>
<td>0.9915</td>
<td>0.9903</td>
</tr>
<tr>
<td>17</td>
<td>0.9913</td>
<td>0.9900</td>
</tr>
<tr>
<td>18</td>
<td>0.9918</td>
<td>0.9904</td>
</tr>
<tr>
<td>19</td>
<td>0.9913</td>
<td>0.9903</td>
</tr>
<tr>
<td>20</td>
<td>0.9916</td>
<td>0.9903</td>
</tr>
<tr>
<td>21</td>
<td>0.9916</td>
<td>0.9901</td>
</tr>
</tbody>
</table>

<table>
<caption>Accuracy over Rounds (b) CICIDS2017</caption>
<thead>
<tr>
<th>Rounds</th>
<th>Train Accuracy</th>
<th>Val Accuracy</th>
</tr>
</thead>
<tbody>
<tr>
<td>0</td>
<td>0.9750</td>
<td>0.9735</td>
</tr>
<tr>
<td>1</td>
<td>0.9830</td>
<td>0.9828</td>
</tr>
<tr>
<td>2</td>
<td>0.9855</td>
<td>0.9852</td>
</tr>
<tr>
<td>3</td>
<td>0.9870</td>
<td>0.9868</td>
</tr>
<tr>
<td>4</td>
<td>0.9875</td>
<td>0.9872</td>
</tr>
<tr>
<td>5</td>
<td>0.9898</td>
<td>0.9895</td>
</tr>
<tr>
<td>6</td>
<td>0.9902</td>
<td>0.9898</td>
</tr>
<tr>
<td>7</td>
<td>0.9912</td>
<td>0.9908</td>
</tr>
<tr>
<td>8</td>
<td>0.9915</td>
<td>0.9909</td>
</tr>
<tr>
<td>9</td>
<td>0.9918</td>
<td>0.9912</td>
</tr>
<tr>
<td>10</td>
<td>0.9912</td>
<td>0.9910</td>
</tr>
<tr>
<td>11</td>
<td>0.9920</td>
<td>0.9918</td>
</tr>
<tr>
<td>12</td>
<td>0.9918</td>
<td>0.9912</td>
</tr>
<tr>
<td>13</td>
<td>0.9920</td>
<td>0.9915</td>
</tr>
<tr>
<td>14</td>
<td>0.9922</td>
<td>0.9918</td>
</tr>
<tr>
<td>15</td>
<td>0.9928</td>
<td>0.9922</td>
</tr>
<tr>
<td>16</td>
<td>0.9929</td>
<td>0.9925</td>
</tr>
<tr>
<td>17</td>
<td>0.9927</td>
<td>0.9920</td>
</tr>
<tr>
<td>18</td>
<td>0.9928</td>
<td>0.9922</td>
</tr>
<tr>
<td>19</td>
<td>0.9930</td>
<td>0.9925</td>
</tr>
<tr>
<td>20</td>
<td>0.9927</td>
<td>0.9920</td>
</tr>
<tr>
<td>21</td>
<td>0.9935</td>
<td>0.9930</td>
</tr>
<tr>
<td>22</td>
<td>0.9930</td>
<td>0.9925</td>
</tr>
<tr>
<td>23</td>
<td>0.9928</td>
<td>0.9924</td>
</tr>
<tr>
<td>24</td>
<td>0.9930</td>
<td>0.9928</td>
</tr>
<tr>
<td>25</td>
<td>0.9932</td>
<td>0.9929</td>
</tr>
<tr>
<td>26</td>
<td>0.9925</td>
<td>0.9920</td>
</tr>
<tr>
<td>27</td>
<td>0.9932</td>
<td>0.9928</td>
</tr>
<tr>
<td>28</td>
<td>0.9933</td>
<td>0.9928</td>
</tr>
<tr>
<td>29</td>
<td>0.9928</td>
<td>0.9920</td>
</tr>
<tr>
<td>30</td>
<td>0.9930</td>
<td>0.9922</td>
</tr>
<tr>
<td>31</td>
<td>0.9932</td>
<td>0.9925</td>
</tr>
<tr>
<td>32</td>
<td>0.9932</td>
<td>0.9926</td>
</tr>
<tr>
<td>33</td>
<td>0.9932</td>
<td>0.9926</td>
</tr>
</tbody>
</table>

**Fig. 5.** Convergence of FedPGO across datasets: (a) CICIoT2023 and (b) CICIDS2017.

imbalanced datasets is used to achieve this [37]. As a result, the model achieved improved minority class detection in a multiclass classification of IoT attacks. Further, the classification reports for CICIoT2023 and CICIDS2017

scores for dominant classes and strong generalization to minority and low-support categories. Despite severe class imbalance, FedPGO maintains high macro and weighted averages, indicating stable effectiveness across diverse traffic patterns. These results highlight the robustness of our approach and its ability to preserve accuracy even under challenging Non-IID conditions, see Tables 5 and 6.

of IoT attacks.

Further, the classification reports for CICIoT2023 and CICIDS2017 show that FedPGO achieves consistently high detection performance across heterogeneous attack scenarios, with perfect or near-perfect

<page_number>9</page_number>

*Computer Networks 280 (2026) 112210*
D.D. Bikila and J. Čapek

<table>
<caption>(a) CICIOT2023.</caption>
<thead>
<tr>
<th>True \ Predicted</th>
<th>Benign</th>
<th>BruteForce</th>
<th>DDoS</th>
<th>DoS</th>
<th>Mirai</th>
<th>Recon</th>
<th>Spoofing</th>
<th>Web</th>
</tr>
</thead>
<tbody>
<tr>
<td>Benign</td>
<td>12085</td>
<td>0</td>
<td>32</td>
<td>6</td>
<td>0</td>
<td>202</td>
<td>568</td>
<td>45</td>
</tr>
<tr>
<td>BruteForce</td>
<td>42</td>
<td>33</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>65</td>
<td>28</td>
<td>0</td>
</tr>
<tr>
<td>DDoS</td>
<td>2</td>
<td>0</td>
<td>398645</td>
<td>91</td>
<td>5</td>
<td>0</td>
<td>10</td>
<td>0</td>
</tr>
<tr>
<td>DoS</td>
<td>3</td>
<td>0</td>
<td>17</td>
<td>94652</td>
<td>4</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>Mirai</td>
<td>3</td>
<td>0</td>
<td>31</td>
<td>61</td>
<td>30901</td>
<td>17</td>
<td>9</td>
<td>0</td>
</tr>
<tr>
<td>Recon</td>
<td>751</td>
<td>0</td>
<td>0</td>
<td>13</td>
<td>16</td>
<td>3211</td>
<td>228</td>
<td>0</td>
</tr>
<tr>
<td>Spoofing</td>
<td>1048</td>
<td>1</td>
<td>20</td>
<td>0</td>
<td>0</td>
<td>552</td>
<td>4016</td>
<td>113</td>
</tr>
<tr>
<td>Web</td>
<td>14</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>186</td>
<td>113</td>
</tr>
</tbody>
</table>

**Fig. 6.** Confusion matrices using FedPGO: (a) CICIoT2023 and (b) CICIDS2017 datasets.

**Table 5** Performance of FedPGO on CICIoT2023.

particularly the adverse effects of Non-IID client data and the pursuit of robust global performance and generalizability. In contrast, by leveraging full gradient decomposition, dynamic weighting, and personalization-aware aggregation, FedPGO achieves enhanced personalization and robust convergence in diverse IoT settings. FedPGO demonstrated a significant improvement in personalization

<table>
<thead>
<tr>
<th>Class</th>
<th>Precision</th>
<th>Recall</th>
<th>F1-score</th>
<th>Support</th>
</tr>
</thead>
<tbody>
<tr>
<td>Benign</td>
<td>0.87</td>
<td>0.93</td>
<td>0.90</td>
<td>12,938</td>
</tr>
<tr>
<td>BruteForce</td>
<td>0.97</td>
<td>0.20</td>
<td>0.33</td>
<td>168</td>
</tr>
<tr>
<td>DDoS</td>
<td>1.00</td>
<td>1.00</td>
<td>1.00</td>
<td>398,753</td>
</tr>
<tr>
<td>DoS</td>
<td>1.00</td>
<td>1.00</td>
<td>1.00</td>
<td>94,676</td>
</tr>
<tr>
<td>Mirai</td>
<td>1.00</td>
<td>1.00</td>
<td>1.00</td>
<td>31,022</td>
</tr>
<tr>
<td>Recon</td>
<td>0.79</td>
<td>0.76</td>
<td>0.78</td>
<td>4219</td>
</tr>
<tr>
<td>Spoofing</td>
<td>0.80</td>
<td>0.70</td>
<td>0.74</td>
<td>5750</td>
</tr>
<tr>
<td>Web</td>
<td>0.42</td>
<td>0.36</td>
<td>0.39</td>
<td>313</td>
</tr>
<tr>
<td>Accuracy</td>
<td colspan="2">–</td>
<td>0.99</td>
<td>547,839</td>
</tr>
<tr>
<td>Macro Avg</td>
<td>0.86</td>
<td>0.74</td>
<td>0.77</td>
<td>547,839</td>
</tr>
<tr>
<td>Weighted Avg</td>
<td>0.99</td>
<td>0.99</td>
<td>0.99</td>
<td>547,839</td>
</tr>
</tbody>
</table>

alization and robust convergence in diverse IoT settings.

FedPGO demonstrated a significant improvement in personalization performance when evaluated using macro-averaged F1, which reflects per-class robustness under Non-IID data distributions. On CICIoT2023, FedPGO achieved a macro-F1 of 0.77, outperforming FedAvg (0.58), Fed-Prox (0.62), pFedMe (0.64), FedCross (0.69), FedPAC (0.73), and FedCP (0.72). On CICIDS2017, FedPGO further improved macro-level performance, attaining a macro-F1 of 0.93, surpassing FedAvg (0.72), FedProx (0.80), pFedMe (0.82), FedCross (0.85), FedPAC (0.88), and FedCP (0.87). These results indicate that FedPGO not only excels in weighted global metrics shown in Table 7, but also delivers markedly stronger class-level generalization, which is critical for intrusion detection in heterogeneous IoT environments.

The full gradient decomposition and parameter-split structure of

**Table 6** Performance of FedPGO on CICIDS2017.
___

<table>
  <thead>
    <tr>
      <th>Class</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>Support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>BENIGN</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>319,029</td>
    </tr>
    <tr>
      <td>Botnet</td>
      <td>0.96</td>
      <td>0.64</td>
      <td>0.77</td>
      <td>155</td>
    </tr>
    <tr>
      <td>Brute Force</td>
      <td>1.00</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>1385</td>
    </tr>
    <tr>
      <td>DDoS</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>18,822</td>
    </tr>
    <tr>
      <td>DoS</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>33,669</td>
    </tr>
    <tr>
      <td>DoS Slowloris</td>
      <td>0.97</td>
      <td>0.93</td>
      <td>0.95</td>
      <td>762</td>
    </tr>
    <tr>
      <td>Heartbleed</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1</td>
    </tr>
    <tr>
      <td>Infiltration</td>
      <td>0.78</td>
      <td>0.78</td>
      <td>0.78</td>
      <td>9</td>
    </tr>
    <tr>
      <td>Infiltration - Portscan</td>
      <td>0.95</td>
      <td>0.93</td>
      <td>0.94</td>
      <td>14,360</td>
    </tr>
    <tr>
      <td>Portscan</td>
      <td>0.97</td>
      <td>0.98</td>
      <td>0.97</td>
      <td>31,771</td>
    </tr>
    <tr>
      <td>Web Attack</td>
      <td>0.94</td>
      <td>0.74</td>
      <td>0.83</td>
      <td>23</td>
    </tr>
    <tr>
      <td>Accuracy</td>
      <td colspan="2">–</td>
      <td>0.99</td>
      <td>419,986</td>
    </tr>
    <tr>
      <td>Macro Avg</td>
      <td>0.96</td>
      <td>0.91</td>
      <td>0.93</td>
      <td>419,986</td>
    </tr>
    <tr>
      <td>Weighted Avg</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>419,986</td>
    </tr>
  </tbody>
</table>

IoT environments.

The full gradient decomposition and parameter-split structure of FedPGO allowed clients to fine-tune their personalized components without destabilizing the shared model, resulting in more accurate and robust predictions across heterogeneous clients. These results confirm that FedPGO not only enhances overall accuracy but also delivers superior personalization, a critical requirement for IoT intrusion detection under realistic Non-IID settings.

## *5.3. Ablation study*

This subsection presents the ablation studies that were conducted by removing the components of AGO and PGF which together constitute the core of the proposed FedPGO framework. Specifically, we analyze the contribution of We compare the complete FedPGO model with variants that exclude these components. We report these ablation variants (w/o, a short form of without): (i) w/o AGO, (ii) w/o PGF and (iii) w/o AGO & PGF correspond to removing both components. The performance of each variant was evaluated on both CICIoT2023 and CICIDS2017 keeping all other settings similar.

As shown in Table 8, both gradient orthogonalization (AGO) and per-

## *5.2. Performance comparison*

In our experimental evaluation of single-model FL, we benchmarked several representative algorithms, namely FedAvg [25], FedProx [26], FedCross [38], pFedMe [33], FedPAC [39], and FedCP [40]. These approaches embody distinct strategies, discussed in experimental setup section under compared methods, for mitigating key challenges in FL,

other settings similar.

As shown in Table 8, both gradient orthogonalization (AGO) and personalized gradient fusion (PGF) contribute positively to the overall performance, with the complete FedPGO framework achieving the highest

<page_number>10</page_number>

*Computer Networks 280 (2026) 112210*
D.D. Bikila and J. Čapek

**Table 7** Performance comparison of FedPGO with FL and PFL baselines on CICIoT2023 and CICIDS2017 datasets.
___



<table>
  <thead>
    <tr>
      <th rowspan="2">Method</th>
      <th colspan="4">CICIoT2023</th>
      <th colspan="4">CICIDS2017</th>
    </tr>
    <tr>
      <th>Acc (%)</th>
      <th>Prec (%)</th>
      <th>Rec (%)</th>
      <th>F1 (%)</th>
      <th>Acc (%)</th>
      <th>Prec (%)</th>
      <th>Rec (%)</th>
      <th>F1 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FedAvg [25]</td>
      <td>86.56</td>
      <td>87.48</td>
      <td>86.98</td>
      <td>87.22</td>
      <td>89.42</td>
      <td>90.99</td>
      <td>88.36</td>
      <td>89.67</td>
    </tr>
    <tr>
      <td>FedProx [26]</td>
      <td>89.42</td>
      <td>89.99</td>
      <td>88.26</td>
      <td>89.66</td>
      <td>94.64</td>
      <td>94.33</td>
      <td>93.79</td>
      <td>94.15</td>
    </tr>
    <tr>
      <td>pFedMe [33]</td>
      <td>90.85</td>
      <td>91.20</td>
      <td>89.94</td>
      <td>90.56</td>
      <td>95.32</td>
      <td>95.10</td>
      <td>94.68</td>
      <td>94.89</td>
    </tr>
    <tr>
      <td>FedCross [38]</td>
      <td>92.69</td>
      <td>94.59</td>
      <td>91.54</td>
      <td>92.94</td>
      <td>96.77</td>
      <td>95.97</td>
      <td>94.89</td>
      <td>95.59</td>
    </tr>
    <tr>
      <td>FedPAC [39]</td>
      <td>96.96</td>
      <td>97.10</td>
      <td>95.88</td>
      <td>96.99</td>
      <td>98.10</td>
      <td>97.45</td>
      <td>96.21</td>
      <td>96.82</td>
    </tr>
    <tr>
      <td>FedCP [40]</td>
      <td>96.48</td>
      <td>96.64</td>
      <td>95.27</td>
      <td>96.36</td>
      <td>97.78</td>
      <td>97.02</td>
      <td>95.96</td>
      <td>96.48</td>
    </tr>
    <tr>
      <td><strong>FedPGO</strong></td>
      <td><strong>99.24</strong></td>
      <td><strong>99.22</strong></td>
      <td><strong>99.24</strong></td>
      <td><strong>99.22</strong></td>
      <td><strong>99.50</strong></td>
      <td><strong>99.50</strong></td>
      <td><strong>99.50</strong></td>
      <td><strong>99.49</strong></td>
    </tr>
  </tbody>
</table>

**Table 8** Ablation study results of FedPGO and its component variants on CICIoT2023 and CICIDS2017 datasets.



<table>
  <thead>
    <tr>
      <th rowspan="2">Model Variants</th>
      <th colspan="4">CICIoT2023</th>
      <th colspan="4">CICIDS2017</th>
    </tr>
    <tr>
      <th>Acc (%)</th>
      <th>Prec (%)</th>
      <th>Rec (%)</th>
      <th>F1 (%)</th>
      <th>Acc (%)</th>
      <th>Prec (%)</th>
      <th>Rec (%)</th>
      <th>F1 (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FedPGO</td>
      <td>99.24</td>
      <td>99.22</td>
      <td>99.24</td>
      <td>99.22</td>
      <td>99.50</td>
      <td>99.50</td>
      <td>99.50</td>
      <td>99.49</td>
    </tr>
    <tr>
      <td>w/o AGO</td>
      <td>99.10</td>
      <td>99.09</td>
      <td>99.10</td>
      <td>99.08</td>
      <td>99.26</td>
      <td>99.26</td>
      <td>99.26</td>
      <td>99.24</td>
    </tr>
    <tr>
      <td>w/o PGF</td>
      <td>98.91</td>
      <td>98.94</td>
      <td>98.91</td>
      <td>98.85</td>
      <td>99.15</td>
      <td>99.14</td>
      <td>99.15</td>
      <td>99.14</td>
    </tr>
    <tr>
      <td>w/o AGO & PGF</td>
      <td>98.80</td>
      <td>98.81</td>
      <td>98.80</td>
      <td>98.78</td>
      <td>99.10</td>
      <td>99.09</td>
      <td>99.10</td>
      <td>99.08</td>
    </tr>
  </tbody>
</table>

**Table 9** Comparison of the proposed model with state-of-the-art approaches across both datasets.





accuracy across both CICIoT2023 (99.24%) and CICIDS2017 (99.50%). This demonstrated that the proposed method effectively balances generalization and personalization, enabling the model to better accommodate heterogeneous client distributions. The removal of either AGO or PGF leads to a measurable reduction in performance, with the absence of PGF showing a comparatively larger decline for instance accuracy decreases to 98.91% in CICIoT2023 and 99.15% on CICIDS2017 which shows that personalized fusion plays a more prominent role in addressing client diversity. Furthermore, the variant without both components performs the worst on both datasets, indicating that gradient alignment and personalized aggregation are mutually beneficial; the full FedPGO framework is necessary to maintain reliable federated intrusion detection under Non-IID IoT environments.

5.4 Comparison with other state-of-the-art works

pared methods, as summarized in Table 10. Methods that involve additional personalized optimization loops or auxiliary model components, such as pFedMe, FedPAC, and FedCP, incur higher per-round training time due to extra local updates or the exchange of class-level statistics. In contrast, FedPGO introduces personalization solely through gradient orthogonalization and scalar-based personalized aggregation, which operate on local gradients without requiring additional backpropagation steps or auxiliary networks. As a result, FedPGO maintains a per-round training time comparable to FedAvg and FedProx while achieving improved personalization performance. Moreover, since FedPGO exchanges only standard model updates, its per-round communication cost remains identical to FedAvg and FedProx, and is lower than FedPAC and FedCP, which require transmitting additional prototype or conditional statistics. These results demonstrate that FedPGO improves personalization and robustness under Non-IID IoT data distributions without sacrificing training efficiency or increasing communication overhead.

tion under Non-IID IoT environments.

**5.4. Comparision with other state-of-the-art works**

We have conducted a comparative analysis, with the state-of-the-art, to further evaluate the performance of the proposed model. We have used accuracy, recall, precision, F1-score, FPR and test time metrics for the comparison. All models used in the comparison employed the same CICIoT2023 or CICIDS2017 datasets. This guarantees to demonstrate a fair and significant comparison of related works. Moreover, it ensures fairness and avoids a biased comparison which is expected when different datasets are used to develop a model. The result of the comparison show that the propose model outperforms the other approaches in terms of matrices used for this comparison. Table 9, presents the comparative analysis.

## *5.6. Discussion*

The experiments were conducted on the pre-processed CICIoT2023 and CICIDS2017 datasets. To ensure the reliability of our method, we conducted a series of comprehensive experiments and evaluated the performance our model in multiple runs. The results, presented in Table 2, demonstrate that our method has the potential to deliver improved or comparable performance compared to state-of-the-art approaches. The proposed FedPGO framework demonstrates a substantial im-

comparable performance compared to state-of-the-art approaches.

The proposed FedPGO framework demonstrates a substantial improvement in addressing heterogeneity across clients by decomposing gradients into parallel and orthogonal components. This decomposition enables the model to capture both global knowledge and client-specific variations without compromising generalization. Our empirical results confirm that FedPGO consistently achieves higher classification

## *5.5. Communication and computation overheads*

We focus on the training phase and report the total training time, per-round training time, and per-round communication cost for all com-

<page_number>11</page_number>

D.D. Bikila and J. Čapek *Computer Networks 280 (2026) 112210*

**Table 10**
Computation and communication cost analysis of FL and PFL methods under identical Conv1D architecture and training settings
for CICIoT2023 and CICIDS2017, to achieve the results indicated in Table 7.

<table>
<thead>
<tr>
<th>Method</th>
<th>CICIoT2023</th>
<th></th>
<th></th>
<th>CICIDS2017</th>
<th></th>
<th></th>
</tr>
<tr>
<th></th>
<th>Param./round</th>
<th>Train time(S)</th>
<th>Model size (MB)</th>
<th>Param./round</th>
<th>Train time(S)</th>
<th>Model size (MB)</th>
</tr>
</thead>
<tbody>
<tr>
<td>FedAvg</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>36.2</td>
<td>0.351</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>28.4</td>
<td>0.665</td>
</tr>
<tr>
<td>FedProx</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>37.8</td>
<td>0.351</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>29.6</td>
<td>0.665</td>
</tr>
<tr>
<td>pFedMe</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>44.9</td>
<td>0.351</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>35.2</td>
<td>0.665</td>
</tr>
<tr>
<td>FedCross</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>41.3</td>
<td>0.351</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>32.8</td>
<td>0.665</td>
</tr>
<tr>
<td>FedPAC</td>
<td>$2 \sum_{k=1}^{N} (dC_k + \theta_k + w_k) + H_k$</td>
<td>43.7</td>
<td>0.351</td>
<td>$2 \sum_{k=1}^{N} (dC_k + \theta_k + w_k) + H_k$</td>
<td>34.6</td>
<td>0.665</td>
</tr>
<tr>
<td>FedCP</td>
<td>$2 \sum_{k=1}^{N} (\theta_k + G_{cs})$</td>
<td>42.9</td>
<td>0.351</td>
<td>$2 \sum_{k=1}^{N} (\theta_k + G_{cs})$</td>
<td>33.9</td>
<td>0.665</td>
</tr>
<tr>
<td>FedPGO</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>39.28</td>
<td>0.351</td>
<td>$2 \sum_{k=1}^{N} \theta_k$</td>
<td>30.45</td>
<td>0.665</td>
</tr>
</tbody>
</table>

$\theta_k$ denotes the number of trainable parameters of client $k$.
$d$ is embedding dimension and $C_k$ is number of local classes.
$w_k$ represents client-specific classifier head parameters.
$H_k$ and $G_{cs}$ denote additional class-level and conditional statistics exchanged by FedPAC and FedCP, respectively.

accuracy, macro F1, and robustness compared to conventional federated averaging and personalization baselines. Notably, the integration of orthogonal gradient components ensures that local adaptations are preserved while mitigating negative transfer from conflicting updates, a limitation observed in standard FL approaches.

The comparison with state-of-the-art and baselines shows that FedPGO offers a more principled mechanism in balancing collaboration and personalization. FedPGO leverages gradient geometry to regulate knowledge sharing, thereby providing a scalable and theoretically grounded solution. Our ablation studies further highlight the sensitivity of performance to $\lambda$-weighting parameters, suggesting that controlling the contribution of similarity, orthogonality, and loss improvement plays a critical role in achieving optimal personalization. This observation underscores the flexibility of FedPGO in adapting to diverse IoT attack patterns and client distributions, where Non-IID characteristics are particularly challenging.

The findings also hold practical significance for real-world IoT security and other distributed applications where data privacy and heterogeneity are central concerns. By effectively aligning global collaboration with local adaptation, FedPGO offers a pathway to deploy lightweight yet robust models at the edge, reducing reliance on centralized solutions. Moreover, the methodological contributions of FedPGO open new directions for gradient-space personalization strategies in FL. Moreover, the use of real-world datasets further improves its relevance and applicability, enabling seamless integration into existing IoT security infrastructures.

### 6. Conclusion

In this work, we proposed a novel PFL framework tailored for non-IID time-series IoT data, leveraging a parameter-split 1D-CNN architecture. By separating global feature extraction from locally personalized classification, our approach preserves both generalizable patterns and client-specific nuances. To address the challenges of statistical heterogeneity, we introduce FedPGO, a personalization-aware optimization strategy that decomposes local gradients into aligned and orthogonal components with respect to the global gradient. A PGF was then used to dynamically weigh client contributions, enhancing both convergence and robustness. Our extensive experiments demonstrate that FedPGO achieved competitive performance in comparison with baseline and state-of-the-art methods. The use of focal loss, weight decay, gradient clipping, and regularization further stabilizes training in highly heterogeneous data.

FedPGO improves personalization and robustness under Non-IID IoT data distributions without sacrificing training efficiency or increasing communication overhead. This confirms its scalability in adapting diverse real-world IoT attack patterns and client heterogeneity while remaining deployable within existing IoT security infrastructures. For future research work, the framework will be tested on real-world decentralized edge deployments with diverse hardware, models and data conditions.

**CRediT authorship contribution statement**

**Dawit Dejene Bikila:** Writing – original draft, Visualization, Software, Methodology, Investigation, Formal analysis, Data curation, Conceptualization; **Jan Čapek:** Writing – review & editing, Validation, Supervision, Resources, Project administration, Formal analysis, Conceptualization.

**Data availability**

Data will be made available on request.

**Declaration of competing interest**

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

**Aknowledgment**

This research was supported by a grant from the Institute of System Engineering and Informatics, Faculty of Economics and Administration, University of Pardubice, grant No. SGS_2025_018.

**References**

[1] B. Olanrewaju-George, B. Pranggono, Federated learning-based intrusion detection system for the internet of things using unsupervised and supervised deep learning models, Cyber Secur. Appl. 3 (2025) <sup>100068.</sup>
[2] R. Beuran, et al., FedMSE: semi-supervised federated learning approach for IoT network intrusion detection, Comput. Secur. <sub>151</sub> (2025) 104337.
[3] B.S. Guendouzi, S. Ouchani, H.E.L. Assaad, M.E.L. Zaher, A systematic review of federated learning: challenges, aggregation methods, and development tools, J. Netw. Comput. Appl. <sup>220</sup> (2023) <sup>103714.</sup>
[4] <sub>N.H.</sub> Quyen, <sub>P.T.</sub> Duy, <sub>N.T.</sub> Nguyen, <sub>N.H.</sub> Khoa, <sub>V.-H.</sub> Pham, FedKD-IDS: a robust intrusion detection system using knowledge distillation-based semi-supervised federated learning and anti-poisoning attack mechanism, Inf. Fusion 117 (2025) <sup>102807.</sup>
[5] L. Ni, X. Gong, J. Li, Y. Tang, Z. Luan, J. Zhang, Rfedfw: secure and trustable aggregation scheme for byzantine-robust federated learning in internet of things, Inf. Sci. 653 (2024) <sub>119784.</sub>
[6] U.A. Isma’ila, K.U. Danyaro, A.A. Muazu, U.D. Maiwada, Review on approaches of federated modeling in anomaly-based intrusion detection for IoT devices, IEEE Access 12 (2024) 30941–30961.
[7] X. Zhang, A. Mavromatis, A. Vafeas, R. Nejabati, D. Simeonidou, Federated feature selection for horizontal federated learning in iot networks, IEEE Internet Things J. 10 (11) (2023) 10095–10112.
[8] A. Chahal, P. Gulia, N.S. Gill, D. Rani, Design of a federated ensemble model for intrusion detection in distributed <sub>IIoT</sub> networks for enhancing cybersecurity, J. Ind. Inf. Integr. 44 (2025) 100800.



is embedding dimension and is number of local classes.

Dawit Dejene Bikila: Writing – original draft, Visualization, Soft- ware, Methodology, Investigation, Formal analysis, Data curation, Con- Jan ˇ ceptualization; Capek: Writing – review & editing, Validation, Su- pervision, Resources, Project administration, Formal analysis, Concep- tualization.

12

D.D. Bikila and J. Čapek

Computer Networks 280 (2026) 112210

[9] D. Chiaro, E. Prezioso, M. Ianni, F. Giampaolo, FL-enhance: a federated learning framework for balancing non-IID data with augmented and shared compressed samples, Inf. Fusion 98 (2023) 101836.

[10] P. Qi, D. Chiaro, F. Piccialli, Small models, big impact: a review on the power of lightweight federated learning, Future Gener. Comput. Syst. 162 (2024) 107484.

[11] J. Zhou, J. Hu, J. Xue, S. Zeng, Secure fair aggregation based on category grouping in federated learning, Inf. Fusion 117 (2025) 102838.

[12] S. Zheng, Q. Zhu, Q. Lin, S. Liu, K.-C. Wong, J. Li, Personalized federated learning with multiple classifier aggregation, Knowl. Based Syst. 311 (2025) 113073.

[13] J. Zhang, Y. Hua, H. Wang, T. Song, Z. Xue, R. Ma, H. Guan, FedALA: adaptive local aggregation for personalized federated learning, Proc. AAAI Conf. Artif. Intell., 37, 2023, pp. 11237–11244.

[14] Y. Fan, W. Xi, H. Zhu, J. Zhao, MiniPFL: mini federations for hierarchical personalized federated learning, Future Gener. Comput. Syst. 157 (2024) 41–50.

[15] J. Bai, L. Cao, J. Li, J. Wan, X. Du, FedWDP: a Wasserstein-distance-based federated learning for privacy and heterogeneous data in IoT, Internet Things 31 (2025) 101532.

[16] O.D. Okey, D.Z. Rodriguez, J.H. Kleinschmidt, Enhancing IoT intrusion detection with federated learning-based CNN-GRU and LSTM-GRU ensembles, in: 2024 19th International Symposium on Wireless Communication Systems (ISWCS), 2024, pp. 1–6. [https://doi.org/10.1109/ISWCS61526.2024.10639159](https://doi.org/10.1109/ISWCS61526.2024.10639159)

[17] I. Sharma, V. Khullar, Blockchain-enabled federated learning-based privacy preservation framework for secure IoT in precision agriculture, J. Ind. Inf. Integr. 44 (2025) 100765.

[18] Z. Jin, Y. Ding, X. Wu, X. Chen, Z. Liu, G. Li, Federated dual correction intrusion detection system: efficient aggregation for heterogeneous data, Comput. Netw. 259 (2025) 111116.

[19] L. Yi, X. Shi, N. Wang, G. Wang, X. Liu, Z. Shi, H. Yu, PfedKT: personalized federated learning with dual knowledge transfer, Knowl. Based Syst. 292 (2024) 111633.

[20] S.E. Maoudj, A. Belghiat, A deep learning-based approach with two-step minority classes prediction for intrusion detection in internet of things networks, Knowl. Based Syst. 312 (2025) 113143.

[21] D. Torre, A. Chennamaneni, J. Jo, G. Vyas, B. Sabrsula, Toward enhancing privacy preservation of a federated learning CNN intrusion detection system in IoT: method and empirical study, ACM Trans. Softw. Eng. Method. 34 (2) (2025) 1–48.

[22] H.B. Akande, A.L. Imoize, T.C. Adeniran, C.-C. Lee, J.B. Awotunde, RF-FLIDS: a novel hybrid intrusion detection model for enhanced anomaly detection in IoT networks, Secur. Priv. 8 (3) (2025) e70041.

[23] N. Alsabilah, D.B. Rawat, Joint rough set theory and XGBoost based learning for network intrusion detection system, IEEE Internet Things J. 12 (7) (2025) 7930–7937.

[24] P. Sinha, D. Sahu, S. Prakash, T. Yang, R.S. Rathore, V.K. Pandey, A high performance hybrid LSTM CNN secure architecture for IoT environments using deep learning, Sci. Rep. 15 (1) (2025) 9684.

[25] B. McMahan, E. Moore, D. Ramage, S. Hampson, B.A. y Arcas, Communication-efficient learning of deep networks from decentralized data, in: Artificial Intelligence and Statistics, PMLR, 2017, pp. 1273–1282.

[26] T. Li, A.K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith, Federated optimization in heterogeneous networks, Proc. Mach. Learn. Syst. 2 (2020) 429–450.

[27] S.P. Karimireddy, S. Kale, M. Mohri, S. Reddi, S. Stich, A.T. Suresh, SCAFFOLD: stochastic controlled averaging for federated learning, in: International Conference on Machine Learning, PMLR, 2020, pp. 5132–5143.

[28] R. Bensaid, N. Labraoui, A.A.A. Ari, H. Saidi, J.H.M. Emati, L. Maglaras, SA-FLIDS: secure and authenticated federated learning-based intelligent network intrusion detection system for smart healthcare, PeerJ Comput. Sci. 10 (2024) e2414.

[29] R. Yang, H. He, Y. Wang, Y. Qu, W. Zhang, Dependable federated learning for IoT intrusion detection against poisoning attacks, Comput. Secur. 132 (2023) 103381.

[30] Z. Wei, J. Wang, Z. Zhao, K. Shi, Toward data efficient anomaly detection in heterogeneous edge–cloud environments using clustered federated learning, Future Gener. Comput. Syst. 164 (2025) 107559.

[31] S. Sun, L. Zhou, Z. Wang, L. Han, Robust intrusion detection based on personalized federated learning for IoT environment, Comput. Secur. 154 (2025) 104442.

[32] W. Lu, A. Ye, P. Xiao, Y. Liu, L. Yang, D. Zhu, Z. Liu, Stones from other hills: intrusion detection in statistical heterogeneous IoT by self-labeled personalized federated learning, IEEE Internet Things J. 12 (2025) 14348–14361.

[33] T.D. Canh, N. Tran, J. Nguyen, Personalized federated learning with moreau envelopes, Adv. Neural Inf. Process. Syst. 33 (2020) 21394–21405.

[34] E.C.P. Neto, S. Dadkhah, R. Ferreira, A. Zohourian, R. Lu, A.A. Ghorbani, CICIoT2023: a real-time dataset and benchmark for large-scale attacks in IoT environment, Sensors 23 (13) (2023) 5941.

[35] I. Sharafaldin, A.H. Lashkari, A.A. Ghorbani, et al., Toward generating a new intrusion detection dataset and intrusion traffic characterization, ICISSp 1 (2018) (2018) 108–116.

[36] J. Li, T. Chen, S. Teng, A comprehensive survey on client selection strategies in federated learning, Comput. Netw. 251 (2024) 110663.

[37] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Dollár, Focal loss for dense object detection, in: Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2980–2988.

[38] M. Hu, P. Zhou, Z. Yue, Z. Ling, Y. Huang, A. Li, Y. Liu, X. Lian, M. Chen, FedCross: towards accurate federated learning via multi-model cross-aggregation, in: 2024 IEEE 40th International Conference on Data Engineering (ICDE), IEEE, 2024, pp. 2137–2150.

[39] J. Xu, X. Tong, S.-L. Huang, Personalized federated learning with feature alignment and classifier collaboration, arXiv:2306.11867 (2023).

[40] J. Zhang, Y. Hua, H. Wang, T. Song, Z. Xue, R. Ma, H. Guan, FedCP: separating feature information for personalized federated learning via conditional policy, in: Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp.3249–3261.

<page_number>13</page_number>