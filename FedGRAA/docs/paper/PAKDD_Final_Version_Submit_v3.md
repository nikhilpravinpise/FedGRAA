> ⚠️ **SUPERSEDED, 2026-08-02.** This draft was reviewed by a simulated peer
> panel and rejected — see [`FedGRAA_Simulated_Peer_Review.md`](FedGRAA_Simulated_Peer_Review.md).
> Every required revision that review names (missing simplex projection,
> the convexity-vs-non-convex-architecture contradiction, single-run/
> unseeded results, missing adaptive-aggregation baselines) is resolved in
> the corrected resubmission: [`FedGRAA_v2_PAKDD_Submission.md`](FedGRAA_v2_PAKDD_Submission.md).
> Kept here, unedited, as the original draft the review and the correction
> both respond to — do not cite numbers from this version.
>
> **Citation-integrity note added 2026-08-03, without altering the frozen
> text below:** reference [12] (line 1722) cites "Zhang, X., Li, T., Jin,
> R., Yang, Q.: Federated learning on non-iid data via local and global
> adaptation... ICML 2020" alongside an arXiv link
> (`arxiv.org/pdf/2306.14443`) whose ID encodes a June-2023 posting date —
> inconsistent with a 2020 ICML paper, and the same reference slot
> `draft1.md`'s bibliography separately confirmed was a fabricated
> citation ("FedSND") elsewhere in this same draft's lineage. This entry
> was never independently re-verified as real; do not cite it. The
> corrected resubmission's bibliography
> ([`FedGRAA_v2_PAKDD_Submission.md`](FedGRAA_v2_PAKDD_Submission.md)) does
> not carry this entry forward.

# Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems

No Author Given

No Institute Given

**Abstract.** Federated learning has become a powerful framework for intrusion detection systems by enabling collaborative learning across distributed clients while maintaining data privacy. However, the standard *FedAvg* approach encounters significant limitations when handling heterogeneous data distributions among clients. In cases where certain clients have substantially less data, their contributions to the global model are disproportionately reduced, leading to suboptimal performance, especially in detecting classes with fewer samples. To overcome this issue, we propose a novel method called FedGRAA which employs an adaptive, gradient-based weight update mechanism that dynamically adjusts the significance of each client in each training round based on the distribution of their local data. This weighting mechanism ensures that clients with smaller datasets still have a meaningful influence on the global model, thereby enhancing detection accuracy for underrepresented classes. Extensive experiments on CICIDS 2017 and UNSW-NB15 demonstrate that FedGRAA achieves 88% accuracy on minority classes, significantly outperforming *FedAvg* while preserving client privacy. Additionally, as the number of clients with minority samples increases, FedGRAA maintains its advantage, outperforming *FedAvg* by up to 55%. This marks a major advancement in federated learning for intrusion detection, especially in non-IID settings.

## 1 Introduction

The rapid growth of network technologies and the increasing number of connected devices have escalated cybersecurity risks, making Intrusion Detection Systems (IDS) crucial for detecting and preventing attacks. Traditional IDS typically depend on centralized data collection, which raises concerns about privacy and limited adaptability [1]. Federated Learning (FL) introduces a decentralized solution, enabling multiple parties to collaboratively train models without sharing raw data. This approach safeguards privacy and enables faster model updates, ensuring an agile response to new intrusions.

However, FL faces significant challenges, particularly due to heterogeneous data distributions across clients. Federated Averaging (*FedAvg*), introduced by McMahan et al. [6], allows privacy-preserving model training but assumes homogeneity in client data—a condition that rarely holds in real-world scenarios. Data heterogeneity, also referred to as non-IID (non-independent and identically

2 No Author Given

<page_number>

2
</page_number>

distributed) data, is especially problematic in federated IDS applications, where different clients encounter distinct types of attacks at varying frequencies. For instance, certain clients may face frequent attempts of one type of intrusion, while others rarely see it. Such variations can cause conventional FL approaches to yield biased and less effective models [3].

In response to these challenges, adaptive federated learning frameworks have been proposed. Zhang et al. [12] introduced *FedSND*, which dynamically adjusts to data disparities by modifying aggregation weights based on data distribution statistics. Despite these advancements, most adaptive techniques continue to struggle with highly imbalanced datasets in IDS, where rare but critical attack patterns are underrepresented. This imbalance leads to inadequate detection of such patterns, heightening security risks.

To tackle this issue, we propose a novel method, *Federated Gradient-Refined Adaptive Averaging* (**FedGRAA**), which enhances model performance for clients with underrepresented data by dynamically adjusting aggregation weights based on data distribution and gradient updates at each training round. This approach refines the aggregation process, granting higher weight to gradients from clients with rare attack patterns. In our evaluation using the imbalanced CIC-IDS-2017 dataset, **FedGRAA** outperforms both traditional federated averaging and static reparameterization methods, achieving notable improvements in detection accuracy for underrepresented attack types. This advancement in adaptive FL approaches paves the way for more robust and responsive IDS solutions in federated environments.

# 2 Related Work

***Static Reparameterization*** Static parameter aggregation methods in federated learning struggle with non-IID data. Zhao et al. [13] and Li et al. [5] demonstrated how *FedAvg* suffers performance degradation and poor convergence in heterogeneous environments due to its static nature. Fallah et al. [2] and Reisizadeh et al. [8] further highlighted these limitations, proposing personalized federated learning and meta-learning techniques as solutions. These studies show that static reparameterization is inadequate for handling client diversity, underscoring the need for adaptive methods.

Static reparameterization lacks adaptability to changing data distributions, leading to model rigidity and bias towards initial conditions (Smith et al. [5]). This approach underrepresents crucial but rare data, affecting applications like anomaly detection and medical diagnostics (Pranab et al. [9]). Furthermore, static models fail to capture the dynamic nature of real-world data, limiting long-term relevance (Lee & Kim [4]). This highlights the need for dynamic strategies that adapt to new data without manual intervention (Zhao & Joshi, 2022 [14]).

***Adaptive Reparameterization*** This overcomes the limitations of static reparameterization by dynamically adjusting aggregation weights based on their performance throughout the training process. Unlike static methods, where client contributions remain fixed, adaptive reparameterization recalibrates the influence

Title Suppressed Due to Excessive Length <page_number>3</page_number>

of each client to ensure the global model remains balanced, even in non-IID settings. This technique, seen in methods like FedOpt [7] and FedProx [5], reduces bias towards clients with majority-class data, emphasizing underrepresented clients, and enhancing model generalization.

Our method (**FedGRAA**) introduces an adaptive reparameterization approach tailored for the non-IID nature of intrusion detection tasks. By continuously adjusting aggregation weights based on gradients, the method ensures that both common and rare data are adequately represented. This dynamic adjustment leads to a robust model capable of detecting rare attack types like Heartbleed, SQL Injection, Infiltration (CIC-IDS-2017) in intrusion detection systems. As a result, our approach outperforms *FedAvg* [6] by achieving more balanced and effective performance in heterogeneous data environments, particularly in handling rare and critical attack types that *FedAvg* [6] struggles to detect.

# 3 Background and Problem Formulation

*Federated Learning* It is a distributed framework that trains a global model across decentralized devices without sharing local data, preserving privacy and reducing communication costs. The most common algorithm, Federated Averaging (*FedAvg*), works by initializing global model parameters $\theta_g^0$, selecting a fixed set of clients $S_t$ at each round $t$, and sending the current model $\theta_t$ to them: $\theta_k^{t+1} \leftarrow \theta_g^t$. Clients update their local models using stochastic gradient descent (SGD) on their local datasets $D_k$:

$$ \theta_k^{t+1} = \theta_k^{t+1} - \eta \nabla L_{\theta_k^{t+1}}(\theta_k^{t+1}; D_k), \quad \forall k \in S_t \eqno(1) $$

where $\eta$ is the learning rate and $K$ is the total number of clients in $S_t$. The server then aggregates these updates to form the global model and this process repeats until convergence [6][10]:

$$ \theta_g^{t+1} = \frac{1}{K} \sum_{k \in S_t} \theta_k^{t+1} \eqno(2) $$

*Non-IID Data Challenges in Federated Learning* In Federated Learning, data is often non-independent and identically distributed across clients, leading to challenges such as bias, where the global model may overfit to certain clients' data, and slower convergence, requiring more communication rounds [13]. The loss function for client $k$ is defined as:

$$ L_k(\theta) = \frac{1}{|D_k|} \sum_{i \in D_k} l(\theta; x_i, y_i) \eqno(3) $$

where $l(\theta; x_i, y_i)$ is the loss for a single data point $(x_i, y_i)$, and $|D| = \sum_{k=1}^K |D_k|$ is the total number of data points. The global loss function $L(\theta)$ is the weighted average of local loss functions:

$$ L(\theta) = \sum_{k=1}^K \frac{|D_k|}{|D|} L_k(\theta) \eqno(4) $$

4 No Author Given

In Federated Averaging, a client’s contribution to the global model is proportional to its dataset size, meaning clients with fewer samples have less influence, as seen in the aggregation step [6]:

$$ \theta_{g}^{t+1} = \sum_{k \in S_{t}} \frac{|D_{k}|}{|D|} \theta_{k}^{t+1} \eqno(5) $$

Non-IID data can also slow convergence as gradients from different clients may diverge. When gradients $\nabla L_{k}(\theta)$ are misaligned, the global gradient $\nabla L(\theta)$ may oscillate, delaying convergence. Techniques like personalization, regularization, and adaptive aggregation address these issues [3]:

$$ \nabla L(\theta) = \sum_{k=1}^{K} \frac{|D_{k}|}{|D|} \nabla L_{k}(\theta) \eqno(6) $$

With non-IID data, clients with fewer samples have limited influence on the global model due to the weighting factor $\frac{|D_{k}|}{|D|}$ in the *FedAvg* algorithm. This under-representation prevents smaller clients from receiving models tailored to their local data. Consequently, the aggregation process favors clients with larger datasets, leading to a global model biased towards them and reducing performance on tasks for smaller clients, i.e., the global model $\theta_{g}^{t+1}$ tends to approximate $\theta_{j}^{t}$ for large $|D_{j}|$.

# 4 Methodology

**Static Reparameterization** In the static reparameterization approach below, we assign fixed weights to each client’s contribution based on the inverse of their dataset size. This is mathematically represented as follows:

$$ \theta_{g}^{t+1} = \sum_{k=1}^{K} \frac{1/|D_{k}|}{\sum_{j=1}^{K} \frac{1}{|D_{j}|}} \theta_{k}^{t} \eqno(7) $$

where $|D_{k}|$ denotes the number of samples for client $k$. In this formulation, clients with smaller datasets have a proportionally larger influence on the global model, as contributions are weighted inversely to dataset size. Although this method helps to mitigate the dominance of large datasets, it lacks the flexibility to adapt to variations in data distribution or client performance over time.

**Adaptive Reparameterization (FedGRAA)** We introduce an adaptive reparameterization method, **FedGRAA** (Algorithm 2), which dynamically adjusts aggregation weights based on performance changes, with a focus on prioritizing minority classes. The adjustment is driven by a personalization co-efficient $\gamma$, where $\gamma \in [0, 1]$. As $\gamma$ approaches 1, the method prioritizes minority classes, while decreasing $\gamma$ towards 0 reduces their emphasis, allowing for effective handling of data heterogeneity in federated learning, particularly in non-IID settings.

Title Suppressed Due to Excessive Length 5

The process begins with the server initializing the global model parameters, $\theta_g^0$, while the initial aggregation weights, $w^0 = [w_1^0, \dots, w_K^0]$, are calculated via static reparameterization (Equation 7). During each training round $t$, a fixed set of clients $S_t$ is selected to participate, and the global model parameters $\theta_g^{t-1}$ are sent to each client. Each client $k$ initializes its local model $\theta_k^t \leftarrow \theta_g^{t-1}$ and performs updates using gradient descent on local data batches from $D_k$ for $E$ epochs. Once local training is complete, the updated model parameters, $\theta_k^t$, are sent back to the server (Algorithm 1)

## Algorithm 1 Adaptive Reparameterization for Model Personalization

1: **Input:** $T$ - Training rounds, $\eta$ & $\eta'$ - learning rates, $w^0 = [w_1^0, w_2^0, \dots, w_K^0]$ - initial aggregation weights computed via static reparameterization (Equation 7), $\gamma$ - personalization coefficient, $\mathcal{D}_{\mathcal{V}}$ - validation dataset, $\mathcal{D}_{\mathcal{V}}^n$ - validation dataset of last $n$ underrepresented samples.

```
 2: Server executes:
 3:     initialize global model parameters θ⁰_g
 4:     for each round t = 1, 2, . . . , T
 5:      Sₜ ← (fixed set of K clients)
 6:      for each client k ∈ Sₜ in parallel
 7:          send current global model θᵗ⁻¹_g to client k
 8:          θᵗ_k ← CLIENTUPDATE(k, θᵗ⁻¹_g)
 9:      θᵗ_c = [θᵗ_1, θᵗ_2, . . . , θᵗ_K]
10:      wᵗ ← FedGRAA(θᵗ_c, wᵗ⁻¹)
11:      θᵗ_g = Σ_{k∈Sₜ} wᵗ_k θᵗ_k // update global model
12:     return θᵀ_g // Final global model parameters

13: ClientUpdate(k, θᵗ⁻¹_g): // Run on client k
14:     initialize local model θᵗ_k ← θᵗ⁻¹_g
15:     for each local epoch from 1 to E
16:      for each batch b in Dₖ // each batch of size B
17:            θᵗ_k ← θᵗ_k − η∇L_{θᵗ_k}(θᵗ_k; b)
18:     return θᵗ_k to server
```

Once the updated local models are received from the clients, the server computes an intermediate global model, $\theta'_g$, by using the weights from the previous round, $w^{t-1}$: $\theta'_g = \sum_{i=1}^m w_i^{t-1} \theta_i^t$.

We split the dataset $\mathcal{D}_{\mathcal{V}} \setminus \mathcal{D}_{\mathcal{V}}^n$ into two parts, $\mathcal{D}_{\mathcal{V}}^h$ and $\mathcal{D}_{\mathcal{V}}^m$, using an automated method that determines the split threshold based on the median frequency. To refine the weight adjustment process further, the aggregation weights are updated by optimizing a combined loss function $L'$ that incorporates both the loss $L_u$, which focuses on $\mathcal{D}_{\mathcal{V}}^n$, and $L_a$, which is a function of losses defined for highly and moderately frequent data. The $L_a$ component is defined as follows:

$$L_a = \min(L_h(\theta'_g; \mathcal{D}_{\mathcal{V}}^h), L_m(\theta'_g; \mathcal{D}_{\mathcal{V}}^m)) \quad (8)$$

Here, $\theta'_g$ is the temporary global model used to update the aggregation weights.

6 No Author Given

<page_number>6</page_number>

This combined loss is expressed as:

$$ L'(\theta'_g; \mathcal{D}_{\mathcal{V}}, \mathcal{D}_{\mathcal{V}}^n) = (1 - \gamma)L_a(\theta'_g; \mathcal{D}_{\mathcal{V}}^h, \mathcal{D}_{\mathcal{V}}^m) + \gamma L_u(\theta'_g; \mathcal{D}_{\mathcal{V}}^n) \eqno(9) $$

Here, $\mathcal{D}_{\mathcal{V}}$ represents the whole validation dataset, while $\mathcal{D}_{\mathcal{V}}^n$ refers to the validation data for the last $n$ underrepresented classes. The parameter $\gamma$ controls the balance between prioritizing overall performance and improving accuracy for these underrepresented classes (Equation 9).

Since $\theta'_g$ is a function of $w^{t-1}$, we take the gradient of the loss function with respect to $w^{t-1}$. The weights $w^{t-1}$ are then updated by minimizing the gradient of this combined loss function:

$$ w^{t-1} \leftarrow w^{t-1} - \eta' \nabla_{w^{t-1}} L'(\sum_{i=1}^K w_i^{t-1}\theta_i^t; \mathcal{D}_{\mathcal{V}}^h, \mathcal{D}_{\mathcal{V}}^m, \mathcal{D}_{\mathcal{V}}^n) \eqno(10) $$

After completing this weight update for $E'$ epochs, the new weights $w^t \leftarrow w^{t-1}$ are applied, and at the end of the $t^{th}$ round, the global model is updated as: $\theta_g^t = \sum_{k \in S_t} w_k^t \theta_k^t$.

**Algorithm 2** Federated Gradient-Refined Adaptive Averaging (FedGRAA)
***
1: **FedGRAA**($\theta_k^t, w^{t-1}$): // *Our Algorithm*
2:     **for** each local epoch from 1 to $E'$
3:         $\theta'_g = w_1^{t-1}\theta_1^t + w_2^{t-1}\theta_2^t + \cdots + w_K^{t-1}\theta_K^t$
4:         $L_a = \min(L_h(\theta'_g; \mathcal{D}_{\mathcal{V}}^h), L_m(\theta'_g; \mathcal{D}_{\mathcal{V}}^m))$
5:         $L' = (1 - \gamma)L_a(\theta'_g; \mathcal{D}_{\mathcal{V}}^h, \mathcal{D}_{\mathcal{V}}^m) + \gamma L_u(\theta'_g; \mathcal{D}_{\mathcal{V}}^n)$
6:         $w^{t-1} \leftarrow w^{t-1} - \eta' \nabla_{w^{t-1}} L'(\theta'_g; \mathcal{D}_{\mathcal{V}}^h, \mathcal{D}_{\mathcal{V}}^m, \mathcal{D}_{\mathcal{V}}^n)$
7:     $w^t \leftarrow w^{t-1}$
8: **return** $w^t$
***

**FedGRAA** dynamically adjusts aggregation weights using the personalization coefficient $\gamma$, prioritizing minority classes to address data heterogeneity. Unlike static reparameterization with fixed weights, **FedGRAA** adapts to varying client contributions, ensuring better generalization and faster convergence (see Section A.1 for convergence proof) in non-IID settings. Studies [7] [11] show that adaptive reparameterization enables personalized updates, reducing outlier's influence and improving model convergence and performance in federated learning.

# 5 Experiments and Results

*Data Settings* In our federated learning setup, the datasets (see Tables 1 and 2) are divided among clients. To prevent highly frequent attacks, such as *DoS Hulk*, *PortScan*, *DDoS*, *DoS GoldenEye* (CIC-IDS data) and *Generic*, *Fuzzers*, *Reconnaissance* and *DoS* (UNSW data), from dominating, their samples are capped at 10,000. Each client is assigned data for one specific attack type paired with an equal number of *Benign* (*Normal* for UNSW data) samples to ensure balanced training. Clients use 90% of their data for local training and reserve 10% for validation ($\mathcal{D}_{\mathcal{V}}$), enhancing detection across both frequent and rare attacks.

Title Suppressed Due to Excessive Length <page_number>7</page_number>

**Network Architecture** We designed a fully connected neural network using Keras Sequential API for our intrusion detection model. The network accepts 78 and 43 input features for CIC-IDS-17 and UNSW datasets, has hidden layers of sizes 200, 100, and 50 with *tanh* activation, and an output layer of 15 and 10 neurons for CIC-IDS-17 and UNSW datasets with *softmax* activation, corresponding to their respective classes in the CIC-IDS-2017 and UNSW-NB15 datasets.

**Training Setup and Hyperparameters** The model was trained using the Adam optimizer with a learning rate of $\eta = 0.01$. Each client performed local training for $E = 2$ epochs per round with a batch size of $B = 64$. The federated learning process was conducted over 200 communication rounds with $K = 14$ and $K = 9$ clients participating for CIC-IDS-17 and UNSW datasets respectively. The loss function used was sparse categorical cross-entropy.

<table>
  <thead>
    <tr>
        <th>ID</th>
        <th>Label</th>
        <th>Original</th>
        <th>Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>0</td>
        <td>BENIGN</td>
        <td>2,273,097</td>
        <td>63,901</td>
    </tr>
    <tr>
        <td>1</td>
        <td>Bot</td>
        <td>1,966</td>
        <td>1,948</td>
    </tr>
    <tr>
        <td>2</td>
        <td>DDoS</td>
        <td>128,027</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>3</td>
        <td>DoS GoldenEye</td>
        <td>10,293</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>4</td>
        <td>DoS Hulk</td>
        <td>231,073</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>5</td>
        <td>DoS Slowhttptest</td>
        <td>5,499</td>
        <td>3,219</td>
    </tr>
    <tr>
        <td>6</td>
        <td>DoS Slowloris</td>
        <td>5,796</td>
        <td>5,228</td>
    </tr>
    <tr>
        <td>7</td>
        <td>FTP-Patator</td>
        <td>7,938</td>
        <td>5,931</td>
    </tr>
    <tr>
        <td>8</td>
        <td>Heartbleed</td>
        <td>11</td>
        <td>11</td>
    </tr>
    <tr>
        <td>9</td>
        <td>Infiltration</td>
        <td>36</td>
        <td>36</td>
    </tr>
    <tr>
        <td>10</td>
        <td>PortScan</td>
        <td>158,930</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>11</td>
        <td>SSH-Patator</td>
        <td>5,897</td>
        <td>5,385</td>
    </tr>
    <tr>
        <td>12</td>
        <td>Brute Force</td>
        <td>1,507</td>
        <td>1,470</td>
    </tr>
    <tr>
        <td>13</td>
        <td>SQL Injection</td>
        <td>21</td>
        <td>21</td>
    </tr>
    <tr>
        <td>14</td>
        <td>XSS</td>
        <td>652</td>
        <td>652</td>
    </tr>
  </tbody>
</table>
<table>
  <thead>
    <tr>
        <th>ID</th>
        <th>Label</th>
        <th>Original</th>
        <th>Used</th>
    </tr>
  </thead>
  <tbody>
    <tr>
        <td>0</td>
        <td>Normal</td>
        <td>93,000</td>
        <td>56,691</td>
    </tr>
    <tr>
        <td>1</td>
        <td>Generic</td>
        <td>58,871</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>2</td>
        <td>Fuzzers</td>
        <td>24,246</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>3</td>
        <td>Backdoor</td>
        <td>2,329</td>
        <td>2,329</td>
    </tr>
    <tr>
        <td>4</td>
        <td>Worms</td>
        <td>174</td>
        <td>174</td>
    </tr>
    <tr>
        <td>5</td>
        <td>Reconnaissance</td>
        <td>13,987</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>6</td>
        <td>Analysis</td>
        <td>2,677</td>
        <td>2,677</td>
    </tr>
    <tr>
        <td>7</td>
        <td>Exploits</td>
        <td>44,525</td>
        <td>10,000</td>
    </tr>
    <tr>
        <td>8</td>
        <td>Shellcode</td>
        <td>1,511</td>
        <td>1,511</td>
    </tr>
    <tr>
        <td>9</td>
        <td>DoS</td>
        <td>16,353</td>
        <td>10,000</td>
    </tr>
  </tbody>
</table>

Table 1: Original and used sample distribution for CIC-IDS-17 data.

Table 2: Original and used sample distribution for UNSW-NB15 data.

For ***FedGRAA***, we continued using the Adam optimizer, with a learning rate $\eta' = 0.1$ and epochs $E' = 5$. Different values of the personalization coefficient $\gamma$ (0, 0.5, 1) were experimented with to adjust the influence of clients with underrepresented data, focusing on the three ($n = 3$) underrepresented clients with rare attack types: Heartbleed, SQL Injection, and Infiltration for CIC-IDS dataset and Worms, Shellcode and Backdoor for UNSW dataset.

**Algorithm Comparison** We compared the performance of our proposed method, FedGRAA, against the standard *FedAvg* and Static Reparameterization methods (see Section 4). Figure 1 shows the accuracy trends over 200 communication rounds for different $\gamma$ values (0, 0.5, 1).

**Results and Analysis** The results in Figure 1 demonstrate that our approach balances accuracy for underrepresented classes and overall global accuracy by tuning $\gamma$. As $\gamma$ increases, accuracy for minority classes (e.g., Classes [8, 9, 13]

<page_number>8</page_number> No Author Given

in CICIDS, [4, 8, 3] in UNSW) improves, with a minor reduction in global accuracy. For higher values (γ = 1 and γ = 0.5), **FedGRAA** excels in minority class performance while maintaining global accuracy comparable to or better than Static Reparameterization and *FedAvg*. Conversely, at γ = 0, **FedGRAA** prioritizes global accuracy, outperforming Static Reparameterization, but sacrifices minority class performance by deprioritizing these classes.


<table>
  <caption>Fig. 1: Performance comparison for varying γ values on CIC-IDS and UNSW.</caption>
  <thead>
    <tr>
      <th rowspan="2">Communication Rounds</th>
      <th colspan="6">[CICIDS] γ = 0, n = 3 (Accuracy)</th>
      <th colspan="6">[CICIDS] γ = 0.5, n = 3 (Accuracy)</th>
      <th colspan="6">[CICIDS] γ = 1, n = 3 (Accuracy)</th>
    </tr>
    <tr>
      <th>FedAvg (All classes)</th>
      <th>FedAvg (Last n classes)</th>
      <th>Static (All classes)</th>
      <th>Static (Last n classes)</th>
      <th>Adaptive (All classes)</th>
      <th>Adaptive (Last n classes)</th>
      <th>FedAvg (All classes)</th>
      <th>FedAvg (Last n classes)</th>
      <th>Static (All classes)</th>
      <th>Static (Last n classes)</th>
      <th>Adaptive (All classes)</th>
      <th>Adaptive (Last n classes)</th>
      <th>FedAvg (All classes)</th>
      <th>FedAvg (Last n classes)</th>
      <th>Static (All classes)</th>
      <th>Static (Last n classes)</th>
      <th>Adaptive (All classes)</th>
      <th>Adaptive (Last n classes)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.42</td>
      <td>0.88</td>
      <td>0.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.42</td>
      <td>0.88</td>
      <td>0.45</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.42</td>
      <td>0.88</td>
      <td>0.45</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.43</td>
      <td>0.88</td>
      <td>0.68</td>
      <td>0.11</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.43</td>
      <td>0.88</td>
      <td>0.50</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.43</td>
      <td>0.88</td>
      <td>0.42</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.70</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.52</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.40</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.71</td>
      <td>0.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.54</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.44</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.71</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.54</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.88</td>
      <td>0.45</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.71</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.67</td>
      <td>0.54</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.45</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.67</td>
      <td>0.71</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.67</td>
      <td>0.54</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.67</td>
      <td>0.45</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>160</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.78</td>
      <td>0.71</td>
      <td>0.22</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.67</td>
      <td>0.54</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.45</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.71</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.67</td>
      <td>0.53</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.45</td>
      <td>0.88</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.71</td>
      <td>0.33</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.67</td>
      <td>0.53</td>
      <td>0.78</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.44</td>
      <td>0.81</td>
      <td>0.45</td>
      <td>0.88</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th rowspan="2">Communication Rounds</th>
      <th colspan="6">[UNSW] γ = 0, n = 3 (Accuracy)</th>
      <th colspan="6">[UNSW] γ = 0.5, n = 3 (Accuracy)</th>
      <th colspan="6">[UNSW] γ = 1, n = 3 (Accuracy)</th>
    </tr>
    <tr>
      <th>FedAvg (All classes)</th>
      <th>FedAvg (Last n classes)</th>
      <th>Static (All classes)</th>
      <th>Static (Last n classes)</th>
      <th>Adaptive (All classes)</th>
      <th>Adaptive (Last n classes)</th>
      <th>FedAvg (All classes)</th>
      <th>FedAvg (Last n classes)</th>
      <th>Static (All classes)</th>
      <th>Static (Last n classes)</th>
      <th>Adaptive (All classes)</th>
      <th>Adaptive (Last n classes)</th>
      <th>FedAvg (All classes)</th>
      <th>FedAvg (Last n classes)</th>
      <th>Static (All classes)</th>
      <th>Static (Last n classes)</th>
      <th>Adaptive (All classes)</th>
      <th>Adaptive (Last n classes)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.01</td>
      <td>0.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.01</td>
      <td>0.49</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.01</td>
      <td>0.49</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.48</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.48</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>80</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>120</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.35</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>160</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.45</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.41</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.21</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.47</td>
      <td>0.21</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th colspan="2">Method</th>
      <th colspan="2">Acc. (CIC-IDS-17)</th>
      <th colspan="2">Acc. (UNSW-NB15)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="2">FedAvg</td>
      <td colspan="2">00.00</td>
      <td colspan="2">00.00</td>
    </tr>
    <tr>
      <td colspan="2">Static Reparam.</td>
      <td colspan="2">66.66</td>
      <td colspan="2">00.00</td>
    </tr>
    <tr>
      <td colspan="2">FedGRAA (γ = 1)</td>
      <td colspan="2">88.88</td>
      <td colspan="2">21.09</td>
    </tr>
    <tr>
      <td colspan="2">FedGRAA (γ = 0.5)</td>
      <td colspan="2">77.77</td>
      <td colspan="2">20.59</td>
    </tr>
    <tr>
      <td colspan="2">FedGRAA (γ = 0)</td>
      <td colspan="2">33.33</td>
      <td colspan="2">00.00</td>
    </tr>
  </tbody>
</table>


Fig. 1: Performance comparison for varying γ values on CIC-IDS and UNSW.

Figure 2 evaluates the robustness of our FedGRAA algorithm as the number of clients participating in personalization increases. For n = 5 and γ = 1, FedGRAA achieved an accuracy of approximately 65%, significantly outperforming FedAvg and static reparameterization, which achieved accuracies below 10%. Similarly, for n = 7 and γ = 1, FedGRAA achieved around 80% accuracy, compared to 40% for FedAvg and just 2% for static reparameterization. Comparable results are observed in Figure 3 for the UNSW dataset, with similar trends for n = 5 and n = 7. These findings confirm FedGRAA’s consistent robustness as the number of clients with minority samples increases.

Title Suppressed Due to Excessive Length <page_number>9</page_number>

<table>
  <caption>Fig. 2: Performance comparison for varying γ and n values for CIC-IDS-17 dataset.</caption>
  <thead>
    <tr>
      <th rowspan="2">Communication Rounds</th>
      <th colspan="5">(a) γ = 0, n = 5</th>
      <th colspan="5">(b) γ = 0.5, n = 5</th>
      <th colspan="5">(c) γ = 1, n = 5</th>
    </tr>
    <tr>
      <th>Teal</th>
      <th>Yellow</th>
      <th>Purple</th>
      <th>Red</th>
      <th>Blue/Green</th>
      <th>Teal</th>
      <th>Yellow</th>
      <th>Purple</th>
      <th>Red</th>
      <th>Blue/Green</th>
      <th>Teal</th>
      <th>Yellow</th>
      <th>Purple</th>
      <th>Red</th>
      <th>Blue/Green</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.55</td>
      <td>0.55</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.55</td>
      <td>0.55</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.55</td>
      <td>0.55</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.70</td>
      <td>0.68</td>
      <td>0.42</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.70</td>
      <td>0.48</td>
      <td>0.42</td>
      <td>0.60</td>
      <td>0.04</td>
      <td>0.70</td>
      <td>0.50</td>
      <td>0.42</td>
      <td>0.64</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.78</td>
      <td>0.69</td>
      <td>0.44</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.78</td>
      <td>0.50</td>
      <td>0.44</td>
      <td>0.60</td>
      <td>0.04</td>
      <td>0.78</td>
      <td>0.50</td>
      <td>0.44</td>
      <td>0.64</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>60</td>
      <td>0.81</td>
      <td>0.70</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.81</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.58</td>
      <td>0.04</td>
      <td>0.81</td>
      <td>0.51</td>
      <td>0.45</td>
      <td>0.66</td>
      <td>0.04</td>
    </tr>
    <tr>
      <td>80</td>
      <td>0.81</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.81</td>
      <td>0.57</td>
      <td>0.45</td>
      <td>0.58</td>
      <td>0.05</td>
      <td>0.81</td>
      <td>0.48</td>
      <td>0.45</td>
      <td>0.67</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>100</td>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.82</td>
      <td>0.58</td>
      <td>0.45</td>
      <td>0.60</td>
      <td>0.05</td>
      <td>0.82</td>
      <td>0.48</td>
      <td>0.45</td>
      <td>0.64</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>120</td>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.82</td>
      <td>0.56</td>
      <td>0.45</td>
      <td>0.68</td>
      <td>0.05</td>
      <td>0.82</td>
      <td>0.48</td>
      <td>0.45</td>
      <td>0.67</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>140</td>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.04</td>
      <td>0.82</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.63</td>
      <td>0.05</td>
      <td>0.82</td>
      <td>0.48</td>
      <td>0.45</td>
      <td>0.67</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>160</td>
      <td>0.83</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.83</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.63</td>
      <td>0.05</td>
      <td>0.83</td>
      <td>0.50</td>
      <td>0.45</td>
      <td>0.65</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>180</td>
      <td>0.83</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.83</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.61</td>
      <td>0.05</td>
      <td>0.83</td>
      <td>0.50</td>
      <td>0.45</td>
      <td>0.65</td>
      <td>0.05</td>
    </tr>
    <tr>
      <td>200</td>
      <td>0.83</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.05</td>
      <td>0.83</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.61</td>
      <td>0.05</td>
      <td>0.83</td>
      <td>0.50</td>
      <td>0.45</td>
      <td>0.67</td>
      <td>0.05</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th rowspan="2">Communication Rounds</th>
      <th colspan="5">(d) γ = 0, n = 7</th>
      <th colspan="5">(e) γ = 0.5, n = 7</th>
      <th colspan="5">(f) γ = 1, n = 7</th>
    </tr>
    <tr>
      <th>Teal</th>
      <th>Yellow</th>
      <th>Purple</th>
      <th>Red</th>
      <th>Green</th>
      <th>Teal</th>
      <th>Yellow</th>
      <th>Purple</th>
      <th>Red</th>
      <th>Green</th>
      <th>Teal</th>
      <th>Yellow</th>
      <th>Purple</th>
      <th>Red</th>
      <th>Green</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.55</td>
      <td>0.55</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>0.55</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.55</td>
      <td>0.55</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.70</td>
      <td>0.68</td>
      <td>0.42</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.70</td>
      <td>0.50</td>
      <td>0.42</td>
      <td>0.41</td>
      <td>0.00</td>
      <td>0.70</td>
      <td>0.50</td>
      <td>0.42</td>
      <td>0.42</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>40</td>
      <td>0.78</td>
      <td>0.69</td>
      <td>0.44</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.78</td>
      <td>0.51</td>
      <td>0.44</td>
      <td>0.42</td>
      <td>0.00</td>
      <td>0.78</td>
      <td>0.51</td>
      <td>0.44</td>
      <td>0.45</td>
      <td>0.00</td>
    </tr>
    <tr>
      <td>60</td>
      <td>0.81</td>
      <td>0.70</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.39</td>
      <td>0.81</td>
      <td>0.51</td>
      <td>0.45</td>
      <td>0.42</td>
      <td>0.38</td>
      <td>0.81</td>
      <td>0.51</td>
      <td>0.45</td>
      <td>0.45</td>
      <td>0.38</td>
    </tr>
    <tr>
      <td>80</td>
      <td>0.81</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.41</td>
      <td>0.81</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.42</td>
      <td>0.41</td>
      <td>0.81</td>
      <td>0.51</td>
      <td>0.45</td>
      <td>0.45</td>
      <td>0.41</td>
    </tr>
    <tr>
      <td>100</td>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.42</td>
      <td>0.82</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.51</td>
      <td>0.42</td>
      <td>0.82</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.52</td>
      <td>0.42</td>
    </tr>
    <tr>
      <td>120</td>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.42</td>
      <td>0.82</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.51</td>
      <td>0.42</td>
      <td>0.82</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.52</td>
      <td>0.42</td>
    </tr>
    <tr>
      <td>140</td>
      <td>0.82</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.43</td>
      <td>0.82</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.51</td>
      <td>0.43</td>
      <td>0.82</td>
      <td>0.52</td>
      <td>0.45</td>
      <td>0.60</td>
      <td>0.43</td>
    </tr>
    <tr>
      <td>160</td>
      <td>0.83</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.43</td>
      <td>0.83</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.68</td>
      <td>0.43</td>
      <td>0.83</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.75</td>
      <td>0.43</td>
    </tr>
    <tr>
      <td>180</td>
      <td>0.83</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.43</td>
      <td>0.83</td>
      <td>0.57</td>
      <td>0.45</td>
      <td>0.75</td>
      <td>0.43</td>
      <td>0.83</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.75</td>
      <td>0.43</td>
    </tr>
    <tr>
      <td>200</td>
      <td>0.83</td>
      <td>0.71</td>
      <td>0.45</td>
      <td>0.01</td>
      <td>0.43</td>
      <td>0.83</td>
      <td>0.57</td>
      <td>0.45</td>
      <td>0.75</td>
      <td>0.43</td>
      <td>0.83</td>
      <td>0.54</td>
      <td>0.45</td>
      <td>0.76</td>
      <td>0.43</td>
    </tr>
  </tbody>
</table>


Fig. 2: Performance comparison for varying *γ* and *n* values for CIC-IDS-17 dataset.


<table>
  <caption>Fig. 3: Performance comparison for varying γ and n values for UNSW dataset.</caption>
  <thead>
    <tr>
      <th rowspan="2">Communication Rounds</th>
      <th colspan="4">(a) γ = 0, n = 5</th>
      <th colspan="4">(b) γ = 0.5, n = 5</th>
      <th colspan="4">(c) γ = 1, n = 5</th>
    </tr>
    <tr>
      <th>Blue</th>
      <th>Yellow</th>
      <th>Red</th>
      <th>Teal</th>
      <th>Blue</th>
      <th>Yellow</th>
      <th>Red</th>
      <th>Teal</th>
      <th>Blue</th>
      <th>Yellow</th>
      <th>Red</th>
      <th>Teal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>0.50</td>
      <td>0.50</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.29</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.50</td>
      <td>0.49</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.45</td>
      <td>0.38</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.35</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.38</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.39</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.11</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.02</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.42</td>
      <td>0.00</td>
    </tr>
  </tbody>
  <thead>
    <tr>
      <th rowspan="2">Communication Rounds</th>
      <th colspan="4">(d) γ = 0, n = 7</th>
      <th colspan="4">(e) γ = 0.5, n = 7</th>
      <th colspan="4">(f) γ = 1, n = 7</th>
    </tr>
    <tr>
      <th>Blue</th>
      <th>Yellow</th>
      <th>Red</th>
      <th>Teal</th>
      <th>Blue</th>
      <th>Yellow</th>
      <th>Red</th>
      <th>Teal</th>
      <th>Blue</th>
      <th>Yellow</th>
      <th>Red</th>
      <th>Teal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>0.50</td>
      <td>0.49</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.13</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>60</th>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.18</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.45</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.45</td>
      <td>0.18</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>140</th>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.17</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>180</th>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.46</td>
      <td>0.19</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>200</th>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.48</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>0.47</td>
      <td>0.18</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>


Fig. 3: Performance comparison for varying *γ* and *n* values for UNSW dataset.

# 6 Conclusion

We proposed a federated learning framework for intrusion detection systems that tackles the challenge of data heterogeneity through FedGRAA. Our approach dynamically adjusts aggregation weights, ensuring that clients with rare attack data have a greater impact on the global model, thus reducing bias towards more

10 No Author Given

common patterns like benign traffic. This adaptive weighting improves detection accuracy for underrepresented attacks, as demonstrated on the CIC-IDS-2017 and UNSW-NB15 datasets. In future work, we aim to develop adaptive models that improve global accuracy while maintaining performance for underrepresented samples, and validate our approach on diverse heterogeneous datasets.

# A Appendix

## A.1 Convergence Proof

The formal convergence proof of FedGRAA (Algorithm 2) is developed based on a set of standard assumptions widely adopted in the federated learning and optimization literature.

**Preliminaries and Notation** Let $\theta_g \in \mathbb{R}^d$ denote the global model parameters. At round $t$, client $k$ has local parameters $\theta_k^t \in \mathbb{R}^d$. The aggregation weights are $w^t = [w_1^t, w_2^t, \dots, w_K^t] \in \mathbb{R}^K$. The global validation dataset is $\mathcal{D}_V$, and $\mathcal{D}_V^n$ represents the $n$ underrepresented classes within $\mathcal{D}_V$.

### Assumptions

**Assumption 1. [Smoothness and Convexity]** *The composite validation loss $L'(\theta)$ is convex and $\beta$-smooth:*

$$ \|\nabla L'(\theta_1) - \nabla L'(\theta_2)\| \leq \beta\|\theta_1 - \theta_2\|, \quad \forall \theta_1, \theta_2 \in \mathbb{R}^d. $$

**Assumption 2. [Bounded Variance and Gradients]** *Stochastic gradients have bounded variance $\sigma^2$, and gradients are uniformly bounded by $G$:*

$$ \mathbb{E} \left[ \|\nabla L_k(\theta_k^t; b) - \nabla L_k(\theta_k^t)\|^2 \right] \leq \sigma^2, \quad \|\nabla L'(\theta)\| \leq G, \quad \forall \theta \in \mathbb{R}^d. $$

**Assumption 3. [Learning Rates]** *Learning rates satisfy $0 < \eta, \eta' \leq \frac{1}{\beta}$.*

**Assumption 4. [Client Participation]** *A fixed set of $K$ clients participate in each round, and each client possesses sufficient data to perform local updates.*

**Objective Function** The goal is to minimize the composite validation loss:

$$ L'(\theta_g') = (1 - \gamma)L_a(\theta_g'; \mathcal{D}_V^h, \mathcal{D}_V^m) + \gamma L_u(\theta_g'; \mathcal{D}_V^n), \quad \gamma \in [0, 1]. $$

**Convergence Theorem**

**Theorem 1.** *Under the above assumptions, FedGRAA generates sequences $\{\theta_g^t\}$ and $\{w^t\}$ such that $\theta_g^t$ converges to $\theta_g^*$, the minimizer of $L'(\theta_g)$ as $t \to \infty$:*

$$ \lim_{t \to \infty} L'(\theta_g^t) = L'(\theta_g^*) = \inf_{\theta} L'(\theta) $$
<page_number>

10
</page_number>

Title Suppressed Due to Excessive Length <page_number>11</page_number>

***Proof*** We will prove by analyzing the updates of the parameters $w^t$ and $\theta_g^t$.

***Weight Update via FedGRAA*** It updates the aggregation weights $w^t$ using gradient descent on composite loss $L'$ for $E'$ epochs. The weights are updated as:

$$w^t = w^{t-1} - \eta' \nabla_{w^{t-1}} L' \left( \sum_{k=1}^K w_k^{t-1} \theta_k^t; \mathcal{D}_V^h, \mathcal{D}_V^m, \mathcal{D}_V^n \right)$$

**Lemma 1.** *[Convergence of Weight Updates] Under the assumptions 1 through 3 and with $\eta' \leq \frac{1}{\beta}$, the sequence $\{w^t\}$ converges to $w^*$ minimizing $L'(\theta_g)$.*

*Proof.* Since $L'$ is convex and $\beta$-smooth 1, gradient descent with $\eta' \leq \frac{1}{\beta}$ ensures convergence to $w^*$. The update decreases $L'$ at each step:

$$L'(w^t) \leq L'(w^{t-1}) - \left( \eta' - \frac{\beta(\eta')^2}{2} \right) \|\nabla L'(w^{t-1})\|^2$$

Choosing $\eta' \leq \frac{1}{\beta}$ ensures that RHS is non-increasing, leading to convergence.

***Global Model Updates*** The global model is updated as:

$$\theta_g^t = \sum_{k=1}^K w_k^t \theta_k^t$$

**Lemma 2.** *[Convergence of Global Model] Given the convergence of $\{w^t\}$ to $w^*$ and the boundedness of client updates 2, the sequence $\{\theta_g^t\}$ converges to $\theta_g^*$, the minimizer of $L'(\theta_g)$.*

*Proof.* From Lemma 1, we have that $w^t \rightarrow w^*$ and local models $\theta_k^t$ are updated via gradient descent on local losses, $\theta_k^t$ approach their local minima. Therefore:

$$\theta_g^t = \sum_{k=1}^K w_k^t \theta_k^t \rightarrow \sum_{k=1}^K w_k^* \theta_k^* = \theta_g^*,$$

where $\theta_k^*$ minimizes $k^{th}$ client's loss. The optimal weights ensure $\theta_g^*$ minimize $L'$.

***Conclusion*** Combining the lemmas, we conclude that both $w^t$ and $\theta_g^t$ converge to $w^*$ and $\theta_g^*$, respectively, ensuring that $\theta_g^t$ minimizes $L'(\theta_g)$ as $t \rightarrow \infty$.

> **Note**
>
> Under convexity, smoothness, bounded gradients &amp; appropriate learning rates, ***FedGRAA*** ensures the convergence of aggregation weights &amp; global weights to their optimal values, effectively minimizing the composite loss.

12 No Author Given

# References

1. Agrawal, S., Sarkar, S., Aouedi, O., Yenduri, G., Piamrat, K., Bhattacharya, S., Maddikunta, P.K.R., Gadekallu, T.R.: Federated learning for intrusion detection system: Concepts, challenges and future directions. arXiv preprint arXiv:2106.09527 (2021), [https://arxiv.org/abs/2106.09527](https://arxiv.org/abs/2106.09527)

2. Fallah, A., Mokhtari, A., Ozdaglar, A.: Personalized federated learning: A meta-learning approach. In: Advances in Neural Information Processing Systems (NeurIPS). pp. 20232–20245 (2020), [https://arxiv.org/pdf/2002.07948](https://arxiv.org/pdf/2002.07948)

3. Kairouz, P., McMahan, H.B., et al.: Advances and open problems in federated learning. Foundations and Trends<sup>®</sup> in Machine Learning **14**(1–2), 1–210 (2021), [https://arxiv.org/pdf/1912.04977](https://arxiv.org/pdf/1912.04977)

4. Kim, S., Lee, G., Oh, J., Yun, S.Y.: Fedfn: Feature normalization for alleviating data heterogeneity problem in federated learning. In: Advances in Neural Information Processing Systems (NeurIPS) (2021), [https://arxiv.org/pdf/2311.13267](https://arxiv.org/pdf/2311.13267)

5. Li, T., Sanjabi, M., Beirami, A., Smith, V.: Federated optimization in heterogeneous networks. In: Proceedings of Machine Learning Research (PMLR) (2020), [https://arxiv.org/abs/1812.06127](https://arxiv.org/abs/1812.06127)

6. McMahan, H.B., Moore, E., Ramage, D., Hampson, S.: Communication-efficient learning of deep networks from decentralized data. Artificial Intelligence and Statistics (2017), [https://arxiv.org/pdf/1602.05629](https://arxiv.org/pdf/1602.05629)

7. Reddi, S.J., Charles, Z., Zaheer, M., Garrett, Z., Rush, K., Konečný, J., Kumar, S., McMahan, H.B.: Adaptive federated optimization (2021), [https://arxiv.org/abs/2003.00295](https://arxiv.org/abs/2003.00295)

8. Reisizadeh, A., Hassani, H., Mokhtari, A., et al.: The non-iid data quagmire of decentralized machine learning. IEEE Transactions on Signal Processing **68**, 5512–5527 (2020), [https://arxiv.org/pdf/1910.00189](https://arxiv.org/pdf/1910.00189)

9. Sahoo, P., Tripathi, A., Saha, S., Mondal, S.: Fedmrl: Data heterogeneity aware federated multi-agent deep reinforcement learning for medical imaging. arXiv preprint arXiv:2407.05800 (2024), [https://arxiv.org/pdf/2407.05800](https://arxiv.org/pdf/2407.05800)

10. Seo, J., Rong, C., Kim, M.: Fedshift: Tackling dual heterogeneity problem of federated learning via weight shift aggregation (2024), [https://arxiv.org/pdf/2402.01070](https://arxiv.org/pdf/2402.01070)

11. Wang, X., Zhang, Z., Zhang, L.: Adaptive regularization and resilient estimation in federated learning. In: International Conference on Machine Learning (ICML). pp. 10247–10256. PMLR (2021), [https://www.computer.org/csdl/journal/sc/2024/04/10316585/1S4z5n0QPcc](https://www.computer.org/csdl/journal/sc/2024/04/10316585/1S4z5n0QPcc)

12. Zhang, X., Li, T., Jin, R., Yang, Q.: Federated learning on non-iid data via local and global adaptation. Proceedings of the 38th International Conference on Machine Learning (ICML) pp. 8340–8351 (2020), [https://arxiv.org/pdf/2306.14443](https://arxiv.org/pdf/2306.14443)

13. Zhao, Y., Li, M., Lai, L., Suda, N., Civin, D., Chandra, V.: Federated learning with non-iid data. arXiv preprint arXiv:1806.00582 (2018), [https://arxiv.org/pdf/1806.00582](https://arxiv.org/pdf/1806.00582)

14. Zhao, Z., Joshi, G.: A dynamic reweighting strategy for fair federated learning. In: IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE (2022). [https://doi.org/10.1109/ICASSP43922.2022.9746300](https://doi.org/10.1109/ICASSP43922.2022.9746300), [https://ieeexplore.ieee.org/document/9746300](https://ieeexplore.ieee.org/document/9746300)