# 10. SOTA Comparison — Adaptive Aggregation in Federated Learning

**Scope of this document:** Papers that do *adaptive aggregation* — i.e., dynamically adjust how much each client's model contributes to the global model — and are most related to FedGRAA's approach.

**FedGRAA's key mechanism (recap):**
- Server side only — no extra client cost.
- After each round, compute a temporary global model `θ'_g = Σ w_k θ_k`.
- Evaluate a composite validation loss `L' = (1-γ)·min(L_h, L_m) + γ·L_u` on a server-held validation set.
- Update aggregation weights `w` via Adam gradient descent on `L'` for `E'` epochs.
- The gradient flows: `dL'/dw_k = <∇_P L', θ_k>` (inner product of the loss gradient w.r.t. model params and the client's params).
- Personalization knob `γ ∈ [0,1]` tunes how much weight goes to minority vs majority classes.

---

## Paper 1 — Agnostic Federated Learning (AFL)

| Field | Details |
|---|---|
| Authors | Mehryar Mohri, Gary Sivek, Ananda Theertha Suresh |
| Venue | ICML 2019 |
| arXiv | [1902.00146](https://arxiv.org/abs/1902.00146) |

### What they do

AFL frames FL as an *agnostic* problem: rather than minimizing the weighted average loss (as FedAvg does), it finds a global model that minimizes the **worst-case loss** over all possible *mixtures* of the client distributions.

Formally, let `λ ∈ Δ_K` be a probability simplex vector (mixture weights). AFL solves:

```
min_θ  max_{λ ∈ Δ_K}  Σ_k λ_k · L_k(θ)
```

This is a min-max problem. The inner max finds the **hardest mixture** (the distribution over clients that the current model performs worst on), and the outer min trains the model against it.

**How λ is updated:** The mixture weights λ are updated by stochastic gradient ascent on each round:

```
λ ← project(λ + α · ℓ)   where ℓ_k = L_k(θ)
```

The weights are pushed toward clients with higher loss — essentially focusing on whoever is hardest.

### Similarity to FedGRAA

| Dimension | AFL | FedGRAA |
|---|---|---|
| Who updates weights | Server | Server |
| What drives the update | Gradient ascent on per-client loss | Gradient descent on composite validation loss |
| Objective | Minimax fairness (worst-case distribution) | Minority-class accuracy while maintaining global accuracy |
| Weight signal | Per-client loss L_k(θ) | Composite L' = (1-γ)·L_a + γ·L_u |
| Optimizer | SGD (projected gradient ascent on λ) | Adam gradient descent on w |

### Key difference

AFL is **adversarial**: it pushes weights toward the *hardest* clients to make the model robust to worst-case distributions. FedGRAA is **targeted**: it explicitly pushes weights toward clients that improve minority-class detection, governed by γ. FedGRAA's composite loss lets you tune the minority/majority trade-off; AFL has no such control.

AFL also has no notion of class-level imbalance — it works at the client level, not the class level.

---

## Paper 2 — q-FedAvg: Fair Resource Allocation in Federated Learning

| Field | Details |
|---|---|
| Authors | Tian Li, Maziar Sanjabi, Ahmad Beirami, Virginia Smith |
| Venue | ICLR 2020 |
| arXiv | [1905.10497](https://arxiv.org/abs/1905.10497) |

### What they do

q-FedAvg aims for **fairness across devices** — specifically, a more uniform accuracy distribution so that no single client is left far behind. The objective (q-FFL) is:

```
min_θ  (1/(q+1)) Σ_k L_k(θ)^{q+1}
```

where `q ≥ 0` is a hyper-parameter. When `q = 0`, this reduces to FedAvg. As `q → ∞`, it approximates minimax fairness (AFL).

The aggregation weights in q-FedAvg are:

```
w_k ∝ L_k(θ)^q · |D_k|
```

Clients with **higher loss get higher weight** — the model is penalized more for leaving them behind.

### Similarity to FedGRAA

| Dimension | q-FedAvg | FedGRAA |
|---|---|---|
| Who updates weights | Server | Server |
| What drives the update | Per-client loss value L_k^q | Gradient of composite validation loss |
| Objective | Uniform accuracy across clients (fairness) | Accuracy for minority classes in IDS |
| Weight signal | Scalar per-client loss (no validation set needed) | Gradient requiring a held-out validation set |
| Optimizer | Closed-form reweighting per round | Adam gradient descent for E' epochs |

### Key difference

q-FedAvg weights are computed from **closed-form loss values** — no validation set, no gradient through the aggregation. FedGRAA computes weights by **differentiating through** the aggregated model on a held-out validation set. This gives FedGRAA a richer signal (which clients help minority classes specifically) at the cost of needing a server validation set.

q-FedAvg focuses on per-device fairness (each device performs well); FedGRAA focuses on per-class fairness (each *attack class* is detected), which is the right framing for IDS.

---

## Paper 3 — FedAdp: Fast-Convergent Federated Learning with Adaptive Weighting

| Field | Details |
|---|---|
| Authors | Hongda Wu, Ping Wang |
| Venue | arXiv preprint (2020), IEEE Transactions on Mobile Computing 2022 |
| arXiv | [2012.00661](https://arxiv.org/abs/2012.00661) |

### What they do

FedAdp assigns per-client weights based on the **angle between the client's local gradient vector and the global gradient vector**. The intuition: a client whose gradient points in the same direction as the global gradient is contributing positively; a client pointing in the opposite direction is diverging and should receive less weight.

The weight for client `k` in round `t` is:

```
w_k ∝ f(cos(θ_k_local_grad, θ_global_grad))
```

where `f(·)` is a nonlinear mapping function that **amplifies positive alignment** and **suppresses negative alignment** (clients that diverge from the global direction get downweighted).

### Similarity to FedGRAA

| Dimension | FedAdp | FedGRAA |
|---|---|---|
| Who updates weights | Server | Server |
| What drives the update | Angle between local and global gradient | Gradient of composite validation loss |
| Objective | Faster convergence under non-IID | Better minority-class accuracy in IDS |
| Weight signal | Gradient cosine similarity (geometric) | Validation loss gradient (task-specific) |
| Validation set needed | No (uses gradient geometry) | Yes (server-held validation set) |

### Key difference

FedAdp is a **convergence-focused** method: it selects clients that help the model converge faster. It doesn't care about which classes are being detected — a client with rare attacks and divergent gradients might be downweighted. FedGRAA is **accuracy-focused on underrepresented classes**: it explicitly upweights clients based on their contribution to improving rare-class detection, even if those clients' gradients are "noisy" from a convergence standpoint.

---

## Paper 4 — FedAtt: Attentive Federated Aggregation

| Field | Details |
|---|---|
| Authors | Shaoxiong Ji, Shirui Pan, Guodong Long, Xue Li, Jing Jiang, Zi Huang |
| Venue | IJCNN 2019 |
| arXiv | [1812.07108](https://arxiv.org/abs/1812.07108) |

### What they do

FedAtt uses an **attention mechanism** to compute aggregation weights. Each client's weight is proportional to the "distance" (or similarity) between its model and the current server model. The attention weight is computed as:

```
α_k ∝ exp(-d(θ_g, θ_k))
```

where `d(·,·)` is a distance measure between server and client parameters. The server model is then updated by minimizing a weighted distance objective:

```
min_{θ_g}  Σ_k α_k · ||θ_g - θ_k||^2
```

This is solved by iterative parameter updates on the server side, going beyond a simple one-shot weighted average.

### Similarity to FedGRAA

| Dimension | FedAtt | FedGRAA |
|---|---|---|
| Who updates weights | Server | Server |
| What drives the update | Distance between server and client params | Gradient of composite validation loss |
| Objective | Language model perplexity / global performance | Minority-class accuracy in IDS |
| Weight signal | Geometric distance in parameter space | Task-specific validation loss gradient |
| Iterative server update | Yes (multiple iterations on server) | Yes (E' weight update epochs) |

### Key difference

FedAtt weights clients by **geometric closeness** to the server model — clients with similar parameters are trusted more. This doesn't capture *what* a client contributes (e.g., its effect on rare classes). FedGRAA weighs clients by **functional contribution** to a specific objective (the composite loss), which directly measures whether a client's model helps detect rare attacks.

---

## Paper 5 — FedOpt / FedAdam: Adaptive Federated Optimization

| Field | Details |
|---|---|
| Authors | Sashank Reddi, Zachary Charles, Manzil Zaheer, et al. (Google) |
| Venue | ICLR 2021 |
| arXiv | [2003.00295](https://arxiv.org/abs/2003.00295) |

### What they do

FedOpt reframes the server-side aggregation as an **optimization step** rather than a simple average. After receiving client updates, the server treats the aggregated update as a pseudo-gradient and applies an adaptive optimizer (Adam, Yogi, or Adagrad) to the global model:

```
Δ_t = Σ_k w_k (θ_g^{t-1} - θ_k^t)   // pseudo-gradient
θ_g^t = θ_g^{t-1} - ServerOptimizer(Δ_t)
```

FedAdam uses Adam on the server; FedAdagrad uses Adagrad, etc. This adapts the **learning rate** for each parameter dimension, not the per-client weights.

### Similarity to FedGRAA

| Dimension | FedOpt | FedGRAA |
|---|---|---|
| Who updates weights | Server (updates model, not client weights) | Server (updates aggregation weights) |
| What's adapted | Per-parameter learning rates (momentum/variance) | Per-client aggregation weights |
| Objective | Faster / more stable convergence | Minority-class accuracy |
| Mechanism | Adaptive gradient descent on θ | Gradient descent on w |

### Key difference

FedOpt adapts **how the global model parameters are updated** (learning rate adaptation). FedGRAA adapts **how much each client contributes** (aggregation weight adaptation). These are orthogonal dimensions — FedGRAA could theoretically use FedAdam as the server optimizer while also learning the weights.

---

## Paper 6 — SCAFFOLD: Stochastic Controlled Averaging for FL

| Field | Details |
|---|---|
| Authors | Karimireddy, Kale, Mohri, Reddi, Stich, Suresh |
| Venue | ICML 2020 |
| arXiv | [1910.06378](https://arxiv.org/abs/1910.06378) |

### What they do

SCAFFOLD addresses **client drift**: in non-IID settings, clients' local gradients diverge from the global gradient, so FedAvg can converge to a biased solution. SCAFFOLD fixes this by introducing **control variates** — correction terms that each client adds to its local gradient to align it with the global gradient:

```
θ_k^{t+1} = θ_k^t - η(∇L_k(θ_k^t) - c_k + c)
```

where `c_k` is the client's control variate, and `c` is the server's control variate. These are updated each round and uploaded alongside the model.

### Similarity to FedGRAA

| Dimension | SCAFFOLD | FedGRAA |
|---|---|---|
| Who updates weights | N/A — uses fixed aggregation | Server |
| What's adapted | Gradient correction per client | Per-client aggregation weight |
| Objective | Eliminate client drift, recover IID convergence rate | Better minority-class accuracy |
| Extra communication | Yes (control variates per client) | No extra communication |

### Key difference

SCAFFOLD doesn't change aggregation weights at all — it changes the gradients clients compute before sending updates. It's a **gradient correction** method, not an **aggregation weight** method. FedGRAA keeps client training unchanged and instead adjusts how the server combines the resulting models.

---

## Paper 7 — FedNova: Tackling Objective Inconsistency in Heterogeneous FL

| Field | Details |
|---|---|
| Authors | Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor |
| Venue | NeurIPS 2020 |
| arXiv | [2007.07481](https://arxiv.org/abs/2007.07481) |

### What they do

FedNova identifies **objective inconsistency**: when clients perform different numbers of local steps, naive averaging creates a biased objective. FedNova normalizes each client's update by the number of local steps it took:

```
θ_g^{t+1} = θ_g^t - η_g · (Σ_k a_k · Δ_k^t / τ_k)
```

where `τ_k` is the effective number of local steps for client `k` and `a_k` is the data-size weight. This ensures the global model converges to the *true* global optimum.

### Key difference from FedGRAA

FedNova fixes a **normalization bug** in FedAvg for heterogeneous local updates. It uses fixed data-size weights `a_k ∝ |D_k|` — no adaptation for non-IID class distributions. FedGRAA has learnable weights and is specifically designed for class-imbalanced non-IID settings.

---

## Paper 8 — pFedMe: Personalized FL with Moreau Envelopes

| Field | Details |
|---|---|
| Authors | Canh T. Dinh, Nguyen H. Tran, Tuan Dung Nguyen |
| Venue | NeurIPS 2020 |
| arXiv | [2006.08848](https://arxiv.org/abs/2006.08848) |

### What they do

pFedMe decouples **personalized model learning** from **global model aggregation** using the Moreau envelope as a regularizer. Each client learns a personalized model `θ_k*` that is close to the global model `w`:

```
θ_k* = argmin_θ  L_k(θ) + (λ/2)||θ - w||^2
```

The global model `w` is then updated by aggregating the personalized models. This allows each client to specialize while still benefiting from the shared global model.

### Key difference from FedGRAA

pFedMe produces **per-client personalized models** — each client gets a different model. FedGRAA produces a single **global model** that performs better across all clients, especially on rare classes. pFedMe is about personalization; FedGRAA is about better global aggregation for class-imbalanced data.

---

## Summary Comparison Table

| Paper | Year | Venue | Aggregation weights | Weight update mechanism | Needs validation set | Handles class imbalance | Target domain |
|---|---|---|---|---|---|---|---|
| **FedGRAA (ours)** | 2024 | PAKDD | Learnable per-client | Adam grad descent on composite loss | Yes (minority/majority split) | Yes (explicit γ parameter) | IDS (CIC-IDS, UNSW) |
| AFL | 2019 | ICML | Learned mixture weights | SGD gradient ascent (minimax) | No (uses per-client loss) | Indirect (worst-case) | General |
| q-FedAvg | 2020 | ICLR | Loss-based reweighting | w_k ∝ L_k^q | No | Indirect (fairness) | General |
| FedAdp | 2022 | TMC | Gradient angle-based | Cosine similarity mapping | No | No | General (MNIST, Fashion) |
| FedAtt | 2019 | IJCNN | Attention (distance-based) | Iterative param-distance minimization | No | No | NLP (language modeling) |
| FedOpt | 2021 | ICLR | Fixed (data-size) | N/A (adapts LR not weights) | No | No | General |
| SCAFFOLD | 2020 | ICML | Fixed (equal) | N/A (gradient correction) | No | No | General |
| FedNova | 2020 | NeurIPS | Fixed (data-size + normalized) | N/A (normalization fix) | No | No | General |
| pFedMe | 2020 | NeurIPS | Global model only | Moreau envelope bilevel | No | No | Personalization |

---

## FedGRAA's Unique Contributions vs. SOTA

### 1. Gradient differentiates *through* the aggregation
FedGRAA computes `dL'/dw_k = <∇_θ L', θ_k>` — the gradient flows through the entire aggregation operation. No other paper in this table does this exact operation (AFL does gradient ascent on λ but using scalar loss, not inner products of gradient and client params).

### 2. Composite loss with explicit minority control
The loss `L' = (1-γ)·min(L_h, L_m) + γ·L_u` is unique:
- `L_u` = loss on **underrepresented** classes (rare attacks: Heartbleed, SQL Injection, Infiltration)
- `L_a = min(L_h, L_m)` = loss on majority classes (the `min` avoids over-penalizing one group)
- `γ` gives direct, interpretable control over the minority/majority trade-off

None of AFL, q-FedAvg, FedAdp, or FedAtt have this kind of class-specific, configurable objective.

### 3. IDS-specific design
All other papers are evaluated on general ML tasks (MNIST, CIFAR, language modeling). FedGRAA is designed for — and evaluated on — intrusion detection datasets (CIC-IDS-2017, UNSW-NB15) where class imbalance is extreme (Heartbleed: 11 samples vs. DoS Hulk: 231K samples).

---

## Recommended SOTA Baselines for Experimental Comparison

For the paper's experimental section, the most defensible SOTA baselines are:

| Priority | Paper | Reason |
|---|---|---|
| **Must-have** | AFL (Mohri et al. 2019) | Most mechanistically similar — both do gradient-based weight learning on the server |
| **Must-have** | q-FedAvg (Li et al. 2020) | Standard benchmark for FL fairness / minority handling; widely cited |
| **Must-have** | FedAdp (Wu & Wang 2022) | Direct gradient-based adaptive weighting; recent, same non-IID motivation |
| **Should have** | FedOpt / FedAdam (Reddi et al. 2021) | Standard adaptive FL baseline; already implied by FedOpt reference [7] in paper |
| **Nice to have** | FedAtt (Ji et al. 2019) | Attention-based aggregation weights; shows FedGRAA beats geometric approaches |
| **Context** | FedNova / SCAFFOLD | Show that *normalization/correction* approaches don't solve class imbalance |

**Note:** FedProx (already in the paper as [5]) is already covered.
**Correction (2026-08-03):** this note previously also named "FedSND
(already cited as [12])" as covered. That citation was later confirmed
**fabricated** — three independent bibliographic tools found zero trace of
it — and has been removed from the paper's bibliography and replaced with
FedNova (see `docs/paper/draft1.md`'s reference notes and
`FEDGRAA_V2_MASTER_PLAN.md` F6). Do not cite it. `StaticDistributionAwareTrainer`
in `fedgraa/baselines.py` was itself formerly named after this same
hallucinated citation and has since been renamed for the same reason.

---

## How to Position FedGRAA in the Related Work

**The key narrative:**
> Prior adaptive aggregation methods (AFL, q-FedAvg) optimize distribution weights using scalar loss values and converge to worst-case or fair-distribution objectives. Gradient-angle methods (FedAdp) improve convergence but ignore class semantics. Attention methods (FedAtt) use geometric proximity without regard for which classes are improved. FedGRAA is the first to (1) differentiate through the aggregation operation using a **class-stratified validation loss**, (2) provide an **explicit minority/majority trade-off** via γ, and (3) demonstrate superior performance on **extreme class-imbalanced IDS datasets** where rare attacks have as few as 11 training samples.

---

## Next Steps

1. **Implement AFL and q-FedAvg** in the `fedgraa/` codebase as additional baselines
2. **Run experiments** with these baselines on CIC-IDS-2017 (and optionally UNSW-NB15)
3. **Add to Related Work section** of the paper: position each baseline and state why FedGRAA differs
4. **Add to Experiments section**: comparison table showing minority-class accuracy and global accuracy
