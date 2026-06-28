# 3. The Methodology — Static Reparameterization and FedGRAA

This is the heart of the paper. We build up in three steps:
1. **Static Reparameterization** — a simple first fix (a baseline the paper also compares against).
2. **The big idea of FedGRAA** — *learn* the aggregation weights instead of fixing them.
3. **The FedGRAA algorithm, line by line.**

---

## 3.1 Step 1 — Static Reparameterization (the simple fix)

FedAvg weighted clients by their data size $|D_k|$, which crushes small clients. The most obvious counter-move is to do the **opposite**: weight clients by the **inverse** of their data size, so small clients get *more* influence, not less.

$$\theta_g^{t+1} = \sum_{k=1}^{K} \frac{1/|D_k|}{\sum_{j=1}^{K} 1/|D_j|}\, \theta_k^{t} \quad \text{(eq. 7)}$$

Reading it:
- $1/|D_k|$ is large when $|D_k|$ is small. So the Heartbleed client (11 samples) gets a big number, and a big client (10,000 samples) gets a tiny number.
- The denominator $\sum_j 1/|D_j|$ just **normalizes** so all the weights add up to 1.

**Effect:** small/rare-attack clients now have a *large* say in the global model. This already beats FedAvg on rare classes (in the results, "Static" reaches ~88% on the last-n classes for CIC-IDS at some rounds, vs FedAvg's 0%).

**Why it is not enough — the word "Static":** these weights are **computed once from data sizes and never change.** They are fixed for all 200 rounds. The paper's criticism:

> "it lacks the flexibility to adapt to variations in data distribution or client performance over time."

A client might be helping a lot early on and very little later; a fixed weight cannot react. We want weights that **adapt** based on how well the model is actually doing. That is FedGRAA.

> ⚠️ Note on terminology: confusingly, the *inverse-size* formula is itself called "Static Reparameterization," and FedGRAA *uses it only to initialize* its weights (round 0). After that, FedGRAA keeps adjusting them.

---

## 3.2 Step 2 — The big idea of FedGRAA

**FedGRAA = Federated Gradient-Refined Adaptive Averaging.**

The breakthrough is a change in *what kind of thing* the aggregation weights are:

| | FedAvg | Static Reparam. | **FedGRAA** |
|---|--------|-----------------|-------------|
| Weight $w_k$ = | data share $\frac{|D_k|}{|D|}$ | inverse-size share | **a learnable parameter, trained by gradient descent every round** |

In FedGRAA, the aggregation weights $w = [w_1, \dots, w_K]$ are treated like the model parameters of a *tiny extra model*. The server **trains them** to minimize a special loss that rewards good performance — especially on the rare classes.

### The three ingredients FedGRAA needs

**(a) A validation set $\mathcal{D}_V$ held by the server.**
Each client reserves 10% of its data; collectively these form a validation set the server can evaluate on. (Privacy note: in this paper's setup the server uses a validation set to score the global model. This is how it "knows" whether the rare classes are being detected.)

**(b) Splitting that validation set by class frequency.** The validation data is divided into three buckets:
- $\mathcal{D}_V^n$ — the **n underrepresented (rare) classes** we want to protect (e.g. the last $n=3$: Heartbleed, SQL Injection, Infiltration).
- The remaining classes $\mathcal{D}_V \setminus \mathcal{D}_V^n$ are split, using the **median frequency** as an automatic threshold, into:
  - $\mathcal{D}_V^h$ — **highly frequent** classes (above median).
  - $\mathcal{D}_V^m$ — **moderately frequent** classes (below median).

**(c) The personalization coefficient $\gamma$ (gamma), with $\gamma \in [0,1]$.** This is the tuning knob:
- $\gamma = 1$ → care **only** about the rare classes.
- $\gamma = 0$ → care **only** about the frequent classes (overall accuracy).
- $\gamma = 0.5$ → balance the two.

### The loss FedGRAA minimizes

First define a loss for the frequent classes. The paper uses the **minimum** of the high- and moderate-frequency losses:

$$L_a = \min\big(L_h(\theta_g'; \mathcal{D}_V^h),\; L_m(\theta_g'; \mathcal{D}_V^m)\big) \quad \text{(eq. 8)}$$

(Taking the `min` means "the loss on whichever of the two frequent groups is currently doing better" — it keeps the frequent-class term well-behaved.)

Then combine it with the rare-class loss $L_u$ (loss on $\mathcal{D}_V^n$) using $\gamma$:

$$L' = (1-\gamma)\, L_a(\theta_g'; \mathcal{D}_V^h, \mathcal{D}_V^m) \;+\; \gamma\, L_u(\theta_g'; \mathcal{D}_V^n) \quad \text{(eq. 9)}$$

This is a **weighted blend of two objectives**:
- $(1-\gamma) \times$ "do well on common classes"
- $+\ \gamma \times$ "do well on rare classes"

Turning $\gamma$ up slides the priority toward rare classes. This single equation is the conceptual core of the whole paper.

### The trick: the loss depends on the weights $w$

Here is why it's clever. The temporary global model is built from the aggregation weights:

$$\theta_g' = w_1\,\theta_1 + w_2\,\theta_2 + \dots + w_K\,\theta_K$$

So $\theta_g'$ is a **function of $w$**. Therefore the loss $L'(\theta_g')$ is *also* a function of $w$. That means we can take the gradient of $L'$ **with respect to the weights $w$** and do gradient descent **on the weights**:

$$w \leftarrow w - \eta' \, \nabla_{w} L'\Big(\sum_{i=1}^K w_i \theta_i;\ \mathcal{D}_V^h, \mathcal{D}_V^m, \mathcal{D}_V^n\Big) \quad \text{(eq. 10)}$$

In plain English: *"Nudge each client's aggregation weight up or down in whatever direction makes the blended validation loss go down."* If trusting the Heartbleed client more reduces the rare-class loss, its weight $w$ automatically grows. This is the **"Gradient-Refined Adaptive"** part of the name.

- $\eta'$ (eta-prime) is a **separate learning rate** for the weights (the paper uses $\eta' = 0.1$, larger than the model's $\eta = 0.01$).
- This weight update is repeated for $E'$ inner epochs (the paper uses $E' = 5$).

---

## 3.3 Step 3 — The full algorithm, line by line

The paper has two algorithm boxes. Algorithm 1 is the outer federated loop; Algorithm 2 (FedGRAA) is the weight-learning subroutine the server calls each round.

### Algorithm 1 — the outer loop (server + clients)

```
INPUT: T rounds, learning rates η and η',
       initial weights w⁰ (from Static Reparam, eq. 7),
       personalization coefficient γ,
       validation set D_V, rare-class validation set D_V^n

SERVER:
  initialize global model θ_g⁰
  for each round t = 1 … T:
      S_t ← the fixed set of K clients
      for each client k in S_t  (in parallel):
          send the current global model θ_g^{t-1} to client k
          θ_k^t ← CLIENTUPDATE(k, θ_g^{t-1})     # client trains locally
      θ_c^t = [θ_1^t, θ_2^t, … , θ_K^t]            # collect all client models
      w^t ← FedGRAA(θ_c^t, w^{t-1})               # ← LEARN the weights (Algo 2)
      θ_g^t = Σ_{k∈S_t} w_k^t · θ_k^t              # aggregate with learned weights
  return θ_g^T                                     # final global model

CLIENTUPDATE(k, θ):                                # runs on client k
  θ_k ← θ                                          # start from global model
  for each local epoch 1 … E:                      # E = 2 in the paper
      for each batch b in D_k (size B = 64):
          θ_k ← θ_k − η · ∇L(θ_k; b)               # ordinary SGD step
  return θ_k                                        # send model back (no raw data)
```

The **only** difference from FedAvg is line `w^t ← FedGRAA(...)`. FedAvg would have used fixed data-share weights; here the weights are recomputed by training them.

### Algorithm 2 — FedGRAA (the weight learner, runs on the server)

```
FedGRAA(θ_c^t, w^{t-1}):          # θ_c^t = all client models this round
  for each inner epoch 1 … E':    # E' = 5
      # 1) Build a temporary global model from current weights
      θ_g' = w_1^{t-1}·θ_1^t + w_2^{t-1}·θ_2^t + … + w_K^{t-1}·θ_K^t

      # 2) Compute the frequent-class loss
      L_a = min( L_h(θ_g'; D_V^h),  L_m(θ_g'; D_V^m) )

      # 3) Blend frequent + rare losses with γ
      L' = (1−γ)·L_a + γ·L_u(θ_g'; D_V^n)

      # 4) Gradient-descend the WEIGHTS (not the model!)
      w^{t-1} ← w^{t-1} − η' · ∇_{w} L'(θ_g'; D_V^h, D_V^m, D_V^n)

  w^t ← w^{t-1}
  return w^t
```

Walk through what happens each inner epoch:
1. Mix the client models with the current weights → a candidate global model $\theta_g'$.
2. Score it on the frequent classes and on the rare classes separately.
3. Combine the two scores using $\gamma$ (your priority knob).
4. Adjust the **weights** so the combined score improves. Repeat 5 times.

After 5 inner epochs the refined weights $w^t$ are returned, and Algorithm 1 uses them to form the real new global model $\theta_g^t = \sum_k w_k^t \theta_k^t$.

---

## 3.4 Why this works — intuition

- **It is performance-driven, not size-driven.** Weights are set by "does this client help me detect what I care about?" rather than "how big is this client?". A tiny Heartbleed client that is the *only* source of Heartbleed knowledge will get a large weight because dropping it spikes the rare-class loss $L_u$.
- **It is adaptive.** Weights are re-learned every round, so they track the model's current strengths and weaknesses instead of being frozen.
- **It is controllable.** One number $\gamma$ slides you along the trade-off between overall accuracy and rare-class accuracy — no retraining of the architecture required.
- **It is still private.** The server only ever combines client *models* and evaluates on a held-out validation set. Raw client data never moves.

---

## 3.5 The one-paragraph summary

FedGRAA replaces FedAvg's fixed, data-size-based averaging weights with **weights that are themselves trained by gradient descent** every round. The training objective is a $\gamma$-controlled blend of "accuracy on frequent classes" and "accuracy on rare classes." Because the global model is a weighted sum of client models, the loss is differentiable with respect to those weights, so the server can nudge each client's influence up or down to maximize detection of the rare, critical attacks. The result: rare-attack accuracy climbs from ~0% (FedAvg) to ~88%, while overall accuracy stays competitive.

Continue to [04_experiments_and_results.md](04_experiments_and_results.md).
