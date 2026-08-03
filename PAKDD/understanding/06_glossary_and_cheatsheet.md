# 6. Glossary and One-Page Cheat Sheet

## 6.1 Symbol glossary

| Symbol | Meaning |
|--------|---------|
| $\theta$ | Model parameters (the neural network's weights) |
| $\theta_g$ (or $\theta_g^t$) | The **global** model held by the server (at round $t$) |
| $\theta_k$ (or $\theta_k^t$) | Client $k$'s **local** model |
| $\theta_g'$ | A **temporary** global model FedGRAA builds while learning the weights |
| $K$ | Number of clients (14 for CIC-IDS, 9 for UNSW) |
| $S_t$ | The set of clients participating in round $t$ |
| $D_k$ | Client $k$'s private local dataset |
| $|D_k|$ | Number of samples client $k$ has |
| $|D|$ | Total samples across all clients = $\sum_k |D_k|$ |
| $t$, $T$ | Current round, total rounds ($T = 200$) |
| $\eta$ (eta) | Model learning rate (= 0.01) |
| $\eta'$ (eta-prime) | **Weight** learning rate, used inside FedGRAA (= 0.1) |
| $E$ | Local training epochs per round (= 2) |
| $E'$ | Inner epochs for weight learning in FedGRAA (= 5) |
| $B$ | Batch size (= 64) |
| $L$, $L_k$ | Loss; loss on client $k$'s data |
| $w = [w_1,\dots,w_K]$ | **Aggregation weights** — how much each client influences the global model |
| $\mathcal{D}_V$ | Validation set (server-side, from clients' reserved 10%) |
| $\mathcal{D}_V^n$ | Validation data for the $n$ **rare/underrepresented** classes |
| $\mathcal{D}_V^h$ | Validation data for **highly** frequent classes (above median) |
| $\mathcal{D}_V^m$ | Validation data for **moderately** frequent classes (below median) |
| $L_u$ | Loss on the rare classes ($\mathcal{D}_V^n$) — the thing we want to push down |
| $L_h$, $L_m$ | Losses on highly / moderately frequent classes |
| $L_a$ | Frequent-class loss = $\min(L_h, L_m)$ (eq. 8) |
| $L'$ | **Composite loss** = $(1-\gamma)L_a + \gamma L_u$ (eq. 9) — what FedGRAA minimizes |
| $\gamma$ (gamma) | **Personalization coefficient** $\in [0,1]$: 1 = favor rare classes, 0 = favor common classes |
| $n$ | Number of rare/protected client classes (tested at 3, 5, 7) |
| $\nabla$ | Gradient ("the direction of steepest increase") |

## 6.2 Key equations at a glance

- **FedAvg aggregation (the baseline):** $\theta_g^{t+1} = \sum_k \frac{|D_k|}{|D|}\theta_k^{t+1}$  (eq. 5) — weight by data size. Fails on rare classes.
- **Static Reparameterization:** $\theta_g^{t+1} = \sum_k \frac{1/|D_k|}{\sum_j 1/|D_j|}\theta_k^t$  (eq. 7) — weight by *inverse* size. Helps but is rigid.
- **Composite loss:** $L' = (1-\gamma)L_a + \gamma L_u$  (eq. 9) — the tunable objective.
- **Weight update (the key idea):** $w \leftarrow w - \eta' \nabla_w L'(\sum_i w_i \theta_i)$  (eq. 10) — *learn* the weights by gradient descent.
- **Final aggregation:** $\theta_g^t = \sum_k w_k^t \theta_k^t$ — combine using the learned weights.

## 6.3 The three methods in one line each

- **FedAvg** — average client models weighted by data size. Big clients dominate; rare attacks → 0% accuracy.
- **Static Reparameterization** — average weighted by *inverse* data size. Boosts small clients, but weights are frozen for all rounds.
- **FedGRAA (proposed)** — *learn* the averaging weights every round via gradient descent on a $\gamma$-controlled blend of common-class and rare-class validation loss. Adaptive, tunable, and convergence-proven.

## 6.4 The whole paper in five sentences

1. Federated learning trains a shared model across clients without sharing raw data, which is ideal for privacy-sensitive intrusion detection.
2. But real clients have **non-IID, imbalanced** data, and standard **FedAvg** weights clients by data size, so rare-but-critical attacks (Heartbleed, SQL Injection) are essentially ignored (≈0% detection).
3. The paper proposes **FedGRAA**, which treats the aggregation weights as **learnable parameters** and trains them by gradient descent to minimize a loss that explicitly rewards rare-class accuracy, controlled by a knob $\gamma$.
4. On CIC-IDS-2017 and UNSW-NB15, FedGRAA lifts rare-attack accuracy from ~0% to ~88% (CIC) while keeping overall accuracy competitive, and its advantage grows as more clients hold rare data. *(This is the paper's own claimed number — see the correction below; it is not this project's current position.)*
5. A convergence proof (under standard convexity/smoothness assumptions) guarantees the learned weights and global model settle at the loss-minimizing optimum. *(Also superseded — see below.)*

> **⚠️ Correction (2026-08-03).** Sentences 4 and 5 above, and "convergence-proven"
> in §6.3, restate the paper's own claims as this project's summary of them —
> but this project later found both do not hold as stated. The algorithm as
> printed has no simplex projection on its aggregation weights, which
> provably leave the simplex on real data starting at round 1, invalidating
> the non-increasing-loss assumption sentence 5 relies on; the corrected
> algorithm recovers a real but smaller accuracy gain (not the ~88%/55%-over-FedAvg
> figures above) under a weaker, honest convergence guarantee (stationarity,
> not global optimality). See
> [`../docs/current/fedgraa_convergence_defect.md`](../docs/current/fedgraa_convergence_defect.md)
> and [`../docs/paper/FedGRAA_v2_PAKDD_Submission.md`](../docs/paper/FedGRAA_v2_PAKDD_Submission.md)
> for the corrected numbers and theory.

## 6.5 Questions to test your own understanding

1. Why does FedAvg give the Heartbleed client a near-zero aggregation weight? *(Because its weight = data share, and Heartbleed has only 11 samples.)*
2. What makes the loss $L'$ differentiable **with respect to the weights $w$**? *(Because $\theta_g' = \sum w_k \theta_k$, so the loss is a function of $w$.)*
3. What does setting $\gamma = 0$ do, and why would you ever want it? *(Ignores rare classes, optimizes overall accuracy — useful if you only care about aggregate performance.)*
4. Why is Static Reparameterization called "static"? *(Its inverse-size weights are computed once and never change across rounds.)*
5. What is the difference between $\eta$ and $\eta'$? *($\eta$ trains the model on clients; $\eta'$ trains the aggregation weights on the server.)*

Back to [00_START_HERE.md](00_START_HERE.md).
