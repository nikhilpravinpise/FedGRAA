# 1. Federated Learning Basics (from zero)

## 1.1 The motivation: privacy

Suppose 14 different companies each run a network and want a machine-learning model that detects cyber-attacks (an **Intrusion Detection System**, or IDS). A model gets better the more data it sees. So the obvious idea is: *collect everyone's network traffic into one big server and train there.*

The problem: **network traffic is sensitive.** It contains private information about each company's systems, users, and vulnerabilities. No company wants to ship its raw traffic to a central server. This is the "centralized data collection" concern the paper mentions in its introduction.

**Federated Learning (FL)** solves this. The key idea:

> Don't move the data to the model. Move the model to the data.

Each company (called a **client**) keeps its data locally. Only **model parameters** (the trained weights/numbers) travel over the network — never the raw data.

## 1.2 The actors

- **Clients** ($k = 1, 2, \dots, K$): the participating parties. Each client $k$ owns a private local dataset $D_k$. In this paper $K = 14$ for the CIC-IDS-2017 dataset and $K = 9$ for the UNSW-NB15 dataset.
- **Server**: a central coordinator. It never sees raw data. It only holds the shared **global model** $\theta_g$ and combines updates from clients.

The symbol $\theta$ (theta) just means "the model's parameters" — the big list of numbers that define a neural network. $\theta_g$ = the global (server) model; $\theta_k$ = client $k$'s local copy.

## 1.3 The training loop (one "communication round")

FL works in repeated **rounds**. The paper runs $T = 200$ rounds. One round looks like this:

```
SERVER:  starts with the current global model θ_g
   │
   ├──► sends θ_g down to every selected client
   │
CLIENTS (in parallel, each on its own private data):
   │     start local model θ_k = θ_g
   │     train for a few epochs on local data D_k  (ordinary gradient descent)
   │     get an improved local model θ_k
   │
   ├──◄ each client sends ONLY θ_k back up (no raw data)
   │
SERVER:  combines all the θ_k into a new global model θ_g
   │
   └──► repeat for the next round
```

After many rounds, the global model has effectively "learned" from everyone's data without that data ever leaving its owner.

### The local training step

On a client, training is just standard neural-network training. For each batch $b$ of local data:

$$\theta_k \leftarrow \theta_k - \eta \, \nabla L_{\theta_k}(\theta_k; b)$$

In words: *adjust the weights a little bit in the direction that reduces the loss.*
- $\eta$ (eta) is the **learning rate** — the step size.
- $\nabla L$ is the **gradient** — the direction of steepest increase of the loss; we step the opposite way (the minus sign).
- $L$ is the **loss** — a number measuring how wrong the model currently is.

This is equation (1) in the paper, and lines 13–18 of Algorithm 1.

## 1.4 FedAvg: the standard way to combine models

The combining step ("aggregation") is the heart of FL. The original and most famous method is **Federated Averaging (FedAvg)**, by McMahan et al. (2017), reference [6] in the paper.

FedAvg's rule: **average the client models, weighted by how much data each client has.**

$$\theta_g^{t+1} = \sum_{k \in S_t} \frac{|D_k|}{|D|} \, \theta_k^{t+1}$$

This is equation (5). Decoding it:
- $|D_k|$ = number of data samples client $k$ has.
- $|D| = \sum_k |D_k|$ = total samples across all clients.
- $\frac{|D_k|}{|D|}$ = client $k$'s **share** of the data. These shares add up to 1.
- So the global model is a weighted blend: a client with more data pulls the global model more strongly toward its own model.

**Example:** If client A has 9,000 samples and client B has 1,000 samples, then A's weight is 0.9 and B's is 0.1. The global model is `0.9 × (A's model) + 0.1 × (B's model)`. A dominates.

This is intuitive and works *when every client has similar data*. The next file explains why it breaks when they don't — which is exactly the situation in intrusion detection.

## 1.5 Key terms introduced so far

- **Client / Server** — the parties and the coordinator.
- **Global model** $\theta_g$ — the shared model the server maintains.
- **Local model** $\theta_k$ — each client's private copy during a round.
- **Round** — one cycle of send-down → local-train → send-up → aggregate.
- **Aggregation** — how the server merges client models.
- **FedAvg** — aggregate by data-size-weighted average.

Continue to [02_the_problem.md](02_the_problem.md).
