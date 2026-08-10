# 2. The Problem: Non-IID Data and Why FedAvg Fails

## 2.1 What "non-IID" means

IID = **Independent and Identically Distributed**. It is a statistics assumption meaning: *every client's data looks like a random sample from the same overall population.* If data were IID, every client would have roughly the same mix of attack types in the same proportions.

In the real world of intrusion detection, this is false. The paper calls this **non-IID** data ("non-independent and identically distributed"). Different organizations see different attacks at different rates:

> "certain clients may face frequent attempts of one type of intrusion, while others rarely see it."

In this paper's experimental design, the non-IID-ness is made extreme on purpose: **each client is given data for exactly ONE attack type** (plus an equal amount of benign/normal traffic). So:
- Client 1 only knows about "DoS Hulk".
- Client 8 only knows about "Heartbleed".
- ...and so on.

No single client has a complete picture. The global model must stitch their knowledge together.

## 2.2 The class-imbalance problem on top of non-IID

Look at Table 1 (CIC-IDS-2017) in the paper. The number of samples per attack type is wildly unequal:

| Attack | Samples used |
|--------|-------------:|
| BENIGN | 63,901 |
| DDoS | 10,000 |
| DoS Hulk | 10,000 |
| PortScan | 10,000 |
| ... | ... |
| XSS | 652 |
| **Infiltration** | **36** |
| **SQL Injection** | **21** |
| **Heartbleed** | **11** |

Some attacks are **common** (tens of thousands of samples). Some are **rare** — Heartbleed has just **11** samples in the entire dataset. But rare does not mean unimportant: Heartbleed, SQL Injection, and Infiltration are dangerous, critical attacks. A security system that ignores them is useless precisely where it matters most.

These rare classes are the **minority classes** / **underrepresented classes** the paper keeps referring to.

## 2.3 Why FedAvg fails here — the math

Recall FedAvg's aggregation weight for client $k$ is its data share $\frac{|D_k|}{|D|}$ (equation 5).

The client holding Heartbleed has only ~11 samples. Its weight is therefore microscopic:

$$\frac{|D_{\text{Heartbleed}}|}{|D|} = \frac{11}{\text{(tens of thousands)}} \approx 0.0001$$

So when the server builds the global model, the Heartbleed client's contribution is almost **multiplied by zero**. Its hard-won knowledge of what Heartbleed looks like barely affects the global model. The paper states this directly:

> "clients with fewer samples have less influence ... the aggregation process favors clients with larger datasets, leading to a global model biased towards them."

Formally (the paper's wording): the global model $\theta_g^{t+1}$ tends to approximate the model of whichever client $j$ has the largest $|D_j|$. The big clients win; the rare-attack clients are drowned out.

**Consequence:** FedAvg gets near **0% accuracy on the rare attack classes** — you can see this in the result tables, where the "FedAvg (Last n classes)" column is `0.00` everywhere.

## 2.4 The second symptom: slow / unstable convergence

There is a related issue. Each client trains on a different attack, so their gradients (the directions they want to push the model) point in **different directions**. The paper writes:

$$\nabla L(\theta) = \sum_{k=1}^{K} \frac{|D_k|}{|D|} \nabla L_k(\theta) \quad \text{(eq. 6)}$$

When these per-client gradients $\nabla L_k$ are **misaligned**, their weighted sum can **oscillate** instead of moving smoothly toward a good solution. Training takes more rounds and can be unstable. This is the "slower convergence" challenge.

## 2.5 What we actually want

We want an aggregation rule that:
1. Does **not** punish a client just for having little data.
2. Deliberately **amplifies** the clients holding rare-but-critical attack knowledge.
3. Can be **tuned**: sometimes you care more about overall accuracy, sometimes more about catching the rare attacks.
4. Still **converges** reliably.

FedAvg fails (1) and (2) by design. A first naive fix — "Static Reparameterization" — partly addresses (1) and (2) but fails (3). FedGRAA, the paper's contribution, addresses all four. That is the subject of the next file.

Continue to [03_methodology_fedgraa.md](03_methodology_fedgraa.md).
