# 4. Experiments and Results — and How to Read the Tables

## 4.1 The datasets

Two standard intrusion-detection datasets:

- **CIC-IDS-2017** — 78 input features, 15 classes (1 benign + 14 attack types). See Table 1.
- **UNSW-NB15** — 43 input features, 10 classes (1 normal + 9 attack types). See Table 2.

Both are heavily **imbalanced**: a few attacks have tens of thousands of samples; others have a handful (Heartbleed = 11, SQL Injection = 21, Infiltration = 36).

## 4.2 How the data is shared out (the non-IID setup)

This is important for understanding the whole experiment:

1. **Cap the common attacks.** Very frequent attacks (DoS Hulk, PortScan, DDoS, DoS GoldenEye for CIC; Generic, Fuzzers, Reconnaissance, DoS for UNSW) are capped at **10,000 samples** so they don't completely dominate.
2. **One attack per client.** Each client gets data for **one specific attack type**, paired with an **equal number of benign/normal** samples (so each client sees a balanced 2-way slice).
3. **90 / 10 split.** Each client uses 90% of its data for local training and reserves 10% as validation $\mathcal{D}_V$.

This deliberately creates a worst-case non-IID scenario — exactly the setting where FedAvg struggles and FedGRAA is meant to shine.

## 4.3 The model and training settings

- **Network:** a fully-connected (dense) neural net built with Keras Sequential.
  - Input: 78 features (CIC) / 43 features (UNSW).
  - Hidden layers: sizes **200 → 100 → 50**, all with **tanh** activation.
  - Output: **15** neurons (CIC) / **10** neurons (UNSW) with **softmax** (one probability per class).
- **Optimizer:** Adam, model learning rate $\eta = 0.01$.
- **Local training:** $E = 2$ epochs per round, batch size $B = 64$.
- **Rounds:** $T = 200$ communication rounds.
- **Clients:** $K = 14$ (CIC), $K = 9$ (UNSW).
- **Loss:** sparse categorical cross-entropy.

**FedGRAA-specific settings (the weight learner):**
- Weight learning rate $\eta' = 0.1$, inner epochs $E' = 5$.
- $\gamma$ tested at **0, 0.5, 1**.
- Number of protected rare clients $n$ tested at **3, 5, 7**. The default ($n=3$) protects Heartbleed, SQL Injection, Infiltration (CIC) and Worms, Shellcode, Backdoor (UNSW).

## 4.4 The three methods being compared

| Name in tables | What it is |
|----------------|------------|
| **FedAvg** | Standard data-size-weighted averaging (the baseline that fails) |
| **Static** | Inverse-data-size fixed weights (eq. 7) — a partial fix |
| **Adaptive** | **FedGRAA** — the proposed method |

In the tables each is reported two ways:
- **(All classes)** = overall accuracy across every class.
- **(Last n classes)** = accuracy on just the rare/protected classes — *this is the column that matters most.*

## 4.5 How to read Figure 1's table (γ = 0, 0.5, 1; n = 3)

Focus on the **"(Last n classes)"** columns — rare-attack detection.

**CIC-IDS-2017 highlights:**
- **FedAvg (Last n) = 0.00 at every round.** The baseline never detects the rare attacks at all. This is the core failure the paper targets.
- **Static (Last n)** reaches **0.88** at many rounds — the inverse-weighting fix already helps a lot on rare classes.
- **FedGRAA (Adaptive, Last n)** at **γ = 1** holds **0.88** consistently while *also* keeping reasonable "All classes" accuracy. At **γ = 0** its rare-class accuracy drops (e.g. 0.33) because you told it not to care about rare classes.

**The trade-off in one sentence:** raising $\gamma$ raises rare-class accuracy, at the cost of a *small* dip in overall accuracy — exactly the controllable behavior the method promises.

The summary sub-table reports final rare-class accuracy:

| Method | Acc. CIC-IDS-17 (rare) | Acc. UNSW (rare) |
|--------|----------------------:|-----------------:|
| FedAvg | 0.00 | 0.00 |
| Static Reparam. | 66.66 | 0.00 |
| **FedGRAA (γ = 1)** | **88.88** | **21.09** |
| FedGRAA (γ = 0.5) | 77.77 | 20.59 |
| FedGRAA (γ = 0) | 33.33 | 0.00 |

Read it as: at $\gamma = 1$, FedGRAA detects ~89% of the rare CIC attacks where FedAvg detects 0%. On UNSW the absolute numbers are lower (UNSW rare classes are harder), but FedGRAA is still the only method that detects them at all (21% vs 0%).

## 4.6 How to read Figures 2 and 3 (robustness as n grows)

These ask: *what if MORE clients hold rare data?* They increase $n$ to 5 and 7. The colored columns (Teal, Yellow, Purple, Red, Green/Blue) are per-client accuracy curves over the 200 rounds.

Key reported findings (from the text):
- **n = 5, γ = 1:** FedGRAA ≈ **65%** accuracy, vs FedAvg and Static both **< 10%**.
- **n = 7, γ = 1:** FedGRAA ≈ **80%**, vs FedAvg ≈ **40%** and Static ≈ **2%**.
- UNSW (Figure 3) shows the same trend.

**Takeaway:** FedGRAA's advantage doesn't just survive when more clients hold rare data — it **grows**. The abstract's headline "outperforming FedAvg by up to 55%" comes from these larger-$n$ comparisons.

## 4.7 What the results prove

1. FedAvg is genuinely broken for rare-but-critical attacks (consistent 0%).
2. Inverse-size Static weighting helps on rare classes but is rigid and can be unstable across rounds.
3. **FedGRAA gives the best, controllable balance:** strong rare-class detection (tunable via $\gamma$) while keeping overall accuracy competitive, and it stays robust as the number of minority-holding clients increases.

## 4.8 Honest caveats (good to mention to your mentor)

- The non-IID setup (one attack per client) is **synthetic/extreme**; it's designed to showcase the method. Real deployments are messier.
- The server needs a **validation set with the rare classes in it** to compute $L_u$. Where that validation data comes from, and whether it's truly privacy-clean, is an assumption worth questioning.
- UNSW absolute rare-class accuracy (~21%) is still low — the method helps relative to baselines but doesn't "solve" the hardest dataset.
- You must **choose $\gamma$ and $n$** — they are hyperparameters, not learned. Picking them is a design decision.

Continue to [05_convergence_proof.md](05_convergence_proof.md).
