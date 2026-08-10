# 7. My Questions for the Mentor (Critical Review of FedGRAA)

This file collects the open questions and concerns about the paper's design choices. The researched answers are in [08_answers_researched.md](08_answers_researched.md).

## A. Data handling

1. **Why didn't the authors use SMOTE (or ADASYN / other resampling techniques)** to fix the class imbalance instead of (or in addition to) the aggregation-weight trick?
2. **Why not use a stratified split** of the data so every split keeps the rare classes in proportion?
3. **Why split frequent classes using the *median frequency*** as the threshold (into $\mathcal{D}_V^h$ and $\mathcal{D}_V^m$)? Why the median specifically?
4. **Why a 90% / 10% train–validation split?** Can we do better (e.g. k-fold cross-validation, different ratio)?

## B. Model architecture

5. **Why `tanh` activation in the hidden layers and not `ReLU`** (the modern default)?
6. **Why `softmax` on the output layer?** Are there better options?

## C. Optimization & training settings

7. **Why the `Adam` optimizer?** Are there better options (SGD+momentum, AdamW, RMSprop)? Note FedAvg classically uses SGD.
8. **Why these exact training settings** ($\eta = 0.01$, $E = 2$, $B = 64$, $T = 200$, $\eta' = 0.1$, $E' = 5$)? Are they justified or arbitrary?
9. **Why no hyperparameter tuning / search** (grid, random, Bayesian) for $\gamma$, $n$, learning rates, architecture?

## D. Theory

10. **Why the convergence assumptions** (convexity, β-smoothness, bounded gradients, fixed full client participation)? Several are unrealistic for neural networks — **how can the analysis be improved**?

## E. Caveats already noted in Section 4.8 (carried over)

- The non-IID setup (one attack per client) is **synthetic/extreme**; designed to showcase the method. Real deployments are messier.
- The server needs a **validation set containing the rare classes** to compute $L_u$. Where that data comes from, and whether it is truly privacy-clean, is an assumption worth questioning.
- UNSW absolute rare-class accuracy (~21%) is still low — the method helps relative to baselines but does not "solve" the hardest dataset.
- $\gamma$ and $n$ are **hand-chosen hyperparameters, not learned** — picking them is a manual design decision.

## F. Follow-up questions these raise (for discussion)

- Could resampling (SMOTE) and adaptive aggregation (FedGRAA) be **combined**, and would that help or double-count?
- Is the server-side validation set a **privacy leak** relative to the "no raw data leaves the client" promise of federated learning?
- Would the conclusions hold with a **more realistic non-IID** partition (mixtures of attacks per client) instead of one-attack-per-client?
- Does using **Adam locally on each client** (instead of plain SGD) interfere with the convergence proof, which is written for gradient descent?
