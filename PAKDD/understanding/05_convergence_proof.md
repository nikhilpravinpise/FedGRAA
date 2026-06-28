# 5. The Convergence Proof (Appendix A.1), Explained Simply

A new training algorithm should come with a guarantee that it actually **converges** — i.e. that repeating the rounds drives the model toward the best possible solution rather than wandering forever. Appendix A.1 provides this guarantee for FedGRAA. You don't need to be a mathematician; here is the logic.

## 5.1 What "converge" means here

The goal is to **minimize the composite validation loss** $L'$ (equation 9 — the $\gamma$-blend of frequent and rare losses). "Converges" means:

$$\lim_{t \to \infty} L'(\theta_g^t) = L'(\theta_g^*) = \inf_\theta L'(\theta)$$

In words: as the number of rounds $t$ grows, the loss of the global model approaches the **lowest possible** loss $L'(\theta_g^*)$. The model stops getting worse and settles at the best achievable point. **Theorem 1** states exactly this.

## 5.2 The assumptions (the "fine print")

The proof holds *if* four standard conditions are true. These are the usual textbook assumptions in optimization/FL theory:

1. **Smoothness & Convexity (Assumption 1).** The loss $L'$ is **convex** (bowl-shaped — one global bottom, no misleading local dips) and **β-smooth** (its slope doesn't change too abruptly). Convexity is what lets gradient descent reliably reach *the* minimum.
2. **Bounded variance & gradients (Assumption 2).** The noise in the stochastic gradients is bounded by $\sigma^2$, and gradients never blow up (bounded by $G$). This keeps steps from being wildly erratic.
3. **Learning rates small enough (Assumption 3).** $0 < \eta, \eta' \le \tfrac{1}{\beta}$. If the step size is small enough relative to the smoothness, each step makes progress instead of overshooting.
4. **Stable client participation (Assumption 4).** The same $K$ clients take part each round and each has enough data to do its local update.

> Caveat to note: real neural-network losses are **not** convex, so Assumption 1 is an idealization. This is standard in the FL literature — the proof gives theoretical reassurance, not an ironclad real-world guarantee.

## 5.3 The proof in two steps

The proof splits into two lemmas — one for the **weights** $w$, one for the **model** $\theta_g$ — then combines them.

### Lemma 1 — the weights converge

FedGRAA updates the weights by gradient descent (eq. 10). A classic result says: *gradient descent on a convex, β-smooth function with step size $\eta' \le 1/\beta$ is guaranteed to converge to the minimizer.*

The proof shows each weight update **decreases** the loss:

$$L'(w^t) \le L'(w^{t-1}) - \Big(\eta' - \tfrac{\beta (\eta')^2}{2}\Big)\,\|\nabla L'(w^{t-1})\|^2$$

The bracket $\big(\eta' - \tfrac{\beta(\eta')^2}{2}\big)$ is **positive** when $\eta' \le 1/\beta$. So the term being subtracted is positive — the loss goes **down every step** and keeps going down until the gradient is zero (the minimum). Hence the weights $w^t \to w^*$ (the optimal weights).

### Lemma 2 — the global model converges

The global model is just the weighted sum of client models:

$$\theta_g^t = \sum_{k=1}^K w_k^t\, \theta_k^t$$

We now know two things:
- From Lemma 1, the weights converge: $w^t \to w^*$.
- The client models, trained by ordinary gradient descent on their local losses, approach their own optima $\theta_k^*$ (and are bounded, by Assumption 2).

A weighted sum of converging things, with converging weights, itself converges:

$$\theta_g^t = \sum_k w_k^t \theta_k^t \;\longrightarrow\; \sum_k w_k^* \theta_k^* = \theta_g^*$$

And because the weights $w^*$ are the ones that **minimize** $L'$, the resulting $\theta_g^*$ is the loss-minimizing global model.

### Combining

Both the weights and the model converge to their optimal values, so $\theta_g^t$ minimizes $L'$ as $t \to \infty$. That is Theorem 1. ∎

## 5.4 The intuition without symbols

- FedGRAA has **two nested learning processes**: clients learn their models, and the server learns the aggregation weights.
- Each process is plain gradient descent on a (assumed) nice bowl-shaped loss with a cautious step size.
- Gradient descent on a bowl always rolls to the bottom.
- So both processes reach their bottoms, and the combined global model reaches the bottom of the composite loss $L'$.

The paper's closing note sums it up: *under convexity, smoothness, bounded gradients, and appropriate learning rates, FedGRAA guarantees that both the aggregation weights and the global weights converge to their optimal values, minimizing the composite loss.*

Continue to [06_glossary_and_cheatsheet.md](06_glossary_and_cheatsheet.md).
