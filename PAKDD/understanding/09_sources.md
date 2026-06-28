# 9. Sources Consulted

These are the documentation pages and references I checked to ground the answers in [08_answers_researched.md](08_answers_researched.md). Claims tied to these are **fact**; anything labeled "analysis" in that file is my reasoning about the authors' likely intent (the paper does not justify these choices itself).

## Official documentation verified

- **SMOTE / ADASYN mechanics** — imbalanced-learn User Guide, "Over-sampling": confirms SMOTE generates samples by interpolation `x_new = x_i + λ(x_zi − x_i)` between a sample and its *k* nearest neighbors; multi-class uses a one-vs-rest scheme needing cross-class neighborhood statistics; default `k_neighbors`. <https://imbalanced-learn.org/stable/over_sampling.html>
- **Stratified split** — scikit-learn `train_test_split`: the `stratify` parameter splits "in a stratified fashion" preserving class proportions; default `test_size`/`train_size` complement to 0.25/0.75. <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html>
- **tanh** — TensorFlow/Keras: `tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)`, smooth, bounded, zero-centered. <https://www.tensorflow.org/api_docs/python/tf/keras/activations/tanh>
- **Adam optimizer** — Keras `Adam`: "stochastic gradient descent method based on adaptive estimation of first-order and second-order moments"; defaults `learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7`; cites Kingma & Ba (2014). <https://keras.io/api/optimizers/adam/>

## Well-established literature referenced (knowledge, not freshly fetched)

- **SMOTE** — Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," JAIR 2002.
- **Adam** — Kingma & Ba, "Adam: A Method for Stochastic Optimization," 2014 (arXiv:1412.6980).
- **Median-frequency balancing** — Eigen & Fergus, "Predicting Depth, Surface Normals and Semantic Labels...," ICCV 2015 (origin of median-frequency class weighting).
- **Adaptive federated optimizers (FedAdam/FedYogi/FedAdagrad)** — Reddi et al., "Adaptive Federated Optimization," ICLR 2021 — *this is reference [7] in the paper itself*.
- **FedAvg on non-IID convergence** — Li et al., "On the Convergence of FedAvg on Non-IID Data," ICLR 2020 (example of partial-participation + heterogeneity-bound analysis).
- **PL condition for non-convex convergence** — Karimi, Nutini & Schmidt, "Linear Convergence of Gradient and Proximal-Gradient Methods Under the Polyak–Łojasiewicz Condition," ECML-PKDD 2016.

## What I did NOT verify (flagged for honesty)

- The **authors' actual motivations** for tanh, Adam, 90/10, and the median threshold — the paper does not state them, so those parts of the answers are clearly labeled **analysis / likely reasoning**, not fact.
- Exact per-client sample counts after splitting (the paper gives totals in Tables 1–2, not the per-client post-split breakdown).
