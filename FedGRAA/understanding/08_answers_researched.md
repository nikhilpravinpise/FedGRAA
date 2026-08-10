# 8. Researched Answers to the Critical Questions

> **How to read this file.** The paper itself does *not* justify most of these choices — it just states them. So each answer separates two things:
> - **Established fact** — backed by official documentation / well-known literature (with sources listed in §9).
> - **Likely reasoning / my analysis** — the most defensible explanation for *why this paper* made the choice, clearly flagged as inference, not something the paper claims.
>
> This honesty is deliberate: you asked me not to hallucinate. Where the paper is silent, I say so.

---

## Q1. Why not SMOTE / ADASYN / other resampling?

**What SMOTE actually is (fact, imbalanced-learn docs).** SMOTE (Synthetic Minority Over-sampling Technique) creates *new* synthetic minority samples by **interpolating between a sample and its *k* nearest neighbors of the same class**:

$$x_{new} = x_i + \lambda \,(x_{zi} - x_i), \qquad \lambda \in [0,1]$$

where $x_{zi}$ is one of $x_i$'s nearest neighbors. ADASYN is similar but generates more samples in hard-to-learn regions. Default `k_neighbors = 5` in imbalanced-learn. Crucially, for **multi-class** problems SMOTE/ADASYN use a **one-vs-rest** scheme: they "need information regarding the neighbourhood of each sample" and compute statistics *against the rest of the dataset*.

**Why SMOTE is a poor fit for *this* paper (analysis, grounded in the above facts):**

1. **Not enough minority samples to interpolate.** SMOTE needs at least `k_neighbors + 1` samples of the minority class to even run (you can't find 5 neighbors among 11 — and after the per-client split, the Heartbleed client may hold only a handful). With 11 Heartbleed / 21 SQL-Injection / 36 Infiltration samples total, interpolation just **copies near-duplicates** and invents network-traffic feature vectors that may not correspond to any real, valid attack packet. Synthetic IDS records can be physically meaningless (e.g., impossible flag/port combinations).
2. **It fights the federated/privacy design.** SMOTE's one-vs-rest neighbor search needs a **view of the whole dataset**. In FL the data is **decentralized and private** — there is no central pool to interpolate across. You could run SMOTE *locally* on each client, but each client here holds **one attack type only**, so local SMOTE just oversamples that one class and does nothing about the cross-client imbalance the paper targets.
3. **It solves a *different* problem.** SMOTE rebalances a **training set** before training a *single* model. The paper's problem is **aggregation across clients** — how much each client's *model* influences the global model. These are orthogonal; SMOTE would not change the FedAvg weighting that crushes small clients.
4. **The paper's contribution is the aggregation rule.** Mixing in SMOTE would confound the experiment: you couldn't tell whether gains came from FedGRAA or from resampling. Keeping the data pipeline simple isolates the contribution.

**Honest counter-point (for your mentor):** SMOTE and FedGRAA are **not mutually exclusive**. A reasonable extension is *local resampling + FedGRAA*, or federated variants of SMOTE (e.g., Fed-SMOTE / privacy-preserving synthetic generation). The paper simply doesn't explore this — a legitimate gap to raise.

---

## Q2. Why not a stratified split?

**What "stratify" means (fact, scikit-learn docs).** `train_test_split(..., stratify=y)` splits data "in a stratified fashion" so that the **class proportions are preserved** in both train and test subsets.

**Analysis:**
- For the *ultra-rare* classes this barely helps and can be impossible. Stratification cannot give you a sensible split of a class that has, e.g., 11 samples — you can't preserve a meaningful proportion across train/val with single-digit counts.
- More importantly, **the paper's split is along a different axis.** It does **not** do one global stratified split; it **partitions by client/attack-type on purpose** (one attack per client) to *manufacture* the non-IID condition. Stratifying would *destroy* the very heterogeneity the method is designed to handle.
- Within a client, the 90/10 split is across just two classes (one attack + benign), which are already balanced by construction (equal benign samples), so stratification adds little.

**Honest point:** for the **server-side validation set** $\mathcal{D}_V$, you *would* want the rare classes guaranteed to appear (otherwise $L_u$ can't be computed) — that is effectively a stratification requirement, and the paper relies on it implicitly without naming it.

---

## Q3. Why the *median frequency* as the high/moderate split threshold?

**Background (fact / known technique).** "Median-frequency balancing" is an established idea from imbalanced learning and semantic segmentation (popularized by Eigen & Fergus, 2015), where class weights are scaled by `median_frequency / class_frequency`. Using the **median** of class frequencies as a reference point is a standard, robust choice.

**Why median and not mean (analysis):**
- The **median is robust to outliers/skew**. Class frequencies here span four orders of magnitude (11 → 63,901). The **mean** would be dragged upward by the few huge classes, putting almost everything below it and giving a useless split. The **median** sits at the middle class by rank, so it cleanly separates the "upper half" (highly frequent, $\mathcal{D}_V^h$) from the "lower half" (moderately frequent, $\mathcal{D}_V^m$) regardless of how extreme the tails are.
- It is **automatic and parameter-free** — exactly what the paper advertises ("an automated method that determines the split threshold based on the median frequency"). No magic constant to tune.

**Honest point:** the median is a *heuristic*, not optimized. Other thresholds (quantiles, clustering of frequencies, or a learned boundary) might split high/moderate better; the paper doesn't ablate this choice.

---

## Q4. Why 90% / 10% train–validation? Can we do better?

**Fact.** There is **no theoretically optimal split ratio**; 80/20, 90/10, and 70/30 are all common conventions. The trade-off: more training data → better-fit model but noisier validation estimate; more validation data → more reliable evaluation but less to train on.

**Why 90/10 here (analysis):**
- Data is **scarce for rare classes**, so the authors keep as much as possible for **training** (90%) — sensible when some classes have only a handful of samples.
- $\mathcal{D}_V$ is used **every round** to drive the FedGRAA weight updates (eq. 10). A small but consistent validation set is enough to *compare* clients; it doesn't need to be large.

**Can we do better? (grounded suggestions):**
- **k-fold cross-validation** gives a less noisy estimate, but in FL it is **very expensive**: each fold = a full 200-round federated run × many clients. Usually impractical, which is why a single hold-out is standard.
- **Stratified hold-out** to guarantee rare classes appear in $\mathcal{D}_V$ (see Q2).
- For the tiniest classes (11 samples), *any* hold-out is statistically fragile — the 10% validation estimate for Heartbleed rests on ~1 sample. This is a genuine limitation, not fixable by ratio alone; it needs more data or leave-one-out style evaluation.

---

## Q5. Why `tanh` and not `ReLU`?

**Facts.**
- `tanh(x) = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)`, output range **(−1, 1)**, **zero-centered**, smooth and bounded (TensorFlow docs).
- `ReLU(x) = max(0, x)`, range **[0, ∞)**, not zero-centered, cheap, and the modern default for *deep* networks because it avoids the vanishing-gradient problem.

**Why tanh is reasonable *here* (analysis):**
- The network is **shallow** (only 3 hidden layers: 200→100→50). The vanishing-gradient problem that motivates ReLU mainly bites in *deep* networks; in a shallow net, tanh's saturation is much less harmful.
- Inputs are **normalized tabular features**. `tanh` is **zero-centered**, which keeps activations balanced around 0 and historically pairs well with normalized inputs — often giving smooth, stable convergence on small tabular models.
- **ReLU's "dying neuron" problem**: with ReLU, a neuron stuck in the negative region outputs 0 and its gradient is 0 forever. In a **federated** setting, where each client trains only a couple of epochs on a narrow slice of data, dead units could be more likely; tanh never fully "dies" (its gradient is non-zero everywhere except at saturation).

**Honest point:** this is almost certainly an **empirical/legacy choice** the paper doesn't justify. `ReLU`, `LeakyReLU`, `ELU`, or `GELU` might match or beat tanh — and would be a cheap ablation to run. tanh is *defensible*, not *proven optimal*.

---

## Q6. Why `softmax`? Better options?

**Fact.** `softmax` converts the output vector into a **probability distribution** over the classes (all outputs in (0,1), summing to 1). It is the **standard, correct** choice for **single-label multi-class** classification (each sample belongs to exactly one class), and it pairs mathematically with **categorical / sparse-categorical cross-entropy** — which is exactly the loss the paper uses.

**Is it the right choice here? Yes.** Intrusion classification here is single-label (a flow is one attack type *or* benign). Softmax + sparse categorical cross-entropy is the textbook-correct combination; there is no better general option for this setup.

**Nuances / alternatives (analysis):**
- If the task were **multi-label** (a flow could be several things at once), you'd use **sigmoid** outputs + binary cross-entropy instead. Not the case here.
- For severe imbalance, people sometimes keep softmax but swap the *loss* for **focal loss** or **class-weighted cross-entropy** to emphasize rare classes. That is an alternative *to the loss*, not to softmax — and it's conceptually a "competitor" to FedGRAA's $\gamma$-weighting. Worth mentioning to your mentor as a baseline the paper omits.

---

## Q7. Why `Adam`? Better options?

**Fact (Keras docs / Kingma & Ba, 2014).** Adam = **Ada**ptive **M**oment estimation: it keeps running estimates of the 1st moment (mean) and 2nd moment (variance) of the gradients and adapts a per-parameter learning rate. Defaults: `learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7`. The paper *overrides* the LR to `0.01` (model) and `0.1` (weights). Adam is described as "computationally efficient, little memory requirement, ... well suited for problems large in data/parameters."

**Why Adam (analysis):**
- It is the **robust default** — works well with little tuning, which matters when you *aren't* doing a hyperparameter search.
- Adaptive per-parameter rates help when **features have different scales/sparsity** (common in network-traffic data).
- For the **FedGRAA weight-learning** step (eq. 10), Adam's adaptivity is useful because the gradient w.r.t. the aggregation weights $w$ can be poorly scaled.

**Better options? (grounded, with a caveat specific to FL):**
- **SGD (+ momentum)** is actually the **classical choice for federated learning** — the original FedAvg uses plain SGD on clients, and much FL convergence theory (including this paper's proof, see Q10) is written for **gradient descent**, not Adam. Using Adam *locally* introduces per-client adaptive state that the simple proof doesn't model — a real inconsistency to raise.
- **AdamW** (decoupled weight decay) usually generalizes better than Adam when regularization matters.
- **Adaptive federated optimizers** — **FedAdam / FedYogi / FedAdagrad** (Reddi et al., 2021, which this paper *cites* as ref [7]) apply adaptivity at the **server** aggregation step and are arguably the most "correct" modern option for FL. The paper compares against FedAvg but not against these adaptive FL optimizers — a notable missing baseline.

---

## Q8. Why these exact training settings?

The paper just lists them; none are justified. Here is what each is and whether it's reasonable (analysis grounded in standard practice):

| Setting | Value | Standard? | Comment |
|---|---|---|---|
| Model LR $\eta$ | 0.01 | Common | 10× Adam's default (0.001); aggressive but fine for a shallow net + few local epochs |
| Local epochs $E$ | 2 | Typical for FL | Few local epochs reduces **client drift** in non-IID data (more local epochs → clients overfit their one attack → worse aggregation) |
| Batch size $B$ | 64 | Very standard | Common default; balances gradient noise vs. speed |
| Rounds $T$ | 200 | Reasonable | Enough for the accuracy curves to flatten (visible in the tables) |
| Weight LR $\eta'$ | 0.1 | Method-specific | Larger than $\eta$ because the weight vector $w$ is tiny (K numbers) and needs faster movement to react each round |
| Weight epochs $E'$ | 5 | Method-specific | A few inner steps to refine $w$ per round without overfitting the small validation set |

**Bottom line:** these are **sensible conventional defaults**, not derived or tuned. The choice of **small $E=2$** is the one with real theoretical backing (limiting client drift under non-IID). The rest are "standard values that work."

---

## Q9. Why no hyperparameter tuning?

**Analysis (grounded in FL cost realities):**
- **Federated tuning is extremely expensive.** One configuration = a full **200-round** run across **9–14 clients**. A grid/random search over LRs, batch size, architecture, $\gamma$, and $n$ multiplies that by hundreds of runs — often computationally infeasible for a conference paper.
- The paper **does** sweep its *two key* method hyperparameters — $\gamma \in \{0, 0.5, 1\}$ and $n \in \{3, 5, 7\}$ — which is the part that matters for its claim. It just doesn't tune the *generic* training settings.
- Using **standard defaults** for generic settings is also a **fairness** argument: the same architecture/optimizer is used for FedAvg, Static, and FedGRAA, so the comparison is apples-to-apples. Heavy tuning of one method could bias the comparison.

**How to do better (grounded suggestions):**
- **Bayesian optimization** (e.g. Optuna) or **Hyperband/ASHA** are far more sample-efficient than grid search and are the right tools if you can afford even a modest budget.
- Tune on a **cheaper proxy** (fewer rounds / a subset of clients) then confirm the best config at full scale.
- At minimum, run **ablations** on tanh-vs-ReLU and Adam-vs-SGD to show robustness — currently absent.

---

## Q10. Why those convergence assumptions, and how to improve the analysis?

**The four assumptions (recap) and why they're made (fact — these are the standard toolkit of convex-optimization / FL theory):**

| Assumption | Why it's assumed | How realistic |
|---|---|---|
| **Convexity + β-smoothness** | Convexity guarantees a *single* global minimum so gradient descent provably reaches *the* optimum; β-smoothness bounds how fast the gradient changes, which is what licenses the "loss decreases each step" inequality. | **Unrealistic for neural nets** — their loss surfaces are highly non-convex. This is the weakest assumption. |
| **Bounded variance ($\sigma^2$) + bounded gradients ($G$)** | Needed to control the noise of stochastic gradients so steps don't explode; standard in SGD analysis. | Mostly reasonable in practice (often enforced via gradient clipping). |
| **Small learning rates ($\eta,\eta' \le 1/\beta$)** | The classic condition under which a gradient step is guaranteed to *decrease* a β-smooth loss rather than overshoot. | Reasonable, but $1/\beta$ is unknown in practice; you tune empirically. |
| **Fixed full client participation** | Simplifies the math: the same $K$ clients every round means the aggregation is a stable weighted sum. | **Unrealistic** — real FL has clients dropping in/out (partial participation). |

**Why authors use them:** they are the **minimal set that yields a clean, closed-form convergence proof**, and they are **conventional** in the FL literature (the paper explicitly says they are "widely adopted"). A proof under these assumptions is the standard "sanity certificate" for a new aggregation rule.

**How the analysis could be improved (grounded in modern optimization/FL theory):**
1. **Drop full convexity → use the Polyak–Łojasiewicz (PL) condition** (or other "non-convex but benign" conditions). PL-based proofs give convergence guarantees for many non-convex losses and are the standard way to make NN convergence results more honest.
2. **Allow partial/stochastic client participation** — model clients sampling per round (as real FedAvg analyses do, e.g. Li et al. 2020 "On the Convergence of FedAvg on Non-IID Data") and bound the extra variance it adds.
3. **Add a data-heterogeneity bound** — explicitly quantify non-IID-ness (e.g. bounded gradient dissimilarity between clients) and show how the convergence rate degrades with it. This directly addresses the paper's own motivation (non-IID data) instead of assuming it away.
4. **Give a convergence *rate*, not just a limit.** The current proof shows $\theta_g^t \to \theta_g^*$ but not *how fast*. Modern results state $O(1/T)$ or $O(1/\sqrt{T})$ rates, which are far more informative.
5. **Model the actual optimizer.** The proof assumes plain gradient descent, but the experiments use **Adam** locally (Q7). Either analyze Adam-style updates or run experiments with SGD to match the theory.

---

## TL;DR table

| # | Question | One-line answer |
|---|----------|-----------------|
| 1 | No SMOTE? | Too few rare samples to interpolate, clashes with decentralized/private FL, and solves a *different* (data-level, not aggregation-level) problem. Could be combined as future work. |
| 2 | No stratified split? | The split is *deliberately* by-attack-per-client to create non-IID; stratifying would erase the heterogeneity the method targets. |
| 3 | Why median? | Robust to extreme skew (unlike the mean) and fully automatic/parameter-free. |
| 4 | Why 90/10? | Conserves scarce rare-class data for training; validation only needs to *rank* clients. k-fold is too costly in FL. |
| 5 | tanh not ReLU? | Shallow net + normalized features; tanh is zero-centered and avoids dead neurons. Empirical, unjustified — a good ablation target. |
| 6 | Why softmax? | Correct, standard choice for single-label multi-class; pairs with cross-entropy. Right call. |
| 7 | Why Adam? | Robust adaptive default needing little tuning. But SGD/FedAdam are more "FL-native" and missing as baselines; Adam also mismatches the GD-based proof. |
| 8 | Why these settings? | Standard defaults; only small $E=2$ has real theory (limits client drift). |
| 9 | No tuning? | FL tuning is hugely expensive; they *did* sweep the key knobs $\gamma$ and $n$; defaults keep the comparison fair. Bayesian opt/ASHA would improve it. |
| 10 | The assumptions? | Standard minimal set for a clean proof; convexity & full participation are unrealistic. Improve via PL condition, partial participation, heterogeneity bounds, convergence *rates*, and modeling Adam. |

Sources are listed in [09_sources.md](09_sources.md).
