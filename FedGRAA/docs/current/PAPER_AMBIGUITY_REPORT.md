# FedGRAA — Paper Ambiguity Report

**Paper:** *"Federated Gradient-Refined Adaptive Averaging for Intrusion
Detection Systems"* — [`PAKDD_Final_Version_Submit_v3.md`](../paper/PAKDD_Final_Version_Submit_v3.md).

**Purpose of this document.** This is a **paper-only** critique. It does not
discuss the code, does not reference `fedgraa/`, and does not propose fixes to
any implementation. It exists purely to catalogue every place the paper's own
text is ambiguous, self-contradictory, mathematically under-specified, or
internally inconsistent, so that any reproduction attempt (this one or a
future one, in any language or framework) runs into the exact same open
questions and can see them collected in one place. For the corresponding
code-level audit (how the current implementation resolves each of these), see
[`REPRODUCTION_AUDIT.md`](../audit/REPRODUCTION_AUDIT.md).

**Method.** Every equation, every algorithm-box line, every sentence of §5
("Experiments and Results"), every table, every figure caption, and the
Appendix A.1 convergence proof were read against each other (not against any
code) looking for: (a) notation used inconsistently across two parts of the
paper, (b) quantities referenced but never defined, (c) experimental
procedures described only partially, and (d) numbers in tables/figures that
look internally suspicious.

---

## Severity legend

| Tag | Meaning |
|---|---|
| **CONTRADICTION** | Two parts of the paper state genuinely incompatible things about the same object |
| **UNDEFINED** | The paper uses a term/quantity/procedure central to its own claims without ever defining it mathematically or operationally |
| **UNDERSPECIFIED** | The paper describes *a* procedure but leaves out details necessary to execute it identically twice |
| **SUSPICIOUS DATA** | Numbers reported in the paper's own tables/figures that are implausible as genuine measurements |
| **NOTATION** | Internally inconsistent or malformed mathematical notation that doesn't change the intended meaning but is technically wrong as printed |

---

## 1. Optimizer identity: SGD notation vs. Adam prose — CONTRADICTION

**Eq. 1:** $\theta_k^{t+1} = \theta_k^{t+1} - \eta\nabla L_{\theta_k^{t+1}}(\theta_k^{t+1};D_k)$

**Algorithm 1, line 17:** `θᵗ_k ← θᵗ_k − η∇L_{θᵗ_k}(θᵗ_k; b)`

Both are, notationally, plain (mini-batch) gradient descent — `param -= lr *
grad`, with no momentum or per-parameter adaptive scaling of any kind.

**§5, "Training Setup and Hyperparameters":** *"The model was trained using
the **Adam optimizer** with a learning rate of η = 0.01."*

Adam's update rule is categorically different from `param -= lr * grad` (it
maintains first and second moment estimates with bias correction). These two
descriptions cannot both be literally describing the same update rule. The
paper never acknowledges this tension or clarifies which one is figurative.
**This is a direct contradiction between the paper's own equations/algorithm
box and its own prose**, not a case of the algorithm box merely
"abbreviating" Adam — nothing about Adam is expressible as `θ - η∇L`.

## 2. Weight-update optimizer: same contradiction one level up — CONTRADICTION

**Eq. 10 / Algorithm 2, line 6:** $w^{t-1}\leftarrow w^{t-1}-\eta'\nabla_{w^{t-1}}L'(\ldots)$ — again literally plain gradient descent, this time on the weight vector $w$.

**§5:** *"For FedGRAA, we continued using the **Adam optimizer**, with a
learning rate η′ = 0.1 and epochs E′ = 5."*

Identical contradiction to §1, at the weight-learner level instead of the
client-model level. Notably the paper is at least *consistent* in which side
it picks each time (algorithm boxes always say plain GD, prose always says
Adam) — but it never states that the algorithm boxes are meant only as
"schematic" descriptions of a more sophisticated update rule.

## 3. FedAvg aggregation: two different formulas, both called "FedAvg" — CONTRADICTION

**Eq. 2:** $\theta_g^{t+1} = \frac{1}{K}\sum_{k\in S_t}\theta_k^{t+1}$ — a plain, unweighted arithmetic mean over the $K$ participating clients.

**Eq. 5:** $\theta_g^{t+1} = \sum_{k\in S_t}\frac{|D_k|}{|D|}\theta_k^{t+1}$ — the historically standard McMahan et al. (2017) **data-size-weighted** average.

These produce different numbers whenever client dataset sizes differ, which
is guaranteed in this experiment by construction (attack classes range from
11 samples to 10,000 samples per Tables 1–2, and every client's size is tied
directly to its attack class's sample count). The paper introduces Eq. 2 as
"the" FedAvg update in §3 (Background) and then, three equations later in the
same section, restates "FedAvg" with a materially different formula (Eq. 5),
without ever flagging that these disagree or which one is the operative
definition used in the actual experiments of §5. A reader trying to
independently reproduce "FedAvg" from this paper alone, using only Eq. 2, would
get a **different global model trajectory** than a reader using Eq. 5 — and
the paper gives no way to determine which curve in Figure 1's "FedAvg" columns
actually corresponds to which equation.

## 4. "Accuracy on the last n classes" — UNDEFINED

This exact phrase (and its column header in Figure 1's data table, "Last n
classes (Accuracy)") is the paper's **central reported metric** — it is the
number behind the abstract's headline "88% accuracy on minority classes" claim
— and it is never defined mathematically anywhere in the paper. At minimum two
readings are equally consistent with everything the paper says:

1. **Mean per-class recall**: average, over the $n$ protected classes, of
   (correct predictions for that class) / (total samples of that class) — each
   class weighted equally regardless of its absolute sample count.
2. **Pooled accuracy**: (correct predictions across all $n$ classes combined) /
   (total samples across all $n$ classes combined) — classes with more samples
   dominate the number.

For CIC-IDS-2017's default $n=3$ (Heartbleed=11, SQL Injection=21,
Infiltration=36 samples in the "Used" column), reading 2 gives roughly 3× more
weight to Infiltration than to Heartbleed; reading 1 gives them identical
weight. The two readings can differ by many percentage points on exactly this
kind of extremely skewed rare-class set, and the paper's own headline number
(88%) could be either. **No formula, no footnote, and no appendix passage
anywhere in the paper resolves this.**

## 5. Median-frequency split threshold — UNDERSPECIFIED

**§4 (Methodology):** *"We split the dataset $\mathcal{D}_\mathcal{V}
\setminus \mathcal{D}_\mathcal{V}^n$ into two parts, $\mathcal{D}_\mathcal{V}^h$
and $\mathcal{D}_\mathcal{V}^m$, using an automated method that determines the
split threshold based on the median frequency."*

This sentence never specifies:

- Whether "median frequency" means the median of *per-class sample counts*
  among the non-rare classes, or the median of *per-sample* class-frequency
  values (equivalent for balanced classes but not in general), or something
  else ("automated method" is left completely open).
- Whether classes **exactly at** the median go to $\mathcal{D}_\mathcal{V}^h$
  or $\mathcal{D}_\mathcal{V}^m$ (a boundary condition that matters whenever
  ties occur, e.g. multiple classes with counts capped at exactly 10,000).
- What happens when **all** non-rare classes have the same count (ties at the
  median for every class), which would make one of $\mathcal{D}_\mathcal{V}^h$
  / $\mathcal{D}_\mathcal{V}^m$ empty under a naive "strictly above" reading —
  a degenerate case the paper never anticipates or addresses, even though
  Eq. 8's $\min(L_h, L_m)$ becomes degenerate ($\min(L,\infty)=L$ trivially, or
  undefined if both are empty) whenever this happens.

## 6. Sub-gradient of $L_a = \min(L_h, L_m)$ — UNDEFINED

**Eq. 8:** $L_a = \min(L_h(\theta_g';\mathcal{D}_\mathcal{V}^h),
L_m(\theta_g';\mathcal{D}_\mathcal{V}^m))$.

**Eq. 10** then differentiates $L'$ (which contains $L_a$) with respect to $w$
as though this were routine: $w^{t-1}\leftarrow w^{t-1}-\eta'\nabla_{w^{t-1}}L'(\ldots)$.

$\min(f,g)$ of two functions is **not differentiable** at any point where
$f=g$ — a measure-zero set in general, but one that is actively and
deliberately approached by the training dynamics here, since the entire point
of Eq. 8 is to balance $L_h$ against $L_m$ (i.e., the optimization is pushing
the system *toward* the non-differentiable boundary, not away from it). Even
away from that boundary, where a sub-gradient does technically exist, the
paper never states which branch's gradient is meant, nor whether some convex
combination of both branches should be used. Appendix A.1's Assumption 1
explicitly assumes $L'$ is "convex and $\beta$-smooth" everywhere — which is
false for $\min$ of two arbitrary smooth functions at their crossing points —
so the convergence proof's own foundational assumption is inconsistent with
the very loss function (Eq. 8) the proof is supposed to be about.

## 7. Algorithm 2's inner-loop recomputation — UNDERSPECIFIED / plausibly CONTRADICTION with itself

Algorithm 2 prints (lines 2–6):

```
for each local epoch from 1 to E'
    θ'_g = w_1^{t-1}θ_1^t + ... + w_K^{t-1}θ_K^t
    L_a = min(L_h(θ'_g; ...), L_m(θ'_g; ...))
    L' = (1-γ)L_a(...) + γL_u(...)
    w^{t-1} ← w^{t-1} − η'∇_{w^{t-1}} L'(...)
```

The right-hand side of the assignment on line 3 uses the symbol $w^{t-1}$ —
the *same* symbol that line 6 is actively re-assigning inside the very loop
that line 3 also sits inside. Read maximally literally, this says: recompute
$\theta_g'$ using "$w^{t-1}$" every iteration, where "$w^{t-1}$" is
simultaneously being updated by line 6 within the same loop — i.e. the
notation for "the previous round's weights" and "the currently-being-refined
weights within this round's E′ inner iterations" is the **exact same symbol**,
never disambiguated with a separate index (e.g. an inner-epoch counter $e$).
There are at least two non-equivalent readings:

- **Iterative:** each inner epoch re-reads the *current* (already partially
  updated within this round) $w$ to recompute $\theta_g'$ and the gradient —
  i.e., $w^{t-1}$ in line 3 silently means "$w$ as of the current inner
  epoch," reusing notation from the outer round index.
- **Static:** $\theta_g'$ and its gradient are computed **once**, using the
  literal previous-round value $w^{t-1}$ before any inner-epoch updates, and
  that same fixed gradient direction is then applied for all $E'$ steps.

These give materially different weight trajectories whenever $E'>1$ (E′=5 is
the paper's own stated default), and the paper's own printed algorithm box —
by reusing one symbol for two different things — is genuinely, not just
stylistically, ambiguous about which is meant.

## 8. Figure 1's own summary table contains suspiciously round, repeating-decimal numbers — SUSPICIOUS DATA

The paper's Figure 1 caption includes a compact results table:

| Method | Acc. (CIC-IDS-17) | Acc. (UNSW-NB15) |
|---|---|---|
| FedAvg | 00.00 | 00.00 |
| Static Reparam. | 66.66 | 00.00 |
| FedGRAA (γ=1) | 88.88 | 21.09 |
| FedGRAA (γ=0.5) | 77.77 | 20.59 |
| FedGRAA (γ=0) | 33.33 | 00.00 |

`66.66`, `88.88`, `77.77`, and `33.33` are, respectively, $\frac{2}{3}$,
$\frac{8}{9}$-ish/$8\times11.11$, $7\times11.11$, and $\frac{1}{3}$ scaled to
percentages with a repeating-decimal pattern ($x.xx$ where all digits after
the decimal repeat the integer part's leading digit) — a pattern far more
consistent with **hand-picked placeholder values** (e.g., simple fractions
like 1/3, 2/3, 8/9, 7/9 typed as repeating decimals) than with genuine
measured neural-network accuracies, which essentially never land on exact
repeating-decimal boundaries by chance. This is not proof of fabrication, but
it is a legitimate, citable statistical oddity: **real accuracy measurements
from a stochastic training process reproducing to two decimal places as
33.33/66.66/77.77/88.88 simultaneously, across two unrelated methods and two
γ settings, is extraordinarily unlikely.** If these are indeed placeholder
values (e.g., left over from a table template before final numbers were
filled in, or simplified for exposition), then **bit-exact numerical
reproduction of this specific table is impossible in principle**, independent
of how faithful any reproduction's code is to the paper's methodology — a
distinct concern from every other item in this report, which are all about
method ambiguity rather than data authenticity.

## 9. Figures 2–3's per-round tables show implausibly static values over long stretches — SUSPICIOUS DATA (weaker)

In Figure 1's full per-round table (γ=0/0.5/1, n=3), several columns are
**exactly** flat for 60–100+ consecutive rounds (e.g., CIC "Static (All
classes)" stays at exactly `0.44` from round 60 through round 200 in the γ=0
panel; "Static (Last n classes)" stays at exactly `0.88` for rounds 20–100).
Neural network accuracy on held-out validation data during continued training
essentially never freezes to the same 2-decimal value across dozens of
evaluation checkpoints with zero fluctuation, even after convergence,
because mini-batch stochasticity and the specific validation sample
composition should produce at least small round-to-round noise. This is a
milder version of finding 8 above (round two-decimal values, suspiciously
static plateaus) and reinforces the same underlying concern: **at least some
of the paper's reported numbers do not have the statistical texture of raw
measured training curves.**

## 10. UNSW-NB15 genuinely does not support an "n=7 minority clients" experiment — CONTRADICTION with the paper's own data

**§5** states the $n$-sweep is run for $n\in\{3,5,7\}$ on **both** datasets
(Figures 2 and 3), and separately (Tables 1–2) that UNSW-NB15's frequent
attacks — Generic, Fuzzers, Reconnaissance, DoS — are capped at 10,000 samples
each, explicitly to *prevent* them from being treated as anything other than
dominant/frequent classes. But UNSW-NB15 (per the paper's own Table 2) has
only **five** attack classes that are not capped at the frequent ceiling:
Worms (174), Shellcode (1,511), Backdoor (2,329), Analysis (2,677), and
Exploits (10,000 used, but 44,525 original — i.e. itself a frequently-capped
class, only "not extremely dominant" relative to the true minority classes).
There simply are not 7 classes on UNSW-NB15 that are simultaneously (a)
distinct from Normal and (b) not among the frequency-capped majority classes.
**Running an "n=7 protected minority clients" experiment on UNSW-NB15, as the
paper's own Figure 3 claims to do, requires including at least 2 classes that
the paper's own Data Settings paragraph elsewhere identifies as
frequency-capped, non-minority classes** — a direct contradiction between
what §5's data-settings paragraph defines as "minority" and what Figure 3's
own experimental sweep (n up to 7) requires the term to include for this
particular dataset.

## 11. "Fixed set of $K$ clients" — NOTATION / mildly UNDERSPECIFIED

Algorithm 1, line 5: $S_t \leftarrow$ (fixed set of $K$ clients). Since $K$ is
already defined elsewhere as *the total number of clients* (§3, Eq. 1's
description), and $S_t$ is introduced as "a fixed set of clients... selected"
at each round $t$ — the notation implies client *selection* (suggesting a
subset could be chosen each round, as in the general federated learning
literature FedAvg is drawn from), but then immediately says the set is
"fixed" and sized exactly $K$ (i.e., every client, every round, always). The
paper never explains why it bothers introducing per-round selection notation
($S_t$, subscripted by $t$) for a quantity it also says never changes across
rounds — a purely notational inconsistency (using time-varying notation for a
time-invariant quantity) that does not change the intended meaning but is
technically confusing as printed.

## 12. Algorithm 1, line 17 — malformed self-referential notation

`θ^t_k ← θ^t_k − η∇L_{θ^t_k}(θ^t_k; b)`

The left-hand side of an assignment and the "current value" term on the
right-hand side use the **identical symbol** $\theta_k^t$ — there is no
distinguishing notation (like a prime, or a separate batch-index superscript)
between "the value before this gradient step" and "the value after this
gradient step," even though they are different quantities being equated by an
assignment. This is standard informal pseudocode shorthand (`x = x - 1` is
universally understood despite the same abuse of notation), so it does not
create genuine ambiguity about the intended computation — but it is
technically malformed as *mathematical* notation (as opposed to as pseudocode),
and it is listed here for completeness since it appears inside an "Algorithm"
box the paper presents as formal.

## 13. Convergence proof's convexity assumption is inconsistent with the model class the paper actually uses — CONTRADICTION

**Appendix A.1, Assumption 1:** *"The composite validation loss $L'(\theta)$
is convex and $\beta$-smooth."*

**§5, Network Architecture:** the model is a multi-layer `tanh`-activated
dense neural network (200→100→50 hidden units) with a softmax output.

The composite validation loss $L'$ is, by construction, a (weighted,
min-combined) function of this network's output on validation data — i.e., a
loss surface over the parameters of a non-linear, multi-layer neural network.
Such loss surfaces are **not convex** in general (this is one of the most
well-established facts in deep learning theory), and the paper gives no
argument, citation, or special-case reasoning for why its particular
architecture or loss combination would be an exception. The convergence proof
is therefore built on an assumption that is inconsistent with the very
network the paper's own §5 describes using for every reported experiment —
the proof, as written, does not actually apply to the system the paper claims
it applies to.

## 14. "Automated method" for the median split is asserted but never specified as an algorithm — UNDERSPECIFIED (restates/sharpens #5)

Beyond the ambiguity in #5 about *which* median is meant, the paper's phrase
"automated method that determines the split threshold" promises there is a
specific, presumably non-trivial procedure being referenced, but no such
procedure — not even a name, not even a citation to prior work — is given
anywhere in the paper (main text, related work, or appendix). This reads as
though a citation or a footnote was dropped during editing; as printed, the
paper is asserting the existence of a method it never actually discloses.

## 15. Rare-class identities for $n=5$ and $n=7$ are never named — UNDEFINED (both datasets)

**§5** explicitly names the $n=3$ rare classes for both datasets (Heartbleed,
SQL Injection, Infiltration for CIC; Worms, Shellcode, Backdoor for UNSW) but
never names which additional classes are added when $n$ is increased to 5 or
7 for the Figure 2/3 sweeps. Given finding #10 above (UNSW doesn't have 7
genuinely rare classes at all), this omission is not merely a missing detail
but plausibly conceals the exact data problem in #10 — a reader cannot tell
from the paper alone whether the authors quietly included a frequency-capped
class in their own "minority" set for the $n=7$ UNSW curve, which is one
possible resolution of #10 the paper simply never discloses either way.

## 16. No random seed, no repetition count, no variance/error bars reported — UNDERSPECIFIED

Every number in Figures 1–3 and the Fig. 1 summary table is presented as a
single point estimate with **no error bars, no confidence intervals, no
standard deviation, and no stated number of repeated runs.** Neural network
training with random initialization and stochastic mini-batch ordering
produces run-to-run variance; the paper gives no way to know whether the
reported gaps between methods (e.g., FedAvg 0.00 vs. FedGRAA 88.88 on CIC) are
robust to that variance or whether they reflect a single lucky/unlucky seed
for one or more of the compared methods. This also compounds finding #16 with
finding #8 above (the plausibility of the exact reported values).

## 17. Batch size interaction with tiny classes is unaddressed — UNDERSPECIFIED

$B=64$ is the stated batch size (§5), but several classes have far fewer than
64 *training* samples once split 90/10 and possibly further reduced (e.g.
CIC's Heartbleed at 11 total samples → 9-10 training samples after a 90/10
split). The paper never addresses what "a batch of size $B=64$" means when a
client's entire training set is smaller than $B$ — whether the last partial
batch is used as-is (a batch smaller than 64), padded, or dropped, is left
entirely to the reader's imagination, despite this being a direct,
unavoidable consequence of the paper's own combination of stated
hyperparameters ($B=64$) and stated data setup (some classes as small as
11 samples).

---

## Summary table

| # | Ambiguity | Type | Where in paper |
|---|---|---|---|
| 1 | SGD notation (Eq.1/Alg.1-L17) vs. Adam prose (§5) for client update | CONTRADICTION | §3, Alg.1, §5 |
| 2 | Plain-GD notation (Eq.10/Alg.2-L6) vs. Adam prose (§5) for weight update | CONTRADICTION | §4, Alg.2, §5 |
| 3 | Uniform FedAvg (Eq.2) vs. size-weighted FedAvg (Eq.5), both called "FedAvg" | CONTRADICTION | §3 |
| 4 | "Accuracy on the last n classes" never defined (mean-recall vs pooled) | UNDEFINED | §5, Fig.1-3 |
| 5 | Median-frequency split: which median, tie-breaking rule undefined | UNDERSPECIFIED | §4 |
| 6 | Sub-gradient of $L_a=\min(L_h,L_m)$ never specified | UNDEFINED | Eq.8, Eq.10 |
| 7 | Algorithm 2's inner loop reuses $w^{t-1}$ for two different things | UNDERSPECIFIED / self-contradictory notation | Alg.2 |
| 8 | Fig.1 summary table has suspiciously round repeating-decimal numbers | SUSPICIOUS DATA | Fig.1 caption table |
| 9 | Long flat plateaus at exact 2-decimal values in per-round tables | SUSPICIOUS DATA | Fig.1-3 tables |
| 10 | UNSW-NB15 lacks 7 genuinely non-capped "minority" classes for the n=7 sweep | CONTRADICTION w/ own data | §5, Fig.3 |
| 11 | $S_t$ notated as time-varying "selection" but defined as always-fixed/full | NOTATION | Alg.1 L5 |
| 12 | Alg.1 L17 uses identical symbol for pre- and post-update θ | NOTATION | Alg.1 L17 |
| 13 | Convergence proof assumes convexity; actual model (tanh MLP) is non-convex | CONTRADICTION | App. A.1 vs §5 |
| 14 | "Automated method" for median split never actually disclosed | UNDERSPECIFIED | §4 |
| 15 | Rare-class identities for n=5/7 never named for either dataset | UNDEFINED | §5, Fig.2-3 |
| 16 | No seed, no repetition count, no variance reported anywhere | UNDERSPECIFIED | §5, Fig.1-3 |
| 17 | Batch size B=64 vs. classes with <64 training samples: undefined behaviour | UNDERSPECIFIED | §5 |

---

## Conclusion

Setting aside any implementation, this paper contains **three direct
contradictions between its own stated equations/algorithm boxes and its own
prose** (#1, #2, #3), **one contradiction between its stated proof assumptions
and its stated model class** (#13), **one contradiction between its stated
experimental sweep and its own published dataset table** (#10), **six
central-to-the-results terms or procedures that are never mathematically or
operationally defined** (#4, #5, #6, #14, #15, and the notation issue in #7),
two purely notational inconsistencies that don't change meaning but are
technically malformed as printed (#11, #12), and two independent statistical
red flags in its own reported numbers suggesting at least some of the
published table values may not be genuine raw measurements (#8, #9), compounded
by a complete absence of variance/seed reporting (#16) and an unaddressed
edge case in its own stated hyperparameters (#17). Any reproduction — this
one included — must make an explicit, arbitrary choice at each of these points,
and no reproduction can be verified as "the" correct one without access to the
original authors' training code, seed, and raw run logs, none of which the
paper discloses.
