# Understanding FedGRAA — Start Here

> **⚠️ Read this before the rest of the series (added 2026-08-03).** This
> 11-file walkthrough was written before the project's own later findings
> and is frozen at that earlier, since-corrected state — most importantly,
> **05_convergence_proof.md's "mathematically guaranteed to converge" framing
> and the ~88%/55%-over-FedAvg headline numbers below and in
> 06_glossary_and_cheatsheet.md are no longer this project's own position.**
> A rigorous, statistically-tested audit later found the paper's Algorithm 2
> has no simplex projection on its aggregation weights, that this causes
> real divergence on real data (not a hypothetical), and that the
> convergence guarantee this proof-walkthrough explains does not hold for
> the non-convex model actually used. A corrected algorithm, corrected
> convergence theory, and corrected experimental numbers now exist. Before
> treating anything numeric or convergence-related in this series as
> current, read: [`../docs/current/fedgraa_convergence_defect.md`](../docs/current/fedgraa_convergence_defect.md)
> (the defect and corrected theory), [`../docs/paper/FedGRAA_Simulated_Peer_Review.md`](../docs/paper/FedGRAA_Simulated_Peer_Review.md)
> (the independent critique that reached the same conclusion), and
> [`../docs/paper/FedGRAA_v2_PAKDD_Submission.md`](../docs/paper/FedGRAA_v2_PAKDD_Submission.md)
> (the corrected manuscript, with corrected headline numbers). This series
> remains useful for the *mechanics* of the original algorithm as
> published — how the pieces fit together — which is unchanged; it is the
> *correctness/convergence* claims that need the pointer above.

This folder explains the paper **"Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems" (FedGRAA)** from zero, assuming no prior knowledge of federated learning.

Read the files in this order:

| # | File | What it teaches |
|---|------|-----------------|
| 1 | [01_federated_learning_basics.md](01_federated_learning_basics.md) | What federated learning is, why it exists, how FedAvg works — explained from scratch |
| 2 | [02_the_problem.md](02_the_problem.md) | The non-IID / data-imbalance problem this paper attacks, and why FedAvg fails at it |
| 3 | [03_methodology_fedgraa.md](03_methodology_fedgraa.md) | The core contribution: Static Reparameterization and the FedGRAA algorithm, line by line |
| 4 | [04_experiments_and_results.md](04_experiments_and_results.md) | The datasets, setup, and how to read the result tables |
| 5 | [05_convergence_proof.md](05_convergence_proof.md) | Why the method is mathematically guaranteed to converge |
| 6 | [06_glossary_and_cheatsheet.md](06_glossary_and_cheatsheet.md) | Every symbol, term, and a one-page summary |
| 7 | [07_my_questions.md](07_my_questions.md) | Critical questions about the design choices (for the mentor) |
| 8 | [08_answers_researched.md](08_answers_researched.md) | Researched, documentation-grounded answers to those questions |
| 9 | [09_sources.md](09_sources.md) | Sources verified, separating fact from analysis |

## The 30-second version

- **Goal:** Train one shared intrusion-detection model across many organizations *without* anyone sharing their raw network traffic (privacy).
- **The standard method (FedAvg)** averages each client's model weighted by how much data they have. Big clients dominate; rare attack types (Heartbleed, SQL Injection) get ignored.
- **This paper's idea (FedGRAA):** Instead of fixing the averaging weights by data size, *learn* the averaging weights with gradient descent on a validation set, so that rare/minority attack classes are deliberately boosted. A knob `γ` (gamma) controls how much you favor the rare classes.
- **Result:** Detection accuracy on rare attacks jumps from ~0% (FedAvg) to ~88% on CIC-IDS-2017, while overall accuracy stays competitive.

## The one mental model to keep

Normal federated learning asks: *"How much data does each client have?"* and weights them by that.

FedGRAA instead asks: *"How much does trusting each client **help me detect the rare attacks** I care about?"* and weights them by that — and it figures this out automatically every round.

Continue to [01_federated_learning_basics.md](01_federated_learning_basics.md).
