# Meeting Prep — Index & Live Script

**Compiled:** 2026-08-12, for a mentor meeting on 2026-08-13. Grounded entirely
in the repo's own docs as of commit `a3f871c` (main). Every number below
traces to a specific file; where two sources disagree, that's called out
rather than smoothed over.

**How to use this folder.** This is the one to have open *during* the
meeting. The other five files are the deep-dive backing material — read them
tonight, don't read them live. If you get asked something you can't recall,
the "Where to look" column below tells you which file has it.

| # | File | What's in it |
|---|---|---|
| 00 | *(this file)* | Opening script, the 4 mentor questions verbatim, quick-reference number tables |
| 01 | `01_STAGE_A_REPRODUCTION.md` | The paper, the algorithms, the datasets, the central bug, the numbers |
| 02 | `02_STAGE_B_THEORY_AND_ABLATION.md` | The corrected proof, the 10-seed factorial, the dataset-dependent finding |
| 03 | `03_STAGE_C_FEDGRAA_V2.md` | The privacy-compliant redesign, the 960-cell campaign, adversarial robustness |
| 04 | `04_MENTOR_AND_PROVENANCE.md` | Full provenance evidence (GitHub repo + the mentor's own 2025 paper), the meeting script's long form |
| 05 | `05_PAPERS_CODE_AND_AI_INTEGRITY.md` | The 17 paper ambiguities, the peer-review verdict, the code map, the "did AI ruin this" evidence |

---

## 1. The one-line status, if asked "where are things at"

> *"The technical work is done — reproduction, critique, and the Stage C
> extension are all built, tested, and audited. What's left needs your
> input, not more engineering."*

(Verbatim from `docs/current/MENTOR_CONVERSATION_2026-08-04.md`.)

## 2. The two-minute recap (say this first, near-verbatim)

> "This is the FedGRAA reproduction project. Short version: I faithfully
> reimplemented the paper's algorithm line-for-line, and it doesn't converge
> as claimed — there's a missing projection step (the aggregation weights
> should stay on the probability simplex; the paper's update rule lets them
> leave it in round one, every time, on both datasets), and that collapses
> rare-class accuracy from a claimed ~89% to ~0%. I found the exact
> mechanism, proved why the paper's own convergence proof doesn't cover it,
> and validated a minimal fix with 10-seed statistical testing — 111 of 112
> comparisons significant. That's Stage A and B, both complete and
> independently audited twice.
>
> I then built a Stage C extension — a privacy-compliant redesign (Option
> B+D from the literature review) — end to end: 960 compute cells across
> both datasets, baselines against 19 competing methods, adversarial
> robustness testing, and a full resubmission-ready manuscript. That's also
> done.
>
> What's not done is a handful of decisions only you can make, and I built
> ahead of getting your input on purpose, as a bet — I want to walk through
> those now before going further."

## 3. Ground rules for the meeting itself

- **This exact conversation has been "the next thing to do" for three-plus
  project passes running** — the goal today is answers, not another
  analysis pass (`MENTOR_CONVERSATION_2026-08-04.md` §0).
- **The provenance evidence (§4 below) is circumstantial — present it as a
  question, not an accusation.** Don't presuppose the answer.
- **Take the questions in order.** Provenance changes what everything after
  it means, so don't let the conversation jump to Stage C results first.
- **Write down the answers right after the meeting**, in
  `04_MENTOR_AND_PROVENANCE.md`'s copy of this script or in the live repo's
  own `MENTOR_CONVERSATION_2026-08-04.md` — that file has "record the answer
  here" blanks for exactly this purpose, and it's been sitting unfilled for
  over a week.

## 4. The four questions, verbatim, in order

### Q1 — Provenance (ask this first; it's the sharpest, most falsifiable one)

> "Your 2025 INFOCOM paper, 'Overcoming Data Skew: Adaptive Federated
> Reparameterization in Edge Intelligence,' proposes an algorithm called
> FedGRAA and reports 88.88% minority-class accuracy — is that the same
> FedGRAA, or the source of the same FedGRAA, that this project has been
> reproducing and critiquing?"

If ambiguous, follow up: *"Is `krishnaAiGen/Federated-Learning` on GitHub an
earlier version of what became FedGRAA?"* And if either lands yes: *"What's
the actual structure here — are you a PhD student, and who's the professor
this project is meant to be working under?"*

**Why this framing, not "where did FedGRAA come from":** it's a yes/no
question about a specific, real, DOI-backed paper (`10.1109/
INFOCOMWKSHPS65812.2025.11152908`), not an abstract provenance question the
mentor has to reconstruct from memory. Full evidence in `04`.

### Q2 — The Stage C angle (confirm, don't re-litigate)

> "I went with the privacy-compliant redesign (Option B+D) as the Stage C
> angle, based on the literature review's recommendation, without checking
> with you first — the implementation and manuscript are built around it.
> Does that angle work for you, or should this be reframed before it goes
> further?"

### Q3 — Compute

> "Stage C's baseline reproduction is bigger than anything I've run before —
> what compute does the lab actually have available, and is it worth
> reconsidering GPU now that the scale is changing?"

### Q4 — Authorship

> "Whatever Stage C becomes — and depending on how the provenance question
> above lands — who should be an author on it, and in what order?"

Full branch tables for what each answer changes: `04_MENTOR_AND_PROVENANCE.md` §2.

## 5. Closing checklist

1. All four answers recorded (or a follow-up date set for any left open).
2. The next concrete action named, and who owns it.
3. Ask whether the mentor wants to look at the manuscript
   (`docs/paper/FedGRAA_v2_PAKDD_Submission.md`) or the core-finding
   walkthrough directly, now or as a follow-up.

---

## 6. Quick-reference number tables (for when you need one fast)

### Headline result, all three stages

| | Paper's claim | Stage A literal reproduction | Stage B combined fix (10-seed) | Stage C FedGRAA-v2 (best) |
|---|---|---|---|---|
| CIC-IDS-2017 rare-class acc. | **88.88%** | **0.00%** (200-round) | **89.3%** [83.6, 94.6] (γ=1) | **93.47%** (m=3) |
| UNSW-NB15 rare-class acc. | **21.09%** | **33.33%** (200-round) | **69.2%** [63.4, 73.8] (γ=1) | **69.66%** (m=3) |

### The mechanism, one sentence each

- **The bug:** Algorithm 2 updates the aggregation-weight vector `w` by
  gradient descent and never projects it back onto the probability simplex
  (`w_k ≥ 0`, `Σw_k = 1`) — verified by direct grep of `fedgraa/aggregation.py`,
  zero matches for any projection call in the strict-reproduction class.
- **The measured consequence:** `w` leaves the simplex in **round 1**, at
  every tested γ, on both datasets — 120/120 unprojected cells at n=10 seeds,
  zero exceptions. Worst blowup: **1.86×10⁶×** (CIC, γ=0, seed 8) vs.
  **46.85×** (UNSW, γ=1, seed 9).
- **The fix:** Euclidean simplex projection + η′ cut 10× (0.1→0.01), `E′`
  unchanged. Diagnostic-only — never merged into the strict-reproduction
  default, per the project's "no silent patching" rule.
- **Which lever does the work is dataset-dependent** — the η′ cut dominates
  on CIC, projection dominates on UNSW. Neither works alone on the other
  dataset. Only `both` is never catastrophically wrong on either metric,
  either dataset, any γ.

### Stage C in one breath

- Removed 2 privacy flaws (server oracle knowledge of rare classes;
  server-held labeled validation set) → SecAgg-summed histograms + a
  federated backward pass, exact to `~2×10⁻⁷` relative error.
- 830 main-frontier cells + 90 adversarial-robustness cells + 40 γ-sweep
  addendum = **960 total**, all on Modal (cloud), real spend (~$26–29 for
  the full T1–T7 campaign).
- Beats **all 19 baselines** (14 domain-specific + 5 classic FL optimizers).
  111/112 significance comparisons hold at Holm-corrected α=0.05.
- Communication cost: 6.00× a plain-FedAvg baseline at the freshest refresh
  interval (m=1), dropping with `m`. **Flag:** the v2 manuscript itself
  states m=3→3.00× in one place (§6) and m=5→"≈2×" in another (§4.3) without
  reconciling them — see `03` §7 for the exact passages; worth resolving
  before submission, and honest to mention if it comes up.
- The mechanism's own up-weighting rule is a real, measured attack surface:
  an attacker's aggregation weight climbs to 2.7–3.1× its fair share.
  Every defense that stops the attack also destroys the rare-class signal
  that's the whole point of the method — "no free win," reported plainly.

### Provenance, in one breath

Two independent findings point the same direction: (1) the mentor's public
GitHub repo implements a closed-form ancestor of the same idea on the
identical architecture and dataset; (2) the mentor's own real, DOI-verified
2025 INFOCOM paper is *literally titled around an algorithm called FedGRAA*
and reports the *exact same* 88.88% number this project's paper carries —
a number this project's own ambiguity report had already flagged, from the
opposite direction, as a suspicious repeating-decimal placeholder. Not
proof, but strong enough that the question should be asked as a
yes/no, not left as an open mystery. Full detail: `04`.

### AI integrity, in one breath

Every headline number in every master report is machine-generated from raw
JSON via a builder script, then independently re-derived a second time by a
separate audit script that recomputes statistics from scratch (not by
re-reading the builder's own output) — and those audits found and fixed
real defects (a heavy-tailed mean reported where a median belonged; a
fabricated citation shipped in the paper's own bibliography; a stale status
doc). The audit scripts themselves were negative-control tested — a known
wrong number was injected and confirmed caught before any of them were
trusted. Full detail and exact incidents: `05` §3.
