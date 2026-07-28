# Stage C — FastTrack + ARS Final Saturation Pass — 2026-07-26

> **What this is.** A final, tool-diversified verification pass over [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) (last updated 2026-07-25) and its two addenda, run via the ARS `deep-research` skill (fact-check mode) plus the **FastTrack Literature Open** MCP server (OpenAlex + Semantic Scholar, ~250M records — a different index from the arXiv-API/DBLP/web-search stack every prior pass used). Purpose: (1) attempt full-text retrieval of the two blocking reads flagged in the 2026-07-25 pass — **FedPGO** and **CELM**; (2) run FastTrack's purpose-built instruments (Duplication Test, Gap Saturation, Topic Debate Map) against the core novelty claim as an independent cross-check; (3) one last broad sweep for anything an arXiv/DBLP-only search could structurally miss. This is an addendum — it does not edit the master doc. §7 lists proposed edits.
>
> **Verification tags**, unchanged from prior passes: ✅ Confirmed (primary source fetched), ⚠️ Corrected/Clarified, ❌ Not found, ❓ Open.

---

## 1. Headline results

1. **CELM is now fully read — full PDF text obtained and analysed (§3).** This closes half of the 2026-07-25 blocking item. The master doc's characterization holds on the objective (data-free, class-wise, minority-upweighting) but was imprecise on mechanism: CELM does **not** use gradients at all — it uses server-side **activation/logit maximization on synthetic probe inputs**, and it computes weights once during a short warm-up (~5% of rounds) and **freezes** them, rather than adapting every round. This is a sharper, more favorable differentiation for FedGRAA than the master doc currently states, not a weaker one — see §3.4.
2. **FedPGO remains inaccessible in full text**, now confirmed independently via a second and third channel (FastTrack/OpenAlex metadata-only; an author SSRN preprint predecessor exists but returns HTTP 403 on both the DOI-redirect and the direct SSRN URL). This is now a **documented access constraint**, not a gap in search effort — three independent tools (arXiv/DBLP web stack, FastTrack/OpenAlex, direct SSRN fetch) have all been tried. See §2.
3. **FastTrack's Duplication Test and Gap Saturation Check, run independently on a different literature index (OpenAlex/Semantic Scholar vs. the arXiv/DBLP stack used in every prior pass), corroborate rather than overturn the 2026-07-25 findings.** No paper closer to FedGRAA than FedPGO surfaced. The gap-saturation publication curve (§4) shows the space **heating up sharply in 2025–2026**, which is quantitative, cross-index confirmation of what the 2026-07-25 pass argued qualitatively.
4. **Two new adjacent (not occupying) papers surfaced**: **HADA-FL** (IEEE Access 2025, 17 citations already — unusually fast uptake) and **TACMA Fed** (Symmetry 2026, explicitly extends AMAFed with "rare-class-aware weighting"). Neither uses gradient-refinement; both add to the crowding evidence. See §5.
5. **A sharpened reviewer-risk finding on XAI-SOH-FL**, now confirmed via full abstract text: XAI-SOH-FL's own stated motivation is that its base method SOH-FL suffers from *"reliance on a manually tuned aggregation parameter γ"* — the **same Greek letter FedGRAA uses for its own manually-set trade-off coefficient** — and XAI-SOH-FL's fix is to make that γ dynamic via similarity-thresholding and Bayesian optimization. This is a concrete, citable, near-verbatim reviewer question FedGRAA should pre-empt: *why is your γ fixed rather than tuned, the way the nearest same-domain paper found this exact weakness worth fixing?* See §6.

---

## 2. FedPGO — third independent attempt, still full-text-blocked

- **FastTrack/OpenAlex record** (`get_paper`, DOI `10.1016/j.comnet.2026.112210`): ✅ independently corroborates title, authors (Dawit Dejene Bikila, Jan **Čapek** — note the diacritic, absent from prior passes' plain-ASCII rendering), venue (*Computer Networks*), year 2026. `cited_by_count: 2`, both in 2026 — i.e. it already has two citations, which is fast for a paper this new and worth tracking. Not OA; no PDF URL via OpenAlex.
- **New find: an author preprint predecessor.** FastTrack `search_papers` surfaced an SSRN preprint by the same two authors — **"A Personalized Federated Learning Framework for IoT Intrusion Detection,"** Dawit Dejene Bikila & Jan Čapek, SSRN Electronic Journal, 2025, **DOI 10.2139/ssrn.5637328** — almost certainly an earlier draft of what became FedPGO (title matches the "personalized federated learning framework for IoT intrusion detection" phrase from FedPGO's own title, before the "gradient orthogonalization" framing was finalized). This was **not in any prior pass of the master doc** and is a genuinely new lead.
- **Attempted retrieval, twice, both blocked:** `WebFetch` on `https://doi.org/10.2139/ssrn.5637328` correctly redirected to `https://www.ssrn.com/abstract=5637328`; a direct fetch of that URL and of the alternate SSRN URL pattern (`papers.ssrn.com/sol3/papers.cfm?abstract_id=5637328`) both returned **HTTP 403** (SSRN's bot-protection, not a paywall in the Elsevier sense — but functionally the same outcome for this pass).
- **Verdict: FedPGO's full text is now confirmed unobtainable through three independent channels** (Elsevier direct, per the 2026-07-25 pass; FastTrack/OpenAlex, this pass; SSRN preprint mirror, this pass). Recommend: obtain via institutional library access (the master doc already flags this) or request the preprint directly from the authors (Bikila is contactable — University of Pardubice, per the *Computer Networks* affiliation typical for this author). **This item stays open and blocking for Related Work**, but is no longer a search-effort question — it is an access-logistics question.

---

## 3. CELM — full text obtained and read (blocking item CLOSED)

`WebFetch` on the arXiv PDF (`arxiv.org/pdf/2605.18892`) failed to extract text directly (returned raw PDF byte structure), but the binary was cached locally and re-read with the `Read` tool, which does support PDF extraction. Full 22-page paper (main text + appendix) read in full, including Algorithm 1, all four experiment tables, and the limitations section.

### 3.1 What CELM's mechanism actually is (not gradient-based)

This is the most important correction from this pass. The master doc's §1 finding N2 and §7 both describe CELM in language that reads as mechanistically adjacent to FedGRAA ("gradient-refined," "the objective FedGRAA uses"). Having read the full method (§4 of the paper, Eq. 4–14, Algorithm 1), **CELM does not compute or use gradients of any loss function on any data, real or synthetic.** Its mechanism is:

1. During a short warm-up window (`T_w = 0.05T`, i.e. the first 5% of communication rounds), the server treats each client's model as a fixed function and runs **activation maximization**: starting from random Gaussian noise, it does gradient ascent *on a synthetic input* `x` (not on model parameters) to maximize each class's output logit, producing an optimized synthetic image `x*_c` per client per class.
2. The resulting logit value `q_c = s_c(x*_c; w)` is used as a proxy for "how well does this client's model recognize class c" — a **class-competence probe**, not a gradient-informed contribution weight.
3. These probe scores are de-biased against a global-model baseline, normalized per class, averaged into a per-client score, projected onto the simplex, and EMA-smoothed (Eq. 10–13) — **this is the part that matches the master doc's description** ("simplex constraints and momentum smoothing").
4. After warm-up, the weights are **frozen** for the rest of training (Eq. 14) — CELM is deliberately *not* an every-round adaptive scheme; freezing is a design choice to bound server overhead.

### 3.2 Consequence for the master doc's positioning

This sharpens, and slightly *softens*, the threat CELM poses to FedGRAA's novelty claim — in FedGRAA's favor:

- **The "simplex + momentum smoothing" stability fix is still directly citable precedent** for FedGRAA-v2's missing projection (§7 of the master doc) — that part of the master doc's read is correct and should stay.
- **But CELM is not a gradient-refined method**, and should not be described as occupying FedGRAA's *mechanism* — only its *objective* (data-free, minority-class-aware weighting). The accurate differentiation sentence is: *CELM pursues the same objective (upweight informative minority-class clients, without server data) via activation-maximization probing of frozen post-warm-up scores; FedGRAA pursues it via a per-round, gradient-informed, γ-tunable interpolation.* "Gradient-refined" vs. "logit-probed and frozen" is a real, defensible, mechanism-level distinction — arguably a **cleaner** differentiation sentence than the master doc's current one, which risks reading as if CELM and FedGRAA differ only by domain (vision vs. IDS).
- CELM's own **Related Work section** (paper §2) cites **CGSV** (NeurIPS 2021, gradient-cosine-similarity-to-average) and **ShapFed** (IJCAI 2024, class-wise Shapley-style gradient alignment) as its closest data-free comparators — **not** PACT, FedPRS, or CoAst, which the master doc's §4.10/§10 groups into the same "data-free contribution estimation" research line as CELM. This suggests that line may be **two looser sub-clusters** (Shapley/gradient-alignment-descended methods vs. validation-avoidance methods) rather than one tight lineage — worth softening the master doc's framing of it as a single "named research line" (§10.1).
- **CELM shares FedGRAA's exact open gaps.** Its own Conclusion & Limitations section (paper §7, quoted verbatim): *"Future work can therefore explore calibration-aware probing, adaptive warm-up schedules, and theoretical guarantees on contribution identifiability under extreme heterogeneity and adversarial participation."* — i.e., **no convergence theorem and no adversarial-robustness evaluation**, the same two gaps master-doc §9 items #1 and #3 name for FedGRAA. This is worth citing explicitly: it is not just FedGRAA that lacks these; the closest 2026 paper in the objective-adjacent space lacks them too, which is a fairer, less isolated framing of FedGRAA's own gaps.

### 3.3 Datasets and results (for completeness)

FashionMNIST, CIFAR-10, FedISIC (medical, natural non-IID split), plus EMNIST (47-class) and a 20-client scale test in the appendix — confirmed no NIDS/network-security dataset anywhere in the paper. Reported gains are largest under severe skew (e.g. FedISIC balanced accuracy 70.18 vs. next-best 62.31) and on rare-class/"Maverick"-holding clients specifically (rare-class accuracy 90.77 vs. FedAvg's 81.76 on FashionMNIST) — the master doc's read that CELM pursues FedGRAA's own minority-recall objective is correct and unaffected by the mechanism correction above.

### 3.4 Net effect

**Closes the CELM half of the 2026-07-25 blocking item.** The differentiation paragraph in master-doc §8.1 can now be written with primary-source confidence for CELM (not secondary-summary confidence) — recommend tightening the CELM clause per §7 below.

---

## 4. FastTrack Gap Saturation Check — quantitative, cross-index confirmation

Ran `check_gap_saturation` on the narrowed novelty claim (simplified per the tool's own retrieval guidance after an over-specified first attempt returned zero rows — noted as a real caveat: an empty saturation result can be a query-specificity artefact, not evidence of an open gap, and the tool's own documentation says so explicitly). Simplified query: *"gradient-based adaptive aggregation federated intrusion detection minority class"*.

| Year | Count |
|---|---|
| 2022 | 1 |
| 2024 | 1 |
| 2025 | 7 |
| 2026 | 4 (partial year) |

**Reading (mine, not the tool's — FastTrack's own output explicitly disclaims that the researcher must interpret the curve):** a roughly flat 2022–2024 baseline followed by a **7× jump in 2025 and continuing at pace into 2026** is a publication curve consistent with a gap that is actively closing, not one that is stable and open. This is the same conclusion the 2026-07-25 pass reached by close-reading individual papers (FedPGO, CELM, FedAWARE, AdaAggRL all landing in 2025–2026), but this pass reaches it independently, from OpenAlex/Semantic Scholar's index rather than arXiv/DBLP, and from a raw count rather than a narrative read — a useful triangulation. None of the `closest_recent` items returned by the tool (a healthcare-IDS federated framework, an IoT homomorphic-encryption FL paper, a blockchain-IIoT paper, two more general adaptive-aggregation-for-FL-IDS papers) is closer to FedGRAA than FedPGO already established — **no new occupant of the intersection found**, which is itself informative: it means the 2026-07-25 saturation finding survives a second, independently-indexed saturation attempt.

`map_topic_debate` on the same topic, run separately, returned **zero shared `common_ancestors`** distinct from generic FL-IDS survey/foundational literature already in the master doc (Kairouz et al., the PPFL CSUR survey, a 2022 FL-IDS concepts survey) — i.e., no evidence of a citation cluster this document hasn't already mapped.

---

## 5. Two new adjacent (non-occupying) papers, not in the master doc

Both surfaced via `map_topic_debate`'s `recent_debate` list and independently confirmed via `search_papers`. Neither uses gradient-refinement or a γ-style interpolation; both are cited here as crowding evidence and optional Related-Work adds, not as competitors requiring the FedPGO-level urgency.

- **HADA-FL** — Chandu, Thumula, Parag et al., *"Federated Learning for Distributed IoT Security: A Privacy-Preserving Approach to Intrusion Detection,"* **IEEE Access, 2025**, DOI `10.1109/access.2025.3592481`. ✅ confirmed via OpenAlex. Mechanism: "Hybrid Adaptive-Weight Aggregation" combining SHAP-based feature stability with per-device differential-privacy budgets. **17 citations already** (11 already accrued in partial-year 2026) — unusually fast uptake for a 2025 paper in this sub-field, worth noting as a citation-trajectory watch item per the master doc's own Impact Forecasting framing. Not gradient-based; adjacent, not occupying.
- **TACMA Fed** — Aljuaid, *"Trust-Aware Contrastive Meta-Aggregation Federated Learning for Intrusion Detection in the Internet of Things,"* **Symmetry, 2026**, DOI `10.3390/sym18071188`. ✅ confirmed via OpenAlex. Explicitly **extends AMAFed** (already in the master doc, §9 item 2) and adds "**rare-class-aware weighting**" to a trust-scoring + contrastive-representation pipeline; evaluated on ToN-IoT. Worth a one-line addition to §5's competitor table as a second AMAFed-family paper.
- **Possible venue upgrade, unconfirmed — flagged, not asserted:** the topic-debate scan also returned *"Sentinel: Dynamic Knowledge Distillation for Personalized Federated Intrusion Detection in Heterogeneous IoT Networks,"* **IEEE Internet of Things Journal, 2026**, DOI `10.1109/jiot.2026.3650848`, authors Gurpreet Singh, Keshav Sood, P. Rajalakshmi. The master doc already lists **"Sentinel — arXiv:2510.23019, 2025 — Personalized federated IDS, dual-model knowledge distillation."** The core idea (dual-model teacher/student, personalized federated IDS, knowledge distillation) matches closely enough to suspect this is the same paper's journal publication — but **the author lists were not cross-checked against the arXiv preprint in this pass**, so this is recorded as an open lead, not a confirmed correction. Recommend a two-minute check (`get_paper` on the arXiv ID, compare author lists) before merging into the master doc.

---

## 6. XAI-SOH-FL's "manually tuned γ" — a sharpened, near-verbatim reviewer risk

Full abstract retrieved this pass (previously only described secondhand). Verbatim: *"existing approaches such as SOH-FL suffer from two key limitations: reliance on a manually tuned aggregation parameter **γ**, and lack of explainability... we introduce a dynamic γ selection mechanism based on similarity thresholding... Bayesian Optimization is employed to automatically determine optimal γ values, eliminating the need for manual tuning."*

The master doc already lists XAI-SOH-FL (§4.12) but describes it only as "similarity-thresholded dynamic aggregation... replaces SOH-FL's manually-tuned coefficient" — accurate, but it does not flag the specific symbol collision. **FedGRAA's own γ ∈ [0,1] is also manually set** (a fixed hyperparameter interpolating minority-boosting vs. size-proportional weighting; master-doc §2). A reviewer who has read XAI-SOH-FL — a paper in the same sub-field, same base dataset (CICIDS2017), published two months before a FedGRAA submission would land — has a ready-made, near-verbatim question: *"the closest related paper found manual γ tuning worth automating; why does FedGRAA not?"* This is cheap to pre-empt (a sentence in Limitations or Future Work acknowledging the manual-γ choice and citing XAI-SOH-FL as the precedent for automating it) and expensive to be caught flat-footed by. **Recommend adding this as an explicit item to master-doc §9's checklist**, not just a bibliography entry.

---

## 7. Proposed edits to the master doc

None applied — this is an addendum only, per this project's established convention.

1. **§1 / §8.1** — tighten the CELM differentiation clause using §3.2's mechanism correction: CELM is activation-maximization-probed and **frozen after warm-up**, not gradient-informed or per-round-adaptive. This is a stronger, more precise differentiation than the current text, not a weaker one.
2. **§4.7 / bibliography** — add the SSRN preprint predecessor of FedPGO (`10.2139/ssrn.5637328`) as a cross-reference; note it is also inaccessible (403), so it does not currently help close the blocking read.
3. **§9 item 0 (blocking)** — mark **CELM as read in full (this pass, 2026-07-26)**; **FedPGO remains open**, now confirmed blocked via three independent channels (Elsevier, FastTrack/OpenAlex, SSRN) — reframe from "not yet attempted" to "access-constrained, escalate to institutional library or direct author contact."
4. **§9 / §5** — add HADA-FL and TACMA Fed to the competitor/crowding table (§5 of this addendum); flag the possible Sentinel venue-upgrade as an open two-minute check.
5. **§9 checklist** — add a new small item: pre-empt the "why is γ manual, not tuned" question in Limitations/Future Work, citing XAI-SOH-FL as the direct same-domain precedent (§6 of this addendum).
6. **§10.1** — soften "named research line" for data-free contribution estimation to two related-but-distinct sub-clusters (Shapley/gradient-alignment-descended: CGSV, ShapFed, CELM; vs. validation-avoidance: PACT, FedPRS, CoAst), per CELM's own Related Work section not citing the latter group.

---

## 8. Verdict: is Stage C literature research complete enough for Related Work?

**Yes, with one named, access-blocked exception.** This pass used a structurally different literature index (OpenAlex/Semantic Scholar via FastTrack) and a different verification methodology (Duplication Test, Gap Saturation Check, Topic Debate Map — nearest-neighbour and citation-graph tools, not query-and-read) than every prior pass, specifically to catch what an arXiv/DBLP/web-search-only methodology might structurally miss. It found:

- No paper closer to FedGRAA than FedPGO (cross-index confirmation of the 2026-07-25 finding).
- Quantitative confirmation that the space is closing fast (§4).
- Two adjacent, non-threatening additions (§5) and one sharpened, cheap-to-fix reviewer risk (§6).
- Full-text resolution of CELM, with a correction that **helps** FedGRAA's positioning rather than hurting it (§3).
- Confirmation, not resolution, that FedPGO is genuinely hard to obtain — worth one direct email to the authors rather than further search-tool effort, which has now been run to the point of diminishing returns across two independent index/tool stacks.

**Recommend:** proceed to drafting Related Work using CELM at full-source confidence and FedPGO at the existing abstract/DBLP-metadata confidence (sufficient for existence, venue, mechanism-in-outline, and topic; insufficient for characterizing FedPGO's own limitations). Attempt the FedPGO full-text read opportunistically (library access or author email) in parallel with drafting rather than gating drafting on it — further *search* effort on this specific paper has passed the point of returns, but the paper itself is worth one non-search-based attempt (a direct request) before the manuscript is finalized.

---

*Compiled 2026-07-26 via ARS `deep-research` (fact-check mode) + FastTrack Literature Open MCP tools (`search_papers`, `get_paper`, `run_duplication_test`, `check_gap_saturation`, `map_topic_debate`, `get_journal_profile`) + direct WebFetch/PDF retrieval. Companion to [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md), [`STAGE_C_LITERATURE_REAUDIT_2026-07-25.md`](STAGE_C_LITERATURE_REAUDIT_2026-07-25.md), and [`STAGE_C_DEEP_RESEARCH_2026-07-25.md`](STAGE_C_DEEP_RESEARCH_2026-07-25.md).*
