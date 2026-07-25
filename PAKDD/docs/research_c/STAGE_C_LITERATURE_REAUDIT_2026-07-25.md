# Stage C Literature Re-Audit — 2026-07-25

> **What this is.** A targeted re-verification pass on four open threads left in [`STAGE_C_LITERATURE_MASTER.md`](STAGE_C_LITERATURE_MASTER.md) (compiled 2026-07-24), run via the ARS `deep-research` skill's fact-check mode. This does not modify the master doc — it is an addendum. Escalation path used throughout, per the master doc's own standard: general web search → arXiv API (`export.arxiv.org/api/query`) → DBLP direct query (`dblp.org/search/publ/api`).
>
> **Headline result (revised after a second verification pass, same day):** "Fed-LAA" is real — an initial literal-string search (arXiv title search + DBLP query for the exact token "Fed-LAA") returned zero hits and looked like a hallucination, matching the CT-BRFL/FedSND failure signature. A broader, descriptive-phrase DBLP query ("layer-wise adaptive weight aggregation federated") then surfaced the actual paper: Khan, Rizwan, Ahmad, Khan, Li, Kim, *"Accelerating model convergence in federated learning with layer-wise adaptive weight aggregation,"* *Applied Soft Computing* 201:115676 (2026), DOI 10.1016/J.ASOC.2026.115676. **Lesson applied mid-pass: a failed literal-acronym search is not sufficient grounds to declare a hallucination — the escalation path needs a descriptive-phrase retry before concluding non-existence, the same way it needed a DBLP retry before concluding non-existence.** The one caveat that survives: the acronym "Fed-LAA" itself is not confirmed as the authors' own (the abstract is paywalled and could not be read directly) — cite by title/DOI, not by the acronym, per the master doc's own precedent for the "FedAgile" placeholder. **The genuine hallucination-suspect finding from this pass is a different item: pFedCross** (see §4) — it failed the same broadened-query test that rescued Fed-LAA, and is very likely a conflation with the real, general-FL, non-IDS paper **FedCross** (arXiv:2210.08285).

---

## 1. Summary table

| Thread | Verdict | Action needed |
|---|---|---|
| FedVG — real, no IDS follow-on | ✅ Confirmed, unchanged | None |
| **Fed-LAA** — real paper, wrong acronym confidence | ✅⚠️ Confirmed via broadened DBLP query; acronym itself unconfirmed | **Done** — master doc updated with full citation (Khan et al., *Applied Soft Computing* 2026, DOI 10.1016/J.ASOC.2026.115676) and an acronym caveat |
| XAI-SOH-FL, FedDrone-Shield, DS-FedIDS, CACFL — no threatening follow-ons found | ✅ Confirmed, unchanged | None (DS-FedIDS DOI already present in master doc) |
| Saturation check — **pFedCross** surfaced repeatedly | ❌ Hallucination-suspect, survives broadened query test as non-existent | **Done** — added as a "do not cite" entry in master doc §4.10 with the FedCross-conflation hypothesis |
| Saturation check — EdgeDetect | ✅ Already in master doc (§4.10, approach E) | None — already cited with arXiv ID |
| CT-BRFL | ❌ Reconfirmed hallucination-equivalent (4th negative DBLP result) | **Done** — master doc §4.10 status tag updated from ❓ to ❌ |

---

## 2. Thread 1 — FedVG / Fed-LAA re-check

**FedVG**: confirmed real. arXiv:2602.21399 (Feb 2026), *"FedVG: Gradient-Guided Aggregation for Enhanced Federated Learning,"* Devkota, Thrasher, Adjeroh, Bhattarai, Gyawali (West Virginia University). Direct WebFetch of the abstract page confirms: validation-gradient-norm-guided aggregation, evaluated only on CV benchmarks (natural + medical images). **No mention of intrusion detection, network security, or cybersecurity anywhere in the paper.** No IDS-specific follow-on citations found — expected, given the paper is 5 months old.

**Fed-LAA**: initially looked like a hallucination, then confirmed real on a second attempt.
- Literal-string search failed on every tool: general web search returned a fluent description ("Federated Layer-wise Adaptive Aggregation," gradient angular similarity, per-layer weighting) with every underlying link pointing to a *different*, differently-named paper (FedALA, FedLPA, FedAA, FedAH); arXiv API title search (`ti:"Fed-LAA"`) → 0 results; DBLP direct query for the literal token `Fed-LAA` → 64 fuzzy-tokenized hits, all irrelevant.
- This matched the failure signature already used to declare FedSND/FeCWGAN-GP/CT-BRFL non-existent — **but a broadened, descriptive-phrase DBLP query** (`layer-wise adaptive weight aggregation federated`) immediately surfaced the real paper: **Khan, Rizwan, Ahmad, Khan, Li, Kim**, *"Accelerating model convergence in federated learning with layer-wise adaptive weight aggregation,"* *Applied Soft Computing* 201:115676 (2026), **DOI 10.1016/J.ASOC.2026.115676**. Mechanism (layer-wise gradient-alignment weighting + data-volume weighting to prevent small-node domination) matches the master doc's prior description exactly.
- **What remains unconfirmed**: whether the authors themselves use the acronym "Fed-LAA" — the ScienceDirect abstract page returned HTTP 403 (paywalled) both times it was fetched directly, so this could not be checked. **Recommendation: keep the citation, cite by title/DOI, and treat "Fed-LAA" as this document's own shorthand rather than a confirmed author-acronym** — the same convention already used elsewhere in the master doc for the "FedAgile" placeholder (§4.7).
- **Process lesson for future passes**: a failed literal-acronym search alone is not sufficient grounds to declare a hallucination. The escalation path in the master doc (web → arXiv → DBLP) should have an explicit sub-step — *retry DBLP with a descriptive phrase before concluding non-existence* — since acronyms are frequently the search tool's own invention, not the paper's.

## 3. Thread 2 — Follow-on citations for the four newly-surfaced 2026 papers

- **XAI-SOH-FL** (arXiv:2606.00134, submitted 28 May 2026) — confirmed real, authors Aslam, Hassan, Zahra, Shahzad. Too recent (< 2 months old) for any follow-on citations to have accumulated; none found. No change to its assessed relation to FedGRAA.
- **FedDrone-Shield** (*Scientific Reports*, 2026) — not independently re-searched this pass beyond the original confirmation; no contradicting information surfaced.
- **DS-FedIDS** — confirmed real with a concrete identifier this pass that the master doc did not previously have: **DOI 10.3390/app15095067**, *"Enhancing Federated Intrusion Detection with Class-Specific Dynamic Sampling"* (MDPI *Applied Sciences*, DOAJ-indexed). Mechanism confirmed consistent with master doc's description (FedPer + dynamic minority up-/majority down-sampling). **Suggest adding this DOI to the master doc's citation for DS-FedIDS** — it currently lacks one.
- **CACFL** — confirmed matches Springer *Neural Computing and Applications* (2026), DOI 10.1007/s00521-026-11845-2, *"Multi-modal class-aware curriculum federated learning fusion for advanced intrusion detection using conditional generative augmentation and real-time neural optimization."* Consistent with master doc. No new follow-on citations found (too recent).

None of the four individually threaten FedGRAA's novelty claim more than the master doc already assesses — all address minority/imbalance via sampling, augmentation, or curriculum scheduling, not via the specific gradient-refined adaptive-weight mechanism FedGRAA and its true competitor set (FedAPA/FedAWA/FedNolowe/Jiang et al.) use.

## 4. Thread 3 — Saturation check (trying to break the "unoccupied intersection" claim)

Searched directly for the triple intersection: adaptive/gradient-based aggregation weighting + network IDS + non-IID/minority-class framing.

**New, verified, real paper worth adding to related work:**
- **EdgeDetect** (arXiv:2604.14663, April 2026), *"Importance-Aware Gradient Compression with Homomorphic Aggregation for Federated Intrusion Detection."* Confirmed via arXiv abstract + HTML + Hugging Face paper page. Mechanism: median-based gradient binarization (32× compression) + Paillier homomorphic encryption + PCA + "imbalance-aware sampling," evaluated on CIC-IDS-2017 (98.0% accuracy, 97.9% macro-F1). This touches "gradient" + "imbalance-aware" + IDS, but its core contribution is communication/privacy efficiency (compression + encryption), not adaptive minority-boosting weight computation. **Recommend adding to the related-work catalog** (§4 of the master doc, alongside the secure-aggregation cluster — EIFFeL/Armadillo/FreqFed — not the adaptive-weighting cluster), but it does **not** threaten the novelty claim.

**Suspect finding — do not cite without further work:**
- **pFedCross** — described consistently across three separate web searches as "personalized federated cross learning" with a "gradient approximation α-fairness algorithm" for class-imbalanced federated IDS. This description, if true, would be a much closer mechanistic analogue to FedGRAA than most items currently in the master doc's competitor table — which is exactly why it needed hard verification, and why the Fed-LAA lesson (broaden the query before concluding non-existence) was applied here too. **It still failed after broadening**: arXiv title search → 0 results; a literal DBLP query → 0 results; a second, descriptive-phrase DBLP query (`personalized federated cross intrusion detection`) → 500 error on retry, but the equivalent broadened web search still produced **zero links actually titled "pFedCross"** — every result pointed to other, real, differently-named papers (CoperFed, P3Fed, hypernetwork-based PFL-IDS). One of those adjacent results is the likely source of the conflation: **FedCross** — *"Towards Accurate Federated Learning via Multi-Model Cross-Aggregation"* (arXiv:2210.08285; also an IEEE conference publication) — a real, general-purpose (non-IDS) FL paper whose actual mechanism, "multi-model cross-aggregation," matches the "collaborative model cross aggregation algorithm" phrase used to describe pFedCross almost verbatim. **Working hypothesis: "pFedCross" is a search-tool blend of the real FedCross paper with an IDS/imbalance framing pulled from the same paywalled 2026 *Artificial Intelligence Review* survey (Springer DOI 10.1007/s10462-026-11519-4) that also describes DS-FedIDS and CACFL** — the survey itself could not be fetched directly (ResearchGate mirror returned HTTP 403) to check whether it actually contains a distinct "pFedCross" citation or whether that name is purely the search tool's invention. **Do not add pFedCross to the master doc or cite it in a manuscript** — this is the "gray zone = FAIL" rule, not a confirmed new competitor. (Now added to master doc §4.10 as an explicit "do not cite" entry so future passes don't re-chase it from scratch.)
- No other candidate surfaced in this pass combines all three criteria (adaptive/gradient weighting + IDS-specific + non-IID/minority framing) closely enough to threaten the master doc's central novelty claim. **The claim in the master doc's executive summary — that this exact three-way intersection remains unoccupied at A*/A venues — survives this saturation attempt**, with the pFedCross caveat above as the one open thread.

## 5. Thread 4 — CT-BRFL, one more pass

Re-ran the DBLP direct query (`q=CT-BRFL`) fresh this pass: **0 results, explicitly** ("Total hits: 0"). This is the same result the master doc already recorded, now independently reproduced. Separately, a general web search this pass again produced a fluent, confident description of CT-BRFL ("100% detection accuracy... 25% accuracy improvement") — **with zero supporting links actually titled CT-BRFL** in the returned result list (the links were for BRFL, BR-CPPFL, and FLTrust, all real but distinct papers). This is a useful, concrete illustration of exactly the failure mode the master doc's evidentiary standard is designed to catch: a summarizing search tool will confidently describe a paper that does not exist by blending adjacent real results. **No change to the master doc's verdict** — CT-BRFL remains hallucination-equivalent, now with a fourth independent negative result. Recommend applying the same "no link, no citation" test used here to Fed-LAA and pFedCross going forward.

## 6. Edits applied to `STAGE_C_LITERATURE_MASTER.md` (2026-07-25)

All five actions below have been applied directly to the master doc:

1. **§1 Executive summary** — updated the Fed-LAA mention with its full confirmed citation (Khan et al., DOI 10.1016/J.ASOC.2026.115676) and a pointer to this re-audit doc, rather than removing it.
2. **§4.7** — replaced the bare "ScienceDirect journal, 2026" row for Fed-LAA with the full author/venue/DOI citation plus an explicit caveat that the acronym itself is unconfirmed.
3. **§4.10 approach F** — added **pFedCross** as a new "do not cite" row with the FedCross-conflation hypothesis, so a future pass doesn't re-run this same search from scratch.
4. **§4.10 approach F** — updated CT-BRFL's status tag from ❓ to ❌, reflecting the 4th independent negative DBLP result.
5. DS-FedIDS (already had its DOI) and EdgeDetect (already cited in §4.10) needed no changes — both were already correctly present.

**Not applied**: nothing was removed from the master doc. The only correction that survived scrutiny was a citation *upgrade* (Fed-LAA), not a deletion — a useful reminder that this pass's own first-draft conclusion (treat Fed-LAA as a hallucination) was itself wrong until re-checked with a broadened query, which is now recorded as a process lesson in §2 above.

---

*Compiled 2026-07-25 via ARS `deep-research` (fact-check mode). Search tools used: general web search, arXiv API, DBLP publication-search API. Semantic Scholar API was attempted but returned HTTP 429 (rate-limited) and should be retried in a future pass, particularly for citation-count verification of FedVG, XAI-SOH-FL, DS-FedIDS, and CACFL.*
