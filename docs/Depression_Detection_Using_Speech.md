# Depression Detection Using Speech — Exhaustive Research Dossier & Implementation Plan

> **Status:** Living document — the single source of truth for the project.
> **Scope:** Everything needed to go from zero to a defensible, reproducible, ethically‑sound research system that detects indicators of depression from speech.
> **Audience:** The project owner (you), collaborators, and reviewers.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Definition & Clinical Foundations](#2-problem-definition--clinical-foundations)
3. [Why Speech? The Scientific Rationale](#3-why-speech-the-scientific-rationale)
4. [Task Formulations](#4-task-formulations)
5. [Datasets (Exhaustive Catalogue)](#5-datasets-exhaustive-catalogue)
6. [Data Access, Ethics, Governance & Legal](#6-data-access-ethics-governance--legal)
7. [Speech Signal Processing & Feature Engineering](#7-speech-signal-processing--feature-engineering)
8. [Methodologies: Classical ML → Deep Learning → Foundation Models](#8-methodologies-classical-ml--deep-learning--foundation-models)
9. [Multimodal & Linguistic Approaches](#9-multimodal--linguistic-approaches)
10. [Evaluation Protocols & Metrics](#10-evaluation-protocols--metrics)
11. [Known Pitfalls, Biases & Threats to Validity](#11-known-pitfalls-biases--threats-to-validity)
12. [Reproducibility & Engineering Best Practices](#12-reproducibility--engineering-best-practices)
13. [Proposed System Architecture](#13-proposed-system-architecture)
14. [Phased Implementation Roadmap](#14-phased-implementation-roadmap)
15. [Baseline Recipes (Concrete, Runnable)](#15-baseline-recipes-concrete-runnable)
16. [Risk Register](#16-risk-register)
17. [Glossary](#17-glossary)
18. [Annotated Bibliography & Key References](#18-annotated-bibliography--key-references)
19. [Appendices](#19-appendices)

---

## 1. Executive Summary

**Goal.** Build a machine‑learning system that analyzes speech signals and identifies acoustic/linguistic indicators associated with depression, validated against clinically accepted ground truth (e.g., PHQ‑8/PHQ‑9, BDI‑II, HAM‑D).

**The honest framing.** This is a *screening / decision‑support* problem, **not** a diagnostic one. Depression is diagnosed by clinicians (DSM‑5 / ICD‑11). Our system estimates **risk or symptom severity** from voice and must always be positioned as an assistive tool. Overclaiming is the single biggest reputational and ethical risk.

**What the literature converges on:**
- **Best‑studied corpus:** DAIC‑WOZ / E‑DAIC (AVEC 2016–2019 challenges), labeled with PHQ‑8.
- **Strong acoustic feature families:** prosody (F0, energy, speaking rate, pauses), voice quality (jitter, shimmer, HNR), spectral (MFCC, formants), and standardized sets (eGeMAPS via openSMILE, COVAREP).
- **Strong modern approaches:** self‑supervised speech embeddings (wav2vec 2.0, HuBERT, WavLM), ECAPA‑TDNN, CNN/CRNN on log‑mel spectrograms, and multimodal fusion (audio + text + video).
- **Recurring methodological problems:** tiny datasets, speaker leakage, label imbalance, overlap with general emotion/arousal, optimistic reporting, and weak generalization across corpora and languages.

**Our differentiator (how we "leave no stone unturned"):**
1. A rigorous, leakage‑free, speaker‑independent evaluation harness from day one.
2. A reproducible pipeline with pinned dependencies, config‑driven experiments, and tests.
3. Transparent baselines (interpretable acoustic features + linear models) **before** deep models.
4. Explicit fairness, calibration, and clinical‑safety analysis.
5. Documented negative results and ablations, not just the winning number.

---

## 2. Problem Definition & Clinical Foundations

### 2.1 What depression is (operationally)
Major Depressive Disorder (MDD) is a mood disorder characterized by persistent low mood, anhedonia (loss of pleasure), and a cluster of cognitive/somatic symptoms (sleep, appetite, energy, concentration, psychomotor changes, worthlessness, suicidality) lasting ≥ 2 weeks (DSM‑5 criteria).

### 2.2 Ground‑truth instruments you will encounter
| Instrument | Type | Range | Notes |
|---|---|---|---|
| **PHQ‑8 / PHQ‑9** | Self‑report questionnaire | 0–24 / 0–27 | Standard in DAIC‑WOZ/E‑DAIC. PHQ‑9 includes a suicidality item. Common binary threshold: **≥ 10** = clinically significant. |
| **BDI / BDI‑II** | Self‑report | 0–63 | Beck Depression Inventory; used in several corpora (AVEC 2013/2014). |
| **HAM‑D (HDRS)** | Clinician‑rated | 0–52 | Hamilton Depression Rating Scale; clinician interview. |
| **MADRS** | Clinician‑rated | 0–60 | Montgomery–Åsberg; used in MODMA and clinical studies. |
| **QIDS** | Self/clinician | 0–27 | Quick Inventory of Depressive Symptomatology. |

> **Modeling implication.** "Depression detection" usually means either (a) **binary classification** at a questionnaire threshold, or (b) **regression** onto a severity score. Decide and document this early; they require different losses, metrics, and label handling.

### 2.3 Why this matters for ML
- The labels are **noisy and subjective** (self‑report bias, day‑to‑day variability, comorbidity with anxiety/PTSD).
- Thresholding a continuous score into binary classes throws away information and creates borderline‑case noise.
- The same speaker contributes many utterances → **speaker identity can leak** and inflate scores if you split naively.

---

## 3. Why Speech? The Scientific Rationale

Depression produces measurable changes in motor control, cognition, and affect that manifest in the voice. This is sometimes called **"psychomotor retardation"** in speech.

Commonly reported acoustic correlates of depression:
- **Prosody:** reduced pitch (F0) range and variability ("monotone"), reduced loudness/energy, **slower speaking rate**, **longer and more frequent pauses**, increased response latency.
- **Voice quality / source features:** increased jitter & shimmer, lower harmonics‑to‑noise ratio (HNR), breathy/tense phonation, changes in glottal flow.
- **Spectral / articulatory:** flatter formant trajectories, reduced articulatory precision and coordination (captured by formant‑based correlation features), shifts in MFCC distributions.
- **Timing/turn‑taking:** in dialogue, slower turn‑taking, reduced engagement.

**Caveat (the crucial nuance):** many of these overlap with **emotional arousal/valence, fatigue, sedation, age, gender, language, recording channel, and health conditions** (e.g., Parkinson's, COPD). A robust system must control for or be tested against these confounds — otherwise you build a "tiredness detector" or a "microphone detector," not a depression model.

---

## 4. Task Formulations

| Formulation | Output | Typical Loss | Primary Metric |
|---|---|---|---|
| Binary classification | depressed / not (PHQ‑8 ≥ 10) | BCE / focal | F1 (macro), AUROC, AUPRC, sensitivity@specificity |
| Severity regression | PHQ‑8 / BDI score | MSE / MAE / CCC | RMSE, MAE, **CCC**, Pearson r |
| Ordinal / multi‑class | none/mild/moderate/severe | ordinal CE | QWK, macro‑F1 |
| Symptom‑level multilabel | per‑symptom presence | BCE per head | macro‑F1 per symptom |
| Risk ranking | continuous risk score | pairwise/ranking | AUROC, calibration (ECE) |

> AVEC challenges historically used **RMSE** and **CCC** (Concordance Correlation Coefficient) for severity; CCC is preferred because it penalizes both correlation error and bias.

---

## 5. Datasets (Exhaustive Catalogue)

> ⚠️ Almost all clinical speech datasets require a signed **End User License Agreement (EULA)**, an academic/non‑profit affiliation, and sometimes IRB/ethics approval. Budget **weeks** for access. Start the applications immediately (see §6).

### 5.1 Primary, speech‑centric, depression‑labeled corpora

| Dataset | Modality | Label | Size | Language | Access | Notes |
|---|---|---|---|---|---|---|
| **DAIC‑WOZ** (Distress Analysis Interview Corpus – Wizard of Oz) | Audio, transcripts, facial/landmark features (no raw video) | PHQ‑8 | 189 sessions, ~7–33 min each | English (US) | Application form, academic only — USC ICT | **The canonical benchmark.** Virtual interviewer "Ellie." Used in AVEC 2016/2017. Includes pre‑extracted COVAREP & formant features. |
| **E‑DAIC (Extended DAIC)** | Audio, features, transcripts | PHQ‑8 (+ PTSD) | 275 sessions | English | USC ICT application | Used in **AVEC 2019**. Adds semi‑automated/agent interviews. Standard train/dev/test split provided. |
| **AVEC 2013 / 2014 (AViD‑Corpus)** | Audio + video | BDI‑II | 150 / 300 recordings | German | Challenge organizers | Read speech + free response tasks. Older but widely cited. |
| **EATD‑Corpus** (Emotional Audio‑Textual Depression) | Audio + text | SDS (Self‑rating Depression Scale) | 162 volunteers | Mandarin | Public (GitHub, on request) | Student volunteers; positive/neutral/negative prompts. One of few openly accessible audio sets. |
| **MODMA** (Multi‑modal Open Dataset for Mental‑disorder Analysis) | Audio + EEG | MDD clinical dx, PHQ‑9 | ~52 audio subjects | Mandarin | Lanzhou University application | Includes resting EEG + speech tasks. |
| **Androids Corpus** | Audio | Clinical diagnosis | ~112 speakers | Italian | Public/on request | Interview + reading tasks; designed for depression detection benchmarking. |
| **CONVERGE** | Audio | MDD (clinical) | Large (thousands), women | Mandarin | Restricted/clinical collaboration | Recurrent MDD in Han Chinese women; used by Alwan group for speaker‑disentanglement work. |
| **DepAudioNet / Pittsburgh datasets** | Audio | HAM‑D | Varies | English | Restricted | Clinical recordings; some referenced in literature, limited availability. |

### 5.2 Adjacent / proxy corpora (useful for pretraining, augmentation, transfer, confound control)

| Dataset | Why it's useful |
|---|---|
| **IEMOCAP, MSP‑IMPROV, MSP‑Podcast, RAVDESS, CREMA‑D** | Speech emotion recognition (SER). Pretrain/transfer; control for emotion vs. depression overlap. |
| **DementiaBank / ADReSS / ADReSSo** | Cognitive‑decline speech; similar paralinguistic pipeline; methodological templates. |
| **Mozilla Common Voice, LibriSpeech, VoxCeleb** | Large unlabeled/ASR speech for self‑supervised pretraining and speaker‑embedding backbones. |
| **DAIC (full)** | Parent corpus of DAIC‑WOZ. |
| **Social‑media text sets** (CLEF eRisk, Reddit, LT‑EDI‑ACL 2022 "Detecting Signs of Depression") | For text‑only baselines and linguistic methodology; **not speech** but complementary. |

### 5.3 Self‑collected data (optional, high effort)
If you collect your own:
- Use a validated instrument (PHQ‑9) administered properly.
- Standardize tasks: **sustained vowel** ("aaah"), **read passage** (e.g., a standard paragraph), **free response** (e.g., "describe your day"), **picture description**.
- Standardize hardware, room, sampling rate (≥ 16 kHz, ideally 44.1/48 kHz), mono, lossless (WAV/FLAC).
- Capture metadata: age, gender, native language, medications, time of day, device.
- **Mandatory:** IRB/ethics approval, informed consent, a crisis‑response protocol for participants who report suicidality.

---

## 6. Data Access, Ethics, Governance & Legal

### 6.1 Access workflow (do this first)
1. Identify the corpus (start with **DAIC‑WOZ + E‑DAIC**).
2. Confirm eligibility (academic/non‑profit; use an academic email).
3. Complete & sign the EULA; get your supervisor/PI to co‑sign if required.
4. Store data on **encrypted, access‑controlled** storage. Never commit data to git. Never upload to public clouds without permission.
5. Track who has access; honor deletion/retention clauses.

### 6.2 Ethics principles (non‑negotiable)
- **Beneficence & non‑maleficence:** a false "not depressed" can be dangerous; a false "depressed" can stigmatize. Design with clinicians.
- **Transparency:** document intended use, limitations, and failure modes (a **Model Card** + **Datasheet**).
- **Privacy:** voice is biometric PII. De‑identify, minimize retention, encrypt, and consider speaker‑anonymization research (e.g., speaker disentanglement) so models don't memorize identity.
- **Fairness:** measure performance across gender, age, language/accent. Report disparities.
- **Consent & autonomy:** never deploy on people without informed consent.
- **Safety net:** any data collection touching suicidality needs a clinical escalation plan.

### 6.3 Regulatory awareness
- Health‑adjacent AI may fall under **medical‑device** regulation (FDA SaMD, EU MDR) and **GDPR/HIPAA** for data. This project is **research‑grade**; do **not** present it as a clinical product without proper validation and approval.

---

## 7. Speech Signal Processing & Feature Engineering

### 7.1 Preprocessing pipeline
1. **Load & standardize:** resample (16 kHz common for SSL models), mono, peak/RMS normalize, convert to float32.
2. **Voice Activity Detection (VAD):** remove non‑speech (silence, interviewer turns). Tools: `webrtcvad`, `silero-vad`, pyannote.
3. **Speaker diarization (for interviews):** isolate the **participant's** speech from the interviewer's. DAIC‑WOZ provides transcripts with timestamps to help.
4. **Denoising / channel handling (optional):** be careful — aggressive denoising can remove voice‑quality cues that matter.
5. **Segmentation:** fixed windows (e.g., 3–10 s) or utterance‑level segments. Decide segment‑ vs. session‑level labeling and aggregation.

### 7.2 Hand‑crafted feature families
- **Prosodic:** F0 (mean, std, range, contour stats), energy/intensity, speaking rate, articulation rate, pause statistics (count, mean/total duration, ratio), voiced/unvoiced ratio, response latency.
- **Voice quality / source:** jitter (local, RAP, PPQ5), shimmer (local, APQ3/5/11), HNR, CPP (cepstral peak prominence), glottal parameters (via COVAREP/openGLOTTAL).
- **Spectral / cepstral:** MFCCs (+ Δ, ΔΔ), log‑mel spectrogram, spectral centroid/rolloff/flux, formants F1–F4 and bandwidths.
- **Articulatory coordination:** correlation‑structure features from formant/MFCC time series (eigen‑spacing of delay‑embedded correlation matrices) — shown effective for psychomotor changes.

### 7.3 Standardized feature sets (use these for credible baselines)
- **eGeMAPS / GeMAPS** (88 / 62 features) via **openSMILE** — the de‑facto interpretable acoustic standard for clinical/affective speech.
- **ComParE 2016** (6 373 features) via openSMILE — large brute‑force set (risk of overfitting on small data; use with strong regularization/feature selection).
- **COVAREP** — glottal & voice‑quality features (provided pre‑computed in DAIC‑WOZ).

### 7.4 Learned representations (modern default)
- **Self‑supervised embeddings:** `wav2vec 2.0`, `HuBERT`, `WavLM`, `data2vec`. Extract frame/utterance embeddings (mean/attentive pooling) and train a light head, or fine‑tune.
- **Paralinguistic encoders:** TRILL / TRILLsson, x‑vector / ECAPA‑TDNN speaker embeddings (transfer surprisingly well).
- **Spectrogram CNNs:** treat log‑mel as an image; CNN/CRNN/AST (Audio Spectrogram Transformer).

### 7.5 Practical tooling
| Need | Tools |
|---|---|
| Audio I/O & DSP | `librosa`, `torchaudio`, `soundfile`, `scipy` |
| Standard feature sets | `opensmile` (Python), Praat / `parselmouth`, COVAREP (MATLAB) |
| VAD / diarization | `silero-vad`, `webrtcvad`, `pyannote.audio` |
| SSL embeddings | `transformers`, `s3prl`, `speechbrain`, `fairseq` |
| Classical ML | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep learning | `pytorch`, `pytorch-lightning`, `speechbrain` |
| Experiment tracking | `mlflow`, `wandb`, `hydra` configs |

---

## 8. Methodologies: Classical ML → Deep Learning → Foundation Models

### 8.1 Tier 0 — Interpretable baselines (do these FIRST)
- eGeMAPS (openSMILE) → standardize → **Logistic Regression / Linear SVM / Random Forest / XGBoost**.
- Session‑level: aggregate frame features (functionals: mean, std, percentiles) before the classifier.
- Value: interpretable, fast, hard to beat on tiny data, and a sanity check against leakage (if a linear model on 5 features hits 0.95 AUC, suspect leakage).

### 8.2 Tier 1 — Classic deep learning
- **CNN on log‑mel spectrograms** (e.g., DepAudioNet‑style: CNN + LSTM).
- **CRNN / BiLSTM** over frame features to model temporal dynamics and pauses.
- **1D‑CNN on raw waveform** (SincNet front‑end).
- Handle class imbalance: weighted loss, focal loss, **random majority undersampling**, segment‑level sampling.

### 8.3 Tier 2 — Self‑supervised & transformer models (current SOTA direction)
- **wav2vec 2.0 / HuBERT / WavLM** frozen embeddings + attentive pooling + MLP head → strong, low‑data‑friendly.
- **Fine‑tuning** the SSL encoder (careful: small data → overfit; use layer‑wise LR decay, freezing schedules, strong augmentation).
- **ECAPA‑TDNN** backbone (originally speaker verification) adapted for depression — reported effective (Interspeech 2022).
- **Instance‑discriminative / contrastive pretraining** on in‑domain speech before the supervised head.

### 8.4 Tier 3 — Robustness & identity disentanglement (research frontier)
- **Speaker disentanglement:** remove speaker identity so the model learns depression cues, not "who is speaking" — improves generalization and privacy (Alwan group, Interspeech 2022).
- **Domain adaptation / cross‑corpus** training to fight overfitting to one corpus's channel.
- **Multi‑task learning:** jointly predict emotion/arousal + depression to regularize and disentangle.

### 8.5 Augmentation (critical on small data)
- SpecAugment (time/freq masking), additive noise (MUSAN), reverberation (RIRs), pitch/tempo perturbation (use cautiously — can destroy the very cues you study), speed perturbation, mixup at embedding level.

---

## 9. Multimodal & Linguistic Approaches

Speech‑only is the project's core, but the strongest published systems are **multimodal**. Document these even if you start audio‑only.

- **Text (from transcripts/ASR):** lexical/semantic cues (negative affect words, first‑person pronouns, absolutist words). Models: TF‑IDF + linear, BERT/RoBERTa, or LLM embeddings. The LT‑EDI‑ACL 2022 shared task is a strong text reference.
- **Audio + Text fusion:** early (feature concat), late (score fusion), or **cross‑modal attention** (e.g., wav2vec2 + BERT). Hierarchical attention over utterances + articulatory coordination features reported strong on E‑DAIC.
- **Audio + Video:** facial action units, head pose, gaze (DAIC‑WOZ provides facial features, not raw video). Improves accuracy but adds privacy burden.
- **Conversation‑level / turn‑taking:** dyadic interaction features (latency, synchrony) from free‑living or interview audio.

> **Decision:** Phase 1–4 = **audio‑only** (the assignment's core). Phase 5 = optional **audio+text** fusion as a stretch goal, since transcripts ship with DAIC‑WOZ.

---

## 10. Evaluation Protocols & Metrics

### 10.1 Splitting — the make‑or‑break step
- **Speaker‑independent splits ALWAYS.** No speaker appears in more than one of train/dev/test. Violating this is the #1 cause of irreproducible, inflated results.
- Use the **official DAIC‑WOZ / E‑DAIC train/dev/test split** for comparability.
- For small data, use **speaker‑stratified k‑fold cross‑validation** (group k‑fold by speaker), report mean ± std.
- Stratify by label and, where possible, by gender to avoid skewed folds.

### 10.2 Metrics
- **Classification:** macro‑F1 (primary, robust to imbalance), AUROC, **AUPRC** (better under imbalance), sensitivity/recall (clinically important — missing a depressed person is costly), specificity, balanced accuracy, confusion matrix. Report **operating point** explicitly.
- **Regression:** **CCC** (primary, AVEC standard), RMSE, MAE, Pearson r.
- **Calibration:** ECE, reliability diagrams (a screening score must be trustworthy, not just rank‑correct).
- **Per‑group reporting:** metrics broken down by gender/age/language.
- **Statistical rigor:** confidence intervals (bootstrap), significance tests vs. baseline, and **never** tune on the test set.

### 10.3 Reporting template (per experiment)
```
Experiment: <id>
Data: <corpus, split, n_train/dev/test, label def, threshold>
Features/Model: <...>
Preprocessing: <VAD, diarization, segment length, normalization>
Result (dev): F1=..  AUROC=..  AUPRC=..  CCC=..  (CI)
Result (test): ...
Per-group: <gender/age breakdown>
Ablations: <what changed vs. baseline>
Notes / failure modes: <...>
```

---

## 11. Known Pitfalls, Biases & Threats to Validity

A checklist distilled from the literature's hard lessons (Berisha et al. warn that reported accuracies are often **over‑optimistic**):

1. **Speaker leakage** → inflated metrics. *Fix:* group splits by speaker.
2. **Channel/recording leakage** → model learns the microphone/room, not depression. *Fix:* cross‑corpus tests, channel normalization, document devices.
3. **Tiny test sets** → huge variance; a single subject flips the score. *Fix:* CV, CIs, avoid single‑split bragging.
4. **Label thresholding noise** → borderline PHQ‑8 ≈ 9–11 cases dominate errors. *Fix:* consider regression; analyze borderline separately.
5. **Class imbalance** → accuracy looks great, recall is terrible. *Fix:* AUPRC, macro‑F1, resampling, focal loss.
6. **Emotion/arousal confound** → "sad‑speech detector." *Fix:* SER controls, multi‑task disentanglement.
7. **Demographic confounds** (gender/age) → unfair, non‑generalizing models. *Fix:* per‑group metrics, balanced sampling.
8. **Comorbidity** (anxiety/PTSD overlap, esp. in DAIC). *Fix:* document, model jointly if labels exist.
9. **Test‑set tuning / multiple comparisons** → optimistic bias. *Fix:* frozen test set, pre‑registered protocol, dev‑only tuning.
10. **Data dredging on ComParE (6 k features)** → overfit. *Fix:* nested CV feature selection or prefer eGeMAPS.
11. **Train/test temporal or site shift.** *Fix:* report distribution shift, adapt.
12. **Overclaiming clinical utility.** *Fix:* "screening/decision support," model card limitations.

> Treat each item as a test or an experiment in the harness, not just a footnote.

---

## 12. Reproducibility & Engineering Best Practices

- **Config‑driven experiments** (Hydra/YAML); no magic numbers in code.
- **Deterministic seeds** + logged hardware; pin all dependency versions.
- **Data versioning** (DVC or hashed manifests) — but never commit the data.
- **Experiment tracking** (MLflow/W&B): params, metrics, artifacts.
- **Unit + integration tests** for the pipeline (feature shapes, no‑leakage splitter, metric correctness, deterministic preprocessing).
- **CI** that runs tests + lint on synthetic fixtures (no real data needed).
- **Model Card + Datasheet** for every released model.
- **Separation of concerns:** `data/ → features/ → models/ → eval/` with clean interfaces.

---

## 13. Proposed System Architecture

```
                         ┌────────────────────────────────────────────┐
                         │                 Raw Audio                   │
                         │        (DAIC-WOZ / E-DAIC sessions)         │
                         └───────────────────┬────────────────────────┘
                                             │
                        ┌────────────────────▼─────────────────────┐
                        │  Preprocessing                            │
                        │  resample → VAD → diarization (participant│
                        │  only) → segment → normalize              │
                        └────────────────────┬─────────────────────┘
                                             │
                ┌────────────────────────────┼────────────────────────────┐
                │                            │                            │
   ┌────────────▼───────────┐  ┌────────────▼───────────┐  ┌────────────▼───────────┐
   │ Tier-0 Features        │  │ Tier-1/2 Spectrogram   │  │ Tier-2 SSL Embeddings  │
   │ eGeMAPS / COVAREP      │  │ log-mel  → CNN/CRNN     │  │ wav2vec2/HuBERT/WavLM  │
   │ (openSMILE)            │  │                        │  │ → pooling              │
   └────────────┬───────────┘  └────────────┬───────────┘  └────────────┬───────────┘
                │                            │                            │
                └────────────────────────────┼────────────────────────────┘
                                             │
                        ┌────────────────────▼─────────────────────┐
                        │  Classifier / Regressor head              │
                        │  (LogReg/SVM/XGB  |  MLP  |  fine-tune)    │
                        └────────────────────┬─────────────────────┘
                                             │
                        ┌────────────────────▼─────────────────────┐
                        │  Segment → Session aggregation            │
                        │  (mean / attention / majority)            │
                        └────────────────────┬─────────────────────┘
                                             │
                        ┌────────────────────▼─────────────────────┐
                        │  Evaluation harness                       │
                        │  speaker-independent CV, F1/AUROC/CCC,    │
                        │  calibration, per-group fairness          │
                        └───────────────────────────────────────────┘
```

---

## 14. Phased Implementation Roadmap

> Each phase has **deliverables** and **exit criteria**. Don't advance until exit criteria are met.

### Phase 0 — Foundations & access (parallelizable)
- Submit DAIC‑WOZ + E‑DAIC EULAs. Set up encrypted storage.
- Stand up the repo: env, configs, tests, CI, tracking.
- **Exit:** reproducible env; tests pass on synthetic audio; data applications submitted.

### Phase 1 — Data understanding & EDA
- Parse labels (PHQ‑8), official splits, transcripts. Plot label distribution, session lengths, gender balance, missingness.
- Build the **leakage‑free, speaker‑grouped splitter** + tests.
- **Exit:** documented EDA notebook; verified splits with zero speaker overlap.

### Phase 2 — Interpretable baseline (Tier 0)
- openSMILE eGeMAPS → functionals → LogReg/SVM/XGB. Speaker‑independent CV.
- Report F1/AUROC/AUPRC (+ CCC if regression). Calibration + per‑group metrics.
- **Exit:** a credible, leakage‑free baseline number with CIs; ablations on feature groups.

### Phase 3 — Deep audio models (Tier 1)
- log‑mel CNN/CRNN (DepAudioNet‑style); imbalance handling; augmentation.
- **Exit:** matches/beats Tier 0 on dev with honest CV; documented training curves.

### Phase 4 — Self‑supervised models (Tier 2)
- wav2vec2/HuBERT/WavLM frozen embeddings + head; then careful fine‑tuning.
- **Exit:** best single‑model dev result; ablation: frozen vs. fine‑tuned, pooling choices.

### Phase 5 — Robustness, fairness, multimodal stretch
- Speaker disentanglement; cross‑corpus test (train DAIC → test EATD/Androids); fairness audit.
- Optional audio+text fusion using shipped transcripts.
- **Exit:** generalization + fairness report; final model card.

### Phase 6 — Packaging & write‑up
- Reproducible "run‑all" pipeline, model card, datasheet, final report/paper draft, demo (offline, with disclaimers).
- **Exit:** one command reproduces headline results from features; documentation complete.

---

## 15. Baseline Recipes (Concrete, Runnable)

### 15.1 Tier‑0 recipe (pseudocode)
```python
# 1) extract eGeMAPS (functionals) per session with openSMILE
import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)
X = smile.process_files(participant_wavs)        # one 88-dim row per session

# 2) speaker-independent split (group by participant id)
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)

# 3) standardize + linear model inside each fold (no leakage)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
clf = Pipeline([("scaler", StandardScaler()),
                ("lr", LogisticRegression(class_weight="balanced", max_iter=1000))])

# 4) report macro-F1, AUROC, AUPRC with CIs; calibrate; per-group breakdown
```

### 15.2 Tier‑2 recipe (pseudocode)
```python
import torch, torchaudio
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H  # or HF wav2vec2/HuBERT/WavLM
model = bundle.get_model().eval()

def embed(wav16k):
    with torch.inference_mode():
        feats, _ = model.extract_features(wav16k)   # list of layer activations
    # attentive/mean pool a chosen layer → utterance embedding
    return feats[-1].mean(dim=1)

# train an MLP head on pooled embeddings with speaker-grouped CV
```

> A runnable, tested scaffold implementing the **leakage‑free splitter**, **feature aggregation**, and **metrics** ships alongside this document in `src/` and `tests/` (see repository README).

---

## 16. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Data access delayed/denied | High | High | Apply early to multiple corpora (DAIC + EATD + Androids); use public sets as fallback. |
| Speaker/channel leakage inflates results | High | High | Group splits + cross‑corpus tests + linear sanity checks. |
| Overfitting on tiny data | High | High | Strong regularization, augmentation, frozen SSL, CV + CIs. |
| Confounds (emotion/age/gender/device) | High | Medium | Control experiments, per‑group metrics, disentanglement. |
| Ethical/privacy breach | Low | Very High | Encryption, EULA compliance, no data in git, model card. |
| Overclaiming clinical value | Medium | High | "Screening/decision support" framing; clinician review. |
| Irreproducible results | Medium | High | Config + seeds + tracking + tests + pinned deps. |

---

## 17. Glossary

- **PHQ‑8/9:** Patient Health Questionnaire depression scales (self‑report).
- **CCC:** Concordance Correlation Coefficient — agreement metric for regression.
- **eGeMAPS:** extended Geneva Minimalistic Acoustic Parameter Set (interpretable features).
- **openSMILE:** toolkit for extracting standardized acoustic features.
- **COVAREP:** voice‑quality/glottal feature toolkit (shipped with DAIC‑WOZ).
- **SSL:** self‑supervised learning (wav2vec2, HuBERT, WavLM).
- **VAD:** voice activity detection.
- **ECAPA‑TDNN:** strong speaker‑embedding architecture, transferable to paralinguistics.
- **Jitter/Shimmer/HNR:** cycle‑to‑cycle frequency/amplitude perturbation and harmonics‑to‑noise ratio (voice‑quality measures).
- **Psychomotor retardation:** slowing of motor/cognitive function reflected in slowed, monotone speech.

---

## 18. Annotated Bibliography & Key References

> Verify and pull PDFs as you go; this is a curated starting set spanning corpora, challenges, features, and modern methods.

**Corpora & challenges**
- Gratch et al. (2014). *The Distress Analysis Interview Corpus of Human and Computer Interviews* (DAIC), LREC.
- Valstar et al. (2016). *AVEC 2016: Depression, Mood, and Emotion Recognition* (DAIC‑WOZ).
- Ringeval et al. (2019). *AVEC 2019: State‑of‑Mind, Detecting Depression with AI, Cross‑Cultural Affect* (E‑DAIC). arXiv:1907.11510.
- Valstar et al. (2013/2014). *AVEC 2013/2014* (AViD‑Corpus, BDI‑II).
- EATD‑Corpus (Mandarin emotional audio‑textual depression dataset).
- MODMA (Lanzhou University; EEG + audio).
- Androids Corpus (Italian; depression benchmarking).

**Features & signal processing**
- Eyben et al. (2016). *The Geneva Minimalistic Acoustic Parameter Set (GeMAPS/eGeMAPS)*, IEEE T‑AFFC.
- Eyben et al. *openSMILE* toolkit.
- Degottex et al. (2014). *COVAREP*.
- Cummins et al. (2015). *A review of depression and suicide risk assessment using speech analysis*, Speech Communication. **(start here for the feature survey)**

**Deep & self‑supervised methods**
- Ma et al. (2016). *DepAudioNet: An Efficient Deep Model for Audio‑based Depression Classification*, AVEC.
- Baevski et al. (2020). *wav2vec 2.0*. Hsu et al. (2021). *HuBERT*. Chen et al. (2022). *WavLM*.
- Desplanques et al. (2020). *ECAPA‑TDNN*.

**Modern depression‑from‑speech (Interspeech 2022 cluster)**
- Wang et al. (2022). *ECAPA‑TDNN Based Depression Detection from Clinical Speech*.
- Ravi, Wang, Flint, Alwan (2022). *A Step Towards Preserving Speakers' Identity While Detecting Depression via Speaker Disentanglement*.
- Wang, Ravi, Flint, Alwan (2022). *Unsupervised Instance Discriminative Learning for Depression Detection from Speech Signals*.
- Seneviratne & Espy‑Wilson (2022). *Multimodal Depression Severity Prediction Using Articulatory Coordination Features and Hierarchical Attention Text Embeddings*.
- Rutowski et al. (2022). *Toward Corpus Size Requirements for Training and Evaluating Depression Risk Models Using Spoken Language*.
- Lamichhane et al. (2022). *Dyadic Interaction Assessment from Free‑living Audio for Depression Severity Assessment*.
- Fara, Goria, Molimpakis, Cummins (2022). *Speech and the n‑Back task as a lens into depression*.

**Methodological caution (read these to stay honest)**
- Berisha et al. (2022). *Are reported accuracies in the clinical speech ML literature over‑optimistic?* Interspeech.
- Botelho, Schultz, Abad, Trancoso (2022). *Challenges of using longitudinal and cross‑domain corpora on studies of pathological speech*.

**Text/linguistic (for multimodal stretch)**
- Poświata & Perełkiewicz (2022). *Winning solution, LT‑EDI‑ACL 2022 Detecting Signs of Depression from Social Media Text*.

**Open‑source references (for engineering patterns; vet licenses & claims before reuse)**
- `Akshat2430/Detecting-Depression-From-Speech-Signals`, `AmirHoseein99/Depression-Engine` (DAIC‑WOZ), `cosmaadrian/multimodal-depression-from-video`.

---

## 19. Appendices

### Appendix A — DAIC‑WOZ practical notes
- Ships with **participant audio**, **transcripts (timestamped)**, **COVAREP** and **formant** features, and facial features (no raw video).
- Labels: PHQ‑8 total + per‑item; binary `PHQ8_Binary`.
- Official `train/dev/test` CSVs — **use them** for comparability.
- Watch for: interviewer speech contamination, variable session length, a few problematic sessions noted in the community.

### Appendix B — Minimal environment (suggested)
```
python>=3.10
numpy, scipy, pandas, scikit-learn, xgboost
librosa, soundfile, torchaudio, torch
opensmile, praat-parselmouth
transformers, speechbrain            # SSL embeddings
matplotlib, seaborn                  # EDA
pytest, hydra-core, mlflow           # tests, config, tracking
```

### Appendix C — Definition of "done" for this project
1. One command reproduces headline metrics from extracted features.
2. Speaker‑independent evaluation enforced and tested.
3. Tier‑0, Tier‑1, Tier‑2 results reported with CIs and per‑group fairness.
4. Cross‑corpus generalization attempted and documented.
5. Model card + datasheet + limitations written.
6. All code covered by tests; CI green; dependencies pinned.
7. Ethics/consent/access compliance documented.

### Appendix D — Open research questions to pursue
- How much does speaker disentanglement improve cross‑corpus generalization?
- Which eGeMAPS subsets carry the depression signal independent of arousal?
- Do SSL layers encode depression cues at specific depths (probing study)?
- Can symptom‑level (PHQ item) multi‑task heads improve robustness and interpretability?
- What is the minimal audio duration for reliable session‑level prediction?

---

*This document is intended to be updated continuously as experiments are run. Treat every claim as a hypothesis to be tested in the evaluation harness.*
