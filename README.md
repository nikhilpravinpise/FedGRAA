# Research

A collection of research projects and experiments in machine learning and cybersecurity.

## Project Structure

### PAKDD/
**Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems (FedGRAA)**

A from-scratch Keras reproduction of the FedGRAA paper — and a measured,
statistically-tested account of why the published algorithm does not converge
on real intrusion-detection data, plus a corrected convergence guarantee for
the fixed algorithm.

| Stage | Goal | Status |
|---|---|---|
| **A** | Faithful reproduction: every paper table/figure has a persisted, reproduced-or-honestly-diverging counterpart | ✅ complete |
| **B** | The divergence critique as a rigorous, citable component: formal theory + statistically-tested 10-seed ablation | ✅ complete |
| **C** | The novel extension for an A/A*-tier venue | ⏳ not started |

**What's in it:**
- Line-for-line TensorFlow/Keras reimplementation of the paper's architecture,
  both algorithm boxes, and both datasets (CIC-IDS-2017, UNSW-NB15)
- Forensic dataset-provenance audit against raw CSV SHA-256s and canonical
  published distributions
- A measured root-cause diagnosis: the aggregation-weight vector leaves the
  probability simplex in **round 1**, in **120 of 120** measured unprojected
  cells, because the published Algorithm 2 has no projection step
- A corrected convergence proposition (projected sub-gradient descent on the
  simplex → stationarity, honestly weaker than the paper's global-optimality
  claim) with a proof sketch
- A 2×2 factorial ablation over 10 seeds × 3 γ × 2 datasets, with bootstrap
  CIs, paired *t* / Wilcoxon / Cohen's *d*, and Holm–Bonferroni correction
- **142 passing tests** plus two standalone stage-completeness checks

**Quick Start:**
```powershell
cd PAKDD
pip install -r requirements.txt
python -m fedgraa gamma-sweep --dataset CIC-IDS-2017 --rounds 200
```

**Start reading here:**
- [PAKDD/README.md](PAKDD/README.md) — setup, experiments, project layout
- [PAKDD/docs/current/STAGE_A_MASTER_REPORT.md](PAKDD/docs/current/STAGE_A_MASTER_REPORT.md) — the reproduction, exhaustively
- [PAKDD/docs/current/STAGE_B_MASTER_REPORT.md](PAKDD/docs/current/STAGE_B_MASTER_REPORT.md) — the theory + campaign, exhaustively
- [PAKDD/docs/current/fedgraa_convergence_defect.md](PAKDD/docs/current/fedgraa_convergence_defect.md) — the paper-section-shaped critique
- [PAKDD/understanding/00_START_HERE.md](PAKDD/understanding/00_START_HERE.md) — a gentle introduction to FL and this paper

### docs/
General project documentation:
- [`Depression_Detection_Using_Speech.md`](docs/Depression_Detection_Using_Speech.md) — documentation on speech-based depression detection
- [`project_topic.md`](docs/project_topic.md) — project topic overview

## Repository Guidelines

### Ignored Files
- Data files (`.csv`, `.pcap`)
- Python cache and bytecode
- Virtual environments
- IDE settings
- Jupyter checkpoints
- Compressed archives

### Setup

1. **Clone the repository**
   ```powershell
   git clone <repository-url>
   cd Research
   ```

2. **Set up Python environment** (Python 3.13 recommended — TensorFlow 2.21
   has no Python 3.14 wheels yet)
   ```powershell
   python -m venv .venv313
   .venv313\Scripts\Activate.ps1
   ```

3. **Install dependencies** (for the PAKDD project)
   ```powershell
   cd PAKDD
   pip install -r requirements.txt
   ```

4. **Run tests**
   ```powershell
   pytest tests --tb=short          # 142 passed
   python scripts\check_stage_a_complete.py   # exit 0
   python scripts\check_stage_b_complete.py   # exit 0
   ```

## License

[Add license information if applicable]

## Authors

[Add author information if applicable]
