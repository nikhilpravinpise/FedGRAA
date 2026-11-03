# FedGRAA

**Reproducing and evaluating adaptive aggregation for federated intrusion detection.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Docs check](https://github.com/nikhilpravinpise/FedGRAA/actions/workflows/docs-check.yml/badge.svg)](https://github.com/nikhilpravinpise/FedGRAA/actions/workflows/docs-check.yml)

FedGRAA is a Python research repository for reproducing, diagnosing, and extending a gradient-refined aggregation method for federated network intrusion detection. It includes a TensorFlow/Keras implementation, experiment artifacts, tests, and documentation that keeps the literal reproduction separate from later corrections and extensions.

## Research status

The literal reproduction does **not** match the minority-class results reported in the target manuscript. The repository documents the divergence and keeps the projected correction and FedGRAA-v2 extension as separate research paths.

| Stage | Work | Status |
|---|---|---|
| A | Reproduce the manuscript's method on CIC-IDS-2017 and UNSW-NB15 | Complete; reported results diverge from the manuscript's minority-class claims |
| B | Diagnose the aggregation-weight update and evaluate a projected correction | Complete; the weight vector leaves the probability simplex in round 1 in 120 of 120 measured unprojected cells |
| C | Evaluate a privacy-oriented extension, baseline comparisons, and an adversarial attack surface | The repository reports 960 experiment cells and 111 of 112 Holm-corrected significance tests against 14 domain baselines; the few-shot/prototype option remains deferred |

These are project-reported research results, not claims of publication or general performance guarantees. The project's own provenance audit leaves the target manuscript's publication and provenance status unresolved. The FedGRAA-v2 manuscript is also a research draft. See the [Stage B report](FedGRAA/docs/current/STAGE_B_MASTER_REPORT.md) and [Stage C report](FedGRAA/docs/current/STAGE_C_MASTER_REPORT.md).

![Persisted CIC-IDS-2017 sweep comparing FedAvg, static reparameterization, and FedGRAA variants on all-class and rare-class accuracy](FedGRAA/results/figures/CIC-IDS-2017_figure1.png)

*One persisted CIC-IDS-2017 sweep. The literal FedGRAA variants do not sustain rare-class accuracy in this run; details and configuration are in the [Stage A report](FedGRAA/docs/current/STAGE_A_MASTER_REPORT.md).*

## What is included

- A from-scratch TensorFlow/Keras implementation of the manuscript's intrusion-detection model and aggregation methods.
- Separate faithful and projected aggregation paths, with the divergence diagnosis and correction documented.
- Data loaders and experiment scripts for CIC-IDS-2017 and UNSW-NB15. Dataset files are not included.
- Persisted result JSON, figures, stage completeness gates, numeric audits, and tests.
- A FedGRAA-v2 research extension with a secure-aggregation-mediated protocol and adversarial evaluation. Its reported trade-offs and limitations are documented with the results.

## Repository layout

```text
.github/                 Issue and pull-request templates; GitHub Actions workflow
FedGRAA/fedgraa/          Model, aggregation, data, privacy, and training code
FedGRAA/scripts/          Experiment drivers and completeness/correctness checks
FedGRAA/tests/            Automated tests
FedGRAA/results/          Persisted experiment results and figures
FedGRAA/docs/current/     Stage reports and current research status
FedGRAA/docs/audit/       Reproduction, data, and result audits
FedGRAA/docs/paper/       Manuscript drafts and their revision history
internship_report/        Source files and report for the research internship
CITATION.cff              Citation metadata
LICENSE                   MIT license
```

## Quick start

Use Python 3.13 to match the CI environment. Clone the repository and enter the project directory:

```powershell
git clone https://github.com/nikhilpravinpise/FedGRAA.git
cd FedGRAA/FedGRAA
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, create the environment with `python3.13 -m venv .venv` and activate it with `source .venv/bin/activate`.

### Run a reproduction sweep

Download CIC-IDS-2017 from the [official UNB dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) and place its CSV files under `data/CIC-IDS-2017/`. Then, from `FedGRAA/`, run:

```powershell
python run_experiments.py --data-dir data/CIC-IDS-2017 --cache-dir .cache --datasets CIC-IDS-2017 --sweep gamma
```

The input data is not tracked in this repository. The [data provenance audit](FedGRAA/docs/audit/data_provenance_audit.md) records how the experiment datasets were checked. The [project README](FedGRAA/README.md) describes the experiment options and UNSW-NB15 setup.

To check the persisted Stage A artifacts and their reported numbers:

```powershell
python scripts/check_stage_a_complete.py
python scripts/audit_stage_a.py
python scripts/check_doc_links.py
```

The completeness and audit scripts check existing artifacts; they do not download data or rerun the experiments. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full verification commands and [the workflow](.github/workflows/docs-check.yml) for the checks run on pull requests.

## Results and documentation

- [Stage A: faithful reproduction](FedGRAA/docs/current/STAGE_A_MASTER_REPORT.md)
- [Stage B: divergence diagnosis and projected correction](FedGRAA/docs/current/STAGE_B_MASTER_REPORT.md)
- [Stage C: FedGRAA-v2 results and limitations](FedGRAA/docs/current/STAGE_C_MASTER_REPORT.md)
- [Results index](FedGRAA/results/README.md)
- [Documentation map](FedGRAA/docs/README.md)

The Stage C evaluation reports an attack that can exploit its weight-updating rule, as well as defenses that reduce the method's rare-class signal. Those findings and the measured communication costs are included in the Stage C report.

## Citation

Please use the metadata in [CITATION.cff](CITATION.cff). GitHub uses this file for the repository's citation display.

```bibtex
@software{pise_fedgraa,
  author  = {Pise, Nikhil Pravin},
  title   = {{FedGRAA: Reproduction, Diagnosis, and Extension for Federated Intrusion Detection}},
  url     = {https://github.com/nikhilpravinpise/FedGRAA},
  license = {MIT},
  date    = {2026-11-03}
}
```

## Contributing

Bug reports and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the verification workflow and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community expectations.

## License

This repository is licensed under the [MIT License](LICENSE). Dataset files are not included; use each dataset under its own terms.
