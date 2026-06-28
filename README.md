# Research Project

A collection of research projects and experiments in machine learning and cybersecurity.

## Project Structure

### PAKDD/
**Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems (FedGRAA)**

A complete, from-scratch reproduction of the FedGRAA paper. Features:
- NumPy-based implementation (deterministic, lightweight)
- Faithful reproduction of paper specifications
- Synthetic and real-world dataset support (CIC-IDS-2017)
- Federated learning experiments with non-IID data partitioning
- Comprehensive test suite

**Quick Start:**
```powershell
cd PAKDD
pip install -r requirements.txt
python -m fedgraa gamma-sweep --dataset CIC-IDS-2017 --rounds 200
```

See [PAKDD/README.md](PAKDD/README.md) and [PAKDD/understanding/00_START_HERE.md](PAKDD/understanding/00_START_HERE.md) for detailed documentation.

### docs/
General project documentation:
- `Depression_Detection_Using_Speech.md` - Documentation on speech-based depression detection
- `project_topic.md` - Project topic overview

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
   cd Research_Project
   ```

2. **Set up Python environment**
   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

3. **Install dependencies** (for PAKDD project)
   ```powershell
   cd PAKDD
   pip install -r requirements.txt
   ```

4. **Run tests**
   ```powershell
   pytest tests/
   ```

## License

[Add license information if applicable]

## Authors

[Add author information if applicable]
