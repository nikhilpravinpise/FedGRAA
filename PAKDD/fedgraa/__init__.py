"""FedGRAA — a faithful, from-scratch reproduction of
*Federated Gradient-Refined Adaptive Averaging for Intrusion Detection Systems*.

The package implements, with no deviations from the paper's stated design:

* the dense ``200 -> 100 -> 50`` tanh / softmax network (``model``),
* the non-IID one-attack-per-client partition (``partition``),
* FedAvg, Static Reparameterization (eq. 7) and FedGRAA (eq. 8-10)
  aggregation rules (``aggregation``),
* the Algorithm-1 federated training loop (``trainer``),
* the gamma / n experiment sweeps of Figures 1-3 (``experiment``).
"""

from .config import (
    CIC_IDS_2017,
    UNSW_NB15,
    DatasetSpec,
    FedConfig,
    ModelConfig,
)
from .model import MLP, Adam
from .data import load_dataset, make_synthetic_dataset
from .partition import (
    ClientShard,
    FederatedSplit,
    median_frequency_split,
    partition_non_iid,
)
from .aggregation import FedGRAA, fedavg_weights, static_reparam_weights
from .trainer import FederatedTrainer, TrainHistory
from .metrics import accuracy, last_n_class_accuracy
from .experiment import ExperimentResult, run_experiment, run_gamma_sweep

__all__ = [
    "CIC_IDS_2017",
    "UNSW_NB15",
    "DatasetSpec",
    "FedConfig",
    "ModelConfig",
    "MLP",
    "Adam",
    "load_dataset",
    "make_synthetic_dataset",
    "ClientShard",
    "FederatedSplit",
    "median_frequency_split",
    "partition_non_iid",
    "FedGRAA",
    "fedavg_weights",
    "static_reparam_weights",
    "FederatedTrainer",
    "TrainHistory",
    "accuracy",
    "last_n_class_accuracy",
    "ExperimentResult",
    "run_experiment",
    "run_gamma_sweep",
]

__version__ = "1.0.0"
