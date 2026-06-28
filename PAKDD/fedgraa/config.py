"""Configuration objects locking every hyperparameter to the paper.

All values come directly from Section 5 (Experiments and Results) and Tables 1-2
of the FedGRAA paper. Nothing here is tuned or altered relative to the paper —
this module is the single source of truth for the fidelity contract described in
``plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Network architecture (Section 5, "Network Architecture").

    Hidden layers 200 -> 100 -> 50 with tanh, softmax output. The number of
    input features and output classes is filled in per dataset.
    """

    n_features: int
    n_classes: int
    hidden_sizes: Tuple[int, ...] = (200, 100, 50)
    hidden_activation: str = "tanh"
    output_activation: str = "softmax"

    @property
    def layer_sizes(self) -> Tuple[int, ...]:
        return (self.n_features, *self.hidden_sizes, self.n_classes)


@dataclass(frozen=True)
class FedConfig:
    """Federated training hyperparameters (Section 5, "Training Setup").

    Defaults reproduce the paper exactly:
        eta = 0.01, E = 2, B = 64, T = 200, eta' = 0.1, E' = 5.
    """

    # Local model optimization
    eta: float = 0.01           # local Adam learning rate
    local_epochs: int = 2       # E
    batch_size: int = 64        # B
    rounds: int = 200           # T

    # FedGRAA weight-learner optimization
    eta_prime: float = 0.1      # eta' — aggregation-weight learning rate
    weight_epochs: int = 5      # E'

    # Personalization
    gamma: float = 1.0          # personalization coefficient, in [0, 1]
    n_rare: int = 3             # number of protected underrepresented clients

    # Validation split held back on each client for D_V
    val_fraction: float = 0.10  # 90% train / 10% validation

    # Reproducibility
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(
                f"val_fraction must be in (0, 1), got {self.val_fraction}"
            )
        if self.n_rare < 1:
            raise ValueError(f"n_rare must be >= 1, got {self.n_rare}")


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset definition taken from Tables 1 and 2.

    ``used_counts`` is the "Used" column (the sample budget actually fed to the
    pipeline); frequent classes are already capped at 10,000 there. ``labels``
    are ordered by the paper's ID column. ``benign_id`` marks the Benign/Normal
    class that is paired with each attack client.
    """

    name: str
    n_features: int
    labels: List[str]
    used_counts: List[int]
    benign_id: int
    frequent_cap: int = 10_000
    # Rare-class client IDs to protect, ordered most-rare-first, for n in {3,5,7}.
    rare_label_order: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.used_counts):
            raise ValueError("labels and used_counts must have equal length")
        if not 0 <= self.benign_id < len(self.labels):
            raise ValueError("benign_id out of range")

    @property
    def n_classes(self) -> int:
        return len(self.labels)

    @property
    def attack_ids(self) -> List[int]:
        """Class IDs that define a client (every class except Benign/Normal)."""
        return [i for i in range(self.n_classes) if i != self.benign_id]

    @property
    def n_clients(self) -> int:
        """One client per attack type (Section 5, Data Settings)."""
        return len(self.attack_ids)

    def label_id(self, label: str) -> int:
        return self.labels.index(label)

    def used_count(self, label: str) -> int:
        return self.used_counts[self.label_id(label)]

    def rare_client_ids(self, n: int) -> List[int]:
        """The class IDs of the ``n`` protected rare clients.

        Uses ``rare_label_order`` when provided (matching the paper's choices),
        otherwise falls back to the ``n`` smallest attack classes by count.
        """
        if self.rare_label_order:
            chosen = self.rare_label_order[:n]
            return [self.label_id(lbl) for lbl in chosen]
        attacks_by_size = sorted(self.attack_ids, key=lambda i: self.used_counts[i])
        return attacks_by_size[:n]

    def model_config(self) -> ModelConfig:
        return ModelConfig(n_features=self.n_features, n_classes=self.n_classes)


# --------------------------------------------------------------------------- #
# CIC-IDS-2017 (Table 1): 78 features, 15 classes, K = 14 attack clients.
# --------------------------------------------------------------------------- #
CIC_IDS_2017 = DatasetSpec(
    name="CIC-IDS-2017",
    n_features=78,
    labels=[
        "BENIGN",            # 0
        "Bot",               # 1
        "DDoS",              # 2
        "DoS GoldenEye",     # 3
        "DoS Hulk",          # 4
        "DoS Slowhttptest",  # 5
        "DoS Slowloris",     # 6
        "FTP-Patator",       # 7
        "Heartbleed",        # 8
        "Infiltration",      # 9
        "PortScan",          # 10
        "SSH-Patator",       # 11
        "Brute Force",       # 12
        "SQL Injection",     # 13
        "XSS",               # 14
    ],
    used_counts=[
        63_901,  # BENIGN
        1_948,   # Bot
        10_000,  # DDoS
        10_000,  # DoS GoldenEye
        10_000,  # DoS Hulk
        3_219,   # DoS Slowhttptest
        5_228,   # DoS Slowloris
        5_931,   # FTP-Patator
        11,      # Heartbleed
        36,      # Infiltration
        10_000,  # PortScan
        5_385,   # SSH-Patator
        1_470,   # Brute Force
        21,      # SQL Injection
        652,     # XSS
    ],
    benign_id=0,
    # n = 3 protects Heartbleed, SQL Injection, Infiltration (Section 5).
    # Extend to 5 / 7 by adding the next smallest attack classes.
    rare_label_order=[
        "Heartbleed",       # 11
        "SQL Injection",    # 21
        "Infiltration",     # 36
        "XSS",              # 652
        "Brute Force",      # 1470
        "Bot",              # 1948
        "DoS Slowhttptest", # 3219
    ],
)


# --------------------------------------------------------------------------- #
# UNSW-NB15 (Table 2): 43 features, 10 classes, K = 9 attack clients.
# --------------------------------------------------------------------------- #
UNSW_NB15 = DatasetSpec(
    name="UNSW-NB15",
    n_features=43,
    labels=[
        "Normal",          # 0
        "Generic",         # 1
        "Fuzzers",         # 2
        "Backdoor",        # 3
        "Worms",           # 4
        "Reconnaissance",  # 5
        "Analysis",        # 6
        "Exploits",        # 7
        "Shellcode",       # 8
        "DoS",             # 9
    ],
    used_counts=[
        56_691,  # Normal
        10_000,  # Generic
        10_000,  # Fuzzers
        2_329,   # Backdoor
        174,     # Worms
        10_000,  # Reconnaissance
        2_677,   # Analysis
        10_000,  # Exploits
        1_511,   # Shellcode
        10_000,  # DoS
    ],
    benign_id=0,
    # n = 3 protects Worms, Shellcode, Backdoor (Section 5).
    rare_label_order=[
        "Worms",       # 174
        "Shellcode",   # 1511
        "Backdoor",    # 2329
        "Analysis",    # 2677
        "Fuzzers",     # capped 10000 (next by original rarity)
        "DoS",         # 10000
        "Reconnaissance",
    ],
)


DATASETS: Dict[str, DatasetSpec] = {
    CIC_IDS_2017.name: CIC_IDS_2017,
    UNSW_NB15.name: UNSW_NB15,
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(
            f"Unknown dataset {name!r}; choose from {sorted(DATASETS)}"
        ) from exc
