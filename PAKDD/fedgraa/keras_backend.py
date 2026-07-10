"""Compatibility alias for the Keras backend.

The paper describes the model as a Keras Sequential network; ``model.py`` is
now the canonical Keras implementation. This module re-exports its public
symbols so that any code still importing ``fedgraa.keras_backend`` continues to
work.
"""

from __future__ import annotations

from .model import (
    Adam,
    MLP,
    Params,
    aggregate,
    clone_params,
    init_params,
)

__all__ = ["Adam", "MLP", "Params", "aggregate", "clone_params", "init_params"]

