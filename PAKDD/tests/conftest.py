"""Shared pytest fixtures and helpers."""

import os
import sys

import numpy as np
import pytest

# Make the package importable when pytest is run from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: real end-to-end test (process spawn and/or real training), "
        "not just structure/signature checks -- runs by default, "
        "deselect with -m 'not slow' for a fast inner loop.",
    )


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def tiny_layer_sizes():
    # small net for fast, exact gradient checks
    return (5, 8, 4, 3)
