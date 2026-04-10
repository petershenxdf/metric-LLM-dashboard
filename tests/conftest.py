"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Make the project root importable when running `pytest` from anywhere
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_moons


@pytest.fixture
def moons_dataset():
    """Two-moon dataset -- classic non-convex test case."""
    X, _ = make_moons(n_samples=100, noise=0.05, random_state=42)
    return X


@pytest.fixture
def moons_dataframe(moons_dataset):
    return pd.DataFrame(moons_dataset, columns=["x1", "x2"])


@pytest.fixture
def simple_DN():
    """A few hand-picked labels for the moons dataset."""
    return {0: 0, 1: 0, 50: 1, 51: 1}


@pytest.fixture
def simple_DO():
    return set()
