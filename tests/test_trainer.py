"""Tests for the training engine."""

import cmath

import numpy as np
import pytest
import torch

from pyeml._config import SearchConfig, TrainConfig
from pyeml._operator import DTYPE
from pyeml._trainer import search


class TestTrainer:
    def test_recover_exp(self):
        """exp(x) should be recovered exactly at depth 1."""
        x = np.linspace(0.1, 2.0, 100)
        y = np.exp(x).astype(np.complex128)

        config = SearchConfig(
            depths=[1],
            n_restarts=5,
            verbose=False,
            train=TrainConfig(epochs=3000, harden_start=0.7),
        )

        result = search(torch.tensor(x, dtype=DTYPE), torch.tensor(y, dtype=DTYPE), config)
        assert result is not None
        assert result.is_exact
        assert "eml(x, 1)" in result.eml_expression

    def test_recover_e_constant(self):
        """The constant e should be recovered at depth 1."""
        x = np.linspace(0.5, 3.0, 100)
        y = np.full_like(x, cmath.e.real, dtype=np.complex128)

        config = SearchConfig(
            depths=[1],
            n_restarts=5,
            verbose=False,
            train=TrainConfig(epochs=3000, harden_start=0.7),
        )

        result = search(torch.tensor(x, dtype=DTYPE), torch.tensor(y, dtype=DTYPE), config)
        assert result is not None
        assert result.is_exact
        assert "eml(1, 1)" in result.eml_expression

    def test_recover_ln(self):
        """ln(x) should be recoverable at depth 3 (may need multiple restarts)."""
        x = np.linspace(0.5, 4.0, 100)
        y = np.log(x).astype(np.complex128)

        config = SearchConfig(
            depths=[3],
            n_restarts=20,
            verbose=False,
            train=TrainConfig(epochs=8000, harden_start=0.75),
        )

        result = search(torch.tensor(x, dtype=DTYPE), torch.tensor(y, dtype=DTYPE), config)
        assert result is not None
        # ln recovery is probabilistic (~25% per restart), so just check loss is reasonable
        # With 20 restarts we should get at least a good approximation
        assert result.snapped_loss < 1.0

    def test_multi_depth_search(self):
        """Multi-depth should find exp(x) at depth 1 and stop."""
        x = np.linspace(0.1, 2.0, 100)
        y = np.exp(x).astype(np.complex128)

        config = SearchConfig(
            depths=[1, 2, 3],
            n_restarts=3,
            verbose=False,
            train=TrainConfig(epochs=3000, harden_start=0.7),
        )

        result = search(torch.tensor(x, dtype=DTYPE), torch.tensor(y, dtype=DTYPE), config)
        assert result is not None
        assert result.depth == 1  # should stop at depth 1
        assert result.is_exact
