"""Tests for EMLTree module."""

import pytest
import torch

from pyeml._tree import EMLTree
from pyeml._operator import DTYPE


class TestTreeStructure:
    @pytest.mark.parametrize("depth,expected_params", [
        (1, 5 * 2**1 - 6),   # 4
        (2, 5 * 2**2 - 6),   # 14
        (3, 5 * 2**3 - 6),   # 34
        (4, 5 * 2**4 - 6),   # 74
    ])
    def test_parameter_count(self, depth, expected_params):
        """Parameter count must match paper formula: 5 * 2^n - 6."""
        model = EMLTree(depth=depth)
        assert model.n_parameters == expected_params

    def test_output_dtype(self):
        model = EMLTree(depth=2)
        x = torch.tensor([1.0, 2.0], dtype=DTYPE)
        y = model(x)
        assert y.dtype == DTYPE

    def test_output_shape(self):
        model = EMLTree(depth=2)
        x = torch.tensor([1.0, 2.0, 3.0], dtype=DTYPE)
        y = model(x)
        assert y.shape == x.shape

    def test_grad_flows(self):
        model = EMLTree(depth=2)
        x = torch.tensor([1.0, 2.0], dtype=DTYPE)
        y = model(x)
        loss = (y.real**2).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_snap_weights(self):
        model = EMLTree(depth=2)
        model.snap_weights()
        # After snapping, softmax of logits should be ~one-hot
        for logits in [model.leaf_logits, model.mixing_logits]:
            for i in range(logits.shape[0]):
                probs = torch.softmax(logits[i], dim=0)
                assert probs.max().item() > 0.99

    def test_harden(self):
        model = EMLTree(depth=2)
        before = model.leaf_logits.clone()
        model.harden(1.5)
        # Logits should be scaled up
        assert torch.allclose(model.leaf_logits, before * 1.5)

    def test_depth_1_minimal(self):
        """Depth 1: single eml node, 2 leaves, no mixing nodes."""
        model = EMLTree(depth=1)
        assert model.n_leaves == 2
        assert model.n_internal == 1
        assert model.mixing_logits.shape[0] == 0  # no non-root internal nodes
