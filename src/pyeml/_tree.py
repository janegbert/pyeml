"""EMLTree: trainable binary tree of EML operators.

Implements the master formula from Section 4.3 of Odrzywołek (2026).
Every internal node computes eml(left_input, right_input).
Each input is a softmax-weighted mix of {1, x, child_eml_output}.
Leaves mix only {1, x}.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyeml._operator import eml, DTYPE


class EMLTree(nn.Module):
    """Trainable EML binary tree for symbolic regression.

    Args:
        depth: Tree depth. Depth 1 = single eml node with 2 leaves.
            Parameter count: 5 * 2^depth - 6.
    """

    def __init__(self, depth: int = 3):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2**depth
        self.n_internal = 2**depth - 1

        # Leaf nodes: each mixes (1, x) via softmax over 2 logits
        self.leaf_logits = nn.Parameter(
            torch.randn(self.n_leaves, 2, dtype=torch.float64)
        )

        # Internal non-root nodes: each mixes (1, x, child_eml) via softmax over 3 logits
        # Root (index 0) has no mixing — it just computes eml(left, right)
        # So we need logits for nodes 1 .. n_internal-1
        n_mixing = max(0, self.n_internal - 1)
        self.mixing_logits = nn.Parameter(
            torch.randn(n_mixing, 3, dtype=torch.float64)
        )

    @property
    def n_parameters(self) -> int:
        """Number of trainable scalar parameters (should equal 5*2^d - 6)."""
        return self.leaf_logits.numel() + self.mixing_logits.numel()

    def forward(self, x: torch.Tensor, temperature: float = 1.0, gumbel: bool = False) -> torch.Tensor:
        """Evaluate the tree on input x (complex128 tensor).

        Uses iterative bottom-up evaluation.
        """
        total = self.n_internal + self.n_leaves
        values: list[torch.Tensor | None] = [None] * total
        ones = torch.ones_like(x)

        # Compute leaf values: softmax(logits) @ [1, x]
        if gumbel and self.training:
            leaf_w = F.gumbel_softmax(
                self.leaf_logits / temperature, tau=1.0, hard=False
            ).to(DTYPE)
        else:
            leaf_w = F.softmax(self.leaf_logits / temperature, dim=1).to(DTYPE)

        for i in range(self.n_leaves):
            node_idx = self.n_internal + i
            values[node_idx] = leaf_w[i, 0] * ones + leaf_w[i, 1] * x

        # Internal non-root nodes, bottom-up
        if gumbel and self.training:
            mix_w = F.gumbel_softmax(
                self.mixing_logits / temperature, tau=1.0, hard=False
            ).to(DTYPE) if self.mixing_logits.numel() > 0 else None
        else:
            mix_w = F.softmax(
                self.mixing_logits / temperature, dim=1
            ).to(DTYPE) if self.mixing_logits.numel() > 0 else None

        for node in range(self.n_internal - 1, 0, -1):
            left = values[2 * node + 1]
            right = values[2 * node + 2]
            child_eml = eml(left, right)

            # This node's output = alpha*1 + beta*x + gamma*child_eml
            w = mix_w[node - 1]  # offset by 1 since root is excluded
            values[node] = w[0] * ones + w[1] * x + w[2] * child_eml

        # Root node (index 0): just eml(left, right), no mixing
        left = values[1]
        right = values[2]
        values[0] = eml(left, right)

        return values[0]

    def snap_weights(self) -> None:
        """Snap all logits to one-hot (argmax gets +100, rest get -100)."""
        with torch.no_grad():
            for logits in [self.leaf_logits, self.mixing_logits]:
                for i in range(logits.shape[0]):
                    idx = torch.argmax(logits[i])
                    logits[i] = torch.full_like(logits[i], -100.0)
                    logits[i, idx] = 100.0

    def harden(self, factor: float) -> None:
        """Multiply all logits by factor to push softmax toward one-hot."""
        with torch.no_grad():
            self.leaf_logits.mul_(factor)
            self.mixing_logits.mul_(factor)
