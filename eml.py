"""
EML Symbolic Regression Engine

Based on: "All elementary functions from a single binary operator"
by Andrzej Odrzywołek (arXiv:2603.21852)

The EML operator: eml(x, y) = exp(x) - ln(y)
Together with the constant 1, this single operator can express
all elementary functions (sin, cos, sqrt, log, +, -, *, /, etc.)

This engine uses the "master formula" approach from Section 4.3:
- Build a complete binary tree of EML nodes
- Each node input is α*1 + β*x + γ*f (softmax-weighted)
- Train with Adam, then harden weights to snap to {0, 1}
- Read off the exact symbolic expression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

DTYPE = torch.complex128
EXP_CLAMP = 500.0  # clamp exp argument to avoid overflow


def eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The EML operator: exp(x) - ln(y), over complex numbers."""
    # Clamp real part of exp argument to avoid overflow
    x_clamped = torch.complex(
        torch.clamp(x.real, -EXP_CLAMP, EXP_CLAMP),
        x.imag
    )
    return torch.exp(x_clamped) - torch.log(y)


class EMLTree(nn.Module):
    """
    Trainable EML binary tree for symbolic regression.

    Architecture (depth 2 example):
        eml(
            α1 + β1*x + γ1 * eml(α3 + β3*x, α4 + β4*x),
            α2 + β2*x + γ2 * eml(α5 + β5*x, α6 + β6*x)
        )

    Each input is a softmax-weighted combination of (1, x, child_output).
    Leaves have no child, so just softmax(α, β) over (1, x).
    """

    def __init__(self, depth: int = 3):
        super().__init__()
        self.depth = depth

        # Count nodes: internal nodes need 3 logits each (α, β, γ)
        # Leaf nodes need 2 logits each (α, β) - no child output
        # Full binary tree: 2^depth - 1 internal nodes, 2^depth leaves
        n_internal = 2**depth - 1
        n_leaves = 2**depth

        # Logits for internal nodes: (α, β, γ) -> softmax -> weights for (1, x, child)
        self.internal_logits = nn.Parameter(torch.randn(n_internal, 3, dtype=torch.float64))

        # Logits for leaves: (α, β) -> softmax -> weights for (1, x)
        self.leaf_logits = nn.Parameter(torch.randn(n_leaves, 2, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the EML tree on input x.
        x: complex128 tensor of input values
        Returns: complex128 tensor of output values
        """
        return self._evaluate(x, node_idx=0, depth=0)

    def _evaluate(self, x: torch.Tensor, node_idx: int, depth: int) -> torch.Tensor:
        """Recursively evaluate the tree."""
        if depth == self.depth:
            # Leaf node
            leaf_idx = node_idx - (2**self.depth - 1)
            weights = F.softmax(self.leaf_logits[leaf_idx], dim=0).to(DTYPE)
            return weights[0] * torch.ones_like(x) + weights[1] * x

        # Internal node: compute children first
        left_child = 2 * node_idx + 1
        right_child = 2 * node_idx + 2

        left_val = self._evaluate(x, left_child, depth + 1)
        right_val = self._evaluate(x, right_child, depth + 1)

        child_output = eml(left_val, right_val)

        # This node's input mixing
        n_internal = 2**self.depth - 1
        if node_idx < n_internal:
            # Check if this node IS the root (no parent) or has a parent
            # Actually all internal nodes produce output that gets mixed by parent
            # The root's output IS the tree output
            pass

        if node_idx == 0:
            # Root: just return the eml result directly
            return child_output

        # Non-root internal node: return the eml result
        # Parent will mix it with (1, x) using its logits
        return child_output

    def _evaluate_v2(self, x: torch.Tensor) -> torch.Tensor:
        """Iterative bottom-up evaluation - cleaner approach."""
        n_leaves = 2**self.depth
        n_internal = 2**self.depth - 1

        # Compute leaf values
        leaf_weights = F.softmax(self.leaf_logits, dim=1).to(DTYPE)  # (n_leaves, 2)
        ones = torch.ones_like(x)

        # values[i] = output of node i
        # Start from leaves
        node_values = [None] * (n_internal + n_leaves)

        for i in range(n_leaves):
            node_values[n_internal + i] = leaf_weights[i, 0] * ones + leaf_weights[i, 1] * x

        # Bottom-up: compute internal nodes
        for i in range(n_internal - 1, -1, -1):
            left = node_values[2 * i + 1]
            right = node_values[2 * i + 2]

            child_eml = eml(left, right)

            if i == 0:
                # Root node: output is just the eml
                node_values[0] = child_eml
            else:
                # Internal non-root: parent will mix (1, x, this_eml)
                # Store the eml output; parent reads it
                node_values[i] = child_eml

        return node_values[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate using the master formula approach from the paper."""
        return self._master_forward(x, node_idx=0, depth=0)

    def _master_forward(self, x: torch.Tensor, node_idx: int, depth: int) -> torch.Tensor:
        """
        Master formula evaluation.
        Each EML node's two inputs are: α*1 + β*x + γ*eml(child_left, child_right)
        At leaves (deepest level), inputs are just: α*1 + β*x
        """
        n_internal = 2**self.depth - 1

        if depth == self.depth:
            # Leaf: α*1 + β*x
            leaf_idx = node_idx - n_internal
            weights = F.softmax(self.leaf_logits[leaf_idx], dim=0).to(DTYPE)
            return weights[0] * torch.ones_like(x) + weights[1] * x

        # Internal node with index in the internal array
        internal_idx = node_idx

        # Get the two sub-inputs for this EML node
        left_input = self._get_input(x, node_idx, depth, side=0)
        right_input = self._get_input(x, node_idx, depth, side=1)

        return eml(left_input, right_input)

    def _get_input(self, x: torch.Tensor, parent_idx: int, parent_depth: int, side: int) -> torch.Tensor:
        """
        Get one input for an EML node.
        side=0 for left, side=1 for right.
        Input = α*1 + β*x + γ*eml(subtree)
        """
        child_idx = 2 * parent_idx + 1 + side
        n_internal = 2**self.depth - 1

        if child_idx >= n_internal:
            # Child is a leaf: α*1 + β*x (no γ term)
            leaf_idx = child_idx - n_internal
            weights = F.softmax(self.leaf_logits[leaf_idx], dim=0).to(DTYPE)
            return weights[0] * torch.ones_like(x) + weights[1] * x
        else:
            # Child is internal: α*1 + β*x + γ*eml(subtree)
            child_eml = self._master_forward(x, child_idx, parent_depth + 1)
            weights = F.softmax(self.internal_logits[child_idx], dim=0).to(DTYPE)
            return weights[0] * torch.ones_like(x) + weights[1] * x + weights[2] * child_eml

    def get_symbolic(self) -> str:
        """Extract the symbolic expression after training."""
        return self._symbolic(node_idx=0, depth=0)

    def _symbolic(self, node_idx: int, depth: int) -> str:
        n_internal = 2**self.depth - 1

        left_str = self._symbolic_input(node_idx, depth, side=0)
        right_str = self._symbolic_input(node_idx, depth, side=1)

        return f"eml({left_str}, {right_str})"

    def _symbolic_input(self, parent_idx: int, parent_depth: int, side: int) -> str:
        child_idx = 2 * parent_idx + 1 + side
        n_internal = 2**self.depth - 1

        if child_idx >= n_internal:
            leaf_idx = child_idx - n_internal
            weights = F.softmax(self.leaf_logits[leaf_idx], dim=0).detach().numpy()
            snapped = np.round(weights).astype(int)
            terms = []
            if snapped[0]: terms.append("1")
            if snapped[1]: terms.append("x")
            return " + ".join(terms) if terms else "0"
        else:
            weights = F.softmax(self.internal_logits[child_idx], dim=0).detach().numpy()
            snapped = np.round(weights).astype(int)
            child_str = self._symbolic(child_idx, parent_depth + 1)
            terms = []
            if snapped[0]: terms.append("1")
            if snapped[1]: terms.append("x")
            if snapped[2]: terms.append(child_str)
            return " + ".join(terms) if terms else "0"

    def snap_weights(self):
        """Snap all weights to nearest vertex of simplex (one-hot)."""
        with torch.no_grad():
            for i in range(self.leaf_logits.shape[0]):
                idx = torch.argmax(self.leaf_logits[i])
                self.leaf_logits[i] = torch.tensor([-100.0] * self.leaf_logits.shape[1], dtype=torch.float64)
                self.leaf_logits[i, idx] = 100.0

            for i in range(self.internal_logits.shape[0]):
                idx = torch.argmax(self.internal_logits[i])
                self.internal_logits[i] = torch.tensor([-100.0] * self.internal_logits.shape[1], dtype=torch.float64)
                self.internal_logits[i, idx] = 100.0


def train_eml(
    target_fn: Callable,
    depth: int = 3,
    n_points: int = 200,
    x_range: tuple = (0.5, 3.0),
    lr: float = 0.01,
    epochs: int = 5000,
    harden_at: int = 4000,
    n_runs: int = 10,
    verbose: bool = True,
    target_name: str = "f(x)",
) -> Optional[EMLTree]:
    """
    Train an EML tree to recover a target function.

    Uses the paper's approach: Adam optimization with softmax-parameterized
    weights, followed by a hardening phase that anneals toward one-hot.
    Multiple random restarts to handle the non-convex landscape.
    """
    # Generate training data — use non-uniform spacing to avoid aliasing
    x_np = np.linspace(x_range[0], x_range[1], n_points)
    # Add a few random points to break symmetry
    x_extra = np.random.uniform(x_range[0], x_range[1], 20)
    x_np = np.sort(np.concatenate([x_np, x_extra]))
    y_np = np.array([target_fn(xi) for xi in x_np], dtype=np.complex128)

    x_train = torch.tensor(x_np, dtype=DTYPE)
    y_train = torch.tensor(y_np, dtype=DTYPE)

    best_model = None
    best_loss = float("inf")

    for run in range(n_runs):
        model = EMLTree(depth=depth)

        # Two-phase optimization: higher LR first, then fine-tune
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=harden_at)

        run_failed = False
        final_loss = float("inf")
        patience_counter = 0
        best_run_loss = float("inf")

        for epoch in range(epochs):
            optimizer.zero_grad()

            y_pred = model(x_train)

            # Loss: penalize both real and imaginary deviation
            diff = y_pred - y_train
            loss = torch.mean(diff.real**2 + diff.imag**2)

            if torch.isnan(loss) or torch.isinf(loss):
                run_failed = True
                break

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if epoch < harden_at:
                scheduler.step()

            # Hardening: exponentially scale logits toward one-hot
            if epoch >= harden_at:
                progress = (epoch - harden_at) / (epochs - harden_at)
                # Gentle start, aggressive end
                harden_factor = 1.001 + 0.004 * progress
                with torch.no_grad():
                    model.internal_logits.mul_(harden_factor)
                    model.leaf_logits.mul_(harden_factor)

            final_loss = loss.item()

            # Early stopping if converged
            if final_loss < best_run_loss:
                best_run_loss = final_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if final_loss < 1e-28:
                break

        if run_failed:
            if verbose:
                print(f"  Run {run+1}/{n_runs}: DIVERGED")
            continue

        # Snap weights and evaluate
        model.snap_weights()
        with torch.no_grad():
            y_snapped = model(x_train)
            diff = y_snapped - y_train
            snapped_loss = torch.mean(diff.real**2 + diff.imag**2).item()

        if verbose:
            status = "EXACT" if snapped_loss < 1e-20 else ("CLOSE" if snapped_loss < 0.01 else "")
            print(f"  Run {run+1}/{n_runs}: loss={final_loss:.2e} -> snapped={snapped_loss:.2e} {status}")

        if snapped_loss < best_loss:
            best_loss = snapped_loss
            best_model = model

        # Stop early if we found an exact match
        if best_loss < 1e-20:
            if verbose and run < n_runs - 1:
                print(f"  (exact match found, skipping remaining runs)")
            break

    if best_model is not None and verbose:
        print(f"\n  Best snapped loss: {best_loss:.2e}")
        expr = best_model.get_symbolic()
        print(f"  Recovered: {expr}")

        # Verify on test points
        with torch.no_grad():
            x_test = torch.tensor([1.0, 2.0, 0.7], dtype=DTYPE)
            y_test_pred = best_model(x_test)
            for i, xi in enumerate([1.0, 2.0, 0.7]):
                expected = target_fn(xi)
                got = y_test_pred[i]
                err = abs(got - expected)
                match = "EXACT" if err < 1e-10 else ("~" if err < 0.01 else "MISS")
                print(f"    f({xi}) = {expected:.6f}, EML = {got.real:.6f} [{match}]")
    elif verbose:
        print(f"\n  All {n_runs} runs diverged. Try increasing depth or runs.")

    return best_model


def demo():
    """Run demonstrations of EML symbolic regression."""
    import cmath

    print("=" * 65)
    print("  EML SYMBOLIC REGRESSION ENGINE")
    print("  Based on Odrzywołek (2026), arXiv:2603.21852")
    print("  Operator: eml(x,y) = exp(x) - ln(y)")
    print("=" * 65)

    targets = [
        {
            "name": "exp(x)",
            "fn": lambda x: cmath.exp(x),
            "depth": 1, "runs": 5, "epochs": 3000, "harden": 2000,
            "xr": (0.1, 2.0),
            "note": "Simplest: eml(x, 1)"
        },
        {
            "name": "e (constant)",
            "fn": lambda x: cmath.exp(1),
            "depth": 1, "runs": 5, "epochs": 3000, "harden": 2000,
            "xr": (0.5, 3.0),
            "note": "eml(1, 1) = e"
        },
        {
            "name": "ln(x)",
            "fn": lambda x: cmath.log(x),
            "depth": 3, "runs": 20, "epochs": 8000, "harden": 6000,
            "xr": (0.5, 4.0),
            "note": "Paper: eml(1, eml(eml(1,x), 1))"
        },
        {
            "name": "-x (negation)",
            "fn": lambda x: -x,
            "depth": 3, "runs": 30, "epochs": 8000, "harden": 6000,
            "xr": (0.5, 3.0),
            "note": "Needs complex path, depth ~3"
        },
        {
            "name": "x + 1",
            "fn": lambda x: x + 1,
            "depth": 3, "runs": 30, "epochs": 10000, "harden": 7000,
            "xr": (0.5, 3.0),
            "note": "x+y = ln(e^x * e^y), depth ~4"
        },
        {
            "name": "x * x",
            "fn": lambda x: x * x,
            "depth": 4, "runs": 30, "epochs": 10000, "harden": 7000,
            "xr": (0.5, 3.0),
            "note": "x^2 = exp(2*ln(x)), needs depth 4+"
        },
    ]

    for i, t in enumerate(targets, 1):
        print(f"\n[{i}] Target: {t['name']}  ({t['note']})")
        train_eml(
            target_fn=t["fn"],
            depth=t["depth"], n_runs=t["runs"],
            epochs=t["epochs"], harden_at=t["harden"],
            x_range=t["xr"], target_name=t["name"],
        )

    print("\n" + "=" * 65)
    print("  Every recovered expression uses ONLY eml(x,y) = exp(x) - ln(y)")
    print("  and the constant 1. No other operations.")
    print("=" * 65)


def verify_known():
    """Verify known EML decompositions from the paper."""
    import cmath

    print("\n" + "=" * 65)
    print("  VERIFYING KNOWN EML IDENTITIES (from the paper)")
    print("=" * 65)

    def E(x, y):
        return cmath.exp(x) - cmath.log(y)

    checks = [
        ("e^x for x=2",      E(2, 1),                  cmath.exp(2)),
        ("e",                 E(1, 1),                  cmath.e),
        ("e^e",               E(E(1,1), 1),             cmath.exp(cmath.e)),
        ("ln(2)",             E(1, E(E(1, 2), 1)),      cmath.log(2)),
        ("ln(0.5)",           E(1, E(E(1, 0.5), 1)),    cmath.log(0.5)),
        ("ln(e) = 1",         E(1, E(E(1, E(1,1)), 1)), 1.0),
    ]

    # Build x-1: paper Table 4 says depth ~43 via compiler, 11 via search
    # That's too deep for verification here, but let's verify the building blocks

    for name, got, expected in checks:
        err = abs(got - expected)
        status = "OK" if err < 1e-10 else f"ERR ({err:.2e})"
        print(f"  {name:25s} = {got.real:+.10f}{got.imag:+.10f}j  [{status}]")

    # Show how complex numbers emerge
    print("\n  --- Complex numbers from EML ---")
    # e^(e^e) is huge, so e - ln(e^(e^e)) = e - e^e ≈ -12.4 (negative!)
    eee = E(E(E(1,1), 1), 1)  # e^(e^e)
    neg = E(1, eee)             # e - e^e ≈ -12.4
    print(f"  e^(e^e)        = {eee.real:.4f}")
    print(f"  e - e^e        = {neg.real:.4f}  (negative!)")

    cpx = E(1, neg)  # e - ln(-12.4) = complex with Im = -pi
    print(f"  eml(1, above)  = {cpx.real:.4f} {cpx.imag:+.10f}j")
    print(f"  Imaginary part = {cpx.imag:.10f}")
    print(f"  -pi            = {-cmath.pi:.10f}")
    print(f"  Match: {abs(cpx.imag + cmath.pi) < 1e-10}")


if __name__ == "__main__":
    demo()
    verify_known()
