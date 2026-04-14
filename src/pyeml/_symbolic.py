"""Extract and simplify symbolic expressions from trained EML trees."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pyeml._tree import EMLTree


def extract_expression(model: EMLTree) -> str:
    """Extract the symbolic EML expression from a snapped model.

    Returns a string like "eml(x, 1)" or "eml(1, eml(eml(1, x), 1))".
    """
    return _node_expr(model, node_idx=0)


def _node_expr(model: EMLTree, node_idx: int) -> str:
    """Build expression string for a node (root calls this with 0)."""
    left_str = _input_expr(model, node_idx, side=0)
    right_str = _input_expr(model, node_idx, side=1)
    return f"eml({left_str}, {right_str})"


def _input_expr(model: EMLTree, parent_idx: int, side: int) -> str:
    """Build expression for one input of an eml node."""
    child_idx = 2 * parent_idx + 1 + side

    if child_idx >= model.n_internal:
        # Leaf node
        leaf_idx = child_idx - model.n_internal
        w = F.softmax(model.leaf_logits[leaf_idx], dim=0).detach()
        return _weighted_terms(w, has_child=False)

    # Internal non-root node
    child_str = _node_expr(model, child_idx)

    if child_idx == 0:
        # Root as child — shouldn't happen in valid tree
        return child_str

    mix_idx = child_idx - 1  # mixing_logits excludes root
    w = F.softmax(model.mixing_logits[mix_idx], dim=0).detach()
    return _weighted_terms(w, has_child=True, child_str=child_str)


def _weighted_terms(
    w: torch.Tensor,
    has_child: bool = False,
    child_str: str = "",
) -> str:
    """Convert snapped weights to a symbolic term string."""
    snapped = torch.round(w).int().tolist()
    terms = []

    if snapped[0]:
        terms.append("1")
    if snapped[1]:
        terms.append("x")
    if has_child and len(snapped) > 2 and snapped[2]:
        terms.append(child_str)

    return " + ".join(terms) if terms else "0"


# ── Simplification ──────────────────────────────────────────────────────

SIMPLIFICATION_RULES = {
    "eml(x, 1)": "exp(x)",
    "eml(1, 1)": "e",
    "eml(1, eml(eml(1, x), 1))": "ln(x)",
}


def simplify(expr: str) -> str:
    """Apply known simplification rules to an EML expression."""
    result = expr
    for pattern, replacement in SIMPLIFICATION_RULES.items():
        result = result.replace(pattern, replacement)
    return result
