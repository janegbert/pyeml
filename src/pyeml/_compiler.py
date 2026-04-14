"""EML compiler: convert standard math expressions to/from EML form.

Known identities from the paper:
    exp(x) = eml(x, 1)
    ln(x)  = eml(1, eml(eml(1, x), 1))
    x + y  = ln(exp(x) * exp(y))           (then recursively compile)
    x * y  = exp(ln(x) + ln(y))            (then recursively compile)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EMLNode:
    """Node in an EML expression tree (immutable)."""

    kind: Literal["eml", "const", "var"]
    left: EMLNode | None = None
    right: EMLNode | None = None
    value: float | None = None  # for const nodes

    def __str__(self) -> str:
        if self.kind == "const":
            if self.value is not None and math.isfinite(self.value) and self.value == int(self.value):
                return str(int(self.value))
            return str(self.value)
        if self.kind == "var":
            return "x"
        return f"eml({self.left}, {self.right})"

    @property
    def depth(self) -> int:
        """Maximum depth of this tree."""
        if self.kind in ("const", "var"):
            return 0
        return 1 + max(
            self.left.depth if self.left else 0,
            self.right.depth if self.right else 0,
        )

    @property
    def leaf_count(self) -> int:
        """Number of leaves (Kolmogorov complexity K in the paper)."""
        if self.kind in ("const", "var"):
            return 1
        return (
            (self.left.leaf_count if self.left else 0)
            + (self.right.leaf_count if self.right else 0)
        )


# Shorthand constructors
def _eml(left: EMLNode, right: EMLNode) -> EMLNode:
    return EMLNode("eml", left, right)

def _const(v: float) -> EMLNode:
    return EMLNode("const", value=v)

def _var() -> EMLNode:
    return EMLNode("var")

ONE = _const(1)
X = _var()


def compile_exp(arg: EMLNode) -> EMLNode:
    """exp(arg) = eml(arg, 1)"""
    return _eml(arg, ONE)


def compile_ln(arg: EMLNode) -> EMLNode:
    """ln(arg) = eml(1, eml(eml(1, arg), 1))"""
    inner = _eml(ONE, arg)           # eml(1, arg) = e - ln(arg)
    middle = _eml(inner, ONE)        # eml(inner, 1) = exp(e - ln(arg))
    return _eml(ONE, middle)         # eml(1, middle) = e - ln(exp(e - ln(arg))) = ln(arg)


def compile_add(a: EMLNode, b: EMLNode) -> EMLNode:
    """a + b = ln(exp(a) * exp(b)) — compile via exp and ln."""
    # x + y = ln(exp(x) * exp(y))
    # But we need multiplication first. Let's use the direct route:
    # x + y = ln(exp(x) * exp(y)) = ln(exp(x)) + ln(exp(y))... circular
    # Actually: x + y = eml(x, 1/exp(y)) but we need 1/exp(y) = exp(-y)
    # Simpler: build from the paper's known EML forms
    # The paper shows x+y at depth ~19 (direct search), which is impractical to hardcode
    # For now, provide the building blocks and let the search find the tree
    raise NotImplementedError(
        "Addition compilation requires deep trees (~19 nodes). "
        "Use search() to find the EML form instead."
    )


def compile_mul(a: EMLNode, b: EMLNode) -> EMLNode:
    """a * b = exp(ln(a) + ln(b)) — compile via exp, ln, add."""
    raise NotImplementedError(
        "Multiplication compilation requires addition, which requires deep trees. "
        "Use search() to find the EML form instead."
    )


# ── High-level compile function ─────────────────────────────────────────

KNOWN_COMPILATIONS: dict[str, EMLNode] = {}


def _build_known():
    global KNOWN_COMPILATIONS
    KNOWN_COMPILATIONS = {
        "exp(x)": compile_exp(X),
        "ln(x)": compile_ln(X),
        "e": _eml(ONE, ONE),
        "exp(exp(x))": compile_exp(compile_exp(X)),
    }

_build_known()


def compile_expr(expr: str) -> EMLNode:
    """Compile a standard math expression to an EML tree.

    Supports: exp(x), ln(x), e, exp(exp(x)).
    Raises NotImplementedError for expressions that require deep trees.
    """
    expr = expr.strip()
    if expr in KNOWN_COMPILATIONS:
        return KNOWN_COMPILATIONS[expr]
    raise NotImplementedError(
        f"Cannot compile '{expr}'. Supported: {list(KNOWN_COMPILATIONS.keys())}. "
        f"For other expressions, use search() to find the EML form."
    )


def decompile(node: EMLNode) -> str:
    """Best-effort simplification of an EML tree to readable math."""
    s = str(node)
    # Apply pattern-matching simplifications
    s = s.replace("eml(x, 1)", "exp(x)")
    s = s.replace("eml(1, 1)", "e")
    if "eml(1, eml(eml(1, x), 1))" in s:
        s = s.replace("eml(1, eml(eml(1, x), 1))", "ln(x)")
    return s
