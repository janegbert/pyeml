"""Tests for the EML compiler."""

import cmath

import pytest

from pyeml._compiler import compile_expr, decompile, compile_exp, compile_ln, EMLNode, X, ONE
from pyeml._operator import eml_scalar


class TestCompileKnown:
    def test_compile_exp(self):
        node = compile_expr("exp(x)")
        assert str(node) == "eml(x, 1)"
        assert node.depth == 1
        assert node.leaf_count == 2

    def test_compile_ln(self):
        node = compile_expr("ln(x)")
        assert str(node) == "eml(1, eml(eml(1, x), 1))"
        assert node.depth == 3
        assert node.leaf_count == 4

    def test_compile_e(self):
        node = compile_expr("e")
        assert str(node) == "eml(1, 1)"
        assert node.depth == 1

    def test_compile_exp_exp(self):
        node = compile_expr("exp(exp(x))")
        assert node.depth == 2

    def test_unsupported_raises(self):
        with pytest.raises(NotImplementedError):
            compile_expr("sin(x)")


class TestDecompile:
    def test_decompile_exp(self):
        node = compile_expr("exp(x)")
        assert decompile(node) == "exp(x)"

    def test_decompile_e(self):
        node = compile_expr("e")
        assert decompile(node) == "e"

    def test_decompile_ln(self):
        node = compile_expr("ln(x)")
        assert decompile(node) == "ln(x)"


class TestNumericalVerification:
    """Verify compiled trees compute correctly."""

    def _eval_node(self, node: EMLNode, x_val: complex) -> complex:
        """Evaluate an EML node tree numerically."""
        if node.kind == "const":
            return complex(node.value)
        if node.kind == "var":
            return complex(x_val)
        left = self._eval_node(node.left, x_val)
        right = self._eval_node(node.right, x_val)
        return eml_scalar(left, right)

    def test_exp_numerical(self):
        node = compile_expr("exp(x)")
        for x in [0.5, 1.0, 2.0, -1.0]:
            assert abs(self._eval_node(node, x) - cmath.exp(x)) < 1e-12

    def test_ln_numerical(self):
        node = compile_expr("ln(x)")
        for x in [0.5, 1.0, 2.0, 4.0]:
            assert abs(self._eval_node(node, x) - cmath.log(x)) < 1e-12

    def test_e_numerical(self):
        node = compile_expr("e")
        assert abs(self._eval_node(node, 999) - cmath.e) < 1e-12
