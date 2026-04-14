"""Tests for the core EML operator."""

import cmath
import math

import pytest
import torch

from pyeml._operator import eml, eml_scalar, DTYPE


class TestEmlScalar:
    """Test the pure-Python scalar eml."""

    def test_eml_1_1_equals_e(self):
        """eml(1, 1) = exp(1) - ln(1) = e."""
        assert abs(eml_scalar(1, 1) - cmath.e) < 1e-14

    def test_exp_x(self):
        """eml(x, 1) = exp(x) for any x."""
        for x in [0.5, 1.0, 2.0, -1.0, 3.14]:
            assert abs(eml_scalar(x, 1) - cmath.exp(x)) < 1e-12

    def test_ln_identity(self):
        """ln(x) = eml(1, eml(eml(1, x), 1)) — the paper's formula (eq. 5)."""
        for x in [0.5, 1.0, 2.0, 4.0, 0.1]:
            inner = eml_scalar(1, x)          # exp(1) - ln(x) = e - ln(x)
            middle = eml_scalar(inner, 1)     # exp(e - ln(x)) - ln(1) = e^(e-ln(x))
            outer = eml_scalar(1, middle)     # exp(1) - ln(e^(e-ln(x))) = e - (e - ln(x)) = ln(x)
            assert abs(outer - cmath.log(x)) < 1e-12

    def test_e_to_the_e(self):
        """eml(eml(1,1), 1) = e^e."""
        e = eml_scalar(1, 1)
        ee = eml_scalar(e, 1)
        assert abs(ee - cmath.exp(cmath.e)) < 1e-10

    def test_complex_emergence(self):
        """Negative intermediates produce complex results with Im = pi."""
        # eml(1, e^(e^e)) = e - e^e ≈ -12.4 (negative)
        e = eml_scalar(1, 1)
        ee = eml_scalar(e, 1)
        eee = eml_scalar(ee, 1)
        neg = eml_scalar(1, eee)                # e - e^e, negative
        assert neg.real < 0

        cpx = eml_scalar(1, neg)                # e - ln(negative) → complex
        assert abs(cpx.imag + math.pi) < 1e-10  # Im = -pi


class TestEmlTensor:
    """Test the PyTorch tensor eml."""

    def test_basic(self):
        x = torch.tensor([1.0 + 0j], dtype=DTYPE)
        y = torch.tensor([1.0 + 0j], dtype=DTYPE)
        result = eml(x, y)
        assert abs(result.item().real - math.e) < 1e-14

    def test_batch(self):
        """eml should work on batches."""
        x = torch.tensor([0.5, 1.0, 2.0], dtype=DTYPE)
        ones = torch.ones(3, dtype=DTYPE)
        result = eml(x, ones)
        for i, xi in enumerate([0.5, 1.0, 2.0]):
            assert abs(result[i].item().real - math.exp(xi)) < 1e-12

    def test_grad_flows(self):
        """Gradients should flow through eml."""
        x = torch.tensor([1.0 + 0j], dtype=DTYPE, requires_grad=True)
        y = torch.tensor([2.0 + 0j], dtype=DTYPE, requires_grad=True)
        result = eml(x, y)
        result.real.sum().backward()
        assert x.grad is not None
        assert y.grad is not None

    def test_exp_clamp_no_overflow(self):
        """Large inputs should not produce inf thanks to clamping."""
        x = torch.tensor([1000.0 + 0j], dtype=DTYPE)
        y = torch.tensor([1.0 + 0j], dtype=DTYPE)
        result = eml(x, y)
        assert not torch.isinf(result.real).any()
