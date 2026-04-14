"""Core EML operator: eml(x, y) = exp(x) - ln(y)."""

import cmath
import torch

DTYPE = torch.complex128
EXP_CLAMP = 500.0


def eml(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """EML operator over complex128 tensors: exp(x) - ln(y).

    Clamps the real part of x to [-500, 500] to prevent overflow in exp().
    """
    x_clamped = torch.complex(
        x.real.clamp(-EXP_CLAMP, EXP_CLAMP),
        x.imag,
    )
    return torch.exp(x_clamped) - torch.log(y)


def eml_scalar(x: complex, y: complex) -> complex:
    """Pure-Python scalar EML for verification: exp(x) - ln(y)."""
    return cmath.exp(x) - cmath.log(y)
