"""Shared fixtures for pyeml tests."""

import cmath
import numpy as np
import pytest
import torch

DTYPE = torch.complex128


@pytest.fixture
def x_real():
    """Real-valued test points as complex128 tensor."""
    return torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], dtype=DTYPE)


@pytest.fixture
def x_np():
    """Numpy array of test points."""
    return np.linspace(0.5, 3.0, 50)


# Known EML identities from the paper
KNOWN_IDENTITIES = {
    "e": (lambda: cmath.exp(1) - cmath.log(1), cmath.e),
    "e^e": (lambda: cmath.exp(cmath.e) - cmath.log(1), cmath.exp(cmath.e)),
    "ln(2)": (
        lambda: cmath.exp(1) - cmath.log(cmath.exp(cmath.exp(1) - cmath.log(2)) - cmath.log(1)),
        cmath.log(2),
    ),
}
