"""
pyeml — All elementary functions from a single binary operator.

Based on Odrzywołek (2026), arXiv:2603.21852.
The EML operator eml(x,y) = exp(x) - ln(y), together with the constant 1,
can express every elementary function: sin, cos, sqrt, log, +, -, *, /, etc.

Usage:
    from pyeml import discover
    result = discover(x_data, y_data)
    print(result.expression)
"""

__version__ = "0.1.0"

from pyeml._operator import eml, eml_scalar
from pyeml._tree import EMLTree
from pyeml._config import TrainConfig, SearchConfig
from pyeml._trainer import SearchResult, search
from pyeml._api import discover, EMLRegressor
from pyeml._compiler import compile_expr, decompile, EMLNode

__all__ = [
    "eml",
    "eml_scalar",
    "EMLTree",
    "TrainConfig",
    "SearchConfig",
    "SearchResult",
    "search",
    "discover",
    "EMLRegressor",
    "compile_expr",
    "decompile",
    "EMLNode",
]
