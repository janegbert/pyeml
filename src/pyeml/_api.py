"""Public API for pyeml: discover() and EMLRegressor."""

from __future__ import annotations

import numpy as np
import torch

from pyeml._config import TrainConfig, SearchConfig
from pyeml._operator import DTYPE
from pyeml._trainer import search, SearchResult
from pyeml._tree import EMLTree


def discover(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    *,
    max_depth: int = 4,
    n_restarts: int = 10,
    verbose: bool = True,
    x_range: tuple[float, float] | None = None,
) -> SearchResult | None:
    """Discover an EML expression that fits y = f(x).

    Args:
        x: Input values (1D array).
        y: Target values (1D array, same length as x).
        max_depth: Maximum tree depth to search (1 through max_depth).
        n_restarts: Random restarts per depth level.
        verbose: Print progress.
        x_range: Override x range for training data generation.
            If None, uses the range of the provided x data.

    Returns:
        SearchResult with the best model, expression, and loss.
        None if all attempts diverged.
    """
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.complex128)

    config = SearchConfig(
        depths=list(range(1, max_depth + 1)),
        n_restarts=n_restarts,
        verbose=verbose,
        x_range=x_range or (float(x_np.min()), float(x_np.max())),
    )

    x_t = torch.tensor(x_np, dtype=DTYPE)
    y_t = torch.tensor(y_np, dtype=DTYPE)

    return search(x_t, y_t, config)


class EMLRegressor:
    """Sklearn-style EML symbolic regressor.

    Example:
        reg = EMLRegressor(max_depth=3)
        reg.fit(x, y)
        print(reg.expression)       # "exp(x)"
        y_pred = reg.predict(x_new)
    """

    def __init__(
        self,
        max_depth: int = 4,
        n_restarts: int = 10,
        verbose: bool = True,
        train_config: TrainConfig | None = None,
    ):
        self.max_depth = max_depth
        self.n_restarts = n_restarts
        self.verbose = verbose
        self.train_config = train_config or TrainConfig()
        self._result: SearchResult | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> EMLRegressor:
        """Fit to data. X must be 1D or shape (n, 1)."""
        x = np.asarray(X, dtype=np.float64).ravel()
        y = np.asarray(y, dtype=np.complex128).ravel()

        config = SearchConfig(
            depths=list(range(1, self.max_depth + 1)),
            n_restarts=self.n_restarts,
            verbose=self.verbose,
            train=self.train_config,
            x_range=(float(x.min()), float(x.max())),
        )

        x_t = torch.tensor(x, dtype=DTYPE)
        y_t = torch.tensor(y, dtype=DTYPE)

        self._result = search(x_t, y_t, config)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the discovered expression."""
        if self._result is None:
            raise RuntimeError("Call fit() first.")

        x = np.asarray(X, dtype=np.float64).ravel()
        x_t = torch.tensor(x, dtype=DTYPE)

        with torch.no_grad():
            y_t = self._result.model(x_t)

        return y_t.real.numpy()

    @property
    def expression(self) -> str:
        """The simplified discovered expression."""
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result.expression

    @property
    def eml_expression(self) -> str:
        """The raw EML expression."""
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result.eml_expression

    @property
    def is_exact(self) -> bool:
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result.is_exact

    def get_tree(self) -> EMLTree:
        """Access the underlying PyTorch model."""
        if self._result is None:
            raise RuntimeError("Call fit() first.")
        return self._result.model
