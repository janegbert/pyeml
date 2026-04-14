"""Tests for the public API."""

import numpy as np
import pytest

from pyeml import discover, EMLRegressor


class TestDiscover:
    def test_discover_exp(self):
        x = np.linspace(0.1, 2.0, 100)
        y = np.exp(x)
        result = discover(x, y, max_depth=1, n_restarts=5, verbose=False)
        assert result is not None
        assert result.is_exact
        assert "exp" in result.expression

    def test_discover_returns_none_on_impossible(self):
        """Random noise should not produce an exact match at depth 1."""
        np.random.seed(42)
        x = np.linspace(0.5, 3.0, 100)
        y = np.random.randn(100)
        result = discover(x, y, max_depth=1, n_restarts=2, verbose=False)
        if result is not None:
            assert not result.is_exact


class TestEMLRegressor:
    def test_fit_predict(self):
        x = np.linspace(0.1, 2.0, 100)
        y = np.exp(x)

        reg = EMLRegressor(max_depth=1, n_restarts=5, verbose=False)
        reg.fit(x, y)

        assert reg.is_exact
        assert "exp" in reg.expression

        y_pred = reg.predict(x)
        assert y_pred.shape == y.shape
        assert np.allclose(y_pred, y, atol=1e-6)

    def test_predict_before_fit_raises(self):
        reg = EMLRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            reg.predict(np.array([1.0]))

    def test_expression_before_fit_raises(self):
        reg = EMLRegressor()
        with pytest.raises(RuntimeError, match="fit"):
            _ = reg.expression
