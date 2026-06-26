import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest
from optimizer.metrics import sharpe, max_drawdown, portfolio_return
from optimizer.markowitz import efficient_frontier, max_sharpe_weights


def sample_returns(n=252, n_assets=3, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0004, 0.01, (n, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_sharpe_positive_for_positive_returns():
    returns = pd.Series([0.001] * 252)
    assert sharpe(returns) > 0


def test_max_drawdown_is_negative():
    equity = pd.Series([100, 110, 90, 95, 105])
    assert max_drawdown(equity) < 0


def test_portfolio_return_weighted_sum():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [-0.01, 0.01]})
    weights = {"A": 0.5, "B": 0.5}
    port = portfolio_return(returns, weights)
    assert abs(port.iloc[0] - 0.0) < 1e-9
    assert abs(port.iloc[1] - 0.015) < 1e-9


def test_efficient_frontier_returns_list_of_dicts():
    returns = sample_returns()
    frontier = efficient_frontier(returns, n_portfolios=100)
    assert len(frontier) == 100
    assert "sharpe" in frontier[0]
    assert "weights" in frontier[0]


def test_max_sharpe_weights_sum_to_1():
    returns = sample_returns()
    weights = max_sharpe_weights(returns)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
