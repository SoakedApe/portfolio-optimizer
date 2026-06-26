import numpy as np
import pandas as pd
from scipy.optimize import minimize
from optimizer.metrics import sharpe, portfolio_return


def efficient_frontier(returns: pd.DataFrame, n_portfolios: int = 2000) -> list[dict]:
    """Monte Carlo efficient frontier — random weight sampling."""
    rng  = np.random.default_rng(42)
    cols = list(returns.columns)
    n    = len(cols)

    results = []
    for _ in range(n_portfolios):
        w       = rng.dirichlet(np.ones(n))
        weights = dict(zip(cols, w))
        port    = portfolio_return(returns, weights)
        results.append({
            "weights": weights,
            "sharpe":  sharpe(port),
            "vol":     float(port.std() * np.sqrt(252)),
            "ret":     float(port.mean() * 252),
        })

    return sorted(results, key=lambda x: x["sharpe"], reverse=True)


def max_sharpe_weights(returns: pd.DataFrame) -> dict:
    """Find weights that maximise Sharpe via scipy optimisation."""
    cols = list(returns.columns)
    n    = len(cols)
    mu   = returns.mean().values
    cov  = returns.cov().values

    def neg_sharpe(w):
        port_ret = w @ mu * 252
        port_vol = np.sqrt(w @ cov @ w * 252)
        return -port_ret / port_vol if port_vol > 0 else 0.0

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, 1)] * n
    x0          = np.ones(n) / n
    result      = minimize(neg_sharpe, x0, bounds=bounds, constraints=constraints,
                           method="SLSQP")

    return dict(zip(cols, result.x))


def min_variance_weights(returns: pd.DataFrame) -> dict:
    """Find minimum variance portfolio."""
    cols = list(returns.columns)
    n    = len(cols)
    cov  = returns.cov().values

    def port_variance(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, 1)] * n
    x0          = np.ones(n) / n
    result      = minimize(port_variance, x0, bounds=bounds, constraints=constraints,
                           method="SLSQP")

    return dict(zip(cols, result.x))
