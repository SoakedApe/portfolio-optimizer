"""
Grid search portfolio optimiser — sweeps allocation combinations and
scores by Calmar ratio. This is the method used in production for the
ICM strategy portfolio.
"""
import itertools
import numpy as np
import pandas as pd
from optimizer.metrics import calmar, portfolio_return, max_drawdown, sharpe


def grid_search(
    returns: pd.DataFrame,
    step: float = 0.10,
    score_fn: str = "calmar",
) -> dict:
    """
    Sweep all weight combinations (multiples of `step` summing to 1).
    Returns the best allocation dict + metrics.
    score_fn: 'calmar' | 'sharpe'
    """
    cols  = list(returns.columns)
    n     = len(cols)
    total = int(round(1 / step))
    best  = {"score": -np.inf, "weights": {}, "metrics": {}}

    combos = [c for c in itertools.product(range(total + 1), repeat=n)
              if sum(c) == total]

    for combo in combos:
        weights = {cols[i]: combo[i] * step for i in range(n)}
        port    = portfolio_return(returns, weights)
        equity  = (1 + port).cumprod() * 10_000

        if score_fn == "calmar":
            score = calmar(equity)
        else:
            score = sharpe(port)

        if score > best["score"]:
            best = {
                "score":   score,
                "weights": weights,
                "metrics": {
                    "sharpe":       sharpe(port),
                    "max_drawdown": max_drawdown(equity),
                    "calmar":       calmar(equity),
                }
            }

    return best
