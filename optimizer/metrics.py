import numpy as np
import pandas as pd


def sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    excess = returns - risk_free / 252
    return float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float(((equity - peak) / peak).min())


def calmar(equity: pd.Series) -> float:
    ann = annualised_return(equity)
    mdd = abs(max_drawdown(equity))
    return ann / mdd if mdd > 0 else 0.0


def annualised_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0


def portfolio_return(returns: pd.DataFrame, weights: dict) -> pd.Series:
    cols = [c for c in weights if c in returns.columns]
    w    = np.array([weights[c] for c in cols])
    return (returns[cols] * w).sum(axis=1)
