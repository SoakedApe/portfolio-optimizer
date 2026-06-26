# portfolio-optimizer

Portfolio construction toolkit: Markowitz mean-variance optimisation versus a Calmar-optimised grid search. Applied to 10 systematic trading strategies.

## Methods

**Markowitz (mean-variance):** Finds the efficient frontier by minimising portfolio variance for a given expected return. Maximises Sharpe ratio via scipy optimisation.

**Grid search (Calmar-optimised):** Sweeps all weight combinations in 10% increments, scoring each portfolio by Calmar ratio (annual return ÷ max drawdown). This is the method used in the production ICM strategy portfolio.

## Key result

Markowitz maximised Sharpe but concentrated ~60% in two strategies. Grid search spread allocation across 7 strategies, reducing MaxDD from ~12% to 7.49% at the cost of ~0.5 Sharpe points. We used grid search in production — a lower max drawdown is more important for a funded trading account than a marginally higher Sharpe.

## Quick start

```python
import pandas as pd
from optimizer.markowitz import max_sharpe_weights, efficient_frontier
from optimizer.grid_search import grid_search

returns = pd.read_csv("data/strategy_returns.csv", index_col=0, parse_dates=True)

# Markowitz max-Sharpe
weights = max_sharpe_weights(returns)
print(weights)

# Grid search (Calmar-optimised, 10% step)
result = grid_search(returns, step=0.10, score_fn="calmar")
print(result["weights"])
print(result["metrics"])
```

## Data

`data/strategy_returns.csv` — 1512 trading days of synthetic daily returns for 10 strategies (Strategy_A through Strategy_J). Statistical properties (mean, volatility, correlation structure) are calibrated to the real production strategies without revealing the actual return series.

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

5 tests: Sharpe sign, max drawdown sign, weighted portfolio return, efficient frontier structure, max-Sharpe weights sum to 1.
