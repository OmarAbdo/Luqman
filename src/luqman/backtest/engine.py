from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    metrics: dict[str, float]
    equity: pd.Series = field(repr=False)


def backtest_signals(
    realized_returns: pd.Series,
    predictions: pd.Series,
    cost_bps: float = 5.0,
    threshold: float = 0.0,
    allow_short: bool = False,
) -> BacktestResult:
    """Vectorized backtest of forecast-driven positions over aligned log returns.

    At each row t the strategy holds long (1) when prediction > threshold,
    short (-1, only if allow_short) when prediction < -threshold, else flat.
    The position earns that row's realized forward log return; every unit of
    position change pays `cost_bps` (per side, on notional).
    """
    aligned = pd.concat([realized_returns, predictions], axis=1, join="inner").dropna()
    returns = aligned.iloc[:, 0].to_numpy()
    preds = aligned.iloc[:, 1].to_numpy()

    position = np.where(preds > threshold, 1.0, 0.0)
    if allow_short:
        position = np.where(preds < -threshold, -1.0, position)

    trades = np.abs(np.diff(position, prepend=0.0))
    cost = cost_bps / 10_000.0
    strategy_returns = position * returns - trades * cost

    equity = pd.Series(np.cumsum(strategy_returns), index=aligned.index, name="equity")
    metrics = _metrics(strategy_returns, trades)
    metrics["buy_hold_return"] = float(np.sum(returns))
    return BacktestResult(metrics=metrics, equity=equity)


def _metrics(strategy_returns: np.ndarray, trades: np.ndarray) -> dict[str, float]:
    total = float(np.sum(strategy_returns))
    std = float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0
    sharpe = (
        float(np.mean(strategy_returns) / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")
    )
    cum = np.cumsum(strategy_returns)
    drawdown = cum - np.maximum.accumulate(cum)
    return {
        "total_return": total,
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(np.mean(trades)),
        "n_trades": int(np.count_nonzero(trades)),
        "n_periods": int(len(strategy_returns)),
    }
