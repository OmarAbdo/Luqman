import numpy as np
import pandas as pd

TRADING_DAYS = 252


def vol_target_positions(
    forecast_vol: pd.Series,
    target_vol: float,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Position size (in units of 'fully invested') for a vol-targeting
    strategy: size = target_vol / forecast_vol, capped at `max_leverage` and
    floored at 0 (never negative — this is a sizing decision, not a directional
    bet). `forecast_vol` and `target_vol` must be on the same scale (daily
    return std, not annualized) — the caller passes whichever forecast is
    under test.
    """
    size = target_vol / forecast_vol.replace(0.0, np.nan)
    return size.clip(lower=0.0, upper=max_leverage)


def backtest_vol_targeting(
    realized_returns: pd.Series,
    forecast_vol: pd.Series,
    target_vol: float,
    cost_bps: float = 5.0,
    max_leverage: float = 3.0,
) -> dict:
    """Backtest a vol-targeted long-only position (size = target/forecast,
    capped) against realized returns, with rebalancing costs whenever size
    changes. This measures whether a volatility FORECAST translates into
    better risk control than a fixed position — the practical value of Track A
    — not directional PnL (see dataset-plan.md: this feature predicts
    magnitude, not direction, so a directional backtest would be the wrong
    test).

    Returns: realized annualized vol of the sized strategy (how close to
    target_vol did it land), total return, Sharpe, and turnover/cost drag —
    compared against a fixed-size (always size=1) baseline over the same
    return series.
    """
    aligned = pd.concat([realized_returns, forecast_vol], axis=1, join="inner").dropna()
    returns = aligned.iloc[:, 0].to_numpy()
    vol_fc = aligned.iloc[:, 1].to_numpy()

    size = np.clip(target_vol / np.where(vol_fc == 0, np.nan, vol_fc), 0.0, max_leverage)
    size = np.nan_to_num(size, nan=0.0)
    trades = np.abs(np.diff(size, prepend=1.0))  # starts fully invested
    cost = cost_bps / 10_000.0
    sized_returns = size * returns - trades * cost

    fixed_returns = returns  # size == 1 always, no rebalancing cost

    def _annualized_vol(r):
        return float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS))

    def _sharpe(r):
        std = np.std(r, ddof=1)
        return float(np.mean(r) / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")

    return {
        "target_vol_annualized": float(target_vol * np.sqrt(TRADING_DAYS)),
        "sized_realized_vol_annualized": _annualized_vol(sized_returns),
        "fixed_realized_vol_annualized": _annualized_vol(fixed_returns),
        "sized_total_return": float(np.sum(sized_returns)),
        "fixed_total_return": float(np.sum(fixed_returns)),
        "sized_sharpe": _sharpe(sized_returns),
        "fixed_sharpe": _sharpe(fixed_returns),
        "mean_position_size": float(np.mean(size)),
        "turnover": float(np.mean(trades)),
        "n_periods": int(len(returns)),
    }
