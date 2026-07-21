"""Costed long-short portfolio backtest.

Equal-weights the top and bottom quantiles of a factor ranking and charges
costs on portfolio turnover -- the names that actually changed -- rather than
on full notional each day.

A long-short book is market-neutral by construction, so it has no natural
buy-and-hold benchmark. The relevant comparison is against doing nothing, which
is reported explicitly.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class CrossSectionalBacktestResult:
    metrics: dict[str, float]
    daily_returns: pd.Series = field(repr=False)


def backtest_long_short(
    factor_panel: pd.DataFrame,
    forward_return_panel: pd.DataFrame,
    quantile: float = 0.2,
    cost_bps: float = 10.0,
    min_names_per_day: int = 10,
) -> CrossSectionalBacktestResult:
    """Costed long-short decile portfolio backtest: equal-weight long the top
    `quantile` fraction and short the bottom `quantile` fraction, ranked fresh
    each day, holding for the horizon implicit in `forward_return_panel`
    (whatever horizon the caller computed it at). Costs are charged on
    portfolio TURNOVER (fraction of names that entered/exited a leg day to
    day), not on the whole notional every day, since a real long-short
    momentum strategy only pays costs on the names it actually trades.

    A long-short portfolio is market-neutral by construction (long leg minus
    short leg), so there is no natural "buy and hold" benchmark the way a
    single-name backtest has one — the zero-return baseline (do nothing) is
    the relevant comparison, made explicit in the metrics.
    """
    common_dates = factor_panel.index.intersection(forward_return_panel.index)
    factor_panel = factor_panel.loc[common_dates]
    forward_return_panel = forward_return_panel.loc[common_dates]

    daily_returns = []
    return_dates = []
    prev_long, prev_short = set(), set()
    cost = cost_bps / 10_000.0

    for date in common_dates:
        factors = factor_panel.loc[date].dropna()
        returns = forward_return_panel.loc[date].reindex(factors.index).dropna()
        factors = factors.reindex(returns.index)
        if len(factors) < min_names_per_day:
            continue

        ranked = factors.sort_values()
        n = len(ranked)
        cut = max(1, int(n * quantile))
        short_names = set(ranked.index[:cut])
        long_names = set(ranked.index[-cut:])

        long_ret = returns.loc[list(long_names)].mean()
        short_ret = returns.loc[list(short_names)].mean()
        spread = long_ret - short_ret

        # Turnover: fraction of the (long + short) book that changed today,
        # relative to book size — charged once per side that changed.
        turnover = (
            len(long_names.symmetric_difference(prev_long)) + len(short_names.symmetric_difference(prev_short))
        ) / (2 * cut)
        daily_returns.append(spread - turnover * cost)
        return_dates.append(date)
        prev_long, prev_short = long_names, short_names

    returns_arr = np.array(daily_returns)
    total = float(np.sum(returns_arr))
    std = float(np.std(returns_arr, ddof=1)) if len(returns_arr) > 1 else 0.0
    sharpe = float(np.mean(returns_arr) / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan")
    cum = np.cumsum(returns_arr)
    drawdown = cum - np.maximum.accumulate(cum)

    metrics = {
        "total_return": total,
        "annualized_return": total / len(returns_arr) * TRADING_DAYS if len(returns_arr) else float("nan"),
        "annualized_sharpe": sharpe,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else float("nan"),
        "n_periods": int(len(returns_arr)),
        "zero_baseline_return": 0.0,
    }
    return CrossSectionalBacktestResult(
        metrics=metrics,
        daily_returns=pd.Series(daily_returns, index=return_dates),
    )
