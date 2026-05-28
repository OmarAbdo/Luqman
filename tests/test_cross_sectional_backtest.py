import numpy as np
import pandas as pd

from luqman.backtest.cross_sectional import backtest_long_short


def _panel(values: dict, dates) -> pd.DataFrame:
    return pd.DataFrame(values, index=dates)


def test_stable_ranking_has_zero_turnover_after_first_day():
    dates = pd.bdate_range("2022-01-03", periods=30)
    tickers = [f"T{i}" for i in range(20)]
    # Same rank every day -> long/short sets never change after day 1.
    factor_panel = _panel({t: np.full(30, i) for i, t in enumerate(tickers)}, dates)
    return_panel = _panel({t: np.full(30, 0.001 * i) for i, t in enumerate(tickers)}, dates)

    result = backtest_long_short(factor_panel, return_panel, quantile=0.2, cost_bps=10)
    # After day 1, no turnover cost should be charged -> daily spread return
    # should be identical every day past the first.
    assert np.allclose(result.daily_returns.iloc[1:], result.daily_returns.iloc[1])


def test_costs_reduce_return_when_ranking_changes_every_day():
    dates = pd.bdate_range("2022-01-03", periods=30)
    rng = np.random.default_rng(0)
    tickers = [f"T{i}" for i in range(20)]
    factor_panel = _panel({t: rng.normal(size=30) for t in tickers}, dates)  # re-ranked randomly each day
    return_panel = _panel({t: rng.normal(scale=0.01, size=30) for t in tickers}, dates)

    free = backtest_long_short(factor_panel, return_panel, cost_bps=0)
    costed = backtest_long_short(factor_panel, return_panel, cost_bps=50)
    assert costed.metrics["total_return"] < free.metrics["total_return"]


def test_perfect_factor_produces_positive_sharpe():
    dates = pd.bdate_range("2022-01-03", periods=200)
    rng = np.random.default_rng(1)
    n = 20
    tickers = [f"T{i}" for i in range(n)]
    factor_data, return_data = {}, {}
    for i, t in enumerate(tickers):
        rank = i / n
        factor_data[t] = rank + rng.normal(0, 0.01, 200)
        return_data[t] = rank * 0.005 + rng.normal(0, 0.0005, 200)
    result = backtest_long_short(_panel(factor_data, dates), _panel(return_data, dates), cost_bps=5)
    assert result.metrics["annualized_sharpe"] > 1.0
    assert result.metrics["total_return"] > 0


def test_days_below_min_names_are_skipped():
    dates = pd.bdate_range("2022-01-03", periods=10)
    tickers = [f"T{i}" for i in range(5)]
    factor_panel = _panel({t: np.arange(10, dtype=float) for t in tickers}, dates)
    return_panel = _panel({t: np.arange(10, dtype=float) * 0.001 for t in tickers}, dates)
    result = backtest_long_short(factor_panel, return_panel, min_names_per_day=10)
    assert result.metrics["n_periods"] == 0
    assert np.isnan(result.metrics["annualized_sharpe"])


def test_zero_baseline_always_zero():
    dates = pd.bdate_range("2022-01-03", periods=30)
    tickers = [f"T{i}" for i in range(20)]
    factor_panel = _panel({t: np.full(30, i) for i, t in enumerate(tickers)}, dates)
    return_panel = _panel({t: np.zeros(30) for t in tickers}, dates)
    result = backtest_long_short(factor_panel, return_panel, cost_bps=0)
    assert result.metrics["zero_baseline_return"] == 0.0
