import numpy as np
import pandas as pd

from luqman.evaluation.cross_sectional import evaluate_cross_sectional_factor


def _panel(values: dict, dates) -> pd.DataFrame:
    return pd.DataFrame(values, index=dates)


def test_perfect_factor_produces_strong_positive_spread():
    dates = pd.bdate_range("2022-01-03", periods=100)
    rng = np.random.default_rng(0)
    n_tickers = 20
    tickers = [f"T{i}" for i in range(n_tickers)]

    factor_data = {}
    return_data = {}
    for t_idx, ticker in enumerate(tickers):
        base_rank = t_idx / n_tickers  # 0..1, deterministic rank per ticker
        factor_data[ticker] = base_rank + rng.normal(0, 0.01, len(dates))
        # Higher-ranked tickers get higher returns -> real factor relationship
        return_data[ticker] = base_rank * 0.01 + rng.normal(0, 0.001, len(dates))

    factor_panel = _panel(factor_data, dates)
    return_panel = _panel(return_data, dates)

    result = evaluate_cross_sectional_factor(factor_panel, return_panel, "perfect_factor", quantile=0.2)
    assert result.verdict == "ADMITTED"
    assert result.annualized_sharpe > 1.0
    assert result.ic_mean > 0.5
    assert result.ic_positive_rate > 0.9


def test_pure_noise_factor_is_rejected():
    dates = pd.bdate_range("2022-01-03", periods=200)
    rng = np.random.default_rng(1)
    tickers = [f"T{i}" for i in range(20)]

    factor_panel = _panel({t: rng.normal(size=len(dates)) for t in tickers}, dates)
    return_panel = _panel({t: rng.normal(scale=0.01, size=len(dates)) for t in tickers}, dates)

    result = evaluate_cross_sectional_factor(factor_panel, return_panel, "noise_factor", quantile=0.2)
    assert result.verdict == "REJECTED"
    assert abs(result.ic_mean) < 0.15


def test_inverted_factor_produces_negative_spread():
    dates = pd.bdate_range("2022-01-03", periods=100)
    rng = np.random.default_rng(2)
    n_tickers = 20
    tickers = [f"T{i}" for i in range(n_tickers)]

    factor_data, return_data = {}, {}
    for t_idx, ticker in enumerate(tickers):
        base_rank = t_idx / n_tickers
        factor_data[ticker] = base_rank + rng.normal(0, 0.01, len(dates))
        return_data[ticker] = -base_rank * 0.01 + rng.normal(0, 0.001, len(dates))  # inverted

    result = evaluate_cross_sectional_factor(
        _panel(factor_data, dates), _panel(return_data, dates), "inverted", quantile=0.2
    )
    assert result.mean_daily_long_short_return < 0
    assert result.verdict == "REJECTED"


def test_days_with_too_few_names_are_skipped():
    dates = pd.bdate_range("2022-01-03", periods=10)
    tickers = [f"T{i}" for i in range(5)]  # below min_names_per_day default of 10
    factor_panel = _panel({t: np.arange(10, dtype=float) for t in tickers}, dates)
    return_panel = _panel({t: np.arange(10, dtype=float) * 0.001 for t in tickers}, dates)
    result = evaluate_cross_sectional_factor(factor_panel, return_panel, "thin", min_names_per_day=10)
    assert result.n_days == 0
    assert np.isnan(result.mean_daily_long_short_return)


def test_handles_partial_nan_coverage_per_day():
    dates = pd.bdate_range("2022-01-03", periods=50)
    rng = np.random.default_rng(3)
    tickers = [f"T{i}" for i in range(20)]
    factor_panel = _panel({t: rng.normal(size=50) for t in tickers}, dates)
    return_panel = _panel({t: rng.normal(scale=0.01, size=50) for t in tickers}, dates)
    # Blank out half the panel on some days
    factor_panel.iloc[::2, :10] = np.nan
    result = evaluate_cross_sectional_factor(factor_panel, return_panel, "partial", min_names_per_day=10)
    assert result.n_days > 0
