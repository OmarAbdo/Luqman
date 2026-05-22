import numpy as np
import pandas as pd
import pytest

from luqman.backtest.vol_sizing import backtest_vol_targeting, vol_target_positions


def test_vol_target_positions_scales_inversely_to_forecast():
    forecast = pd.Series([0.01, 0.02, 0.005])
    sizes = vol_target_positions(forecast, target_vol=0.01, max_leverage=10.0)
    assert np.isclose(sizes.iloc[0], 1.0)
    assert np.isclose(sizes.iloc[1], 0.5)
    assert np.isclose(sizes.iloc[2], 2.0)


def test_vol_target_positions_capped_at_max_leverage():
    forecast = pd.Series([0.0001])  # would imply huge leverage
    sizes = vol_target_positions(forecast, target_vol=0.01, max_leverage=3.0)
    assert sizes.iloc[0] == 3.0


def test_vol_target_positions_never_negative():
    forecast = pd.Series([0.0, -0.01, 0.01])  # pathological/invalid forecasts
    sizes = vol_target_positions(forecast, target_vol=0.01, max_leverage=3.0)
    assert (sizes.fillna(0.0) >= 0).all()


def test_backtest_constant_forecast_matches_fixed_leverage():
    """A constant forecast equal to target_vol should keep size pinned at 1.0
    for the whole series after the initial (no-trade) period — sized and
    fixed returns should coincide (net of the zero rebalancing cost this
    implies), confirming the sizing math is correct end to end."""
    idx = pd.bdate_range("2023-01-02", periods=50)
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 50), index=idx)
    forecast = pd.Series(0.01, index=idx)  # exactly target_vol every day
    result = backtest_vol_targeting(returns, forecast, target_vol=0.01, cost_bps=0.0)
    assert np.isclose(result["mean_position_size"], 1.0)
    assert np.isclose(result["sized_total_return"], result["fixed_total_return"])
    assert result["turnover"] == 0.0


def test_backtest_reduces_realized_vol_when_forecast_is_informative():
    """A forecast that correctly anticipates a high-vol regime should let
    sizing shrink exposure going into it, reducing realized vol relative to a
    fixed-size position over the same returns."""
    idx = pd.bdate_range("2023-01-02", periods=200)
    rng = np.random.default_rng(1)
    # Regime switch: calm for 100 days, volatile for 100 days.
    calm = rng.normal(0, 0.005, 100)
    volatile = rng.normal(0, 0.03, 100)
    returns = pd.Series(np.concatenate([calm, volatile]), index=idx)
    # Forecast "knows" the regime one day ahead (perfect foresight, for the test).
    forecast = pd.Series(np.concatenate([np.full(100, 0.005), np.full(100, 0.03)]), index=idx)

    result = backtest_vol_targeting(returns, forecast, target_vol=0.01, cost_bps=1.0)
    assert result["sized_realized_vol_annualized"] < result["fixed_realized_vol_annualized"]


def test_backtest_misaligned_indices_intersected():
    idx = pd.bdate_range("2023-01-02", periods=20)
    returns = pd.Series(0.01, index=idx)
    forecast = pd.Series(0.01, index=idx[5:])
    result = backtest_vol_targeting(returns, forecast, target_vol=0.01)
    assert result["n_periods"] == 15
