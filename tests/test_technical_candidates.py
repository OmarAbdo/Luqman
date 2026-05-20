import numpy as np
import pandas as pd

from luqman.features.technical.candidates import (
    _swing_points,
    future_realized_vol,
    macd_divergence,
    momentum_12_1,
    realized_vol,
    reversal_1d,
    volume_breakout,
)


def test_reversal_1d_is_negated_return():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))
    feat = reversal_1d(close)
    assert np.isclose(feat.iloc[1], -np.log(110 / 100))
    assert np.isclose(feat.iloc[2], -np.log(99 / 110))


def test_momentum_12_1_uses_correct_window():
    close = pd.Series(np.linspace(100, 200, 300), index=pd.bdate_range("2023-01-01", periods=300))
    feat = momentum_12_1(close)
    # at row 300 (0-indexed 299): shift(21) and shift(252)
    expected = np.log(close.iloc[299 - 21] / close.iloc[299 - 252])
    assert np.isclose(feat.iloc[299], expected)
    assert feat.iloc[:252].isna().all()


def test_realized_vol_matches_manual_std():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))), index=pd.bdate_range("2023-01-01", periods=100))
    feat = realized_vol(close, window=20)
    logret = np.log(close / close.shift(1))
    expected = logret.rolling(20).std()
    pd.testing.assert_series_equal(feat, expected, check_names=False)
    assert (feat.dropna() >= 0).all()


def test_future_realized_vol_no_lookahead_shape():
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))), index=pd.bdate_range("2023-01-01", periods=100))
    target = future_realized_vol(close, horizon=5, window=5)
    # last 9 rows (horizon + window - 1) cannot have a fully-future window
    assert target.iloc[-9:].isna().all()
    assert target.iloc[:-9].notna().any()


def test_swing_points_detects_local_extrema():
    values = pd.Series([1, 2, 5, 2, 1, 1, 2, 6, 2, 1], dtype=float)
    swings = _swing_points(values, order=2)
    assert swings.iloc[2] == 1  # peak at value 5
    assert swings.iloc[7] == 1  # peak at value 6


def test_macd_divergence_output_is_bounded_and_named():
    rng = np.random.default_rng(2)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 300))), index=pd.bdate_range("2023-01-01", periods=300))
    feat = macd_divergence(close, order=3)
    assert feat.name == "macd_divergence"
    assert feat.isin([-1.0, 0.0, 1.0]).all()
    assert not feat.isna().any()  # ffilled + zero-filled, no NaNs leak through


def test_macd_divergence_bearish_case_detected():
    # Construct a clean higher-high in price with a lower MACD high: strong
    # first rally, weaker second rally to a higher price level.
    n = 60
    idx = pd.bdate_range("2023-01-01", periods=n)
    t = np.arange(n)
    # First swing high at t=15 (strong rally), second swing high at t=45
    # (higher price, weaker momentum) via a flatter approach.
    close = pd.Series(100.0, index=idx)
    close.iloc[:16] = 100 + np.linspace(0, 20, 16)          # strong rally to 120
    close.iloc[16:31] = np.linspace(120, 105, 15)            # pullback
    close.iloc[31:46] = 105 + np.linspace(0, 21, 15)         # weaker rally to a higher high (126) but slower
    close.iloc[46:] = np.linspace(126, 110, n - 46)
    feat = macd_divergence(close, order=3)
    assert (feat == -1.0).any()


def test_volume_breakout_positive_on_up_move_high_volume():
    idx = pd.bdate_range("2023-01-01", periods=40)
    close = pd.Series(100.0, index=idx)
    close.iloc[25:] = 100 * 1.05  # jump
    volume = pd.Series(1_000_000.0, index=idx)
    volume.iloc[25] = 5_000_000.0
    feat = volume_breakout(close, volume, window=20)
    assert feat.iloc[25] > 0


def test_volume_breakout_zero_when_volume_at_baseline():
    idx = pd.bdate_range("2023-01-01", periods=45)
    rng = np.random.default_rng(3)
    close = pd.Series(100.0, index=idx)
    close.iloc[25] = 105.0
    # Volume varies slightly so std > 0, but day 25's volume sits exactly at
    # the trailing mean -> vol_z = 0 -> feature must be exactly 0 regardless
    # of the price move's z-score.
    volume = pd.Series(1_000_000.0 + rng.normal(0, 1000, 45), index=idx)
    volume.iloc[25] = volume.iloc[5:25].mean()
    feat = volume_breakout(close, volume, window=20)
    assert np.isclose(feat.iloc[25], 0.0)
