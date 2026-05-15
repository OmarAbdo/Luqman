import numpy as np
import pandas as pd

from luqman.features.cross_asset import crypto_risk_appetite, gold_equity_divergence, oil_shock


def _prices(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)


def test_gold_equity_divergence_positive_on_flight_to_safety_day():
    idx = pd.bdate_range("2023-01-02", periods=10)
    gold = pd.Series(100.0, index=idx)
    spy = pd.Series(100.0, index=idx)
    gold.iloc[5] = 103.0  # gold jumps
    spy.iloc[5] = 97.0  # equities drop same day
    feat = gold_equity_divergence(gold, spy, idx)
    assert feat.iloc[6] > 0  # signal appears the NEXT day (lagged)
    assert np.isnan(feat.iloc[5]) or feat.iloc[5] == 0  # not leaked into the event day itself


def test_gold_equity_divergence_is_lagged_by_one_day():
    idx = pd.bdate_range("2023-01-02", periods=20)
    gold = _prices(20, seed=1)
    spy = _prices(20, seed=2)
    feat = gold_equity_divergence(gold, spy, idx)
    manual = np.log(gold / gold.shift(1)) - np.log(spy / spy.shift(1))
    pd.testing.assert_series_equal(feat, manual.shift(1), check_names=False)


def test_oil_shock_large_on_big_move_small_on_calm():
    idx = pd.bdate_range("2023-01-02", periods=40)
    rng = np.random.default_rng(4)
    oil = pd.Series(50 * np.exp(np.cumsum(rng.normal(0, 0.005, 40))), index=idx)
    oil.iloc[25] = oil.iloc[24] * 1.20  # 20% jump -> should register as a shock the next day
    feat = oil_shock(oil, idx, window=20)
    assert abs(feat.iloc[26]) > 2  # next day sees a large |z|
    assert abs(feat.iloc[24]) < 2  # calm day just before the shock, no shock yet


def test_oil_shock_sign_matches_direction():
    idx = pd.bdate_range("2023-01-02", periods=40)
    oil_up = pd.Series(50.0, index=idx)
    oil_up.iloc[25] = 60.0
    oil_down = pd.Series(50.0, index=idx)
    oil_down.iloc[25] = 40.0
    feat_up = oil_shock(oil_up, idx, window=20)
    feat_down = oil_shock(oil_down, idx, window=20)
    assert feat_up.iloc[26] > 0
    assert feat_down.iloc[26] < 0


def test_crypto_risk_appetite_negative_on_drawdown():
    idx = pd.bdate_range("2023-01-02", periods=20)
    btc = pd.Series(100.0, index=idx)
    btc.iloc[10:] = 80.0  # sharp drawdown at day 10, then flat
    feat = crypto_risk_appetite(btc, idx, window=5)
    # Day 11: yesterday's (day 10) 5-day trailing return already reflects the
    # drop; the 1-day lag means day 11 is the first day the feature sees it.
    assert feat.iloc[11] < 0


def test_crypto_risk_appetite_no_lookahead():
    idx = pd.bdate_range("2023-01-02", periods=40)
    btc = _prices(40, seed=3)
    full = crypto_risk_appetite(btc, idx, window=5)
    truncated = crypto_risk_appetite(btc.iloc[:25], idx[:25], window=5)
    shared = truncated.dropna().index
    pd.testing.assert_series_equal(full.loc[shared], truncated.loc[shared], check_names=False)
