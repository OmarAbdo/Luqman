import numpy as np
import pandas as pd

from luqman.features.technical.candidates import pct_of_52w_high


def test_at_the_high_equals_one():
    idx = pd.bdate_range("2023-01-02", periods=300)
    close = pd.Series(np.linspace(100, 200, 300), index=idx)  # strictly rising -> today's close is the rolling max
    feat = pct_of_52w_high(close, window=252)
    assert np.isclose(feat.iloc[-1], 1.0)


def test_below_the_high_less_than_one():
    idx = pd.bdate_range("2023-01-02", periods=300)
    close = pd.Series(100.0, index=idx)
    close.iloc[100] = 150.0  # spike to a high, then falls back
    close.iloc[101:] = 100.0
    feat = pct_of_52w_high(close, window=252)
    assert feat.iloc[-1] < 1.0
    assert np.isclose(feat.iloc[-1], 100.0 / 150.0)


def test_no_lookahead():
    idx = pd.bdate_range("2023-01-02", periods=400)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
    full = pct_of_52w_high(close, window=252)
    truncated = pct_of_52w_high(close.iloc[:300], window=252)
    shared = truncated.dropna().index
    pd.testing.assert_series_equal(full.loc[shared], truncated.loc[shared], check_names=False)


def test_values_bounded_zero_to_one():
    idx = pd.bdate_range("2023-01-02", periods=300)
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))), index=idx)
    feat = pct_of_52w_high(close, window=252)
    assert (feat.dropna() <= 1.0 + 1e-9).all()
    assert (feat.dropna() > 0).all()
