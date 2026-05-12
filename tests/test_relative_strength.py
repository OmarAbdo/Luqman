import numpy as np
import pandas as pd

from luqman.features.relative_strength import relative_strength


def test_relative_strength_zero_when_matching_benchmark():
    idx = pd.bdate_range("2023-01-02", periods=60)
    close = pd.Series(100 * np.exp(np.linspace(0, 0.2, 60)), index=idx)
    feat = relative_strength(close, close, window=20)
    assert np.allclose(feat.dropna(), 0.0, atol=1e-9)


def test_relative_strength_positive_when_outperforming():
    idx = pd.bdate_range("2023-01-02", periods=60)
    ticker = pd.Series(100 * np.exp(np.linspace(0, 0.3, 60)), index=idx)
    bench = pd.Series(100 * np.exp(np.linspace(0, 0.1, 60)), index=idx)
    feat = relative_strength(ticker, bench, window=20)
    assert (feat.dropna() > 0).all()


def test_relative_strength_no_lookahead():
    idx = pd.bdate_range("2023-01-02", periods=80)
    rng = np.random.default_rng(0)
    ticker = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))), index=idx)
    bench = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))), index=idx)
    full = relative_strength(ticker, bench, window=20)
    truncated = relative_strength(ticker.iloc[:60], bench.iloc[:60], window=20)
    shared = truncated.dropna().index
    pd.testing.assert_series_equal(full.loc[shared], truncated.loc[shared], check_names=False)
