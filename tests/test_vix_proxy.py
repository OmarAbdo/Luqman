import numpy as np
import pandas as pd

from luqman.features.geopolitical.vix_proxy import vix_change, vix_level


def _vix_series():
    idx = pd.bdate_range("2023-01-02", periods=30)
    return pd.Series(np.linspace(15, 25, 30), index=idx, name="vix")


def test_vix_level_lagged_by_one_day():
    vix = _vix_series()
    daily_index = vix.index
    feat = vix_level(vix, daily_index, lag=1)
    assert np.isnan(feat.iloc[0])
    assert np.isclose(feat.iloc[1], vix.iloc[0])
    assert np.isclose(feat.iloc[-1], vix.iloc[-2])


def test_vix_level_handles_missing_days_via_ffill():
    vix = _vix_series()
    sparse = vix.iloc[::3]  # gaps
    feat = vix_level(sparse, vix.index, lag=1)
    assert feat.notna().sum() > 0
    assert feat.index.equals(vix.index)


def test_vix_change_matches_manual_log_change():
    vix = _vix_series()
    feat = vix_change(vix, vix.index, window=5, lag=1)
    expected_at_10 = np.log(vix.iloc[9] / vix.iloc[4])  # row 10 lagged 1 -> uses rows up to index 9
    assert np.isclose(feat.iloc[10], expected_at_10)


def test_vix_change_no_lookahead():
    vix = _vix_series()
    full = vix_change(vix, vix.index, window=5, lag=1)
    truncated = vix_change(vix.iloc[:20], vix.index[:20], window=5, lag=1)
    shared = truncated.index
    pd.testing.assert_series_equal(full.loc[shared], truncated, check_names=False)
