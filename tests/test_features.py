import numpy as np
import pandas as pd

from luqman.features.features import WARMUP_ROWS, build_features


def test_no_nan_or_inf(ohlcv):
    feats = build_features(ohlcv)
    assert np.isfinite(feats.to_numpy()).all()


def test_warmup_dropped(ohlcv):
    feats = build_features(ohlcv)
    assert feats.index[0] >= ohlcv.index[WARMUP_ROWS]
    assert len(feats) <= len(ohlcv) - WARMUP_ROWS


def test_no_lookahead(ohlcv):
    """Features computed on a truncated history must match the full-history
    values row for row — i.e. no feature uses future data."""
    full = build_features(ohlcv)
    truncated = build_features(ohlcv.iloc[:300])
    shared = truncated.index
    pd.testing.assert_frame_equal(full.loc[shared], truncated)


def test_value_ranges(ohlcv):
    feats = build_features(ohlcv)
    assert feats["rsi_14"].between(0, 1).all()
    assert feats["vol_20d"].ge(0).all()
    for col in ["dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert feats[col].between(-1, 1).all()


def test_ret_1d_matches_log_return(ohlcv):
    feats = build_features(ohlcv)
    close = ohlcv["close"]
    expected = np.log(close / close.shift(1)).loc[feats.index]
    pd.testing.assert_series_equal(feats["ret_1d"], expected, check_names=False)
