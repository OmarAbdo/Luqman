import pandas as pd

from luqman.data.validation import validate_daily_ohlcv


def test_valid_frame_passes(ohlcv):
    report = validate_daily_ohlcv(ohlcv)
    assert report.ok
    assert report.n_rows == len(ohlcv)


def test_empty_frame_fails():
    report = validate_daily_ohlcv(pd.DataFrame())
    assert not report.ok


def test_negative_price_fails(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[10, bad.columns.get_loc("close")] = -1.0
    report = validate_daily_ohlcv(bad)
    assert any("non-positive" in issue for issue in report.issues)


def test_unsorted_index_fails(ohlcv):
    bad = ohlcv.iloc[::-1]
    report = validate_daily_ohlcv(bad)
    assert any("sorted" in issue for issue in report.issues)


def test_incoherent_high_fails(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[5, bad.columns.get_loc("high")] = bad.iloc[5]["low"] * 0.5
    report = validate_daily_ohlcv(bad)
    assert any("high <" in issue for issue in report.issues)


def test_float_noise_at_ohlc_boundary_does_not_fail(ohlcv):
    """Split/dividend-adjustment arithmetic (yfinance auto_adjust) can leave
    ~1e-12-scale float noise between high/low and open/close on bars that are
    economically coherent (e.g. high == close exactly, but not bit-for-bit
    after adjustment). Real XOM/KO data hit this; the tolerance must not flag
    it as a genuine OHLC violation."""
    bad = ohlcv.copy()
    row = bad.iloc[5].copy()
    row["close"] = row["high"]  # close pinned exactly to high
    row["high"] = row["close"] - 7e-15  # then perturbed a few ULPs below by adjustment math
    bad.iloc[5] = row
    report = validate_daily_ohlcv(bad)
    assert not any("high <" in issue for issue in report.issues)


def test_large_gap_flagged(ohlcv):
    bad = ohlcv.drop(ohlcv.index[100:130])
    report = validate_daily_ohlcv(bad)
    assert report.missing_business_days >= 30
    assert any("gap" in issue for issue in report.issues)


def test_nan_fails(ohlcv):
    bad = ohlcv.copy()
    bad.iloc[3, bad.columns.get_loc("open")] = float("nan")
    report = validate_daily_ohlcv(bad)
    assert any("NaN" in issue for issue in report.issues)
