import numpy as np
import pandas as pd

from luqman.features.value import trailing_earnings_yield

DAYS = pd.bdate_range("2023-01-02", periods=260)  # ~1 trading year


def _earnings(report_dates, eps_values):
    return pd.DataFrame({"report_date": report_dates, "eps_actual": eps_values, "eps_estimated": eps_values})


def test_ttm_eps_requires_four_quarters():
    dates = pd.to_datetime(["2022-02-01", "2022-05-01", "2022-08-01", "2022-11-01", "2023-02-01"])
    earnings = _earnings(dates, [1.0, 1.0, 1.0, 1.0, 1.2])
    close = pd.Series(100.0, index=DAYS)
    feat = trailing_earnings_yield(earnings, close, DAYS)
    # By day 1 of DAYS (2023-01-02), 4 quarters (up to 2022-11-01) are available -> ttm = 4.0
    assert np.isclose(feat.iloc[0], 4.0 / 100.0)


def test_earnings_yield_inversely_related_to_price():
    dates = pd.to_datetime(["2022-02-01", "2022-05-01", "2022-08-01", "2022-11-01"])
    earnings = _earnings(dates, [1.0, 1.0, 1.0, 1.0])
    cheap = pd.Series(50.0, index=DAYS)
    expensive = pd.Series(200.0, index=DAYS)
    cheap_yield = trailing_earnings_yield(earnings, cheap, DAYS)
    expensive_yield = trailing_earnings_yield(earnings, expensive, DAYS)
    assert (cheap_yield.dropna() > expensive_yield.dropna()).all()


def test_updates_after_new_quarter_reported():
    dates = pd.to_datetime(["2022-02-01", "2022-05-01", "2022-08-01", "2022-11-01", "2023-02-01"])
    earnings = _earnings(dates, [1.0, 1.0, 1.0, 1.0, 2.0])  # big Q1 2023 beat
    close = pd.Series(100.0, index=DAYS)
    feat = trailing_earnings_yield(earnings, close, DAYS)
    before = feat.loc["2023-01-31"]
    after = feat.loc["2023-02-02"]  # after report_eod + next trading day
    assert after > before  # ttm eps rose from 4.0 to 5.0 after the beat


def test_no_lookahead():
    dates = pd.to_datetime(["2022-02-01", "2022-05-01", "2022-08-01", "2022-11-01", "2023-06-01"])
    earnings = _earnings(dates, [1.0, 1.0, 1.0, 1.0, 5.0])  # future huge beat
    close = pd.Series(100.0, index=DAYS)
    full = trailing_earnings_yield(earnings, close, DAYS)
    truncated = trailing_earnings_yield(earnings.iloc[:4], close, DAYS)
    cutoff = pd.Timestamp("2023-05-31")
    pd.testing.assert_series_equal(full.loc[:cutoff], truncated.loc[:cutoff], check_names=False)


def test_nan_before_four_quarters_available():
    dates = pd.to_datetime(["2022-11-01"])  # only 1 quarter, never reaches 4
    earnings = _earnings(dates, [1.0])
    close = pd.Series(100.0, index=DAYS)
    feat = trailing_earnings_yield(earnings, close, DAYS)
    assert feat.isna().all()
