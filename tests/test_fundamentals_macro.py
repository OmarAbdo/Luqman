import numpy as np
import pandas as pd
import pytest

from luqman.data.fundamentals import parse_income_statements
from luqman.data.macro import parse_av_series
from luqman.features.fundamental_features import build_fundamental_features
from luqman.features.macro_features import build_macro_features

DAYS = pd.bdate_range("2024-01-15", periods=10)  # Mon 15th .. Fri 26th


def _reports():
    # 6 quarters; the last one accepted Wed 2024-01-17 after the close (17:30 ET).
    period_ends = pd.date_range("2022-09-30", periods=6, freq="QE")
    accepted = list(period_ends + pd.Timedelta(days=30))
    accepted[5] = pd.Timestamp("2024-01-17 17:30")
    return pd.DataFrame(
        {
            "period_end": period_ends,
            "accepted": accepted,
            "revenue": [100.0, 110.0, 90.0, 105.0, 120.0, 132.0],
            "net_income": [10.0, 11.0, 9.0, 10.5, 18.0, 13.2],
            "eps": [1.0, 1.1, 0.9, 1.05, 1.2, 1.4],
        }
    )


def test_fundamental_availability_and_values():
    feats = build_fundamental_features(_reports(), DAYS, periods_per_year=4)

    # Accepted Wed 17th after the close -> not usable until Thu 18th; on the
    # 17th the previous quarter's values apply: rev_yoy = 120/100 - 1.
    assert np.isclose(feats.loc[pd.Timestamp("2024-01-17"), "fund_rev_yoy"], 0.2)
    # From Thursday the 18th, the new report applies: 132/110 - 1
    assert np.isclose(feats.loc[pd.Timestamp("2024-01-18"), "fund_rev_yoy"], 132 / 110 - 1)
    assert np.isclose(feats.loc[pd.Timestamp("2024-01-18"), "fund_net_margin"], 13.2 / 132)
    assert np.isclose(feats.loc[pd.Timestamp("2024-01-18"), "fund_eps_chg_yoy"], 1.4 - 1.1)
    # Carried forward unchanged afterwards.
    assert np.isclose(feats.loc[DAYS[-1], "fund_rev_yoy"], 132 / 110 - 1)


def test_fundamental_annual_yoy_lag():
    period_ends = pd.date_range("2021-09-30", periods=3, freq="YE-SEP")
    reports = pd.DataFrame(
        {
            "period_end": period_ends,
            "accepted": period_ends + pd.Timedelta(days=30),
            "revenue": [100.0, 120.0, 150.0],
            "net_income": [10.0, 12.0, 15.0],
            "eps": [1.0, 1.2, 1.5],
        }
    )
    days = pd.bdate_range("2023-11-06", periods=3)
    feats = build_fundamental_features(reports, days, periods_per_year=1)
    assert np.isclose(feats.iloc[-1]["fund_rev_yoy"], 150 / 120 - 1)


def test_fundamental_no_reports_before_first_filing():
    reports = _reports().iloc[4:]
    early_days = pd.bdate_range("2020-01-01", periods=5)
    feats = build_fundamental_features(reports, early_days, periods_per_year=4)
    assert feats.isna().all().all()


def test_parse_income_statements_skips_bad_rows():
    payload = [
        {
            "date": "2024-09-30",
            "filingDate": "2024-11-01",
            "acceptedDate": "2024-11-01 06:01:00",
            "revenue": 5,
            "netIncome": 1,
            "eps": 0.5,
        },
        {"date": "bad", "filingDate": "2024-11-01", "revenue": 5},
        {"date": "2024-06-30", "filingDate": None, "revenue": 5},
    ]
    df = parse_income_statements(payload)
    assert len(df) == 1
    assert df.iloc[0]["revenue"] == 5.0
    assert df.iloc[0]["accepted"] == pd.Timestamp("2024-11-01 06:01:00")


def test_macro_lag_one_trading_day():
    macro = pd.DataFrame(
        {
            "y10": np.arange(10, 20, dtype=float),
            "y3m": np.arange(5, 15, dtype=float),
            "fedfunds": 5.0,
        },
        index=DAYS,
    )
    feats = build_macro_features(macro, DAYS)
    # Day t carries day t-1's value.
    assert np.isclose(feats.loc[DAYS[1], "macro_y10"], 10.0)
    assert np.isclose(feats.loc[DAYS[-1], "macro_y10"], 18.0)
    assert np.isnan(feats.loc[DAYS[0], "macro_y10"])
    assert np.isclose(feats.loc[DAYS[2], "macro_spread_10y3m"], 11.0 - 6.0)


def test_macro_weekend_gaps_ffilled_backward_only():
    observed = pd.DataFrame(
        {"y10": [4.0, 4.5], "y3m": [3.0, 3.1], "fedfunds": [5.0, 5.0]},
        index=pd.to_datetime(["2024-01-12", "2024-01-17"]),  # Fri and Wed
    )
    feats = build_macro_features(observed, DAYS)
    # Mon 15th and Tue 16th know only Friday's 4.0 (lagged); Wed 17th still 4.0;
    # Thursday 18th sees Wednesday's 4.5.
    assert np.isclose(feats.loc[pd.Timestamp("2024-01-16"), "macro_y10"], 4.0)
    assert np.isclose(feats.loc[pd.Timestamp("2024-01-17"), "macro_y10"], 4.0)
    assert np.isclose(feats.loc[pd.Timestamp("2024-01-18"), "macro_y10"], 4.5)


def test_parse_av_series():
    payload = {"data": [{"date": "2024-01-02", "value": "4.1"}, {"date": "2024-01-03", "value": "."}]}
    series = parse_av_series(payload, "y10")
    assert len(series) == 1
    assert np.isclose(series.iloc[0], 4.1)


def test_parse_av_series_error():
    with pytest.raises(RuntimeError, match="rate limit"):
        parse_av_series({"Information": "rate limit reached"}, "y10")
