import numpy as np
import pandas as pd

from luqman.data.earnings import parse_earnings
from luqman.features.pead import build_pead_feature, earnings_surprise

DAYS = pd.bdate_range("2024-01-15", periods=100)  # ~5 months of trading days


def test_earnings_surprise_sign_and_magnitude():
    earnings = pd.DataFrame({"eps_actual": [1.10, 0.90, -0.5], "eps_estimated": [1.00, 1.00, -1.0]})
    surprise = earnings_surprise(earnings)
    assert np.isclose(surprise.iloc[0], 0.10)
    assert np.isclose(surprise.iloc[1], -0.10)
    assert np.isclose(surprise.iloc[2], 0.5)  # beat a loss estimate -> positive surprise


def test_earnings_surprise_handles_zero_estimate():
    earnings = pd.DataFrame({"eps_actual": [0.1], "eps_estimated": [0.0]})
    surprise = earnings_surprise(earnings)
    assert np.isnan(surprise.iloc[0])


def test_pead_feature_decays_linearly_after_report():
    earnings = pd.DataFrame(
        {"report_date": [pd.Timestamp("2024-01-17")], "eps_actual": [1.20], "eps_estimated": [1.00]}
    )
    feat = build_pead_feature(earnings, DAYS, drift_window=20)
    report_pos = DAYS.get_loc(pd.Timestamp("2024-01-18"))  # after-close report -> next day
    assert np.isnan(feat.iloc[report_pos - 1])  # before the report is usable
    assert np.isclose(feat.iloc[report_pos], 0.20)  # full surprise on day 1
    assert np.isclose(feat.iloc[report_pos + 19], 0.20 * (1 / 20))  # near-zero at the last day in-window
    assert np.isnan(feat.iloc[report_pos + 20])  # past the drift window -> NaN, not stale


def test_pead_feature_no_lookahead():
    earnings = pd.DataFrame(
        {
            "report_date": [pd.Timestamp("2024-01-17"), pd.Timestamp("2024-03-01")],
            "eps_actual": [1.20, 0.80],
            "eps_estimated": [1.00, 1.00],
        }
    )
    full = build_pead_feature(earnings, DAYS, drift_window=30)
    truncated = build_pead_feature(earnings.iloc[:1], DAYS, drift_window=30)
    cutoff_pos = DAYS.get_loc(pd.Timestamp("2024-03-01"))
    pd.testing.assert_series_equal(
        full.iloc[:cutoff_pos], truncated.iloc[:cutoff_pos], check_names=False
    )


def test_pead_feature_latest_report_wins_same_day():
    earnings = pd.DataFrame(
        {
            "report_date": [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-16")],
            "eps_actual": [1.0, 1.0],
            "eps_estimated": [1.0, 0.5],  # second report: bigger surprise
        }
    )
    feat = build_pead_feature(earnings, DAYS, drift_window=20)
    report_pos = DAYS.get_loc(pd.Timestamp("2024-01-17"))
    assert np.isclose(feat.iloc[report_pos], 1.0)  # (1.0-0.5)/0.5


def test_parse_earnings_drops_incomplete_rows():
    payload = [
        {"date": "2026-07-30", "epsActual": None, "epsEstimated": 1.88},
        {"date": "2026-04-30", "epsActual": 2.01, "epsEstimated": 1.95},
        {"date": "bad", "epsActual": 1.0, "epsEstimated": 1.0},
    ]
    df = parse_earnings(payload)
    assert len(df) == 1
    assert df.iloc[0]["eps_actual"] == 2.01


def test_pead_feature_dropped_at_join_not_imputed():
    earnings = pd.DataFrame(
        {"report_date": [pd.Timestamp("2024-06-01")], "eps_actual": [1.0], "eps_estimated": [1.0]}
    )
    feat = build_pead_feature(earnings, DAYS, drift_window=20)
    assert feat.iloc[:10].isna().all()
