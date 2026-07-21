"""Tests for session-aware intraday handling.

The central property under test is that no return is ever computed across a
session boundary -- the failure mode that would otherwise inject overnight
moves into an intraday series and manufacture a fake time-of-day pattern.
"""

import numpy as np
import pandas as pd
import pytest

from luqman.data.intraday import (
    add_session_flags,
    audit_intraday,
    forward_session_aware_log_return,
    load_intraday_csv,
    session_aware_log_returns,
)


@pytest.fixture
def two_sessions():
    """Two consecutive trading days, 09:30-10:00 regular bars only."""
    day1 = pd.date_range("2024-01-02 09:30", periods=6, freq="5min")
    day2 = pd.date_range("2024-01-03 09:30", periods=6, freq="5min")
    index = day1.append(day2)
    # Prices step by 1 within each session; day 2 opens far above day 1's close
    # so an accidental cross-session return would be conspicuously large.
    close = pd.Series([100, 101, 102, 103, 104, 105, 200, 201, 202, 203, 204, 205], index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )


def test_session_boundary_return_is_nan(two_sessions):
    flagged = add_session_flags(two_sessions)
    returns = session_aware_log_returns(flagged["close"], flagged["session_date"])

    first_bar_of_day2 = pd.Timestamp("2024-01-03 09:30")
    assert np.isnan(returns.loc[first_bar_of_day2]), "overnight gap must not produce a return"
    # The huge 105 -> 200 jump must appear nowhere in the series.
    assert returns.dropna().abs().max() < 0.02


def test_within_session_returns_are_computed(two_sessions):
    flagged = add_session_flags(two_sessions)
    returns = session_aware_log_returns(flagged["close"], flagged["session_date"])
    expected = np.log(101 / 100)
    assert returns.loc[pd.Timestamp("2024-01-02 09:35")] == pytest.approx(expected)
    # 12 bars, 2 session-opening bars have no predecessor in-session.
    assert returns.notna().sum() == 10


def test_multi_period_return_respects_boundary(two_sessions):
    flagged = add_session_flags(two_sessions)
    returns = session_aware_log_returns(flagged["close"], flagged["session_date"], periods=3)
    # First three bars of each session cannot look back 3 bars within-session.
    assert returns.iloc[:3].isna().all()
    assert returns.iloc[6:9].isna().all()
    assert returns.loc[pd.Timestamp("2024-01-02 09:45")] == pytest.approx(np.log(103 / 100))


def test_forward_target_respects_boundary(two_sessions):
    flagged = add_session_flags(two_sessions)
    target = forward_session_aware_log_return(flagged["close"], flagged["session_date"], horizon=1)
    last_bar_of_day1 = pd.Timestamp("2024-01-02 09:55")
    assert np.isnan(target.loc[last_bar_of_day1]), "target must not span the overnight gap"
    assert target.loc[pd.Timestamp("2024-01-02 09:30")] == pytest.approx(np.log(101 / 100))


def test_forward_and_backward_returns_are_consistent(two_sessions):
    """The forward return at bar t must equal the backward return at bar t+1."""
    flagged = add_session_flags(two_sessions)
    backward = session_aware_log_returns(flagged["close"], flagged["session_date"], periods=1)
    forward = forward_session_aware_log_return(flagged["close"], flagged["session_date"], horizon=1)
    pd.testing.assert_series_equal(
        forward.dropna().reset_index(drop=True),
        backward.dropna().reset_index(drop=True),
        check_names=False,
    )


def test_regular_hours_flagging():
    index = pd.to_datetime([
        "2024-01-02 04:00",  # pre-market
        "2024-01-02 09:25",  # pre-market
        "2024-01-02 09:30",  # first regular bar
        "2024-01-02 15:55",  # last regular bar
        "2024-01-02 16:00",  # after-hours (close is exclusive)
        "2024-01-02 19:55",  # after-hours
    ])
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=index)
    flagged = add_session_flags(df)
    assert flagged["is_regular"].tolist() == [False, False, True, True, False, False]


def test_audit_counts_sessions_and_gaps(two_sessions):
    audit = audit_intraday(two_sessions)
    assert audit.n_sessions == 2
    assert audit.n_regular_bars == 12
    assert audit.n_extended_bars == 0
    assert audit.n_within_session_gaps == 0


def test_audit_detects_missing_bars():
    """A hole inside a session is a real data defect and must be reported,
    unlike the expected overnight gap."""
    index = pd.to_datetime([
        "2024-01-02 09:30",
        "2024-01-02 09:35",
        "2024-01-02 10:30",  # 55-minute hole
        "2024-01-02 10:35",
    ])
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}, index=index)
    audit = audit_intraday(df)
    assert audit.n_within_session_gaps == 1
    assert audit.largest_within_session_gap_minutes == pytest.approx(55.0)


def test_load_handles_both_provider_column_orders(tmp_path):
    """AAPL files are timestamp-first; NVDA/SAP are timestamp-last."""
    a = tmp_path / "a.csv"
    a.write_text(
        "timestamp,open,high,low,close,volume\n2024-01-02 09:30:00,1,2,0.5,1.5,100\n"
    )
    b = tmp_path / "b.csv"
    b.write_text(
        "open,low,high,close,volume,timestamp\n1,0.5,2,1.5,100,2024-01-02 09:30:00\n"
    )
    frame_a = load_intraday_csv(a)
    frame_b = load_intraday_csv(b)
    pd.testing.assert_frame_equal(frame_a, frame_b, check_dtype=False)
    assert list(frame_a.columns) == ["open", "high", "low", "close", "volume"]


def test_load_rejects_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("timestamp,close\n2024-01-02 09:30:00,1.5\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_intraday_csv(bad)
