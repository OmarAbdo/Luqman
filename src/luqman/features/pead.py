"""Post-earnings-announcement drift features.

Mechanism: returns are documented to drift in the direction of an earnings
surprise for weeks after the announcement.

Status: REJECTED. Built on real actual-vs-estimate earnings data (not a proxy)
and tested both single-name and cross-sectionally.
"""

import numpy as np
import pandas as pd

from ..data.asof import NY_TZ, effective_trading_day

MECHANISM = (
    "Post-Earnings-Announcement Drift (PEAD): stocks with large positive "
    "(negative) earnings surprises continue drifting up (down) for roughly "
    "40-60 trading days after the report, because the market underreacts to "
    "the surprise (one of the most replicated anomalies in equity markets)."
)
DRIFT_WINDOW_DAYS = 60


def earnings_surprise(earnings: pd.DataFrame) -> pd.Series:
    """Standardized surprise: (actual - estimate) / |estimate|, per report.
    Using |estimate| in the denominator keeps sign meaningful when the
    estimate itself is negative (loss-making quarters)."""
    surprise = (earnings["eps_actual"] - earnings["eps_estimated"]) / earnings["eps_estimated"].abs()
    return surprise.replace([np.inf, -np.inf], np.nan)


def build_pead_feature(
    earnings: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    drift_window: int = DRIFT_WINDOW_DAYS,
) -> pd.Series:
    """PEAD feature: the most recent earnings surprise, decayed linearly to
    zero over `drift_window` trading days after it becomes available (report
    date treated as end-of-day, so the drift window starts the NEXT trading
    day) — modeling the documented decay of the drift effect rather than a
    flat forward-fill for a full year (the flaw in the original fundamentals
    feature). Days before the first usable report, or more than `drift_window`
    days past the last one, are NaN (dropped at join, not imputed).
    """
    surprise = earnings_surprise(earnings)
    valid = earnings.assign(surprise=surprise).dropna(subset=["surprise"])

    report_eod = pd.Series(
        [
            pd.Timestamp(f"{d.date()}T23:59").tz_localize(NY_TZ).tz_convert("UTC")
            for d in valid["report_date"]
        ],
        index=valid.index,
    )
    valid = valid.assign(day=effective_trading_day(report_eod, daily_index)).dropna(subset=["day"])
    valid = valid.sort_values("day").groupby("day").last()  # same-day multiple reports: latest wins

    feature = pd.Series(np.nan, index=daily_index, dtype=float)
    positions = {date: i for i, date in enumerate(daily_index)}
    for report_day, row in valid.iterrows():
        start = positions.get(report_day)
        if start is None:
            continue
        end = min(start + drift_window, len(daily_index))
        decay = np.linspace(1.0, 0.0, end - start, endpoint=False)
        feature.iloc[start:end] = row["surprise"] * decay

    return feature.rename("pead_surprise_decay")
