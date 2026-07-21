"""Session-aware loading and return computation for intraday (5-minute) bars.

Phase 8 of `.refactor/plan.md`. The intraday asset is a set of Alpha Vantage
5-minute CSVs collected for AAPL, NVDA and SAP; AAPL is the substantial one
(~548k bars, 2012-2024, including extended hours).

The methodological point this module exists to enforce: **intraday bars are not
a continuous series.** Between 15:55 and the next session's 09:30 there is an
overnight gap during which the price moves without any bars being printed. A
naive `close.pct_change()` over the raw file silently treats that gap as if it
were a five-minute move, which:

  - injects the largest returns in the sample at regular daily intervals,
  - makes realized volatility a measure of overnight risk rather than
    intraday risk,
  - and creates a spurious time-of-day seasonality that any model will
    happily learn and that will not generalize.

Every return produced here is therefore NaN across a session boundary, and
callers are expected to drop rather than fill those.

Two provider schemas appear in the stored files: AAPL uses
timestamp-first column ordering, NVDA/SAP use timestamp-last. Both are
normalized to the same canonical frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# US equity regular trading hours, in the exchange-local time the provider
# timestamps are already expressed in.
REGULAR_OPEN = pd.Timestamp("09:30").time()
REGULAR_CLOSE = pd.Timestamp("16:00").time()

INTRADAY_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_intraday_csv(path: str | Path) -> pd.DataFrame:
    """Load one 5-minute CSV into the canonical schema.

    Handles both stored column orderings, sorts, drops duplicate timestamps,
    and returns a frame indexed by a tz-naive DatetimeIndex named 'timestamp'.
    """
    raw = pd.read_csv(path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    if "timestamp" not in raw.columns:
        raise ValueError(f"{path}: no 'timestamp' column in {list(raw.columns)}")

    missing = [c for c in INTRADAY_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    df = raw[["timestamp"] + INTRADAY_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = (
        df.set_index("timestamp")
        .sort_index()
        .pipe(lambda d: d[~d.index.duplicated(keep="last")])
    )
    df.index.name = "timestamp"
    return df


def add_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Attach session metadata: trading date, regular-hours flag, and session
    ordinal (position of the bar within its own regular session).

    `is_regular` is True only for bars inside 09:30-16:00. Extended-hours bars
    are kept rather than dropped, because whether to use them is an analysis
    decision -- but they are marked so that decision is explicit rather than
    accidental.
    """
    out = df.copy()
    times = out.index.time
    out["session_date"] = out.index.normalize()
    out["is_regular"] = (times >= REGULAR_OPEN) & (times < REGULAR_CLOSE)
    out["bar_of_session"] = out.groupby(["session_date", "is_regular"]).cumcount()
    return out


def session_aware_log_returns(
    close: pd.Series,
    session_date: pd.Series,
    periods: int = 1,
) -> pd.Series:
    """Log returns that are NaN wherever the window crosses a session boundary.

    This is the whole point of the module. A return is only defined when the
    bar `periods` steps back belongs to the SAME trading session; otherwise the
    interval spans an overnight (or weekend, or holiday) gap and is not a
    comparable five-minute observation.
    """
    if periods < 1:
        raise ValueError("periods must be >= 1")

    raw = np.log(close / close.shift(periods))
    same_session = session_date == session_date.shift(periods)
    return raw.where(same_session).rename(f"logret_{periods}bar")


def forward_session_aware_log_return(
    close: pd.Series,
    session_date: pd.Series,
    horizon: int = 1,
) -> pd.Series:
    """Forward-looking counterpart used as a prediction target: the return from
    this bar to `horizon` bars ahead, NaN if that crosses a session boundary."""
    raw = np.log(close.shift(-horizon) / close)
    same_session = session_date == session_date.shift(-horizon)
    return raw.where(same_session).rename(f"fwd_logret_{horizon}bar")


@dataclass
class IntradayAudit:
    n_bars: int
    n_sessions: int
    first_timestamp: pd.Timestamp
    last_timestamp: pd.Timestamp
    n_regular_bars: int
    n_extended_bars: int
    median_bars_per_session: float
    n_within_session_gaps: int
    largest_within_session_gap_minutes: float

    def summary(self) -> str:
        return (
            f"bars: {self.n_bars} over {self.n_sessions} sessions "
            f"({self.first_timestamp:%Y-%m-%d} -> {self.last_timestamp:%Y-%m-%d})\n"
            f"regular-hours bars: {self.n_regular_bars} | extended-hours bars: {self.n_extended_bars}\n"
            f"median regular bars per session: {self.median_bars_per_session:.0f} "
            f"(a complete 09:30-16:00 session is 78 five-minute bars)\n"
            f"within-session gaps > 5min: {self.n_within_session_gaps} "
            f"(largest {self.largest_within_session_gap_minutes:.0f} min)"
        )


def audit_intraday(df: pd.DataFrame) -> IntradayAudit:
    """Structural audit of an intraday frame.

    Reports coverage and, importantly, gaps that occur WITHIN a session --
    those are genuine data holes (missing bars), as distinct from the overnight
    gaps that are expected and handled by the session-aware return functions.
    """
    flagged = add_session_flags(df)
    regular = flagged[flagged["is_regular"]]

    within = regular.groupby("session_date").apply(
        lambda g: g.index.to_series().diff().dt.total_seconds().div(60).max(),
        include_groups=False,
    )
    gaps = within[within > 5.0]

    return IntradayAudit(
        n_bars=len(flagged),
        n_sessions=int(flagged["session_date"].nunique()),
        first_timestamp=flagged.index.min(),
        last_timestamp=flagged.index.max(),
        n_regular_bars=int(flagged["is_regular"].sum()),
        n_extended_bars=int((~flagged["is_regular"]).sum()),
        median_bars_per_session=float(regular.groupby("session_date").size().median()),
        n_within_session_gaps=int(len(gaps)),
        largest_within_session_gap_minutes=float(gaps.max()) if len(gaps) else 0.0,
    )
