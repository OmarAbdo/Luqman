"""Value features (trailing earnings yield).

Status: REJECTED, and strongly wrong-signed. Root-caused rather than left
unexplained: a bare earnings-yield ranking labels the momentum winners
"expensive" and genuine value traps "cheap", so it amounted to betting against
the momentum factor that this project had already proven. Documented in the
literature as a structural negative correlation between value and momentum.
"""

import numpy as np
import pandas as pd

from ..data.asof import NY_TZ, effective_trading_day

MECHANISM = (
    "Value: a stock trading cheap relative to its own trailing earnings "
    "(high earnings yield = low P/E) tends to outperform expensive peers over "
    "the medium term (Basu 1977 P/E anomaly; the quantitative core of "
    "Graham/Buffett/Lynch-style value investing) — tested cross-sectionally "
    "as a ranked long-short factor, the same way it is validated in the "
    "academic literature, using only trailing EPS we already have via "
    "data/earnings.py (no new external dependency)."
)


def trailing_earnings_yield(
    earnings: pd.DataFrame,
    close: pd.Series,
    daily_index: pd.DatetimeIndex,
) -> pd.Series:
    """Trailing-twelve-month EPS (sum of the last 4 reported quarters, as of
    each day, respecting report-availability timing) divided by price =
    earnings yield. Higher = cheaper. Uses only `data/earnings.py` data
    (already fetched, decade+ history), not FMP's thin 5-year key-metrics cap.

    Availability: a quarter's EPS becomes usable the first trading day after
    its report date (treated as end-of-day, consistent with pead.py). Before
    4 quarters have been reported, or more than ~100 trading days (~1 quarter
    + buffer) have passed since the last report without a new one landing
    (a real gap, not just quarterly cadence), the value is NaN.
    """
    valid = earnings.dropna(subset=["eps_actual"]).sort_values("report_date").reset_index(drop=True)
    valid["ttm_eps"] = valid["eps_actual"].rolling(4).sum()
    valid = valid.dropna(subset=["ttm_eps"])

    report_eod = pd.Series(
        [pd.Timestamp(f"{d.date()}T23:59").tz_localize(NY_TZ).tz_convert("UTC") for d in valid["report_date"]],
        index=valid.index,
    )
    valid = valid.assign(day=effective_trading_day(report_eod, daily_index)).dropna(subset=["day"])
    valid = valid.sort_values("day").groupby("day").last()

    ttm_eps_series = valid["ttm_eps"].reindex(daily_index).ffill(limit=100)
    price = close.reindex(daily_index)
    yield_ = (ttm_eps_series / price).rename("trailing_earnings_yield")
    return yield_.replace([np.inf, -np.inf], np.nan)
