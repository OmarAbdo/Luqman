"""Features derived from financial statements.

Revenue year-over-year growth, net margin, EPS change -- aligned to the date
each report actually became public.

Status: REJECTED at the daily horizon. Fundamentals plausibly suit much longer
horizons than the ones tested here.
"""

import numpy as np
import pandas as pd

from ..data.asof import NY_TZ, effective_trading_day


def build_fundamental_features(
    reports: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    periods_per_year: int = 1,
) -> pd.DataFrame:
    """Report-derived features, available from the first trading-day close after
    the SEC acceptance timestamp and carried forward until superseded.

    `periods_per_year` sets the YoY comparison lag: 1 for annual reports,
    4 for quarterly. Days before the first usable report are NaN (dropped at
    join, not imputed).
    """
    reports = reports.sort_values("period_end").reset_index(drop=True)
    lag = periods_per_year
    derived = pd.DataFrame(
        {
            "fund_rev_yoy": reports["revenue"] / reports["revenue"].shift(lag) - 1.0,
            "fund_net_margin": reports["net_income"] / reports["revenue"].replace(0.0, np.nan),
            "fund_eps_chg_yoy": reports["eps"] - reports["eps"].shift(lag),
        }
    )

    accepted = pd.to_datetime(reports["accepted"])
    if accepted.dt.tz is None:
        accepted = accepted.dt.tz_localize(NY_TZ)
    derived["day"] = effective_trading_day(accepted.dt.tz_convert("UTC"), daily_index)
    derived = derived.dropna(subset=["day"]).sort_values("day")
    # If several reports become usable the same day, the latest filing wins.
    derived = derived.groupby("day").last()

    return derived.reindex(daily_index).ffill()
