"""Analyst rating level and month-over-month revision features.

Status: REJECTED -- see `data/analyst_grades.py` for why the promising-looking
revision result was an evaluation-cadence artifact rather than alpha.
"""

import numpy as np
import pandas as pd

from ..data.asof import NY_TZ, effective_trading_day

MECHANISM = (
    "Analyst estimate revisions (Chan, Jegadeesh & Lakonishok 1996): the "
    "TREND in analyst sentiment/estimates -- distinct from earnings surprise "
    "at announcement (already tested and rejected as PEAD, Section 6a) -- "
    "has separately documented, independent predictive power. Constructed "
    "here from monthly analyst rating distributions (FMP grades-historical, "
    "genuinely deep free-tier history unlike the 5yr-capped fundamentals "
    "endpoints): a net-bullishness score and its month-over-month change "
    "(the 'revision' -- are analysts turning more or less positive)."
)


def net_rating_score(grades: pd.DataFrame) -> pd.Series:
    """Weighted net rating: (+2*strong_buy + 1*buy - 1*sell - 2*strong_sell)
    normalized by total analyst count, in [-2, +2]. Higher = more bullish
    consensus."""
    total = grades[["strong_buy", "buy", "hold", "sell", "strong_sell"]].sum(axis=1).replace(0, np.nan)
    weighted = (
        2 * grades["strong_buy"] + grades["buy"] - grades["sell"] - 2 * grades["strong_sell"]
    )
    score = (weighted / total).rename("net_rating_score")
    score.index = grades["month"]
    return score


def build_analyst_revision_features(
    grades: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    availability_lag_days: int = 5,
) -> pd.DataFrame:
    """Daily features from monthly analyst grade data:
      - analyst_net_rating: level of the net rating score, carried forward.
      - analyst_rating_revision: month-over-month CHANGE in net rating score
        (the actual 'revision' signal per the mechanism -- level and trend
        are tested as separate columns since the research distinguishes them).

    Each monthly observation is treated as available `availability_lag_days`
    after its month-start date (FMP gives no exact publication timestamp for
    this aggregate; a small conservative lag avoids assuming same-day
    knowledge of a monthly rollup). Days before the first observation are NaN.
    """
    if grades.empty:
        return pd.DataFrame(
            {"analyst_net_rating": np.nan, "analyst_rating_revision": np.nan}, index=daily_index
        )

    score = net_rating_score(grades).sort_index()
    revision = score.diff()

    available_eod = pd.Series(
        [
            pd.Timestamp(f"{(d + pd.Timedelta(days=availability_lag_days)).date()}T23:59")
            .tz_localize(NY_TZ)
            .tz_convert("UTC")
            for d in score.index
        ],
        index=score.index,
    )
    day = effective_trading_day(available_eod, daily_index)
    tagged = pd.DataFrame(
        {"day": day.to_numpy(), "analyst_net_rating": score.to_numpy(), "analyst_rating_revision": revision.to_numpy()}
    ).dropna(subset=["day"])
    tagged = tagged.sort_values("day").groupby("day").last()

    result = tagged.reindex(daily_index)
    result["analyst_net_rating"] = result["analyst_net_rating"].ffill()
    result["analyst_rating_revision"] = result["analyst_rating_revision"].ffill()
    return result[["analyst_net_rating", "analyst_rating_revision"]]
