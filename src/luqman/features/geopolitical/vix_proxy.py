import numpy as np
import pandas as pd

MECHANISMS: dict[str, dict] = {
    "vix_level": {
        "mechanism": (
            "VIX level as a risk-off/stress proxy: elevated implied volatility "
            "reflects broad risk aversion (geopolitical shocks, macro stress) "
            "that historically coincides with equity drawdowns and dispersion."
        ),
        "horizon": 5,
    },
    "vix_change": {
        "mechanism": (
            "VIX spikes (rate of change, not level) mark acute stress events "
            "more precisely than level — a sharp VIX rise is the closest tractable "
            "proxy to a geopolitical shock without a full event feed."
        ),
        "horizon": 3,
    },
}


def vix_level(vix: pd.Series, daily_index: pd.DatetimeIndex, lag: int = 1) -> pd.Series:
    """VIX close lagged by `lag` trading days (available before the next
    close), reindexed onto the target ticker's calendar."""
    aligned = vix.reindex(vix.index.union(daily_index)).ffill().reindex(daily_index)
    return aligned.shift(lag).rename("vix_level")


def vix_change(vix: pd.Series, daily_index: pd.DatetimeIndex, window: int = 5, lag: int = 1) -> pd.Series:
    """`window`-day VIX rate of change, lagged by `lag` trading days."""
    aligned = vix.reindex(vix.index.union(daily_index)).ffill().reindex(daily_index)
    change = np.log(aligned / aligned.shift(window))
    return change.shift(lag).rename("vix_change")
