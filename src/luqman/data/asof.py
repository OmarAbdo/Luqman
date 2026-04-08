import numpy as np
import pandas as pd

NY_TZ = "America/New_York"


def effective_trading_day(
    event_times: pd.Series,
    daily_index: pd.DatetimeIndex,
    close_time: str = "16:00",
) -> pd.Series:
    """Map event timestamps (UTC) to the first trading day whose close they can inform.

    An event published at or before day t's close belongs to day t; after the
    close it belongs to the next trading day. This is the point-in-time rule
    that makes every event-derived feature leakage-safe by construction.
    Events after the last close in `daily_index` map to NaT (unusable yet).
    """
    closes = (
        pd.DatetimeIndex([pd.Timestamp(f"{day.date()}T{close_time}") for day in daily_index])
        .tz_localize(NY_TZ)
        .tz_convert("UTC")
    )
    ts = pd.to_datetime(event_times, utc=True)
    pos = np.searchsorted(closes.asi8, pd.DatetimeIndex(ts).asi8, side="left")
    result = pd.Series(pd.NaT, index=event_times.index, dtype="datetime64[ns]")
    valid = pos < len(daily_index)
    result[valid] = daily_index[pos[valid]]
    return result
