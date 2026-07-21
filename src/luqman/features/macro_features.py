"""Macroeconomic features with an availability lag.

Treasury yields, the yield spread, and the fed funds rate, each lagged one
trading day so that a value is only used once it could have been observed.

Status: REJECTED, with inconsistent sign across tickers.
"""

import pandas as pd


def build_macro_features(macro: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Macro features lagged one trading day: the value used on day t is the
    last one observable through day t-1, so a same-day close decision never
    assumes intraday knowledge of that day's official quote."""
    aligned = (
        macro.reindex(macro.index.union(daily_index)).ffill().reindex(daily_index).shift(1)
    )
    feats = pd.DataFrame(index=daily_index)
    feats["macro_y10"] = aligned["y10"]
    feats["macro_spread_10y3m"] = aligned["y10"] - aligned["y3m"]
    feats["macro_y10_chg_20d"] = aligned["y10"].diff(20)
    feats["macro_fedfunds"] = aligned["fedfunds"]
    return feats
