import numpy as np
import pandas as pd

from ..data.asof import effective_trading_day


def build_news_features(articles: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily news features under the point-in-time rule: an article only enters
    the day whose close it preceded. Days without news keep count 0; sentiment
    windows are NaN until they contain at least one article (dropped at join,
    never imputed).
    """
    day = effective_trading_day(articles["time_published"], daily_index)
    tagged = articles.assign(day=day).dropna(subset=["day"])
    grouped = tagged.groupby("day").agg(
        count=("sentiment", "size"), sent_sum=("sentiment", "sum")
    )
    daily = grouped.reindex(daily_index).fillna(0.0)

    feats = pd.DataFrame(index=daily_index)
    feats["news_count_1d"] = np.log1p(daily["count"])
    for window in (5, 20):
        window_count = daily["count"].rolling(window, min_periods=1).sum()
        feats[f"news_sent_{window}d"] = (
            daily["sent_sum"].rolling(window, min_periods=1).sum()
            / window_count.replace(0.0, np.nan)
        )
    feats["news_sent_trend"] = feats["news_sent_5d"] - feats["news_sent_20d"]
    return feats
