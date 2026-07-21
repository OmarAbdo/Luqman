"""Event-study evaluation: returns around a dated event.

Used to test whether an event type (an earnings surprise, for instance) is
followed by abnormal drift, measured relative to the event date rather than on
a fixed calendar.
"""

import numpy as np
import pandas as pd

from ..data.asof import effective_trading_day
from scipy import stats


def article_level_return_ic(
    articles: pd.DataFrame,
    ohlcv: pd.DataFrame,
    forward_days: int = 1,
    min_relevance: float = 0.0,
) -> dict:
    """Gate 0 for the sentiment redesign (dataset-plan.md Section 2b step 1):
    does an individual article's sentiment score relate AT ALL to the return
    on its effective trading day (or `forward_days` after it), before any
    daily aggregation? If this IC is near zero, no rolling average manufactures
    a signal that isn't there at the source.

    Returns IC (Spearman, article sentiment vs. forward log return from the
    article's effective day), n_articles used, and n_days with >=1 article —
    the count matters because many articles can map to the same trading day,
    which is a form of pseudo-replication the caller should be aware of.
    """
    filtered = articles[articles["relevance"] >= min_relevance].copy()
    filtered["day"] = effective_trading_day(filtered["time_published"], ohlcv.index)
    filtered = filtered.dropna(subset=["day"])

    close = ohlcv["close"]
    fwd_return = np.log(close.shift(-forward_days) / close)
    filtered["target"] = filtered["day"].map(fwd_return)
    filtered = filtered.dropna(subset=["target", "sentiment"])

    if len(filtered) < 3 or filtered["sentiment"].nunique() < 2:
        return {"ic": float("nan"), "n_articles": len(filtered), "n_days": 0}

    rho, pvalue = stats.spearmanr(filtered["sentiment"], filtered["target"])
    return {
        "ic": float(rho),
        "pvalue": float(pvalue),
        "n_articles": len(filtered),
        "n_days": int(filtered["day"].nunique()),
    }
