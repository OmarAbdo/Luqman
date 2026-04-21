import numpy as np
import pandas as pd

from luqman.evaluation.event_study import article_level_return_ic


def _ohlcv(n=60, seed=0):
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def test_article_ic_detects_planted_relationship():
    ohlcv = _ohlcv(n=100, seed=1)
    close = ohlcv["close"]
    fwd = np.log(close.shift(-1) / close)
    # One article per day, sentiment = scaled version of that day's actual
    # forward return -> should recover a strong positive IC.
    articles = pd.DataFrame(
        {
            "time_published": pd.to_datetime(ohlcv.index[:-1]) + pd.Timedelta(hours=10),
            "sentiment": (fwd.iloc[:-1] * 20).to_numpy(),
            "relevance": 1.0,
            "url": [f"https://x/{i}" for i in range(len(ohlcv) - 1)],
        }
    )
    articles["time_published"] = articles["time_published"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
    result = article_level_return_ic(articles, ohlcv, forward_days=1)
    assert result["ic"] > 0.7
    assert result["n_articles"] > 0


def test_article_ic_near_zero_for_random_sentiment():
    ohlcv = _ohlcv(n=200, seed=2)
    rng = np.random.default_rng(3)
    times = pd.to_datetime(ohlcv.index) + pd.Timedelta(hours=10)
    articles = pd.DataFrame(
        {
            "time_published": times.tz_localize("America/New_York").tz_convert("UTC"),
            "sentiment": rng.normal(size=len(ohlcv)),
            "relevance": 1.0,
            "url": [f"https://x/{i}" for i in range(len(ohlcv))],
        }
    )
    result = article_level_return_ic(articles, ohlcv, forward_days=1)
    assert abs(result["ic"]) < 0.15


def test_min_relevance_filters_articles():
    ohlcv = _ohlcv(n=30, seed=4)
    times = pd.to_datetime(ohlcv.index) + pd.Timedelta(hours=10)
    articles = pd.DataFrame(
        {
            "time_published": times.tz_localize("America/New_York").tz_convert("UTC"),
            "sentiment": 0.1,
            "relevance": [0.2] * 15 + [0.9] * 15,
            "url": [f"https://x/{i}" for i in range(len(ohlcv))],
        }
    )
    result_all = article_level_return_ic(articles, ohlcv, min_relevance=0.0)
    result_filtered = article_level_return_ic(articles, ohlcv, min_relevance=0.5)
    assert result_filtered["n_articles"] < result_all["n_articles"]


def test_empty_after_filtering_returns_nan():
    ohlcv = _ohlcv(n=10, seed=5)
    articles = pd.DataFrame(columns=["time_published", "sentiment", "relevance", "url"])
    result = article_level_return_ic(articles, ohlcv)
    assert np.isnan(result["ic"])
