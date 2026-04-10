import numpy as np
import pandas as pd

from luqman.data.asof import effective_trading_day
from luqman.data.news import parse_feed
from luqman.data.store import ParquetStore
from luqman.features.news_features import build_news_features

# 2024-01-15 is a Monday; NY close 16:00 ET = 21:00 UTC in January.
DAYS = pd.bdate_range("2024-01-15", periods=5)


def test_effective_day_cutoff():
    events = pd.Series(
        pd.to_datetime(
            [
                "2024-01-15 20:59:00",  # 15:59 ET Mon -> same day
                "2024-01-15 21:00:00",  # 16:00 ET Mon (at close) -> same day
                "2024-01-15 22:30:00",  # 17:30 ET Mon (after close) -> Tue
                "2024-01-13 12:00:00",  # Saturday -> Mon
                "2024-01-19 22:00:00",  # after last close -> NaT
            ],
            utc=True,
        )
    )
    mapped = effective_trading_day(events, DAYS)
    assert mapped.iloc[0] == DAYS[0]
    assert mapped.iloc[1] == DAYS[0]
    assert mapped.iloc[2] == DAYS[1]
    assert mapped.iloc[3] == DAYS[0]
    assert pd.isna(mapped.iloc[4])


def _articles(times, sentiments):
    return pd.DataFrame(
        {
            "time_published": pd.to_datetime(times, utc=True),
            "sentiment": sentiments,
            "relevance": 1.0,
            "url": [f"https://x/{i}" for i in range(len(times))],
        }
    )


def test_news_feature_aggregation():
    articles = _articles(
        ["2024-01-15 14:00:00", "2024-01-15 15:00:00", "2024-01-16 14:00:00"],
        [0.4, 0.2, -0.3],
    )
    feats = build_news_features(articles, DAYS)
    assert np.isclose(feats.loc[DAYS[0], "news_count_1d"], np.log1p(2))
    assert np.isclose(feats.loc[DAYS[1], "news_count_1d"], np.log1p(1))
    assert feats.loc[DAYS[2], "news_count_1d"] == 0.0
    # 5-day window at day 3 holds all three articles
    assert np.isclose(feats.loc[DAYS[2], "news_sent_5d"], (0.4 + 0.2 - 0.3) / 3)


def test_news_features_no_lookahead():
    articles = _articles(
        ["2024-01-15 14:00:00", "2024-01-18 14:00:00"], [0.5, -0.9]
    )
    full = build_news_features(articles, DAYS)
    truncated = build_news_features(articles.iloc[:1], DAYS)
    cutoff = DAYS[2]  # strictly before the second article's effective day
    pd.testing.assert_frame_equal(full.loc[:cutoff], truncated.loc[:cutoff])


def test_parse_feed_prefers_ticker_sentiment():
    payload = {
        "feed": [
            {
                "time_published": "20240115T140000",
                "overall_sentiment_score": 0.1,
                "url": "https://a",
                "ticker_sentiment": [
                    {"ticker": "AAPL", "ticker_sentiment_score": "0.7", "relevance_score": "0.9"},
                    {"ticker": "MSFT", "ticker_sentiment_score": "-0.2", "relevance_score": "0.1"},
                ],
            },
            {
                "time_published": "20240115T150000",
                "overall_sentiment_score": -0.4,
                "url": "https://b",
                "ticker_sentiment": [],
            },
            {"time_published": "bad", "overall_sentiment_score": 0.0, "url": "https://c"},
        ]
    }
    df = parse_feed(payload, "AAPL")
    assert len(df) == 2
    assert np.isclose(df.iloc[0]["sentiment"], 0.7)
    assert np.isclose(df.iloc[0]["relevance"], 0.9)
    assert np.isclose(df.iloc[1]["sentiment"], -0.4)


def test_news_store_round_trip(tmp_path):
    store = ParquetStore(tmp_path)
    articles = _articles(["2024-01-15 14:00:00", "2024-01-15 14:00:00"], [0.1, 0.1])
    articles.loc[1, "url"] = articles.loc[0, "url"]  # duplicate url
    store.save_news("test", articles)
    loaded = store.load_news("TEST")
    assert len(loaded) == 1  # deduped on url
    assert store.load_news("NOPE") is None
