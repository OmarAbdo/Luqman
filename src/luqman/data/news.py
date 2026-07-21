"""News-article fetching and storage (Alpha Vantage).

Resumable and chunked, because the full history runs to tens of thousands of
articles across many rate-limited requests.

Status: the sentiment signal built from this data was REJECTED. Article-level
testing found the vendor's sentiment score carries essentially zero information
about forward returns at every horizon and relevance filter tried, so no
downstream aggregation could have rescued it. Kept because the stored corpus is
expensive to re-fetch and may support a different featurization later.
"""

import time

import pandas as pd
import requests

API_URL = "https://www.alphavantage.co/query"
PAGE_LIMIT = 1000


class NewsApiLimitError(RuntimeError):
    """Raised when Alpha Vantage returns a rate/quota message instead of data."""


def parse_feed(payload: dict, ticker: str) -> pd.DataFrame:
    """Article-level frame from a NEWS_SENTIMENT payload: UTC publication time,
    ticker-specific sentiment (fallback: overall), relevance, url."""
    rows = []
    for article in payload.get("feed", []):
        published = pd.to_datetime(
            article.get("time_published"), format="%Y%m%dT%H%M%S", utc=True, errors="coerce"
        )
        if published is pd.NaT:
            continue
        sentiment = article.get("overall_sentiment_score")
        relevance = None
        for entry in article.get("ticker_sentiment", []):
            if entry.get("ticker", "").upper() == ticker.upper():
                sentiment = entry.get("ticker_sentiment_score", sentiment)
                relevance = entry.get("relevance_score")
                break
        if sentiment is None:
            continue
        rows.append(
            {
                "time_published": published,
                "sentiment": float(sentiment),
                "relevance": float(relevance) if relevance is not None else float("nan"),
                "url": article.get("url", ""),
            }
        )
    df = pd.DataFrame(rows, columns=["time_published", "sentiment", "relevance", "url"])
    return df.drop_duplicates(subset="url").sort_values("time_published").reset_index(drop=True)


class AlphaVantageNewsFetcher:
    """Chunked, resumable historical news fetch. Pages forward with sort=EARLIEST;
    on a quota error it returns everything gathered so far and records the reason
    in `stopped_early` — a later run resumes from the stored articles."""

    def __init__(self, api_key: str, ticker: str, sleep_seconds: float = 13.0):
        self.api_key = api_key
        self.ticker = ticker
        self.sleep_seconds = sleep_seconds
        self.stopped_early: str | None = None

    def _page(self, time_from: pd.Timestamp, time_to: pd.Timestamp) -> dict:
        response = requests.get(
            API_URL,
            params={
                "function": "NEWS_SENTIMENT",
                "tickers": self.ticker,
                "time_from": time_from.strftime("%Y%m%dT%H%M"),
                "time_to": time_to.strftime("%Y%m%dT%H%M"),
                "sort": "EARLIEST",
                "limit": PAGE_LIMIT,
                "apikey": self.api_key,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if "feed" not in payload:
            message = (
                payload.get("Information")
                or payload.get("Note")
                or payload.get("Error Message")
                or f"unexpected payload keys: {list(payload)[:5]}"
            )
            raise NewsApiLimitError(message)
        return payload

    def fetch_range(self, start: pd.Timestamp, end: pd.Timestamp, on_page=None) -> pd.DataFrame:
        """`on_page(df)` is invoked after every fetched page so callers can persist
        incrementally — a killed process then loses at most one page."""
        self.stopped_early = None
        frames: list[pd.DataFrame] = []
        cursor = pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tz is None else start
        end = pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tz is None else end

        while cursor < end:
            try:
                payload = self._page(cursor, end)
            except NewsApiLimitError as exc:
                self.stopped_early = str(exc)
                break
            page = parse_feed(payload, self.ticker)
            if page.empty:
                break
            frames.append(page)
            if on_page is not None:
                on_page(page)
            print(
                f"  {len(page)} articles {page['time_published'].min():%Y-%m-%d} "
                f"-> {page['time_published'].max():%Y-%m-%d}"
            )
            next_cursor = page["time_published"].max() + pd.Timedelta(minutes=1)
            if next_cursor <= cursor or len(payload["feed"]) < PAGE_LIMIT:
                break
            cursor = next_cursor
            time.sleep(self.sleep_seconds)

        if not frames:
            return pd.DataFrame(columns=["time_published", "sentiment", "relevance", "url"])
        combined = pd.concat(frames, ignore_index=True)
        return (
            combined.drop_duplicates(subset="url")
            .sort_values("time_published")
            .reset_index(drop=True)
        )
