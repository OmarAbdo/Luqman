"""Daily price fetching and schema normalization.

Provider frames are normalized to one canonical shape (tz-naive DatetimeIndex
named `date`, lowercase OHLCV columns, sorted, deduplicated) so that nothing
downstream needs to know which provider the data came from. Prices are
split/dividend adjusted.

Fetches fail loudly on an empty result rather than returning an empty frame,
because a silently empty frame propagates into features and evaluation and
surfaces much later as an unexplained result.
"""

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def normalize_ohlcv(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a provider frame to the canonical daily schema:
    tz-naive DatetimeIndex named 'date', lowercase OHLCV columns, sorted, deduplicated."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    df = df[OHLCV_COLUMNS]
    index = pd.to_datetime(df.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    df.index = index.normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_daily(ticker: str, start: str = "2000-01-01", end: str | None = None) -> pd.DataFrame:
    """Fetch split/dividend-adjusted daily OHLCV. Fails loudly on empty results."""
    import yfinance as yf

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"No daily data returned for {ticker!r}")
    return normalize_ohlcv(raw)
