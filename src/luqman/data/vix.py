import pandas as pd

from .fetcher import normalize_ohlcv


def fetch_vix(start: str = "2000-01-01", end: str | None = None) -> pd.Series:
    """Daily VIX close via yfinance ^VIX — a standard, immediately-available
    risk-off/stress proxy, tested as a cheap stand-in for a full geopolitical
    event feed before investing in one (dataset-plan.md Section 2e)."""
    import yfinance as yf

    raw = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("No VIX data returned")
    return normalize_ohlcv(raw)["close"].rename("vix")
