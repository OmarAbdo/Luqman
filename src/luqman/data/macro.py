"""Daily macroeconomic series (Alpha Vantage).

Restricted to market-quoted series (treasury yields, fed funds) that have an
unambiguous daily observation. Release-based indicators such as CPI are
deliberately excluded: using them correctly requires modelling publication
dates, and using them incorrectly is a straightforward source of look-ahead.

Status: macro features were REJECTED -- see dataset-plan.md.
"""

import time

import pandas as pd
import requests

API_URL = "https://www.alphavantage.co/query"


def parse_av_series(payload: dict, name: str) -> pd.Series:
    """Alpha Vantage economic-indicator payload -> float series indexed by date."""
    if "data" not in payload:
        message = (
            payload.get("Information")
            or payload.get("Note")
            or payload.get("Error Message")
            or f"unexpected payload keys: {list(payload)[:5]}"
        )
        raise RuntimeError(f"Alpha Vantage macro error ({name}): {message}")
    rows = {
        pd.to_datetime(item["date"]): float(item["value"])
        for item in payload["data"]
        if item.get("value") not in (None, ".", "")
    }
    series = pd.Series(rows, name=name).sort_index()
    series.index.name = "date"
    return series


class AlphaVantageMacroFetcher:
    """Daily market-quoted macro series only (treasury yields, fed funds) —
    they have no publication lag ambiguity beyond one day, unlike CPI-style
    releases which need explicit release-date modeling before use."""

    def __init__(self, api_key: str, sleep_seconds: float = 13.0):
        self.api_key = api_key
        self.sleep_seconds = sleep_seconds

    def _series(self, params: dict, name: str) -> pd.Series:
        response = requests.get(
            API_URL, params={**params, "apikey": self.api_key}, timeout=60
        )
        response.raise_for_status()
        return parse_av_series(response.json(), name)

    def fetch(self) -> pd.DataFrame:
        y10 = self._series(
            {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "10year"}, "y10"
        )
        time.sleep(self.sleep_seconds)
        y3m = self._series(
            {"function": "TREASURY_YIELD", "interval": "daily", "maturity": "3month"}, "y3m"
        )
        time.sleep(self.sleep_seconds)
        fedfunds = self._series(
            {"function": "FEDERAL_FUNDS_RATE", "interval": "daily"}, "fedfunds"
        )
        return pd.concat([y10, y3m, fedfunds], axis=1).sort_index()
