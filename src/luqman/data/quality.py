"""Balance-sheet and income-statement inputs for quality factors.

Supports the Novy-Marx gross profitability construction (gross profit / total
assets).

Status: REJECTED, but for an identified and fixable reason rather than an
absence of effect -- the raw ratio is structurally distorted by financials,
whose balance sheets are enormous relative to revenue by the nature of the
business. Excluding financials flipped the sign to the expected direction,
though the magnitude still fell short of the admission bar.
"""

import pandas as pd
import requests

BALANCE_SHEET_URL = "https://financialmodelingprep.com/stable/balance-sheet-statement"
INCOME_STATEMENT_URL = "https://financialmodelingprep.com/stable/income-statement"

# FMP's free tier caps balance-sheet/income-statement/ratios/cash-flow at 5
# annual periods (confirmed by direct testing 2026-07-20) — unlike
# /stable/earnings, which is uncapped. Do not pass `limit`.


def _fetch(url: str, ticker: str, api_key: str) -> list:
    response = requests.get(url, params={"symbol": ticker.upper(), "period": "annual", "apikey": api_key}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        raise RuntimeError(f"FMP error ({url}): {payload.get('Error Message') or payload}")
    return payload


def fetch_gross_profitability_inputs(ticker: str, api_key: str) -> pd.DataFrame:
    """Gross profit (income statement) and total assets (balance sheet) per
    fiscal year end — the exact Novy-Marx (2013) gross profitability
    construction: gross profit / total assets. Free tier caps this at 5
    years for both source statements."""
    income = _fetch(INCOME_STATEMENT_URL, ticker, api_key)
    balance = _fetch(BALANCE_SHEET_URL, ticker, api_key)

    income_by_date = {
        pd.to_datetime(item["date"]): item.get("grossProfit")
        for item in income
        if item.get("date") and item.get("grossProfit") is not None
    }
    assets_by_date = {
        pd.to_datetime(item["date"]): item.get("totalAssets")
        for item in balance
        if item.get("date") and item.get("totalAssets") is not None
    }

    common_dates = sorted(set(income_by_date) & set(assets_by_date))
    rows = [
        {
            "fiscal_year_end": date,
            "gross_profit": float(income_by_date[date]),
            "total_assets": float(assets_by_date[date]),
        }
        for date in common_dates
    ]
    return pd.DataFrame(rows)
