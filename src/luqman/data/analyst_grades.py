"""Analyst rating history (Financial Modeling Prep).

This endpoint has genuinely deep free-tier history (~90 months per ticker),
unlike the statement endpoints.

Status: REJECTED, both the rating level and the month-over-month revision. The
revision signal initially showed a striking Sharpe that was diagnosed as an
artifact: a monthly-updating signal evaluated under daily rebalancing produces
portfolio turnover effects rather than genuine predictive ranking.
"""

import pandas as pd
import requests

GRADES_URL = "https://financialmodelingprep.com/stable/grades-historical"


def fetch_grades_historical(ticker: str, api_key: str) -> pd.DataFrame:
    """Monthly analyst rating distribution (strong buy/buy/hold/sell/strong
    sell counts). Free tier: genuinely deep history (~90+ months for AAPL,
    back to 2018), unlike the 5-year caps on financial-statement endpoints."""
    response = requests.get(GRADES_URL, params={"symbol": ticker.upper(), "apikey": api_key}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        raise RuntimeError(f"FMP error: {payload.get('Error Message') or payload}")
    rows = []
    for item in payload:
        date = pd.to_datetime(item.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        rows.append(
            {
                "month": date,
                "strong_buy": item.get("analystRatingsStrongBuy", 0) or 0,
                "buy": item.get("analystRatingsBuy", 0) or 0,
                "hold": item.get("analystRatingsHold", 0) or 0,
                "sell": item.get("analystRatingsSell", 0) or 0,
                "strong_sell": item.get("analystRatingsStrongSell", 0) or 0,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("month").reset_index(drop=True)
