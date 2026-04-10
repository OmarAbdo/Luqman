import pandas as pd
import requests

FMP_URL = "https://financialmodelingprep.com/stable/income-statement"
FREE_TIER_LIMIT = 5  # FMP free plan rejects limit > 5


def parse_income_statements(payload: list) -> pd.DataFrame:
    """Report rows keyed for point-in-time use: `accepted` is the SEC acceptance
    timestamp (falls back to filing date end-of-day when absent)."""
    rows = []
    for item in payload:
        accepted = pd.to_datetime(item.get("acceptedDate"), errors="coerce")
        if pd.isna(accepted):
            filing = pd.to_datetime(
                item.get("filingDate") or item.get("fillingDate"), errors="coerce"
            )
            accepted = filing + pd.Timedelta(hours=23, minutes=59) if not pd.isna(filing) else pd.NaT
        period_end = pd.to_datetime(item.get("date"), errors="coerce")
        if pd.isna(accepted) or pd.isna(period_end):
            continue
        rows.append(
            {
                "period_end": period_end,
                "accepted": accepted,
                "revenue": float(item.get("revenue") or 0.0),
                "net_income": float(item.get("netIncome") or 0.0),
                "eps": float(item.get("eps") or 0.0),
            }
        )
    df = pd.DataFrame(rows, columns=["period_end", "accepted", "revenue", "net_income", "eps"])
    return df.sort_values("period_end").reset_index(drop=True)


def fetch_income_statements(
    ticker: str, api_key: str, period: str = "annual", limit: int = FREE_TIER_LIMIT
) -> pd.DataFrame:
    response = requests.get(
        FMP_URL,
        params={"symbol": ticker.upper(), "period": period, "limit": limit, "apikey": api_key},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        raise RuntimeError(f"FMP error: {payload.get('Error Message') or payload}")
    df = parse_income_statements(payload)
    if df.empty:
        raise RuntimeError(f"FMP returned no income statements for {ticker!r}")
    df.attrs["period"] = period
    return df
