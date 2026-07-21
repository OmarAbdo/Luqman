"""Historical earnings surprises (actual vs estimated EPS).

Unlike the statement endpoints, FMP's earnings endpoint is not capped at 5
periods on the free tier, so this reaches back decades -- provided no explicit
`limit` parameter is sent.

Status: the PEAD (post-earnings-announcement drift) feature built from this was
REJECTED both single-name and cross-sectionally.
"""

import pandas as pd
import requests

FMP_URL = "https://financialmodelingprep.com/stable/earnings"


def parse_earnings(payload: list) -> pd.DataFrame:
    """Historical EPS actual-vs-estimate rows, oldest first. Rows missing
    either actual or estimated EPS (future/unreported quarters) are dropped —
    a surprise cannot be computed without both."""
    rows = []
    for item in payload:
        date = pd.to_datetime(item.get("date"), errors="coerce")
        actual = item.get("epsActual")
        estimated = item.get("epsEstimated")
        if pd.isna(date) or actual is None or estimated is None:
            continue
        rows.append({"report_date": date, "eps_actual": float(actual), "eps_estimated": float(estimated)})
    df = pd.DataFrame(rows, columns=["report_date", "eps_actual", "eps_estimated"])
    return df.sort_values("report_date").reset_index(drop=True)


def fetch_earnings(ticker: str, api_key: str) -> pd.DataFrame:
    """No `limit` param: FMP's free tier rejects an explicit `limit` above 5,
    but omitting it returns the full available history (empirically 100+
    quarters back to the 1980s for large-caps) — do not add a limit param."""
    response = requests.get(FMP_URL, params={"symbol": ticker.upper(), "apikey": api_key}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        raise RuntimeError(f"FMP error: {payload.get('Error Message') or payload}")
    df = parse_earnings(payload)
    if df.empty:
        raise RuntimeError(f"FMP returned no usable earnings rows for {ticker!r}")
    return df
