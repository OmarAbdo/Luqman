"""Fetch a ~50-name large-cap universe spanning all major sectors, for
cross-sectional factor testing (momentum/PEAD/relative-strength as ranked
long-short portfolios, per dataset-plan.md Section 6/6a follow-up).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luqman.data.fetcher import fetch_daily
from luqman.data.validation import validate_daily_ohlcv
from luqman.data.store import ParquetStore

# ~50 large-caps, GICS-sector-diverse, all liquid enough to trade in practice.
UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "IBM",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT",
    # Consumer discretionary / staples
    "AMZN", "TSLA", "HD", "MCD", "NKE", "WMT", "KO", "PEP", "PG", "COST",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "CAT", "BA", "HON", "UPS", "GE",
    # Communication services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA",
    # Utilities / real estate / materials (thinner but present)
    "NEE", "LIN",
]


def main():
    store = ParquetStore("data")
    ok, failed = [], []
    for ticker in UNIVERSE:
        try:
            df = fetch_daily(ticker, start="2015-01-01")
            report = validate_daily_ohlcv(df)
            if not report.ok:
                print(f"{ticker}: VALIDATION FAILED {report.issues}")
                failed.append(ticker)
                continue
            store.save_raw(ticker, df)
            ok.append(ticker)
            print(f"{ticker}: {len(df)} rows stored")
        except Exception as e:
            print(f"{ticker}: FETCH FAILED {e}")
            failed.append(ticker)
        time.sleep(0.3)

    print(f"\n{len(ok)}/{len(UNIVERSE)} stored successfully.")
    if failed:
        print(f"Failed: {failed}")


if __name__ == "__main__":
    main()
