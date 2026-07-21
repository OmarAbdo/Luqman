"""Fetch a ~50-name large-cap universe spanning all major sectors, for
cross-sectional factor testing (momentum/PEAD/relative-strength as ranked
long-short portfolios, per dataset-plan.md Section 6/6a follow-up).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from luqman.data.fetcher import fetch_daily
from luqman.data.store import ParquetStore
from luqman.data.universe import EQUITY_UNIVERSE
from luqman.data.validation import validate_daily_ohlcv

# The universe definition lives in the package (luqman.data.universe) so that
# the CLI and these scripts rank an identical set of names.
UNIVERSE = EQUITY_UNIVERSE


def main():
    # Wave 3 needs a long history: long-term reversal uses a 5-year formation
    # window, which leaves no usable evaluation period if the data starts in
    # 2015. Survivorship caveat: this universe was selected from 2026 large-caps,
    # so pre-2015 history is survivorship-biased and any result relying on it
    # must say so.
    start = sys.argv[1] if len(sys.argv) > 1 else "2000-01-01"
    print(f"Fetching {len(UNIVERSE)} tickers from {start}\n")

    store = ParquetStore("data")
    ok, failed = [], []
    for ticker in UNIVERSE:
        try:
            df = fetch_daily(ticker, start=start)
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
