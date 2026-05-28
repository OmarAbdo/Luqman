"""Cross-sectional value-factor proof: trailing earnings yield, per owner's
Buffett/Lynch/Munger suggestion (2026-07-20). Not part of the installed package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from fetch_universe import UNIVERSE
from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.evaluation.cross_sectional import evaluate_cross_sectional_factor
from luqman.features.targets import forward_log_return
from luqman.features.value import MECHANISM, trailing_earnings_yield

store = ParquetStore("data")


def main():
    earnings_tickers = [t for t in UNIVERSE if store.load_table(f"{t}_earnings") is not None]
    print(f"Value factor: {len(earnings_tickers)}/{len(UNIVERSE)} tickers have earnings data\n")

    for horizon in [21, 60]:  # value is a slower-moving effect than momentum; test both
        factor_cols, return_cols = {}, {}
        for t in earnings_tickers:
            ohlcv = store.load_raw(t)
            if not validate_daily_ohlcv(ohlcv).ok:
                continue
            earnings = store.load_table(f"{t}_earnings")
            factor_cols[t] = trailing_earnings_yield(earnings, ohlcv["close"], ohlcv.index)
            return_cols[t] = forward_log_return(ohlcv["close"], horizon=horizon)

        factor_panel = pd.DataFrame(factor_cols)
        return_panel = pd.DataFrame(return_cols)
        result = evaluate_cross_sectional_factor(
            factor_panel, return_panel, f"value_earnings_yield_{horizon}d", quantile=0.2, min_names_per_day=6
        )
        print(result.summary())
        print()

    print(f"mechanism: {MECHANISM}")


if __name__ == "__main__":
    main()
