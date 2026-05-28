"""Value + Quality composite proof: gross profitability (Novy-Marx) as a
standalone factor, and as a quality gate combined with earnings yield,
per research-corrected dataset-plan.md Section 9. Not part of the installed
package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from fetch_universe import UNIVERSE
from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.evaluation.cross_sectional import evaluate_cross_sectional_factor
from luqman.features.quality import (
    GROSS_PROFITABILITY_MECHANISM,
    VALUE_QUALITY_MECHANISM,
    build_gross_profitability_feature,
)
from luqman.features.targets import forward_log_return
from luqman.features.value import trailing_earnings_yield

store = ParquetStore("data")


def main():
    tickers = [t for t in UNIVERSE if store.load_table(f"{t}_gross_profitability") is not None]
    print(f"Gross profitability: {len(tickers)}/{len(UNIVERSE)} tickers have data\n")

    gp_panel, ey_panel, return_panels = {}, {}, {}
    for t in tickers:
        ohlcv = store.load_raw(t)
        if not validate_daily_ohlcv(ohlcv).ok:
            continue
        gp_inputs = store.load_table(f"{t}_gross_profitability")
        gp_panel[t] = build_gross_profitability_feature(gp_inputs, ohlcv.index)

        earnings = store.load_table(f"{t}_earnings")
        if earnings is not None:
            ey_panel[t] = trailing_earnings_yield(earnings, ohlcv["close"], ohlcv.index)

        return_panels[t] = ohlcv["close"]

    # --- Standalone gross profitability (Novy-Marx), horizon 21d and 60d ---
    for horizon in [21, 60]:
        factor_panel = pd.DataFrame(gp_panel)
        return_panel = pd.DataFrame(
            {t: forward_log_return(close, horizon=horizon) for t, close in return_panels.items() if t in gp_panel}
        )
        result = evaluate_cross_sectional_factor(
            factor_panel, return_panel, f"gross_profitability_{horizon}d", quantile=0.2, min_names_per_day=6
        )
        print(result.summary())
        print()

    print(f"mechanism (standalone): {GROSS_PROFITABILITY_MECHANISM}\n")

    # --- Value + Quality composite: within below-median-P/E (top-half earnings
    # yield) names, rank by gross profitability. Only tickers with BOTH signals. ---
    common = sorted(set(gp_panel) & set(ey_panel))
    print(f"Value+Quality composite: {len(common)} tickers have both earnings-yield and gross-profitability data")

    ey_df = pd.DataFrame({t: ey_panel[t] for t in common})
    gp_df = pd.DataFrame({t: gp_panel[t] for t in common})

    for horizon in [21, 60]:
        composite_rows = {}
        for date in ey_df.index:
            ey_row = ey_df.loc[date].dropna()
            if len(ey_row) < 6:
                continue
            cheap_half = ey_row[ey_row >= ey_row.median()].index  # above-median earnings yield = "cheap" half
            gp_row = gp_df.loc[date, cheap_half].dropna()
            composite_rows[date] = gp_row  # only cheap names are ranked; expensive names excluded (NaN)

        composite_panel = pd.DataFrame(composite_rows).T.reindex(ey_df.index)
        return_panel = pd.DataFrame(
            {t: forward_log_return(return_panels[t], horizon=horizon) for t in common}
        )
        result = evaluate_cross_sectional_factor(
            composite_panel, return_panel, f"value_quality_composite_{horizon}d", quantile=0.2, min_names_per_day=4
        )
        print(result.summary())
        print()

    print(f"mechanism (composite): {VALUE_QUALITY_MECHANISM}")


if __name__ == "__main__":
    main()
