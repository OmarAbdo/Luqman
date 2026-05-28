"""Analyst estimate revisions proof: net rating level and month-over-month
revision, per research-flagged Chan/Jegadeesh/Lakonishok (1996) mechanism,
distinct from the already-rejected PEAD/earnings-surprise construction.
Not part of the installed package.
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
from luqman.features.analyst_revisions import MECHANISM, build_analyst_revision_features
from luqman.features.targets import forward_log_return

store = ParquetStore("data")


def main():
    tickers = [t for t in UNIVERSE if store.load_table(f"{t}_grades") is not None]
    print(f"Analyst grades: {len(tickers)}/{len(UNIVERSE)} tickers have data\n")

    rating_panel, revision_panel, closes = {}, {}, {}
    for t in tickers:
        ohlcv = store.load_raw(t)
        if not validate_daily_ohlcv(ohlcv).ok:
            continue
        grades = store.load_table(f"{t}_grades")
        feats = build_analyst_revision_features(grades, ohlcv.index)
        rating_panel[t] = feats["analyst_net_rating"]
        revision_panel[t] = feats["analyst_rating_revision"]
        closes[t] = ohlcv["close"]

    for name, panel_dict in [("analyst_net_rating", rating_panel), ("analyst_rating_revision", revision_panel)]:
        for horizon in [5, 21, 60]:
            factor_panel = pd.DataFrame(panel_dict)
            return_panel = pd.DataFrame({t: forward_log_return(c, horizon=horizon) for t, c in closes.items()})
            result = evaluate_cross_sectional_factor(
                factor_panel, return_panel, f"{name}_{horizon}d", quantile=0.2, min_names_per_day=6
            )
            print(result.summary())
            print()

    print(f"mechanism: {MECHANISM}")


if __name__ == "__main__":
    main()
