"""Cross-sectional re-test of momentum, relative-strength, and PEAD as
long-short ranked portfolios across the ~50-name universe, per owner
direction (2026-07-20): re-test what we already built properly before
sourcing new data. Not part of the installed package.
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
from luqman.features.pead import build_pead_feature
from luqman.features.relative_strength import relative_strength
from luqman.features.targets import forward_log_return
from luqman.features.technical.candidates import momentum_12_1

store = ParquetStore("data")


def load_valid(ticker):
    df = store.load_raw(ticker)
    return df if validate_daily_ohlcv(df).ok else None


def build_panel(feature_fn, tickers, horizon):
    """feature_fn(close_series) -> Series. Returns (factor_panel, forward_return_panel), wide by ticker."""
    factor_cols, return_cols = {}, {}
    for t in tickers:
        ohlcv = load_valid(t)
        if ohlcv is None:
            continue
        feature = feature_fn(ohlcv["close"])
        target = forward_log_return(ohlcv["close"], horizon=horizon)
        factor_cols[t] = feature
        return_cols[t] = target
    factor_panel = pd.DataFrame(factor_cols)
    return_panel = pd.DataFrame(return_cols)
    return factor_panel, return_panel


def main():
    print(f"Universe: {len(UNIVERSE)} tickers requested\n")

    # --- Momentum 12-1, horizon 21d (matches its own mechanism) ---
    factor_panel, return_panel = build_panel(momentum_12_1, UNIVERSE, horizon=21)
    result = evaluate_cross_sectional_factor(factor_panel, return_panel, "momentum_12_1_xs", quantile=0.2)
    print(result.summary())
    print()

    # --- Relative strength vs SPY, horizon 20d ---
    spy = store.load_raw("SPY")["close"]
    rs_fn = lambda close: relative_strength(close, spy, window=20)
    factor_panel, return_panel = build_panel(rs_fn, UNIVERSE, horizon=20)
    result = evaluate_cross_sectional_factor(factor_panel, return_panel, "relative_strength_xs", quantile=0.2)
    print(result.summary())
    print()

    # --- PEAD, horizon 20d, only tickers with earnings data ---
    earnings_tickers = [t for t in UNIVERSE if store.load_table(f"{t}_earnings") is not None]
    print(f"PEAD: {len(earnings_tickers)}/{len(UNIVERSE)} tickers have earnings data")

    def pead_fn(ohlcv, ticker):
        earnings = store.load_table(f"{ticker}_earnings")
        return build_pead_feature(earnings, ohlcv.index)

    factor_cols, return_cols = {}, {}
    for t in earnings_tickers:
        ohlcv = load_valid(t)
        if ohlcv is None:
            continue
        factor_cols[t] = pead_fn(ohlcv, t)
        return_cols[t] = forward_log_return(ohlcv["close"], horizon=20)
    factor_panel = pd.DataFrame(factor_cols)
    return_panel = pd.DataFrame(return_cols)
    result = evaluate_cross_sectional_factor(factor_panel, return_panel, "pead_xs", quantile=0.2, min_names_per_day=6)
    print(result.summary())


if __name__ == "__main__":
    main()
