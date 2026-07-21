"""Relative strength versus a benchmark (SPY).

Status: REJECTED, both single-name and cross-sectionally, with IC
indistinguishable from zero in both framings.
"""

import numpy as np
import pandas as pd

MECHANISM = (
    "Relative strength: a stock's return relative to its sector/index benchmark "
    "persists over medium horizons (relative-strength persistence is better "
    "documented than a direct macro rate-level effect on single-name returns)."
)


def relative_strength(ticker_close: pd.Series, benchmark_close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing `window`-day return of ticker minus benchmark — the feature;
    mechanism predicts this persists, i.e. it should relate to the SAME
    metric computed over the subsequent window (tested by the caller)."""
    ticker_ret = np.log(ticker_close / ticker_close.shift(window))
    bench_ret = np.log(benchmark_close / benchmark_close.shift(window))
    aligned_bench = bench_ret.reindex(ticker_ret.index).ffill()
    return (ticker_ret - aligned_bench).rename("relative_strength_20d")
