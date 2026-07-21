"""Why does our idiosyncratic-volatility result have the OPPOSITE sign to the
documented IVOL puzzle?

Ang, Hodrick, Xing & Zhang (2006) document that HIGH idiosyncratic volatility
predicts LOW returns. Our cross-sectional test found the reverse, and with a
healthy Sharpe. Under this project's rules a result that contradicts its own
stated mechanism cannot be admitted on the strength of its statistics alone --
either the mechanism is wrong, the construction is wrong, or we are measuring
something else and calling it IVOL.

This script asks which, by inspecting WHAT the factor is actually holding.
The hypothesis under test: Ang et al. measure the full US cross-section, where
the high-IVOL bucket is dominated by small speculative names (the 'lottery
stock' effect). Our universe is 52 mega-caps by construction, so its high-IVOL
bucket cannot contain lottery stocks -- it will instead contain high-growth
large caps. If so, we have not replicated Ang's effect with a flipped sign; we
have measured a different population and should say so.
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from fetch_universe import UNIVERSE
from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.features.residual import idiosyncratic_vol, residual_return_series

store = ParquetStore("data")


def load_valid(ticker):
    try:
        df = store.load_raw(ticker)
    except FileNotFoundError:
        return None
    return df if validate_daily_ohlcv(df).ok else None


def main():
    factors = store.load_table("ff_factors_daily")
    prices, residuals = {}, {}
    for ticker in UNIVERSE:
        ohlcv = load_valid(ticker)
        if ohlcv is None:
            continue
        prices[ticker] = ohlcv
        resid = residual_return_series(ohlcv["close"], factors)
        if resid.notna().sum() > 500:
            residuals[ticker] = resid

    ivol = pd.DataFrame({t: idiosyncratic_vol(r) for t, r in residuals.items()})

    # Which names actually populate each leg, pooled over the whole sample and
    # over each half (the factor's Sharpe differed 5x between halves).
    def leg_membership(panel, start=None, end=None, quantile=0.2):
        sliced = panel.loc[start:end]
        long_counts, short_counts = Counter(), Counter()
        for date in sliced.index:
            row = sliced.loc[date].dropna()
            if len(row) < 10:
                continue
            ranked = row.sort_values()
            cut = max(1, int(len(ranked) * quantile))
            short_counts.update(ranked.index[:cut])
            long_counts.update(ranked.index[-cut:])
        return long_counts, short_counts

    dates = ivol.dropna(how="all").index
    midpoint = dates[len(dates) // 2]

    for label, (start, end) in {
        "FULL SAMPLE": (None, None),
        f"1st half (<= {midpoint:%Y-%m})": (None, midpoint),
        f"2nd half (>  {midpoint:%Y-%m})": (midpoint, None),
    }.items():
        long_counts, short_counts = leg_membership(ivol, start, end)
        total_days = sum(long_counts.values()) / max(1, len(long_counts))
        print(f"\n=== {label} ===")
        print("LONG leg (highest IVOL) most frequent names:")
        for name, count in long_counts.most_common(10):
            print(f"   {name:6s} {count:5d} days")
        print("SHORT leg (lowest IVOL) most frequent names:")
        for name, count in short_counts.most_common(10):
            print(f"   {name:6s} {count:5d} days")

    # Does the high-IVOL leg simply have higher market beta? If the factor is
    # really a disguised beta bet, that is a compensated risk exposure, not
    # alpha, and it would also explain the strong second half (a decade-plus
    # bull market).
    print("\n=== Beta of each leg (full-sample CAPM beta vs Mkt-RF) ===")
    betas = {}
    for ticker, ohlcv in prices.items():
        if ticker not in residuals:
            continue
        ret = ohlcv["close"].pct_change()
        joined = pd.concat([ret.rename("r"), factors[["mkt_rf", "rf"]]], axis=1).dropna()
        excess = joined["r"] - joined["rf"]
        betas[ticker] = np.polyfit(joined["mkt_rf"], excess, 1)[0]
    beta_series = pd.Series(betas)

    mean_ivol = ivol.mean()
    combined = pd.DataFrame({"mean_ivol": mean_ivol, "beta": beta_series}).dropna()
    high_ivol_names = combined.nlargest(10, "mean_ivol")
    low_ivol_names = combined.nsmallest(10, "mean_ivol")
    print(f"\nHighest-IVOL names  -- mean beta {high_ivol_names['beta'].mean():.3f}")
    print(high_ivol_names.to_string(float_format=lambda v: f"{v: .3f}"))
    print(f"\nLowest-IVOL names   -- mean beta {low_ivol_names['beta'].mean():.3f}")
    print(low_ivol_names.to_string(float_format=lambda v: f"{v: .3f}"))
    print(
        f"\ncorrelation(mean IVOL, market beta) across the universe: "
        f"{combined['mean_ivol'].corr(combined['beta']):+.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
