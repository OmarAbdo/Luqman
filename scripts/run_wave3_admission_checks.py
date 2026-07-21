"""Wave 3 admission checks: costs, and 'is it just momentum?'

Three candidates cleared the statistical gates in run_wave3_residual_proofs.py.
Per this project's standing invariant that the costed backtest is the final
arbiter -- and per the §8a lesson that a signal can score well while merely
proxying a factor we already own -- none of them is admitted until it survives
this script.

Two questions are asked of each survivor:

1. Does it still pay after transaction costs, at a range of cost assumptions?
   Reversal signals in particular rebalance far more often than momentum, so a
   headline Sharpe computed without costs is not evidence of anything.

2. Is it distinct from what we already have? A candidate whose factor returns
   are highly correlated with momentum_12_1_xs is not a third feature; it is
   momentum wearing a different hat, and admitting it would double-count the
   same bet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from fetch_universe import UNIVERSE
from luqman.backtest.cross_sectional import backtest_long_short
from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.features.residual import (
    idiosyncratic_vol,
    residual_momentum,
    residual_return_series,
    residual_reversal,
)
from luqman.features.targets import forward_log_return
from luqman.features.technical.candidates import momentum_12_1

store = ParquetStore("data")
COST_LEVELS = [0.0, 5.0, 10.0, 20.0]


def load_valid(ticker):
    try:
        df = store.load_raw(ticker)
    except FileNotFoundError:
        return None
    return df if validate_daily_ohlcv(df).ok else None


def cost_curve(name, factor_panel, return_panel):
    """Report net Sharpe across a range of per-trade cost assumptions."""
    print(f"\n--- {name}: cost sensitivity ---")
    results = {}
    for bps in COST_LEVELS:
        result = backtest_long_short(factor_panel, return_panel, cost_bps=bps)
        results[bps] = result
        print(
            f"  {bps:5.1f} bps -> Sharpe {result.metrics['annualized_sharpe']:+7.3f}  "
            f"ann.return {result.metrics['annualized_return']:+7.4f}  "
            f"maxDD {result.metrics['max_drawdown']:+7.3f}"
        )
    return results


def main():
    factors = store.load_table("ff_factors_daily")
    prices = {}
    for ticker in UNIVERSE:
        ohlcv = load_valid(ticker)
        if ohlcv is not None:
            prices[ticker] = ohlcv

    print(f"Universe: {len(prices)} tickers. Building residuals...")
    residuals = {}
    for ticker, ohlcv in prices.items():
        resid = residual_return_series(ohlcv["close"], factors)
        if resid.notna().sum() > 500:
            residuals[ticker] = resid

    def forward_panel(horizon, tickers):
        return pd.DataFrame(
            {t: forward_log_return(prices[t]["close"], horizon=horizon) for t in tickers}
        )

    panels = {
        "momentum_12_1_xs (incumbent)": (
            pd.DataFrame({t: momentum_12_1(prices[t]["close"]) for t in residuals}), 21,
        ),
        "residual_momentum_xs": (
            pd.DataFrame({t: residual_momentum(r) for t, r in residuals.items()}), 21,
        ),
        "idiosyncratic_vol_xs": (
            pd.DataFrame({t: idiosyncratic_vol(r) for t, r in residuals.items()}), 21,
        ),
        "residual_reversal_xs": (
            pd.DataFrame({t: residual_reversal(r) for t, r in residuals.items()}), 5,
        ),
    }

    # --- Question 1: costs ---
    print("\n" + "=" * 78)
    print("QUESTION 1 -- does it survive transaction costs?")
    print("=" * 78)
    net_returns = {}
    for name, (panel, horizon) in panels.items():
        returns_panel = forward_panel(horizon, list(residuals))
        results = cost_curve(name, panel, returns_panel)
        net_returns[name] = results[10.0].daily_returns  # 10bps = the level momentum was admitted at

    # --- Question 2: is it distinct from momentum? ---
    print("\n" + "=" * 78)
    print("QUESTION 2 -- is it actually distinct from the momentum we already own?")
    print("=" * 78)
    aligned = pd.DataFrame(net_returns).dropna()
    correlation = aligned.corr()
    print("\nCorrelation of net daily factor returns (10 bps):")
    print(correlation.to_string(float_format=lambda v: f"{v: .3f}"))

    incumbent = "momentum_12_1_xs (incumbent)"
    print("\nCorrelation with the incumbent momentum factor:")
    for name in aligned.columns:
        if name == incumbent:
            continue
        rho = correlation.loc[name, incumbent]
        verdict = (
            "REDUNDANT (same bet)" if rho > 0.7
            else "PARTIALLY OVERLAPPING" if rho > 0.4
            else "DISTINCT"
        )
        print(f"  {name:32s} rho={rho:+.3f}  {verdict}")

    # --- Question 3: is a blend better than just using the best single factor? ---
    print("\n" + "=" * 78)
    print("QUESTION 3 -- does a blend beat the BEST SINGLE FACTOR available?")
    print("=" * 78)
    print(
        "\nBenchmarking a blend against the incumbent alone flatters it: any blend\n"
        "containing a stronger factor will beat the weaker one. The decision a user\n"
        "actually faces is 'blend, or just run the single best factor?', so the best\n"
        "single factor is the benchmark."
    )

    def sharpe_of(series):
        return series.mean() / series.std() * np.sqrt(252)

    solo = {name: sharpe_of(aligned[name]) for name in aligned.columns}
    best_name = max(solo, key=solo.get)
    print("\nSingle-factor Sharpe (10bps):")
    for name, value in sorted(solo.items(), key=lambda kv: -kv[1]):
        marker = "  <-- best single factor" if name == best_name else ""
        print(f"  {name:32s} {value:+.3f}{marker}")

    base = aligned[incumbent]
    print(f"\nBlends, judged against the best single factor ({best_name}, {solo[best_name]:+.3f}):")
    for name in aligned.columns:
        if name == incumbent:
            continue
        combo = (base + aligned[name]) / 2
        combo_sharpe = sharpe_of(combo)
        print(
            f"  50/50 {incumbent} + {name:28s} Sharpe {combo_sharpe:+.3f}  "
            f"vs best single {solo[best_name]:+.3f} -> "
            f"{'WORTH BLENDING' if combo_sharpe > solo[best_name] else 'NOT WORTH IT (dilutes the best factor)'}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
