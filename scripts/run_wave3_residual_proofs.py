"""Wave 3 (dataset-plan.md Section 10): the residualization wave.

Tests six candidates whose common theme is that a raw signal may fail because
it carries unintended factor exposure, and that regressing those factors out
can isolate the intended content. Four of the six are computed from
Fama-French residual returns; the other two (long-term reversal, Amihud
illiquidity) are included because they are cheap given the extended history
fetched for this wave.

Every candidate is evaluated cross-sectionally with the same long-short
methodology used to admit momentum_12_1_xs, plus the split-half sign-stability
check that rejected pct_of_52w_high_xs. Not part of the installed package.
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
from luqman.features.residual import (
    MECHANISMS,
    amihud_illiquidity,
    idiosyncratic_vol,
    long_term_reversal,
    residual_momentum,
    residual_return_series,
    residual_reversal,
)
from luqman.features.targets import forward_log_return

store = ParquetStore("data")


def load_valid(ticker):
    try:
        df = store.load_raw(ticker)
    except FileNotFoundError:
        return None
    return df if validate_daily_ohlcv(df).ok else None


def split_half_check(factor_panel, return_panel, name, quantile=0.2):
    """Re-run the factor test on each chronological half separately.

    This is the check that disqualified the 52-week-high candidate: a factor
    whose long-short Sharpe FLIPS SIGN between two independent multi-year
    windows is not showing a stable effect, however good the pooled number
    looks. Reported for every candidate, pass or fail.
    """
    dates = factor_panel.index
    midpoint = dates[len(dates) // 2]
    first = evaluate_cross_sectional_factor(
        factor_panel.loc[:midpoint], return_panel.loc[:midpoint], f"{name}[1st half]", quantile=quantile
    )
    second = evaluate_cross_sectional_factor(
        factor_panel.loc[midpoint:], return_panel.loc[midpoint:], f"{name}[2nd half]", quantile=quantile
    )
    stable = (
        np.sign(first.annualized_sharpe) == np.sign(second.annualized_sharpe)
        and np.isfinite(first.annualized_sharpe)
        and np.isfinite(second.annualized_sharpe)
    )
    return first, second, stable


def report(factor_panel, return_panel, name, mechanism, horizon, quantile=0.2):
    result = evaluate_cross_sectional_factor(factor_panel, return_panel, name, quantile=quantile)
    first, second, stable = split_half_check(factor_panel, return_panel, name, quantile=quantile)

    print(result.summary())
    print(f"mechanism: {mechanism}")
    print(f"horizon: {horizon}d")
    print(
        f"split-half Sharpe: 1st {first.annualized_sharpe:+.3f} | "
        f"2nd {second.annualized_sharpe:+.3f} | "
        f"sign-stable: {'YES' if stable else 'NO -- disqualifying'}"
    )
    verdict = "ADMITTED" if (result.verdict == "ADMITTED" and stable) else "REJECTED"
    print(f"FINAL VERDICT (incl. split-half): {verdict}")
    print("-" * 78)
    return result, stable, verdict


def build_residual_panel(tickers, factors, window=252):
    """Residual return series for every ticker with enough history."""
    residuals = {}
    for ticker in tickers:
        ohlcv = load_valid(ticker)
        if ohlcv is None:
            continue
        resid = residual_return_series(ohlcv["close"], factors, window=window)
        if resid.notna().sum() > 500:
            residuals[ticker] = resid
    return residuals


def main():
    factors = store.load_table("ff_factors_daily")
    if factors is None:
        print("Missing Fama-French factors. Run: luqman fetch-factors")
        return 1

    print(f"Fama-French factors: {factors.index.min():%Y-%m-%d} -> {factors.index.max():%Y-%m-%d}\n")
    print("Computing rolling factor residuals for the universe (this is the slow step)...")
    residuals = build_residual_panel(UNIVERSE, factors)
    print(f"Residual series built for {len(residuals)}/{len(UNIVERSE)} tickers.\n")

    prices = {t: load_valid(t) for t in residuals}

    def forward_panel(horizon):
        return pd.DataFrame(
            {t: forward_log_return(prices[t]["close"], horizon=horizon) for t in residuals}
        )

    results = {}

    # --- 1. Residual momentum (the headline candidate) ---
    horizon = MECHANISMS["residual_momentum"]["horizon"]
    panel = pd.DataFrame({t: residual_momentum(r) for t, r in residuals.items()})
    results["residual_momentum_xs"] = report(
        panel, forward_panel(horizon), "residual_momentum_xs",
        MECHANISMS["residual_momentum"]["mechanism"], horizon,
    )

    # --- 2. Idiosyncratic volatility ---
    horizon = MECHANISMS["idiosyncratic_vol"]["horizon"]
    panel = pd.DataFrame({t: idiosyncratic_vol(r) for t, r in residuals.items()})
    results["idiosyncratic_vol_xs"] = report(
        panel, forward_panel(horizon), "idiosyncratic_vol_xs",
        MECHANISMS["idiosyncratic_vol"]["mechanism"], horizon,
    )
    # The documented IVOL effect is NEGATIVE (high IVOL underperforms), so the
    # tradable construction is the sign-flipped factor. Reported separately
    # rather than silently flipping, because choosing a sign after seeing the
    # result would be fitting the gate to the data.
    results["low_ivol_xs"] = report(
        -panel, forward_panel(horizon), "low_ivol_xs (sign-flipped: long LOW ivol)",
        "Low-volatility anomaly: the documented direction of the IVOL effect.", horizon,
    )

    # --- 3. Residual short-term reversal ---
    horizon = MECHANISMS["residual_reversal"]["horizon"]
    panel = pd.DataFrame({t: residual_reversal(r) for t, r in residuals.items()})
    results["residual_reversal_xs"] = report(
        panel, forward_panel(horizon), "residual_reversal_xs",
        MECHANISMS["residual_reversal"]["mechanism"], horizon,
    )

    # --- 4. Long-term reversal (needs the extended history) ---
    horizon = MECHANISMS["long_term_reversal"]["horizon"]
    panel = pd.DataFrame({t: long_term_reversal(prices[t]["close"]) for t in residuals})
    results["long_term_reversal_xs"] = report(
        panel, forward_panel(horizon), "long_term_reversal_xs",
        MECHANISMS["long_term_reversal"]["mechanism"], horizon,
    )

    # --- 5. Amihud illiquidity ---
    horizon = MECHANISMS["amihud_illiquidity"]["horizon"]
    panel = pd.DataFrame(
        {t: amihud_illiquidity(prices[t]["close"], prices[t]["volume"]) for t in residuals}
    )
    results["amihud_illiquidity_xs"] = report(
        panel, forward_panel(horizon), "amihud_illiquidity_xs",
        MECHANISMS["amihud_illiquidity"]["mechanism"], horizon,
    )

    print("\n=== WAVE 3 SUMMARY ===")
    for name, (result, stable, verdict) in results.items():
        print(
            f"{name:42s} Sharpe {result.annualized_sharpe:+7.3f}  "
            f"IC {result.ic_mean:+.4f}  stable={'Y' if stable else 'N'}  {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
