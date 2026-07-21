"""Factor-residual returns and idiosyncratic volatility.

The idea in one sentence: instead of ranking stocks by their raw return, rank
them by the part of their return that the known factors do NOT explain.

Motivation from this project's own results (recorded in
`.refactor/dataset-plan.md`): two separate rejected signals were traced to
unintended factor exposure rather than to the absence of an effect. A raw
signal that is quietly a bet on (or against) market/size/value picks up all
the noise of those factors on top of whatever it is actually trying to
measure. Regressing it out isolates the intended content.

Method: for each ticker, run a rolling ordinary-least-squares regression of
daily excess return on the Fama-French factors,

    r_t - rf_t = alpha + b_mkt*MktRF_t + b_smb*SMB_t + b_hml*HML_t + e_t

and keep `e_t` (the residual return) and the rolling standard deviation of
`e_t` (idiosyncratic volatility). Both are strictly backward-looking: the
regression for day t is fit on a window ENDING at t, so no future data enters
the coefficients used to residualize day t.

Implementation note: the rolling regression is solved in closed form with
NumPy over sliding windows rather than by looping a fitted estimator per day.
A per-day loop over ~50 tickers x ~2,500 days is slow enough to discourage
running the proofs; the closed-form version keeps the whole wave cheap to
re-run, which matters more than it sounds when results are meant to be
reproducible on demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MECHANISMS: dict[str, dict] = {
    "residual_momentum": {
        "mechanism": (
            "Residual momentum (Blitz, Huij & Martens 2011): the 12-1 momentum "
            "ranking computed on Fama-French residual returns rather than raw "
            "returns. Stripping market/size/value exposure removes drift that "
            "is unrelated to the firm-specific trend momentum is meant to "
            "capture, historically giving a higher information ratio and much "
            "lower factor exposure than raw momentum."
        ),
        "horizon": 21,
    },
    "idiosyncratic_vol": {
        "mechanism": (
            "Idiosyncratic volatility (Ang, Hodrick, Xing & Zhang 2006): the "
            "volatility of Fama-French residual returns. Documented as a "
            "cross-sectional return predictor (the 'IVOL puzzle', historically "
            "negative: high-IVOL names underperform), distinct from total "
            "volatility because it excludes compensated factor risk."
        ),
        "horizon": 21,
    },
    "residual_reversal": {
        "mechanism": (
            "Residual short-term reversal (Blitz, Huij, Lansdorp & Martens "
            "2013): short-horizon mean reversion measured on residual rather "
            "than raw returns. Raw short-term reversal is documented to have "
            "decayed, in part because the raw version takes unintended "
            "positions against factor momentum over the formation window; the "
            "residual version neutralizes that exposure."
        ),
        "horizon": 5,
    },
    "long_term_reversal": {
        "mechanism": (
            "Long-term reversal (De Bondt & Thaler 1985): returns over a 3-5 "
            "year formation window revert, attributed to prolonged investor "
            "over-reaction correcting. Deliberately skips the most recent year "
            "so the measure does not collide with 12-1 momentum, which "
            "operates in the opposite direction at a shorter horizon."
        ),
        "horizon": 21,
    },
    "amihud_illiquidity": {
        "mechanism": (
            "Amihud (2002) illiquidity: average ratio of absolute return to "
            "dollar volume, i.e. how much the price moves per dollar traded. "
            "Illiquid names should earn a premium as compensation for trading "
            "costs. Expected to be weak here by construction -- the premium is "
            "documented in small and illiquid stocks, and this universe is "
            "deliberately all large, liquid names."
        ),
        "horizon": 21,
    },
}


def rolling_ols_residuals(
    y: pd.Series,
    factors: pd.DataFrame,
    window: int = 252,
    min_periods: int | None = None,
) -> pd.Series:
    """Residuals from a rolling OLS of `y` on `factors` (plus an intercept).

    The residual reported at date t is the residual of the LAST observation in
    a window ending at t, using coefficients fit on that same window. Windows
    ending before `min_periods` observations are available yield NaN.

    Both inputs must already be aligned on a common index by the caller.
    """
    if min_periods is None:
        min_periods = window // 2

    y_values = y.to_numpy(dtype=float)
    x_values = factors.to_numpy(dtype=float)
    n_obs, n_factors = x_values.shape

    # Design matrix with intercept.
    design = np.column_stack([np.ones(n_obs), x_values])
    n_params = n_factors + 1

    residuals = np.full(n_obs, np.nan)
    valid = np.isfinite(y_values) & np.isfinite(design).all(axis=1)

    for end in range(n_obs):
        start = max(0, end - window + 1)
        rows = np.arange(start, end + 1)
        rows = rows[valid[rows]]
        if len(rows) < max(min_periods, n_params + 1) or not valid[end]:
            continue

        x_win = design[rows]
        y_win = y_values[rows]
        # lstsq is used rather than the normal equations because factor columns
        # can be near-collinear in short windows; lstsq degrades gracefully.
        coef, *_ = np.linalg.lstsq(x_win, y_win, rcond=None)
        residuals[end] = y_values[end] - design[end] @ coef

    return pd.Series(residuals, index=y.index, name=f"resid_{y.name or 'ret'}")


def residual_return_series(
    close: pd.Series,
    factors: pd.DataFrame,
    window: int = 252,
) -> pd.Series:
    """Daily Fama-French residual return for one ticker.

    `factors` must contain mkt_rf, smb, hml and rf as decimal daily returns
    (see `data/fama_french.py`). Simple (not log) returns are used because the
    factor series are themselves simple returns; mixing the two would bias the
    regression.
    """
    simple_return = close.pct_change()
    aligned = pd.concat([simple_return.rename("ret"), factors], axis=1, join="inner").dropna()
    if aligned.empty:
        return pd.Series(dtype=float, name="residual_return")

    excess = aligned["ret"] - aligned["rf"]
    excess.name = "excess"
    residual = rolling_ols_residuals(excess, aligned[["mkt_rf", "smb", "hml"]], window=window)
    return residual.rename("residual_return")


def residual_momentum(residual_return: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """12-1 momentum computed on residual returns: the sum of residual returns
    over the formation window, excluding the most recent `skip` days.

    Residuals are summed rather than compounded because they are already
    additive deviations from a factor model, not gross returns.
    """
    cumulative = residual_return.rolling(lookback - skip, min_periods=(lookback - skip) // 2).sum()
    return cumulative.shift(skip).rename("residual_momentum")


def idiosyncratic_vol(residual_return: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of residual returns."""
    return residual_return.rolling(window, min_periods=window // 2).std().rename("idiosyncratic_vol")


def residual_reversal(residual_return: pd.Series, window: int = 5) -> pd.Series:
    """Short-horizon residual reversal: recent cumulative residual return,
    sign-flipped so that a HIGH feature value means 'expected to rise'
    (i.e. the name has recently underperformed its factor exposure)."""
    recent = residual_return.rolling(window, min_periods=window).sum()
    return (-recent).rename("residual_reversal")


def long_term_reversal(close: pd.Series, lookback: int = 1260, skip: int = 252) -> pd.Series:
    """De Bondt-Thaler long-term reversal: cumulative log return over roughly
    five years, skipping the most recent year, sign-flipped so a high value
    means 'past loser, expected to revert upward'.

    Needs `lookback` days of history before producing any value, which is why
    the universe must be re-fetched with a long start date before this is
    testable.
    """
    log_price = np.log(close)
    formation = log_price.shift(skip) - log_price.shift(lookback)
    return (-formation).rename("long_term_reversal")


def amihud_illiquidity(close: pd.Series, volume: pd.Series, window: int = 21) -> pd.Series:
    """Amihud illiquidity: rolling mean of |daily return| / dollar volume.

    Scaled by 1e6 purely to keep the numbers in a readable range; the factor
    test is rank-based, so the scale has no effect on any result.
    """
    ret = close.pct_change().abs()
    dollar_volume = (close * volume).replace(0.0, np.nan)
    illiquidity = (ret / dollar_volume) * 1e6
    return illiquidity.rolling(window, min_periods=window // 2).mean().rename("amihud_illiquidity")
