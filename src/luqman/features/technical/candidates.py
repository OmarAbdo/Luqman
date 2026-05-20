"""Each function is one hypothesis-driven technical candidate, tested in
isolation against dataset-plan.md's proof gates before combination. Every
function returns a single named Series aligned to `close`'s index — no bundled
indicator dumps.
"""

import numpy as np
import pandas as pd

MECHANISMS: dict[str, dict] = {
    "reversal_1d": {
        "mechanism": (
            "Short-term reversal: 1-day return in liquid large-caps shows "
            "documented mean-reversion from microstructure/overreaction effects."
        ),
        "horizon": 1,
    },
    "momentum_12_1": {
        "mechanism": (
            "Momentum (12-1): 12-month return excluding the most recent month "
            "predicts next-month return (Jegadeesh-Titman 1993) — one of the "
            "most replicated cross-sectional equity factors."
        ),
        "horizon": 21,
    },
    "vol_cluster": {
        "mechanism": (
            "Volatility clustering: realized volatility is strongly autocorrelated "
            "(ARCH/GARCH effects) — tested against FUTURE volatility as the "
            "target, not future return, since that is the actual documented effect."
        ),
        "horizon": 5,
    },
    "macd_divergence": {
        "mechanism": (
            "Momentum failing to confirm a new price extreme (swing-high/low "
            "based divergence between MACD and price) historically precedes "
            "short-term reversals. Must be computed on swing points, not "
            "adjacent bars — the original implementation compared bar-to-bar "
            "and had two logically unreachable branches."
        ),
        "horizon": 7,
    },
    "volume_breakout": {
        "mechanism": (
            "A price move confirmed by abnormal volume carries more information "
            "than either alone (order-flow/microstructure claim: informed "
            "trading shows up as volume, not just price)."
        ),
        "horizon": 3,
    },
    "pct_of_52w_high": {
        "mechanism": (
            "52-week high proximity (George & Hwang 2004): ratio of current "
            "price to trailing 52-week high predicts returns, an anchoring-bias "
            "effect (investors slow to update past a salient reference price). "
            "Well-replicated and found to subsume standard price momentum in "
            "explanatory power in the original study -- tested cross-sectionally "
            "as a long-short factor, same construction as momentum_12_1_xs."
        ),
        "horizon": 21,
    },
}


def reversal_1d(close: pd.Series) -> pd.Series:
    """Yesterday's return, sign-flipped as the hypothesis (mean reversion)."""
    ret = np.log(close / close.shift(1))
    return (-ret).rename("reversal_1d")


def momentum_12_1(close: pd.Series) -> pd.Series:
    """Return over the 252->21 trading-day window (12 months excluding the most
    recent month), the classic Jegadeesh-Titman formation window."""
    long_ago = close.shift(252)
    recent = close.shift(21)
    return np.log(recent / long_ago).rename("momentum_12_1")


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Trailing realized volatility — the feature; test its ability to predict
    FUTURE volatility (a separate target, built in the caller), not return."""
    logret = np.log(close / close.shift(1))
    return logret.rolling(window).std().rename("realized_vol_20d")


def future_realized_vol(close: pd.Series, horizon: int, window: int = 5) -> pd.Series:
    """Target for the vol-clustering hypothesis: realized vol over the
    `window` days starting `horizon` days ahead."""
    logret = np.log(close / close.shift(1))
    fwd_vol = logret.shift(-horizon).rolling(window).std().shift(-(window - 1))
    return fwd_vol.rename(f"future_vol_{horizon}d")


def _swing_points(series: pd.Series, order: int = 3) -> pd.Series:
    """+1 at local maxima, -1 at local minima (strict, `order`-bar neighborhoods),
    0 elsewhere. Used to find genuine swing highs/lows instead of adjacent bars."""
    values = series.to_numpy()
    n = len(values)
    labels = np.zeros(n)
    for i in range(order, n - order):
        window = values[i - order : i + order + 1]
        if values[i] == window.max() and np.argmax(window) == order:
            labels[i] = 1
        elif values[i] == window.min() and np.argmin(window) == order:
            labels[i] = -1
    return pd.Series(labels, index=series.index)


def macd_divergence(close: pd.Series, order: int = 3) -> pd.Series:
    """Bearish divergence (-1): price makes a higher swing high while MACD's
    swing high is lower than its prior swing high. Bullish (+1): mirror case
    on swing lows. 0 otherwise. Forward-filled between swing points so every
    row has a defined "current divergence regime" value, since divergence is
    only ever known exactly at swing points but persists as information until
    the next one.
    """
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26

    price_swings = _swing_points(close, order=order)
    macd_swings = _swing_points(macd, order=order)

    signal = pd.Series(0.0, index=close.index)

    price_highs = close[price_swings == 1]
    macd_highs = macd[price_swings == 1]
    for i in range(1, len(price_highs)):
        if price_highs.iloc[i] > price_highs.iloc[i - 1] and macd_highs.iloc[i] < macd_highs.iloc[i - 1]:
            signal.loc[price_highs.index[i]] = -1.0  # bearish divergence

    price_lows = close[price_swings == -1]
    macd_lows = macd[price_swings == -1]
    for i in range(1, len(price_lows)):
        if price_lows.iloc[i] < price_lows.iloc[i - 1] and macd_lows.iloc[i] > macd_lows.iloc[i - 1]:
            signal.loc[price_lows.index[i]] = 1.0  # bullish divergence

    return signal.replace(0.0, np.nan).ffill().fillna(0.0).rename("macd_divergence")


def pct_of_52w_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Current price as a fraction of the trailing `window`-day (default: 1
    trading year) rolling maximum close. Values near 1.0 = near the 52-week
    high; well below 1.0 = far from it. Strictly backward-looking (today's
    close is included in its own rolling window by construction, but the
    ratio only uses information available through today)."""
    rolling_high = close.rolling(window, min_periods=window // 2).max()
    return (close / rolling_high).rename("pct_of_52w_high")


def volume_breakout(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """Signed product of a price move's z-score and volume's z-score vs a
    trailing baseline — large in magnitude only when price move AND volume are
    both abnormal, sign following the price move's direction."""
    ret = np.log(close / close.shift(1))
    ret_z = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
    vol_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std()
    return (ret_z * vol_z.clip(lower=0)).rename("volume_breakout")
