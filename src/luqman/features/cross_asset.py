"""Cross-asset and geopolitical-proxy features.

Gold-equity divergence (flight to safety), oil shocks, and crypto risk
appetite, tested as lead-lag signals into equity returns.

Status: all REJECTED. The oil shock showed a real and perfectly sign-consistent
pattern below the admission bar, so it was followed up with a sharper test
targeting oil-sensitive names specifically -- which came back weaker and
sign-inconsistent, closing the question.
"""

import numpy as np
import pandas as pd

MECHANISMS: dict[str, dict] = {
    "gold_equity_divergence": {
        "mechanism": (
            "Flight-to-safety: during risk-off/geopolitical stress, capital "
            "rotates into gold and out of equities. A sharp POSITIVE gold "
            "return coinciding with a NEGATIVE broad-equity return (SPY) marks "
            "an acute stress day; testing whether this divergence predicts "
            "continued equity weakness (stress persists) or a rebound "
            "(overreaction reverts) over the following days."
        ),
        "horizon": 5,
    },
    "oil_shock": {
        "mechanism": (
            "Oil price shocks (large moves, either direction) are a classic "
            "transmission channel for geopolitical events (supply disruption, "
            "sanctions, conflict) into broad risk sentiment and input costs; "
            "testing whether an oil shock day leads equity returns."
        ),
        "horizon": 5,
    },
    "crypto_risk_appetite": {
        "mechanism": (
            "Crypto (BTC) as a high-beta 'risk appetite' barometer: sharp BTC "
            "drawdowns often coincide with or precede broad de-risking across "
            "other risk assets in the same liquidity-driven selloff; testing "
            "whether BTC weakness leads equity weakness."
        ),
        "horizon": 3,
    },
}


def gold_equity_divergence(gold_close: pd.Series, spy_close: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    """1-day gold return minus 1-day SPY return, lagged 1 day (signal known
    before the next day's close) — positive = gold up while equities lagged
    or fell (flight-to-safety divergence)."""
    gold_ret = np.log(gold_close / gold_close.shift(1))
    spy_ret = np.log(spy_close / spy_close.shift(1))
    divergence = (gold_ret - spy_ret).reindex(daily_index).ffill()
    return divergence.shift(1).rename("gold_equity_divergence")


def oil_shock(oil_close: pd.Series, daily_index: pd.DatetimeIndex, window: int = 20) -> pd.Series:
    """Absolute 1-day oil return, z-scored against its trailing distribution,
    signed by direction — large |z| = a shock day, sign = shock direction.
    Lagged 1 day."""
    ret = np.log(oil_close / oil_close.shift(1))
    z = (ret - ret.rolling(window).mean()) / ret.rolling(window).std()
    return z.reindex(daily_index).ffill().shift(1).rename("oil_shock")


def crypto_risk_appetite(btc_close: pd.Series, daily_index: pd.DatetimeIndex, window: int = 5) -> pd.Series:
    """Trailing `window`-day BTC return, lagged 1 day — negative values signal
    a "risk-off" crypto move that may lead broader de-risking."""
    ret = np.log(btc_close / btc_close.shift(window))
    return ret.reindex(daily_index).ffill().shift(1).rename("crypto_risk_appetite")
