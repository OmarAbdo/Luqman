import pandas as pd

from .technical.candidates import future_realized_vol


def volatility_target(close: pd.Series, horizon: int = 5, window: int = 5) -> pd.Series:
    """Target for the volatility-forecasting track: realized volatility over
    `window` trading days starting `horizon` days ahead. Proven feature
    (`vol_cluster`, registry.py) predicts this, not direction — see
    dataset-plan.md Section 6."""
    return future_realized_vol(close, horizon=horizon, window=window)
