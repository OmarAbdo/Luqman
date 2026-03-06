import numpy as np
import pandas as pd


def forward_log_return(close: pd.Series, horizon: int = 1) -> pd.Series:
    """Target at row t = log(close[t+horizon] / close[t]).

    The last `horizon` rows are NaN (their future is unknown) and must be
    dropped at alignment time — never filled.
    """
    target = np.log(close.shift(-horizon) / close)
    target.name = f"target_logret_{horizon}d"
    return target


def direction(target: pd.Series) -> pd.Series:
    """-1 / 0 / +1 direction label derived from a return target."""
    label = np.sign(target)
    label.name = f"{target.name}_direction"
    return label
