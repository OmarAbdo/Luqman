import numpy as np
import pandas as pd
import pytest


def synthetic_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Geometric random walk with coherent OHLC bars on business days."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n, name="date")
    logret = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(logret))
    open_ = close * np.exp(rng.normal(0, 0.003, n))
    spread = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return synthetic_ohlcv()
