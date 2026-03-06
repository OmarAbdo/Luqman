import numpy as np
import pandas as pd

# Longest backward lookback across all features (BB/vol windows = 20, RSI = 14,
# MACD EMAs effectively ~35). Rows before this are dropped, never imputed.
WARMUP_ROWS = 40


def wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI with Wilder smoothing, scaled to [0, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    return avg_gain / (avg_gain + avg_loss)


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Stationary, strictly backward-looking features from a validated daily OHLCV frame.

    No fitted state: every column is a rolling/ewm transform of past data only,
    so the same function is leakage-safe for any train/test boundary.
    """
    close = ohlcv["close"]
    volume = ohlcv["volume"]
    logret = np.log(close / close.shift(1))

    feats = pd.DataFrame(index=ohlcv.index)

    feats["ret_1d"] = logret
    feats["ret_5d"] = np.log(close / close.shift(5))
    feats["ret_20d"] = np.log(close / close.shift(20))
    feats["vol_20d"] = logret.rolling(20).std()
    feats["volume_ratio"] = np.log1p(volume) - np.log1p(volume.rolling(20).mean())
    feats["rsi_14"] = wilder_rsi(close)

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False).mean()
    feats["macd_hist_norm"] = (macd - signal) / close

    bb_ma = close.rolling(20).mean()
    bb_sd = close.rolling(20).std()
    feats["bb_pct_b"] = (close - (bb_ma - 2 * bb_sd)) / (4 * bb_sd)
    feats["bb_bandwidth"] = (4 * bb_sd) / bb_ma

    dow = ohlcv.index.dayofweek
    month = ohlcv.index.month
    feats["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    feats["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    feats["month_sin"] = np.sin(2 * np.pi * month / 12)
    feats["month_cos"] = np.cos(2 * np.pi * month / 12)

    feats = feats.iloc[WARMUP_ROWS:]
    return feats.replace([np.inf, -np.inf], np.nan).dropna()
