import numpy as np
import pandas as pd


def make_sequences(X: pd.DataFrame, y: pd.Series, seq_len: int):
    """Window (X, y) into (n_samples, seq_len, n_features) input tensors and
    aligned targets: sample i uses rows [i, i+seq_len) of X to predict
    y at row i+seq_len-1 (the target already encodes the forward return from
    that row, so no additional shift happens here).

    Returns (X_seq, y_seq, index) where index[i] is the timestamp of the last
    row in sample i's window — the same timestamp convention as the tabular
    walk-forward pipeline, so seq_len-windowed data can reuse the identical
    split/embargo/backtest machinery without any date-math translation.
    """
    values = X.to_numpy(dtype=np.float32)
    targets = y.to_numpy(dtype=np.float32)
    n = len(X) - seq_len + 1
    if n <= 0:
        raise ValueError(f"Need at least {seq_len} rows, got {len(X)}")
    X_seq = np.stack([values[i : i + seq_len] for i in range(n)])
    y_seq = targets[seq_len - 1 :]
    index = X.index[seq_len - 1 :]
    return X_seq, y_seq, index
