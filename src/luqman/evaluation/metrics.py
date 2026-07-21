"""Forecast metrics, anchored to baselines.

MAE and RMSE are included but are not the point. MASE (error scaled by a naive
forecast's error) and skill-versus-baseline are, because absolute error on
returns is nearly meaningless in isolation -- the original system drove MAE
down 7x while directional accuracy stayed below a coin flip.

Directional accuracy is always reported next to the base rate, since a stock
that rose on 54% of days makes 54% accuracy worthless.
"""

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """MAE scaled by the zero-return (persistence) forecast's MAE on the training
    set. < 1.0 means beating persistence; >= 1.0 means no skill."""
    scale = np.mean(np.abs(y_train))
    if scale == 0:
        return float("nan")
    return mae(y_true, y_pred) / float(scale)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of matching return signs, over rows where both sides take a direction.
    NaN when the model never takes a direction (e.g. the zero baseline)."""
    mask = (y_true != 0) & (y_pred != 0)
    if not mask.any():
        return float("nan")
    return float(np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask])))


def up_base_rate(y_true: np.ndarray) -> float:
    """Fraction of up moves among nonzero returns — the floor an always-up
    guesser achieves, and the number directional accuracy must be judged against."""
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(y_true[mask] > 0))
