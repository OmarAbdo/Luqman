"""Walk-forward evaluation harness.

Fits each candidate model across walk-forward folds and returns one comparison
table with the baselines included as rows, so no result can be presented
without its baseline visible alongside it.
"""

from typing import Callable

import numpy as np
import pandas as pd

from ..splits.walk_forward import WalkForwardSplitter
from .metrics import directional_accuracy, mae, mase, rmse, up_base_rate

ZERO_BASELINE = "zero_return"


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def _select(X, idx: np.ndarray) -> np.ndarray:
    """Positional row selection that works for both a tabular DataFrame
    (n, n_features) and a pre-windowed sequence array (n, seq_len, n_features).
    Both index types share the same fold positions, so a sequence model and a
    tabular model trained on the same underlying series get identical
    train/test boundaries at every fold."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[idx].to_numpy()
    return X[idx]


def walk_forward_evaluate(
    X,
    y: pd.Series,
    model_factories: dict[str, Callable[[], object]],
    splitter: WalkForwardSplitter,
) -> pd.DataFrame:
    """Fit each model fresh per fold, aggregate metrics across folds.

    X may be a DataFrame (tabular models) or a 3D array (sequence models);
    y's index defines fold positions and row alignment in both cases.

    Returns one row per model: mae / rmse / mase / dir_acc (fold means),
    plus skill_vs_zero computed against the zero-return baseline row.
    """
    folds = list(splitter.split(len(y)))
    results: dict[str, dict[str, float]] = {}

    for name, factory in model_factories.items():
        per_fold = []
        for train_idx, test_idx in folds:
            model = factory()
            model.fit(_select(X, train_idx), y.iloc[train_idx].to_numpy())
            pred = np.asarray(model.predict(_select(X, test_idx))).ravel()
            actual = y.iloc[test_idx].to_numpy()
            per_fold.append(
                {
                    "mae": mae(actual, pred),
                    "rmse": rmse(actual, pred),
                    "mase": mase(actual, pred, y.iloc[train_idx].to_numpy()),
                    "dir_acc": directional_accuracy(actual, pred),
                }
            )
        results[name] = {
            key: _nanmean([m[key] for m in per_fold]) for key in per_fold[0]
        }

    table = pd.DataFrame(results).T
    if ZERO_BASELINE in table.index:
        table["skill_vs_zero"] = 1.0 - table["mae"] / table.loc[ZERO_BASELINE, "mae"]

    test_rows = np.concatenate([test_idx for _, test_idx in folds])
    table.attrs["up_base_rate"] = up_base_rate(y.iloc[test_rows].to_numpy())
    return table.sort_values("mae")


def walk_forward_predictions(
    X,
    y: pd.Series,
    model_factory: Callable[[], object],
    splitter: WalkForwardSplitter,
) -> pd.Series:
    """Pooled out-of-fold predictions: each test row is predicted by a model
    fit only on data strictly before it (minus embargo). This is the only kind
    of prediction a backtest may consume."""
    parts = []
    for train_idx, test_idx in splitter.split(len(y)):
        model = model_factory()
        model.fit(_select(X, train_idx), y.iloc[train_idx].to_numpy())
        pred = np.asarray(model.predict(_select(X, test_idx))).ravel()
        parts.append(pd.Series(pred, index=y.index[test_idx]))
    out = pd.concat(parts).sort_index()
    out.name = "prediction"
    return out
