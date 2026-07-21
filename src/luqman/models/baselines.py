"""Baseline models and the standard ridge pipeline.

The baselines are the point of this module. Every model ever evaluated in this
project is reported against zero-return (persistence) and historical-mean
baselines, without exception, because the original system's headline results
came from a model that had learned to echo the previous price -- excellent MAE,
no skill. A model that cannot beat persistence has not demonstrated anything.

Scaling lives inside the pipeline so it is fit per training fold, never across
the full sample.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ZeroReturnBaseline:
    """Persistence in price space: predicted log return is always 0.
    Every model must beat this to claim any skill."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class HistoricalMeanBaseline:
    """Predicts the training-set mean return (drift)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


def make_ridge(alpha: float = 1.0) -> Pipeline:
    """First real model. The scaler lives inside the pipeline so it is re-fit
    on each training fold — scaling can never leak across the fold boundary."""
    return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
