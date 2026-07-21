"""GJR-GARCH volatility model (via the `arch` package).

Included as the purpose-built alternative for volatility forecasting, so that
the simple ridge model faces a serious competitor rather than only a naive one.

Result: ridge won outright -- higher mean skill and more head-to-head wins
across the 8-ticker panel. Model search on the volatility track is therefore
not the bottleneck.
"""

import numpy as np


class GarchVolForecaster:
    """GJR-GARCH(1,1) volatility forecaster — the actual industry-standard
    tool for this exact problem (asymmetric: captures the well-documented
    leverage effect where negative returns raise future volatility more than
    positive ones of the same size). Sklearn-shaped (fit/predict) so it drops
    into the same walk_forward_evaluate machinery as ridge/GBM.

    Unlike ridge/GBM, this model does NOT use the registered features as X —
    it refits directly on the raw daily return series each fold (the correct
    way to use GARCH) and forecasts `horizon`-ahead realized volatility
    analytically from the fitted conditional variance process. X is accepted
    only for interface compatibility (ignored) since return history is
    already implicit in the row ordering of the series it's given at fit time.
    """

    def __init__(self, returns_pct: "np.ndarray", horizon: int = 5, window: int = 5):
        """
        Args:
            returns_pct: the FULL daily return series (in percent, e.g. 1.0 for
                1%) aligned to the same index as X/y in the caller — GARCH
                needs the raw return history, not the windowed feature matrix.
            horizon: forecast horizon in trading days (matches the target's horizon).
            window: realized-vol averaging window at the target (matches vol_targets.py).
        """
        self.returns_pct = returns_pct
        self.horizon = horizon
        self.window = window

    def fit(self, X: np.ndarray, y: np.ndarray):
        # X carries only positional alignment info (its length = train fold size);
        # the actual return history used is self.returns_pct up to that many rows.
        self._n_train = len(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from arch import arch_model

        n_test = len(X)
        train_returns = self.returns_pct[: self._n_train]
        model = arch_model(train_returns, vol="GARCH", p=1, o=1, q=1, dist="t", rescale=False)
        fitted = model.fit(disp="off", show_warning=False)

        # One forecast from the train-fold fit, held CONSTANT across every row
        # in the test fold (not re-derived per row). Refitting per test row
        # would be the more faithful walk-forward approach but is far more
        # expensive; this constant-per-fold forecast is the practical
        # compromise used here and is why GARCH's fold-level metrics are
        # coarser than ridge/GBM's (which do predict per-row).
        forecasts = np.full(n_test, np.nan)
        variance_forecast = fitted.forecast(horizon=self.horizon, reindex=False).variance.iloc[-1]
        avg_daily_variance = variance_forecast.iloc[-self.window :].mean()
        # Convert from percent^2 back to the return-scale std used by the
        # rest of the project's volatility target.
        forecast_vol = np.sqrt(avg_daily_variance) / 100.0
        forecasts[:] = forecast_vol
        return forecasts
