import numpy as np
import pytest

pytest.importorskip("arch")
from luqman.models.garch import GarchVolForecaster  # noqa: E402


def test_garch_fits_and_predicts_finite_positive_vol():
    rng = np.random.default_rng(0)
    n = 800
    returns_pct = rng.normal(0, 1.5, n)  # ~1.5% daily vol, in percent units
    X_dummy = np.zeros((n, 1))

    model = GarchVolForecaster(returns_pct=returns_pct, horizon=5, window=5)
    model.fit(X_dummy[:600], y=np.zeros(600))
    forecast = model.predict(X_dummy[600:650])

    assert forecast.shape == (50,)
    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()
    # Forecast should be in the right ballpark (return-scale std, not percent^2)
    assert 0.001 < forecast[0] < 0.1


def test_garch_forecast_reflects_higher_recent_volatility():
    rng = np.random.default_rng(1)
    calm = rng.normal(0, 0.5, 400)
    volatile = rng.normal(0, 3.0, 400)
    X_dummy = np.zeros((800, 1))

    calm_model = GarchVolForecaster(returns_pct=calm, horizon=5, window=5)
    calm_model.fit(X_dummy[:400], y=np.zeros(400))
    calm_forecast = calm_model.predict(X_dummy[400:410])

    volatile_model = GarchVolForecaster(returns_pct=volatile, horizon=5, window=5)
    volatile_model.fit(X_dummy[:400], y=np.zeros(400))
    volatile_forecast = volatile_model.predict(X_dummy[400:410])

    assert volatile_forecast[0] > calm_forecast[0]
