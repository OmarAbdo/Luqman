"""Tests for the HTTP serving surface.

These use a temporary store seeded with synthetic data rather than the real
one, so they neither depend on what happens to be fetched locally nor take
minutes to run.

The most important test here is not that the endpoints return 200 -- it is
`test_rejected_factor_is_not_servable`, which pins the guarantee that a
candidate the proof process rejected cannot be served no matter how good its
raw statistics looked.
"""

import numpy as np
import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi", reason="serving extra not installed")
from fastapi.testclient import TestClient  # noqa: E402

from luqman.data.store import ParquetStore  # noqa: E402
from luqman.serving import service  # noqa: E402
from luqman.serving.api import app  # noqa: E402


def _synthetic_ohlcv(seed: int, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n, name="date")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=index,
    )


@pytest.fixture
def seeded_store(tmp_path):
    store = ParquetStore(tmp_path)
    for i, ticker in enumerate(["AAPL", "MSFT", "JPM"]):
        store.save_raw(ticker, _synthetic_ohlcv(seed=i))
    return store


@pytest.fixture
def client(tmp_path, monkeypatch, seeded_store):
    """TestClient whose store points at the temporary seeded directory."""
    monkeypatch.setattr("luqman.serving.api._store", lambda: seeded_store)
    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_features_lists_admitted_and_rejected(client):
    payload = client.get("/features").json()
    admitted = {f["name"] for f in payload["admitted_cross_sectional"]}
    rejected = {f["name"] for f in payload["rejected"]}

    assert "momentum_12_1_xs" in admitted
    assert "residual_momentum_xs" in admitted
    # Rejected candidates are surfaced deliberately, as the honest record.
    assert "idiosyncratic_vol_xs" in rejected
    assert admitted.isdisjoint(rejected)


def test_volatility_forecast_shape(client):
    response = client.get("/forecast/volatility/AAPL?horizon=5")
    assert response.status_code == 200
    body = response.json()

    assert body["ticker"] == "AAPL"
    assert body["horizon_days"] == 5
    assert body["forecast_daily_vol"] >= 0
    assert body["expected_move_pct"] > 0
    # Annualization must be internally consistent.
    assert body["forecast_annualized_vol"] == pytest.approx(
        body["forecast_daily_vol"] * np.sqrt(252)
    )


def test_volatility_forecast_states_it_is_not_a_direction_call(client):
    """The framing is load-bearing: this project found no direction edge, so a
    magnitude forecast must never read as a buy/sell signal."""
    body = client.get("/forecast/volatility/AAPL").json()
    assert "EITHER direction" in body["interpretation"]
    assert body["proof"]


def test_unknown_ticker_returns_404(client):
    response = client.get("/forecast/volatility/NOSUCHTICKER")
    assert response.status_code == 404
    assert "fetch" in response.json()["detail"].lower()


def test_rejected_factor_is_not_servable(client):
    """The registry's safety property, enforced at the HTTP boundary.

    idiosyncratic_vol_xs posted the best headline statistics of any candidate
    ever tested here and was still rejected (it was a disguised beta bet). It
    must not be reachable as a tradable ranking.
    """
    for rejected in [
        "idiosyncratic_vol_xs",
        "residual_reversal_xs",
        "amihud_illiquidity_xs",
        "pct_of_52w_high_xs",
    ]:
        response = client.get(f"/factors/{rejected}/ranking")
        assert response.status_code == 404, f"{rejected} must not be servable"


def test_factor_ranking_legs_are_disjoint(client):
    response = client.get("/factors/momentum_12_1_xs/ranking?quantile=0.3")
    assert response.status_code == 200
    body = response.json()
    assert set(body["long_names"]).isdisjoint(body["short_names"])
    assert body["caveat"]
    assert body["proof_run"]


def test_service_layer_rejects_unknown_factor(seeded_store):
    with pytest.raises(KeyError, match="not an admitted factor"):
        service.rank_factor(seeded_store, "totally_made_up_factor")
