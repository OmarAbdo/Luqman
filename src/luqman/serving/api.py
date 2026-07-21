"""FastAPI surface over the proven forecasting components.

Run it with:

    luqman serve                      # or: uvicorn luqman.serving.api:app --reload

Interactive documentation is then at http://127.0.0.1:8000/docs.

Design rule for this module: an endpoint may only expose something that
cleared the proof standard, and every response carries its own provenance
(which proof run backs it) and its own caveats. There is deliberately no
"predict tomorrow's price" or "should I buy" endpoint -- see
`serving/service.py` for why that absence is the point rather than an omission.
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query

from ..config import Settings
from ..data.store import ParquetStore
from . import service

app = FastAPI(
    title="Luqman",
    version="2.0.0a0",
    description=(
        "Forecasting engine serving only signals that beat a real baseline "
        "out-of-sample. Volatility (expected move magnitude) forecasts for risk "
        "sizing, and cross-sectional factor rankings. No single-name direction "
        "predictions -- none were found that beat a coin flip."
    ),
)


def _store() -> ParquetStore:
    return ParquetStore(Settings.load().data_dir)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness check."""
    return {"status": "ok", "service": "luqman", "version": app.version}


@app.get("/features", tags=["meta"])
def features() -> dict:
    """Everything proven, and everything tested and rejected, with mechanisms.

    The rejected list is served alongside the admitted one on purpose: it is
    the honest record of how much was tried relative to how little worked.
    """
    return asdict(service.feature_inventory())


@app.get("/forecast/volatility/{ticker}", tags=["forecast"])
def forecast_volatility(
    ticker: str,
    horizon: int = Query(5, ge=1, le=60, description="Trading days ahead"),
    window: int = Query(5, ge=2, le=60, description="Realized-vol window"),
) -> dict:
    """Expected move MAGNITUDE over the next `horizon` trading days.

    This is the project's most robust result: a ridge model on the `vol_cluster`
    feature beats a persistence baseline on 8/8 daily tickers and again at
    intraday frequency. Use it for position sizing and risk budgeting.

    It is explicitly not a direction call.
    """
    try:
        return asdict(service.forecast_volatility(_store(), ticker, horizon=horizon, window=window))
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No stored data for {ticker!r}. Fetch it first: luqman fetch --ticker {ticker}",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/factors", tags=["factors"])
def list_factors() -> dict:
    """Admitted cross-sectional factors available for ranking and backtesting."""
    from ..features.registry import CROSS_SECTIONAL_REGISTRY

    return {
        "factors": [
            {
                "name": f.name,
                "horizon_days": f.horizon,
                "proof_run": f.proof_run,
                "mechanism": f.mechanism,
            }
            for f in CROSS_SECTIONAL_REGISTRY.values()
        ]
    }


@app.get("/factors/{factor_name}/ranking", tags=["factors"])
def factor_ranking(
    factor_name: str,
    quantile: float = Query(0.2, gt=0.0, lt=0.5, description="Fraction in each leg"),
) -> dict:
    """Today's long/short book for one admitted factor.

    Valid only as a market-neutral portfolio across the whole universe; a
    single name's position in the ranking is not a forecast for that name.
    """
    try:
        return asdict(service.rank_factor(_store(), factor_name, quantile=quantile))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/factors/{factor_name}/backtest", tags=["factors"])
def factor_backtest(
    factor_name: str,
    cost_bps: float = Query(10.0, ge=0.0, le=100.0, description="Per-side turnover cost"),
    quantile: float = Query(0.2, gt=0.0, lt=0.5),
) -> dict:
    """Costed long-short backtest report for one admitted factor."""
    try:
        return service.backtest_factor(_store(), factor_name, cost_bps=cost_bps, quantile=quantile)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
