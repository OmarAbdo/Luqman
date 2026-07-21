"""Business logic behind the serving API, kept free of any HTTP concepts.

Everything exposed here is backed by something that cleared the proof standard
in `.refactor/dataset-plan.md`. Two things are deliberately NOT offered, and
their absence is the most important design decision in this module:

  - **No single-name buy/sell signal.** Ridge, gradient boosting and an
    LSTM+attention network were each evaluated against a persistence baseline
    on daily bars, and a ridge model again on 5-minute bars. None beat the
    baseline at any horizon. Serving a direction forecast would mean shipping
    a number the project's own evidence says is noise.
  - **No price-level forecast.** The original system predicted price levels and
    scored well on MAE while being no better than echoing the last price. The
    targets here are stationary by construction.

What IS offered: a volatility (expected move magnitude) forecast for risk
sizing, and the current ranking of the admitted cross-sectional factors, which
are direction claims only in the relative, long-short sense in which they were
proven.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class VolatilityForecast:
    ticker: str
    as_of: str
    horizon_days: int
    forecast_daily_vol: float
    forecast_annualized_vol: float
    trailing_daily_vol: float
    expected_move_pct: float
    interpretation: str
    proof: str


@dataclass
class FactorRanking:
    factor: str
    as_of: str
    horizon_days: int
    long_names: list[str]
    short_names: list[str]
    scores: dict[str, float]
    mechanism: str
    proof_run: str
    caveat: str


@dataclass
class FeatureInventory:
    admitted_single_name: list[dict] = field(default_factory=list)
    admitted_cross_sectional: list[dict] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)


def feature_inventory() -> FeatureInventory:
    """The registry, rendered as plain data.

    Rejected candidates are included on purpose: the ratio of tested to proven
    is the honest headline of this project, and hiding the failures would
    misrepresent how much of what was tried actually worked.
    """
    from ..features.registry import (
        CROSS_SECTIONAL_REGISTRY,
        REGISTRY,
        REJECTED_CROSS_SECTIONAL_CANDIDATES,
    )

    return FeatureInventory(
        admitted_single_name=[
            {
                "name": f.name,
                "category": f.category,
                "predicts": f.target_kind,
                "horizon_days": f.horizon,
                "mechanism": f.mechanism,
                "proof_run": f.proof_run,
            }
            for f in REGISTRY.values()
        ],
        admitted_cross_sectional=[
            {
                "name": f.name,
                "category": f.category,
                "horizon_days": f.horizon,
                "mechanism": f.mechanism,
                "proof_run": f.proof_run,
            }
            for f in CROSS_SECTIONAL_REGISTRY.values()
        ],
        rejected=[
            {"name": f.name, "why": f.mechanism}
            for f in REJECTED_CROSS_SECTIONAL_CANDIDATES.values()
        ],
    )


def forecast_volatility(store, ticker: str, horizon: int = 5, window: int = 5) -> VolatilityForecast:
    """Fit the proven volatility model on all stored history and forecast forward.

    The model is a ridge regression on the registered `vol_cluster` feature.
    That specific combination was chosen on evidence, not preference: it beat a
    persistence baseline on 8/8 daily tickers (mean +14.9% skill), beat
    purpose-built GJR-GARCH and gradient boosting head to head, and reproduced
    at intraday frequency with +21% to +30% skill.

    Note this fits on all available history to produce a live forecast, which
    is the correct thing for serving but is NOT how the model was validated --
    the reported skill figures come from walk-forward out-of-fold evaluation.
    """
    from ..features.registry import build_registered_features
    from ..features.vol_targets import volatility_target
    from ..models.baselines import make_ridge

    ohlcv = store.load_raw(ticker)
    features = build_registered_features(ohlcv, target_kind="volatility")
    target = volatility_target(ohlcv["close"], horizon=horizon, window=window)

    training = features.join(target, how="inner").dropna()
    if training.empty:
        raise ValueError(f"No aligned training rows for {ticker!r}")

    X_train = training[features.columns]
    y_train = training[target.name]

    model = make_ridge()
    model.fit(X_train.to_numpy(), y_train.to_numpy())

    # The most recent row with complete features is what we forecast from.
    latest_features = features.dropna()
    if latest_features.empty:
        raise ValueError(f"No complete feature row for {ticker!r}")
    latest = latest_features.iloc[[-1]]
    as_of = latest.index[-1]

    daily_vol = float(model.predict(latest.to_numpy())[0])
    daily_vol = max(daily_vol, 0.0)  # a negative volatility is not meaningful
    trailing = float(latest.iloc[0, 0])

    return VolatilityForecast(
        ticker=ticker.upper(),
        as_of=as_of.strftime("%Y-%m-%d"),
        horizon_days=horizon,
        forecast_daily_vol=daily_vol,
        forecast_annualized_vol=daily_vol * np.sqrt(TRADING_DAYS),
        trailing_daily_vol=trailing,
        expected_move_pct=daily_vol * np.sqrt(horizon) * 100,
        interpretation=(
            f"Over the next {horizon} trading days, a typical (one standard deviation) "
            f"move in {ticker.upper()} is about {daily_vol * np.sqrt(horizon) * 100:.2f}%, "
            "in EITHER direction. This is a magnitude forecast for position sizing and "
            "risk budgeting. It says nothing about which way the price will go -- this "
            "project found no reliable single-name direction signal at any horizon tested."
        ),
        proof="vol_cluster, proof run wave1-2026-07-20; see .refactor/dataset-plan.md",
    )


def rank_factor(store, factor_name: str, quantile: float = 0.2) -> FactorRanking:
    """Current long/short ranking for one admitted cross-sectional factor.

    This is the closest thing the project has to a tradable direction view, and
    it is only valid in the form in which it was proven: a market-neutral
    long-short book across the whole universe, rebalanced, held for the
    factor's horizon. The ranking of an individual name carries no standalone
    forecast.
    """
    from ..data.universe import EQUITY_UNIVERSE
    from ..data.validation import validate_daily_ohlcv
    from ..features.registry import CROSS_SECTIONAL_REGISTRY

    if factor_name not in CROSS_SECTIONAL_REGISTRY:
        raise KeyError(
            f"{factor_name!r} is not an admitted factor. "
            f"Available: {list(CROSS_SECTIONAL_REGISTRY)}"
        )
    factor = CROSS_SECTIONAL_REGISTRY[factor_name]

    ohlcv_by_ticker = {}
    for ticker in EQUITY_UNIVERSE:
        try:
            df = store.load_raw(ticker)
        except FileNotFoundError:
            continue
        if validate_daily_ohlcv(df).ok:
            ohlcv_by_ticker[ticker] = df

    if not ohlcv_by_ticker:
        raise ValueError("No stored universe data. Run: python scripts/fetch_universe.py")

    panel = factor.build(ohlcv_by_ticker)
    latest_row = panel.dropna(how="all").iloc[-1].dropna()
    if latest_row.empty:
        raise ValueError(f"No current values for factor {factor_name!r}")

    ranked = latest_row.sort_values()
    cut = max(1, int(len(ranked) * quantile))

    return FactorRanking(
        factor=factor.name,
        as_of=panel.dropna(how="all").index[-1].strftime("%Y-%m-%d"),
        horizon_days=factor.horizon,
        long_names=list(ranked.index[-cut:][::-1]),
        short_names=list(ranked.index[:cut]),
        scores={k: float(v) for k, v in ranked.items()},
        mechanism=factor.mechanism,
        proof_run=factor.proof_run,
        caveat=(
            "Valid only as a market-neutral long-short book across the whole universe, "
            f"held for ~{factor.horizon} trading days. Individual names carry no "
            "standalone forecast, and the reported backtest Sharpe assumes daily "
            "rebalancing with turnover costs."
        ),
    )


def backtest_factor(store, factor_name: str, cost_bps: float = 10.0, quantile: float = 0.2) -> dict:
    """Costed long-short backtest report for one admitted factor."""
    from ..backtest.cross_sectional import backtest_long_short
    from ..data.universe import EQUITY_UNIVERSE
    from ..data.validation import validate_daily_ohlcv
    from ..features.registry import CROSS_SECTIONAL_REGISTRY
    from ..features.targets import forward_log_return

    if factor_name not in CROSS_SECTIONAL_REGISTRY:
        raise KeyError(
            f"{factor_name!r} is not an admitted factor. "
            f"Available: {list(CROSS_SECTIONAL_REGISTRY)}"
        )
    factor = CROSS_SECTIONAL_REGISTRY[factor_name]

    ohlcv_by_ticker = {}
    for ticker in EQUITY_UNIVERSE:
        try:
            df = store.load_raw(ticker)
        except FileNotFoundError:
            continue
        if validate_daily_ohlcv(df).ok:
            ohlcv_by_ticker[ticker] = df

    panel = factor.build(ohlcv_by_ticker)
    returns = pd.DataFrame(
        {t: forward_log_return(df["close"], horizon=factor.horizon) for t, df in ohlcv_by_ticker.items()}
    )
    result = backtest_long_short(panel, returns, quantile=quantile, cost_bps=cost_bps)

    return {
        "factor": factor.name,
        "universe_size": len(ohlcv_by_ticker),
        "horizon_days": factor.horizon,
        "cost_bps": cost_bps,
        "metrics": {k: (float(v) if isinstance(v, float) else v) for k, v in result.metrics.items()},
        "proof_run": factor.proof_run,
        "caveat": (
            "Rebalances daily against overlapping forward returns, which inflates Sharpe "
            "through autocorrelation. Figures are comparable between factors here but not "
            "to externally published values. The universe is survivorship-biased "
            "(selected from 2026 large-caps), which inflates absolute returns."
        ),
    }
