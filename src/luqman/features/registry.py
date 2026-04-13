"""The single source of truth for which features are allowed into a dataset.

Per dataset-plan.md Section 1: a feature reaches a model ONLY if it appears
here, and it only appears here after clearing all four proof gates
(named mechanism, IC audit, 8-ticker stability, marginal contribution). This
makes "an unproven feature reached a model" a registry-lookup failure, not a
discipline lapse.
"""

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .technical.candidates import realized_vol


@dataclass(frozen=True)
class RegisteredFeature:
    """A single-ticker feature: build(ohlcv_for_one_ticker) -> Series."""

    name: str
    category: str
    mechanism: str
    horizon: int
    target_kind: str  # "return" or "volatility" — which target this feature was proven against
    proof_run: str
    build: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class RegisteredCrossSectionalFactor:
    """A factor that needs the WHOLE universe's data at once to compute a
    per-day rank (e.g. momentum: today's factor value for AAPL is only
    meaningful relative to every other stock's value that same day). Distinct
    from RegisteredFeature, which is computable from one ticker in isolation.

    build(ohlcv_by_ticker) -> wide DataFrame, index=date, columns=ticker.
    """

    name: str
    category: str
    mechanism: str
    horizon: int
    proof_run: str
    build: Callable[[dict[str, pd.DataFrame]], pd.DataFrame]


def _vol_cluster_feature(ohlcv: pd.DataFrame) -> pd.Series:
    return realized_vol(ohlcv["close"], window=20)


def _momentum_12_1_panel(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .technical.candidates import momentum_12_1

    return pd.DataFrame({t: momentum_12_1(df["close"]) for t, df in ohlcv_by_ticker.items()})


def _pct_of_52w_high_panel(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .technical.candidates import pct_of_52w_high

    return pd.DataFrame({t: pct_of_52w_high(df["close"]) for t, df in ohlcv_by_ticker.items()})


REGISTRY: dict[str, RegisteredFeature] = {
    "vol_cluster": RegisteredFeature(
        name="vol_cluster",
        category="technical",
        mechanism=(
            "Volatility clustering: realized volatility is strongly autocorrelated "
            "(ARCH/GARCH effects)."
        ),
        horizon=5,
        target_kind="volatility",
        proof_run="wave1-2026-07-20",
        build=_vol_cluster_feature,
    ),
}

CROSS_SECTIONAL_REGISTRY: dict[str, RegisteredCrossSectionalFactor] = {
    "momentum_12_1_xs": RegisteredCrossSectionalFactor(
        name="momentum_12_1_xs",
        category="technical",
        mechanism=(
            "Momentum (12-1): 12-month return excluding the most recent month "
            "predicts next-month return (Jegadeesh-Titman 1993), validated as a "
            "cross-sectional long-short decile factor, not a single-name signal."
        ),
        horizon=21,
        proof_run="cross-sectional-2026-07-20",
        build=_momentum_12_1_panel,
    ),
}

# Tested and REJECTED candidates, kept here (not deleted) as a record of what
# was tried and why it doesn't qualify -- see dataset-plan.md for full proof
# numbers. Never move an entry from here into REGISTRY/CROSS_SECTIONAL_REGISTRY
# without a fresh, recorded proof run; a rejected candidate does not become
# provisionally trustworthy just by sitting in this dict.
REJECTED_CROSS_SECTIONAL_CANDIDATES: dict[str, RegisteredCrossSectionalFactor] = {
    "pct_of_52w_high_xs": RegisteredCrossSectionalFactor(
        name="pct_of_52w_high_xs",
        category="technical",
        mechanism=(
            "52-week high proximity (George & Hwang 2004): anchoring-bias "
            "effect, found in the original paper to subsume standard price "
            "momentum. REJECTED (dataset-plan.md Section 9a): Sharpe -0.31 "
            "overall, and split-half sign FLIPS (+0.83 first half, -1.19 "
            "second half) -- unlike momentum, which was stable both halves. "
            "Not a robust effect in this universe/period."
        ),
        horizon=21,
        proof_run="cross-sectional-2026-07-20-rejected",
        build=_pct_of_52w_high_panel,
    ),
}


def admitted(category: str | None = None, target_kind: str | None = None) -> list[RegisteredFeature]:
    values = REGISTRY.values()
    if category is not None:
        values = [f for f in values if f.category == category]
    if target_kind is not None:
        values = [f for f in values if f.target_kind == target_kind]
    return list(values)


def admitted_cross_sectional(category: str | None = None) -> list[RegisteredCrossSectionalFactor]:
    values = CROSS_SECTIONAL_REGISTRY.values()
    if category is not None:
        values = [f for f in values if f.category == category]
    return list(values)


def build_registered_features(ohlcv: pd.DataFrame, target_kind: str) -> pd.DataFrame:
    """All admitted single-ticker features of the given target_kind, joined into one frame."""
    features = admitted(target_kind=target_kind)
    if not features:
        raise ValueError(f"No registered features for target_kind={target_kind!r}")
    columns = {f.name: f.build(ohlcv) for f in features}
    return pd.DataFrame(columns)
