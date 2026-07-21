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


def _load_ff_factors() -> pd.DataFrame:
    """Fama-French factors from the configured store.

    Residual features need the factor series in addition to price data, which
    the plain build(ohlcv_by_ticker) signature does not carry. Paths are
    resolved through Settings so this stays consistent with the rest of the
    package's single-source-of-paths rule.
    """
    from ..config import Settings
    from ..data.store import ParquetStore

    factors = ParquetStore(Settings.load().data_dir).load_table("ff_factors_daily")
    if factors is None:
        raise FileNotFoundError(
            "Fama-French factors not stored. Run: luqman fetch-factors"
        )
    return factors


def _residual_momentum_panel(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .residual import residual_momentum, residual_return_series

    factors = _load_ff_factors()
    columns = {}
    for ticker, df in ohlcv_by_ticker.items():
        residual = residual_return_series(df["close"], factors)
        columns[ticker] = residual_momentum(residual)
    return pd.DataFrame(columns)


def _idiosyncratic_vol_panel(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .residual import idiosyncratic_vol, residual_return_series

    factors = _load_ff_factors()
    return pd.DataFrame(
        {
            ticker: idiosyncratic_vol(residual_return_series(df["close"], factors))
            for ticker, df in ohlcv_by_ticker.items()
        }
    )


def _residual_reversal_panel(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .residual import residual_return_series, residual_reversal

    factors = _load_ff_factors()
    return pd.DataFrame(
        {
            ticker: residual_reversal(residual_return_series(df["close"], factors))
            for ticker, df in ohlcv_by_ticker.items()
        }
    )


def _long_term_reversal_panel(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .residual import long_term_reversal

    return pd.DataFrame({t: long_term_reversal(df["close"]) for t, df in ohlcv_by_ticker.items()})


def _amihud_panel(ohlcv_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    from .residual import amihud_illiquidity

    return pd.DataFrame(
        {t: amihud_illiquidity(df["close"], df["volume"]) for t, df in ohlcv_by_ticker.items()}
    )


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
    "residual_momentum_xs": RegisteredCrossSectionalFactor(
        name="residual_momentum_xs",
        category="technical",
        mechanism=(
            "Residual momentum (Blitz, Huij & Martens 2011): the 12-1 momentum "
            "ranking computed on Fama-French residual returns instead of raw "
            "returns, which strips the market/size/value drift that adds noise "
            "to raw momentum. Admitted in Wave 3 (dataset-plan.md Section 10): "
            "net Sharpe +1.55 at 10bps vs +1.01 for raw momentum over the same "
            "2000-2026 window, with less than half the max drawdown (-5.79 vs "
            "-13.77), and split-half stable (+1.57 / +1.64). Correlation with "
            "raw momentum is only +0.41, so the two are not the same bet -- but "
            "note that a 50/50 blend scores +1.46, BELOW residual momentum "
            "alone: blending beats raw momentum yet dilutes the better factor. "
            "On the evidence, residual momentum SUPERSEDES raw momentum rather "
            "than complementing it."
        ),
        horizon=21,
        proof_run="wave3-residual-2026-07-21",
        build=_residual_momentum_panel,
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
    "idiosyncratic_vol_xs": RegisteredCrossSectionalFactor(
        name="idiosyncratic_vol_xs",
        category="technical",
        mechanism=(
            "Idiosyncratic volatility (Ang et al. 2006). REJECTED in Wave 3 "
            "(dataset-plan.md Section 10) DESPITE strong statistics (net Sharpe "
            "+1.39 at 10bps, the largest blend improvement of any candidate) "
            "because the result contradicts its own stated mechanism: Ang "
            "documents HIGH idiosyncratic vol predicting LOW returns, and we "
            "measured the opposite sign. Diagnosis: in a 52-name mega-cap "
            "universe the high-IVOL leg is AMD/NVDA/TSLA/NFLX (mean market "
            "beta 1.35) and the low-IVOL leg is JNJ/KO/PG (mean beta 0.73), "
            "with IVOL correlating +0.61 with beta across the universe. This "
            "is a leveraged long-beta growth bet measured through a tech bull "
            "market -- compensated risk, not alpha -- which also explains the "
            "+0.57 -> +2.91 split-half jump. Ang's effect concerns small "
            "speculative names that this universe cannot contain by "
            "construction. Retest only on a universe including small caps."
        ),
        horizon=21,
        proof_run="wave3-residual-2026-07-21-rejected",
        build=_idiosyncratic_vol_panel,
    ),
    "residual_reversal_xs": RegisteredCrossSectionalFactor(
        name="residual_reversal_xs",
        category="technical",
        mechanism=(
            "Residual short-term reversal (Blitz et al. 2013). REJECTED in "
            "Wave 3 on COSTS, the project's final arbiter: gross Sharpe +1.10 "
            "decays to +0.65 at 10bps and +0.20 at 20bps, because a 5-day "
            "signal rebalances far more often than the 21-day factors. The "
            "sign and split-half stability were fine; the edge simply is not "
            "large enough to survive realistic trading frictions."
        ),
        horizon=5,
        proof_run="wave3-residual-2026-07-21-rejected",
        build=_residual_reversal_panel,
    ),
    "long_term_reversal_xs": RegisteredCrossSectionalFactor(
        name="long_term_reversal_xs",
        category="technical",
        mechanism=(
            "Long-term reversal (De Bondt & Thaler 1985). REJECTED in Wave 3: "
            "Sharpe -1.43 with mean IC -0.040, i.e. strongly and consistently "
            "WRONG-signed over 2000-2026 -- past 5-year losers kept losing. "
            "Stable across both halves, so this is a genuine negative result "
            "for this universe, not noise: in mega-caps the long-horizon "
            "pattern runs with continuation, not reversal."
        ),
        horizon=21,
        proof_run="wave3-residual-2026-07-21-rejected",
        build=_long_term_reversal_panel,
    ),
    "amihud_illiquidity_xs": RegisteredCrossSectionalFactor(
        name="amihud_illiquidity_xs",
        category="technical",
        mechanism=(
            "Amihud (2002) illiquidity. REJECTED in Wave 3 by the split-half "
            "sign-stability check that also killed pct_of_52w_high_xs: pooled "
            "Sharpe looks excellent (+2.27) but decomposes into +5.11 in the "
            "first half and -0.12 in the second, i.e. the entire result is a "
            "pre-2013 artifact that has since disappeared. Consistent with the "
            "prior stated before testing -- the illiquidity premium lives in "
            "small/illiquid names, and this universe is large-cap by design."
        ),
        horizon=21,
        proof_run="wave3-residual-2026-07-21-rejected",
        build=_amihud_panel,
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
