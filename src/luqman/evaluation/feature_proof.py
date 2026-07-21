"""Enforcement harness for the feature admission gates.

Implements the proof standard from `.refactor/dataset-plan.md`: IC thresholds,
sign consistency, multi-ticker stability, and marginal contribution. A feature
that does not clear these does not enter the registry, and only registry
features reach a model.
"""

from dataclasses import dataclass, field

import pandas as pd

from .feature_audit import audit_features
from .evaluate import ZERO_BASELINE, walk_forward_evaluate
from ..models.baselines import ZeroReturnBaseline, make_ridge
from ..splits.walk_forward import WalkForwardSplitter

# Sector/liquidity-diverse set used for every proof run so a feature can't pass
# by fitting one name's idiosyncratic history.
PROOF_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT", "NVDA", "KO"]

IC_THRESHOLD = 0.08
CONSISTENCY_THRESHOLD = 0.70
MIN_TICKERS_PASSING = 6  # of len(PROOF_TICKERS)


@dataclass
class FeatureProof:
    feature: str
    mechanism: str
    horizon: int
    per_ticker: pd.DataFrame
    n_tickers_passing_ic: int
    marginal_skill_delta: float | None = None
    verdict: str = field(init=False)

    def __post_init__(self):
        ic_gate = self.n_tickers_passing_ic >= MIN_TICKERS_PASSING
        marginal_gate = self.marginal_skill_delta is None or self.marginal_skill_delta > 0
        self.verdict = "ADMITTED" if (ic_gate and marginal_gate) else "REJECTED"

    def summary(self) -> str:
        lines = [
            f"=== {self.feature} (horizon {self.horizon}d) — {self.verdict} ===",
            f"mechanism: {self.mechanism}",
            self.per_ticker.to_string(float_format=lambda v: f"{v: .4f}"),
            f"tickers passing IC gate (|ic|>={IC_THRESHOLD}, consistency>={CONSISTENCY_THRESHOLD}): "
            f"{self.n_tickers_passing_ic}/{len(PROOF_TICKERS)} (need >= {MIN_TICKERS_PASSING})",
        ]
        if self.marginal_skill_delta is not None:
            lines.append(f"marginal skill_vs_zero delta when added to ridge: {self.marginal_skill_delta:+.4f}")
        return "\n".join(lines)


def _passes_ic_gate(row: pd.Series) -> bool:
    return (
        abs(row["ic_full"]) >= IC_THRESHOLD
        and abs(row["ic_mean_blocks"]) >= IC_THRESHOLD
        and row["ic_sign_consistency"] >= CONSISTENCY_THRESHOLD
    )


def prove_feature(
    feature_name: str,
    mechanism: str,
    horizon: int,
    per_ticker_data: dict[str, tuple[pd.Series, pd.Series]],
) -> FeatureProof:
    """Run the IC gate for one feature across the standard ticker panel.

    `per_ticker_data` maps ticker -> (feature_series, target_series), already
    aligned to the feature's own horizon by the caller (each candidate builds
    its own feature + matching-horizon target, since mechanisms operate at
    different horizons — see dataset-plan.md Section 2).
    """
    rows = {}
    n_passing = 0
    for ticker, (feature_series, target_series) in per_ticker_data.items():
        X = feature_series.to_frame(feature_name)
        audit = audit_features(X, target_series, n_splits=5)
        row = audit.loc[feature_name]
        rows[ticker] = row
        if _passes_ic_gate(row):
            n_passing += 1

    per_ticker = pd.DataFrame(rows).T
    return FeatureProof(
        feature=feature_name,
        mechanism=mechanism,
        horizon=horizon,
        per_ticker=per_ticker,
        n_tickers_passing_ic=n_passing,
    )


def marginal_contribution(
    baseline_X: pd.DataFrame,
    candidate: pd.Series,
    y: pd.Series,
    splitter: WalkForwardSplitter | None = None,
) -> float:
    """skill_vs_zero of ridge(baseline + candidate) minus ridge(baseline alone),
    out-of-fold. Positive means the candidate adds real net-new information on
    top of what's already admitted, not just IC in isolation (which can pass
    for a feature that's actually redundant with something already in the set)."""
    aligned = baseline_X.join(candidate.rename("__candidate__"), how="inner").join(
        y.rename("__target__"), how="inner"
    ).dropna()
    y_aligned = aligned["__target__"]
    without = aligned[baseline_X.columns]
    with_candidate = aligned[list(baseline_X.columns) + ["__candidate__"]]

    splitter = splitter or WalkForwardSplitter(
        n_splits=5, test_size=max(21, len(aligned) // 12), embargo=1,
        min_train_size=max(100, len(aligned) // 3),
    )

    def _skill(X):
        table = walk_forward_evaluate(
            X, y_aligned, {ZERO_BASELINE: ZeroReturnBaseline, "ridge": make_ridge}, splitter
        )
        return float(table.loc["ridge", "skill_vs_zero"])

    if without.shape[1] == 0:
        # No baseline yet (first admitted feature) — compare straight to zero baseline.
        return _skill(with_candidate)
    return _skill(with_candidate) - _skill(without)
