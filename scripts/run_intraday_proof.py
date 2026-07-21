"""Phase 8a: re-run the project's methodology at the intraday (5-minute) horizon.

This does NOT assume anything carries over from daily. It asks the same two
questions the daily work asked, against the same baselines:

  1. Direction: can anything beat the zero-return baseline on the next 5-minute
     bar's return? (Daily answer: no, for every model and feature tried.)
  2. Volatility: does `vol_cluster`, the project's one proven single-name
     feature, still beat a persistence baseline intraday? Volatility clustering
     is documented to be STRONGER at intraday frequencies, so this is the
     hypothesis most likely to transfer.

All returns are session-aware (see luqman/data/intraday.py): nothing is ever
computed across the overnight gap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from luqman.data.intraday import (
    add_session_flags,
    audit_intraday,
    forward_session_aware_log_return,
    load_intraday_csv,
    session_aware_log_returns,
)
from luqman.evaluation.metrics import directional_accuracy, mae, mase
from luqman.splits.walk_forward import WalkForwardSplitter

INTRADAY_DIR = Path("data/raw/intraday")


def build_intraday_features(flagged: pd.DataFrame) -> pd.DataFrame:
    """Stationary, session-aware, strictly backward-looking intraday features.

    Deliberately the intraday analogues of the daily set that was actually
    tested, not a new indicator dump: lagged returns, realized volatility,
    volume ratio, and position within the session (the one genuinely intraday
    variable, since liquidity and volatility follow a well-documented U-shape
    across the trading day).
    """
    close = flagged["close"]
    session = flagged["session_date"]

    features = pd.DataFrame(index=flagged.index)
    for lag in (1, 2, 3, 6, 12):
        features[f"ret_{lag}bar"] = session_aware_log_returns(close, session, periods=lag)

    # min_periods matters here and is not a cosmetic choice: the first bar of
    # every session has no within-session predecessor and is therefore NaN by
    # design. A session is exactly 78 bars, so a plain rolling(78) window would
    # always straddle one of those NaNs and yield NaN for the entire series.
    # Allowing the window to compute from the valid returns it does contain
    # pools recent within-session returns, which is the standard trailing
    # volatility estimate.
    one_bar = session_aware_log_returns(close, session, periods=1)
    features["rvol_12bar"] = one_bar.rolling(12, min_periods=6).std()
    features["rvol_78bar"] = one_bar.rolling(78, min_periods=40).std()  # ~one session
    features["volume_ratio"] = flagged["volume"] / flagged["volume"].rolling(78, min_periods=40).mean()
    # Time-of-day, as a fraction of the regular session elapsed.
    features["session_progress"] = flagged["bar_of_session"] / 78.0
    return features


def evaluate(
    name,
    features,
    target,
    baseline,
    baseline_name,
    report_direction=True,
    n_splits=5,
    test_size=2000,
):
    """Walk-forward ridge against an explicitly supplied baseline.

    The baseline is a parameter rather than hardcoded to zero because the right
    benchmark depends on the target. For returns it is zero (predict no
    change). For VOLATILITY zero is a nonsense benchmark -- the target is
    strictly positive, so predicting zero is trivially terrible and beating it
    proves nothing. The standard volatility benchmark is persistence: tomorrow's
    volatility equals today's trailing volatility.

    Both models are scored on exactly the same out-of-fold rows, which is the
    only way the comparison means anything.
    """
    data = (
        pd.concat([features, target.rename("y"), baseline.rename("baseline")], axis=1)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    X = data.drop(columns=["y", "baseline"])
    y, baseline_aligned = data["y"], data["baseline"]
    if len(data) < test_size * 2:
        print(f"  {name}: not enough clean rows ({len(data)})")
        return

    splitter = WalkForwardSplitter(n_splits=n_splits, test_size=test_size, embargo=1)
    preds_ridge, preds_base, actuals, train_tails = [], [], [], []
    for train_idx, test_idx in splitter.split(len(X)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds_ridge.append(pd.Series(model.predict(X.iloc[test_idx]), index=X.index[test_idx]))
        preds_base.append(baseline_aligned.iloc[test_idx])
        actuals.append(y.iloc[test_idx])
        train_tails.append(y.iloc[train_idx])

    actual = pd.concat(actuals)
    ridge_pred = pd.concat(preds_ridge)
    base_pred = pd.concat(preds_base)
    train_y = pd.concat(train_tails)

    ridge_mae, base_mae = mae(actual, ridge_pred), mae(actual, base_pred)
    skill = (base_mae - ridge_mae) / base_mae if base_mae else float("nan")
    line = (
        f"  {name}: n={len(actual)} | ridge MAE {ridge_mae:.6f} vs "
        f"{baseline_name} {base_mae:.6f} | skill_vs_{baseline_name} {skill:+.2%} | "
        f"MASE {mase(actual, ridge_pred, train_y):.3f}"
    )
    if report_direction:
        # Only meaningful for a signed target; a volatility target is always
        # positive, so "directional accuracy" would be a constant 1.0.
        line += (
            f" | dir_acc {directional_accuracy(actual, ridge_pred):.3f} "
            f"(base rate {(actual > 0).mean():.3f})"
        )
    print(line)


def main():
    files = sorted(INTRADAY_DIR.glob("*_5min_raw*.csv"))
    if not files:
        print(f"No intraday CSVs under {INTRADAY_DIR}")
        return 1

    for path in files:
        ticker = path.stem.split("_")[0]
        print(f"\n{'=' * 78}\n{path.name}\n{'=' * 78}")
        df = load_intraday_csv(path)
        audit = audit_intraday(df)
        print(audit.summary())

        flagged = add_session_flags(df)
        regular = flagged[flagged["is_regular"]].copy()
        print(f"restricting to regular-hours bars: {len(regular)} of {len(flagged)}")
        if len(regular) < 5000:
            print("  too few regular-hours bars for walk-forward evaluation; skipping")
            continue

        features = build_intraday_features(regular)

        print("\n-- Question 1: next-bar DIRECTION (target = forward 1-bar log return)")
        direction_target = forward_session_aware_log_return(
            regular["close"], regular["session_date"], horizon=1
        )
        zero_baseline = pd.Series(0.0, index=regular.index)
        evaluate(
            f"{ticker} 1-bar return", features, direction_target,
            baseline=zero_baseline, baseline_name="zero",
        )

        print("\n-- Question 2: VOLATILITY (target = realized vol over next 12 bars = 1 hour)")
        one_bar = session_aware_log_returns(regular["close"], regular["session_date"])
        future_vol = (
            one_bar.shift(-12).rolling(12, min_periods=6).std().shift(-11).rename("future_vol_12bar")
        )
        vol_features = features[["rvol_12bar", "rvol_78bar"]]
        # Persistence -- "the next hour's volatility equals the last hour's" --
        # is the standard volatility benchmark and the only honest one here.
        evaluate(
            f"{ticker} future vol", vol_features, future_vol,
            baseline=features["rvol_12bar"], baseline_name="persist",
            report_direction=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
