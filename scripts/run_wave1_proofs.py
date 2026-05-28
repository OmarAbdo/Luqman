"""Wave 1 proof run: re-derived technical candidates + VIX geopolitical proxy,
across the 8-ticker panel, per dataset-plan.md Section 1/6. Not part of the
installed package — a one-off proof script whose OUTPUT (this run's printed
report) is the artifact that matters, same spirit as the other tracked runs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.evaluation.feature_proof import PROOF_TICKERS, marginal_contribution, prove_feature
from luqman.features.targets import forward_log_return
from luqman.features.technical.candidates import (
    MECHANISMS as TECH_MECHANISMS,
    future_realized_vol,
    macd_divergence,
    momentum_12_1,
    realized_vol,
    reversal_1d,
    volume_breakout,
)
from luqman.features.geopolitical.vix_proxy import (
    MECHANISMS as VIX_MECHANISMS,
    vix_change,
    vix_level,
)

store = ParquetStore("data")


def load_ohlcv(ticker: str) -> pd.DataFrame:
    df = store.load_raw(ticker)
    report = validate_daily_ohlcv(df)
    assert report.ok, f"{ticker} failed validation: {report.issues}"
    return df


def build_candidate(name: str, ohlcv: pd.DataFrame, vix: pd.Series | None = None) -> tuple[pd.Series, int]:
    close, volume = ohlcv["close"], ohlcv["volume"]
    horizon = {**TECH_MECHANISMS, **VIX_MECHANISMS}[name]["horizon"]

    if name == "reversal_1d":
        return reversal_1d(close), horizon
    if name == "momentum_12_1":
        return momentum_12_1(close), horizon
    if name == "vol_cluster":
        return realized_vol(close), horizon
    if name == "macd_divergence":
        return macd_divergence(close), horizon
    if name == "volume_breakout":
        return volume_breakout(close, volume), horizon
    if name == "vix_level":
        return vix_level(vix, ohlcv.index), horizon
    if name == "vix_change":
        return vix_change(vix, ohlcv.index), horizon
    raise ValueError(name)


def build_target(name: str, ohlcv: pd.DataFrame, horizon: int) -> pd.Series:
    if name == "vol_cluster":
        return future_realized_vol(ohlcv["close"], horizon=horizon, window=5)
    return forward_log_return(ohlcv["close"], horizon=horizon)


def main():
    vix_full = store.load_table("vix")["vix"] if store.load_table("vix") is not None else None

    all_candidates = list(TECH_MECHANISMS) + list(VIX_MECHANISMS)
    proofs = []
    admitted_features: dict[str, pd.Series] = {}
    admitted_target_horizon = 1  # marginal-contribution baseline uses the 1d return target

    for name in all_candidates:
        mech = {**TECH_MECHANISMS, **VIX_MECHANISMS}[name]
        per_ticker_data = {}
        for ticker in PROOF_TICKERS:
            ohlcv = load_ohlcv(ticker)
            feature, horizon = build_candidate(name, ohlcv, vix_full)
            target = build_target(name, ohlcv, horizon)
            aligned = pd.concat([feature, target], axis=1, join="inner").dropna()
            per_ticker_data[ticker] = (aligned.iloc[:, 0], aligned.iloc[:, 1])

        proof = prove_feature(name, mech["mechanism"], mech["horizon"], per_ticker_data)
        proofs.append(proof)
        print(proof.summary())
        print()

    print("=" * 70)
    print("WAVE 1 SUMMARY")
    print("=" * 70)
    for p in proofs:
        print(f"{p.feature:20s} horizon={p.horizon:3d}d  passing={p.n_tickers_passing_ic}/8  verdict={p.verdict}")

    n_admitted = sum(1 for p in proofs if p.verdict == "ADMITTED")
    print(f"\n{n_admitted}/{len(proofs)} candidates admitted at |IC|>=0.08, consistency>=0.70, >=6/8 tickers.")


if __name__ == "__main__":
    main()
