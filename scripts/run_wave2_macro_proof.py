"""Wave 2 macro-track proof: relative strength vs. SPY (mechanism-appropriate
20d+ horizon), and macro rate features re-tested at 20d instead of the
original 1d, per dataset-plan.md Section 2d. Not part of the installed package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.evaluation.feature_proof import PROOF_TICKERS, prove_feature
from luqman.features.macro_features import build_macro_features
from luqman.features.relative_strength import MECHANISM as RS_MECHANISM, relative_strength
from luqman.features.targets import forward_log_return

store = ParquetStore("data")
spy = store.load_raw("SPY")["close"]
macro = store.load_table("macro_us")


def main():
    horizon = 20  # matches the relative-strength/macro-regime mechanism's own timescale

    print(f"=== relative_strength_20d (horizon {horizon}d) ===")
    rs_data = {}
    for ticker in PROOF_TICKERS:
        ohlcv = store.load_raw(ticker)
        assert validate_daily_ohlcv(ohlcv).ok
        feature = relative_strength(ohlcv["close"], spy, window=20)
        target = forward_log_return(ohlcv["close"], horizon=horizon)
        aligned = pd.concat([feature, target], axis=1, join="inner").dropna()
        rs_data[ticker] = (aligned.iloc[:, 0], aligned.iloc[:, 1])
    proof = prove_feature("relative_strength_20d", RS_MECHANISM, horizon, rs_data)
    print(proof.summary())

    for macro_col in ["macro_y10", "macro_spread_10y3m", "macro_y10_chg_20d", "macro_fedfunds"]:
        print(f"\n=== {macro_col} (horizon {horizon}d, re-tested at regime timescale) ===")
        data = {}
        for ticker in PROOF_TICKERS:
            ohlcv = store.load_raw(ticker)
            feats = build_macro_features(macro, ohlcv.index)
            target = forward_log_return(ohlcv["close"], horizon=horizon)
            aligned = pd.concat([feats[macro_col], target], axis=1, join="inner").dropna()
            data[ticker] = (aligned.iloc[:, 0], aligned.iloc[:, 1])
        proof = prove_feature(macro_col, f"Macro regime feature re-tested at {horizon}d", horizon, data)
        print(proof.summary())


if __name__ == "__main__":
    main()
