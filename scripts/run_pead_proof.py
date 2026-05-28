"""PEAD proof run: real earnings-surprise feature (FMP actual vs. estimate),
per dataset-plan.md Section 2c. Not part of the installed package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.evaluation.feature_proof import PROOF_TICKERS, marginal_contribution, prove_feature
from luqman.features.pead import MECHANISM, build_pead_feature
from luqman.features.targets import forward_log_return

store = ParquetStore("data")


def main():
    for horizon in [5, 20]:
        print(f"\n{'='*70}\nPEAD at horizon {horizon}d\n{'='*70}")
        per_ticker_data = {}
        for ticker in PROOF_TICKERS:
            ohlcv = store.load_raw(ticker)
            assert validate_daily_ohlcv(ohlcv).ok
            earnings = store.load_table(f"{ticker}_earnings")
            feature = build_pead_feature(earnings, ohlcv.index)
            target = forward_log_return(ohlcv["close"], horizon=horizon)
            aligned = pd.concat([feature, target], axis=1, join="inner").dropna()
            per_ticker_data[ticker] = (aligned.iloc[:, 0], aligned.iloc[:, 1])
            print(f"  {ticker}: {len(aligned)} rows with an active PEAD window")

        proof = prove_feature("pead_surprise_decay", MECHANISM, horizon, per_ticker_data)
        print()
        print(proof.summary())

        if proof.verdict == "ADMITTED":
            print("\n--- marginal contribution vs zero baseline ---")
            deltas = []
            for ticker, (feat, target) in per_ticker_data.items():
                empty = pd.DataFrame(index=feat.index)
                delta = marginal_contribution(empty, feat, target)
                deltas.append(delta)
                print(f"  {ticker}: {delta:+.4f}")
            print(f"  mean: {sum(deltas)/len(deltas):+.4f}, passing (>0): {sum(1 for d in deltas if d>0)}/{len(deltas)}")


if __name__ == "__main__":
    main()
