"""Cross-asset/geopolitical proof: gold/oil/crypto flight-to-safety signals
predicting SPY (broad market) forward returns, per owner's suggestion
(2026-07-20). Not part of the installed package.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from luqman.data.store import ParquetStore
from luqman.data.validation import validate_daily_ohlcv
from luqman.evaluation.feature_audit import audit_features
from luqman.features.cross_asset import (
    MECHANISMS,
    crypto_risk_appetite,
    gold_equity_divergence,
    oil_shock,
)
from luqman.features.targets import forward_log_return

store = ParquetStore("data")
spy = store.load_raw("SPY")
assert validate_daily_ohlcv(spy).ok


def main():
    gold = store.load_raw("GLD")["close"]
    oil = store.load_raw("USO")["close"]
    btc = store.load_raw("BTC-USD")["close"]

    candidates = {
        "gold_equity_divergence": gold_equity_divergence(gold, spy["close"], spy.index),
        "oil_shock": oil_shock(oil, spy.index),
        "crypto_risk_appetite": crypto_risk_appetite(btc, spy.index),
    }

    for name, feature in candidates.items():
        horizon = MECHANISMS[name]["horizon"]
        target = forward_log_return(spy["close"], horizon=horizon)
        aligned = pd.concat([feature, target], axis=1, join="inner").dropna()
        X = aligned.iloc[:, [0]]
        y = aligned.iloc[:, 1]
        audit = audit_features(X, y, n_splits=5)
        print(f"=== {name} -> SPY forward {horizon}d return ===")
        print(f"mechanism: {MECHANISMS[name]['mechanism']}")
        print(f"n_obs: {len(aligned)}")
        print(audit.to_string(float_format=lambda v: f"{v: .4f}"))
        print()


if __name__ == "__main__":
    main()
