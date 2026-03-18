import json

import numpy as np

from luqman.evaluation.evaluate import ZERO_BASELINE
from luqman.experiments.tracker import data_hash, run_experiment
from luqman.features.features import build_features
from luqman.features.targets import forward_log_return
from luqman.models.baselines import ZeroReturnBaseline
from luqman.models.boosted import make_hist_gbm
from luqman.splits.walk_forward import WalkForwardSplitter


def _aligned(ohlcv):
    feats = build_features(ohlcv)
    target = forward_log_return(ohlcv["close"])
    aligned = feats.join(target, how="inner").dropna()
    return aligned[feats.columns], aligned[target.name]


def test_data_hash_stable_and_sensitive(ohlcv):
    X, y = _aligned(ohlcv)
    aligned = X.assign(**{y.name: y})
    assert data_hash(aligned) == data_hash(aligned.copy())
    perturbed = aligned.copy()
    perturbed.iloc[0, 0] += 1e-6
    assert data_hash(aligned) != data_hash(perturbed)


def test_gbm_is_deterministic(ohlcv):
    X, y = _aligned(ohlcv)
    preds = []
    for _ in range(2):
        model = make_hist_gbm(seed=7)
        model.fit(X.iloc[:300].to_numpy(), y.iloc[:300].to_numpy())
        preds.append(model.predict(X.iloc[300:350].to_numpy()))
    assert np.array_equal(preds[0], preds[1])


def test_run_experiment_writes_record_and_index(tmp_path, ohlcv):
    X, y = _aligned(ohlcv)
    splitter = WalkForwardSplitter(n_splits=2, test_size=40, embargo=1, min_train_size=150)

    table, record_path = run_experiment(
        name="unit",
        ticker="TEST",
        X=X,
        y=y,
        model_factories={ZERO_BASELINE: ZeroReturnBaseline, "gbm": lambda: make_hist_gbm(7)},
        splitter=splitter,
        experiments_dir=tmp_path,
        seed=7,
        horizon=1,
    )

    record = json.loads(record_path.read_text())
    assert record["ticker"] == "TEST"
    assert record["seed"] == 7
    assert record["splitter"]["n_splits"] == 2
    assert set(record["metrics"]) == {ZERO_BASELINE, "gbm"}
    assert record["data"]["n_rows"] == len(X)
    assert len(record["data"]["hash"]) == 16
    assert record["features"] == list(X.columns)

    index = (tmp_path / "index.csv").read_text().splitlines()
    assert len(index) == 3  # header + one row per model
    assert "unit" in index[1]
