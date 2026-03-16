import numpy as np
import pandas as pd

from luqman.evaluation.evaluate import ZERO_BASELINE, walk_forward_evaluate
from luqman.evaluation.metrics import directional_accuracy, mae, mase, up_base_rate
from luqman.features.features import build_features
from luqman.features.targets import forward_log_return
from luqman.models.baselines import HistoricalMeanBaseline, ZeroReturnBaseline, make_ridge
from luqman.splits.walk_forward import WalkForwardSplitter


def test_zero_baseline_mase_is_one():
    y = np.array([0.01, -0.02, 0.005, -0.01])
    pred = np.zeros(4)
    assert np.isclose(mase(y, pred, y), 1.0)


def test_directional_accuracy_known():
    y = np.array([0.01, -0.02, 0.03, -0.04])
    pred = np.array([0.5, 0.5, 0.5, -0.5])
    assert np.isclose(directional_accuracy(y, pred), 0.75)


def test_directional_accuracy_nan_for_zero_predictions():
    y = np.array([0.01, -0.02])
    assert np.isnan(directional_accuracy(y, np.zeros(2)))


def test_up_base_rate():
    y = np.array([0.01, -0.02, 0.03, 0.0])
    assert np.isclose(up_base_rate(y), 2 / 3)


def test_baseline_predictions():
    X = np.zeros((10, 3))
    y = np.arange(10, dtype=float)
    assert (ZeroReturnBaseline().fit(X, y).predict(X) == 0).all()
    assert np.allclose(HistoricalMeanBaseline().fit(X, y).predict(X), 4.5)


def test_walk_forward_evaluate_end_to_end(ohlcv):
    feats = build_features(ohlcv)
    target = forward_log_return(ohlcv["close"])
    aligned = feats.join(target, how="inner").dropna()
    X, y = aligned[feats.columns], aligned[target.name]

    table = walk_forward_evaluate(
        X,
        y,
        {
            ZERO_BASELINE: ZeroReturnBaseline,
            "hist_mean": HistoricalMeanBaseline,
            "ridge": make_ridge,
        },
        WalkForwardSplitter(n_splits=3, test_size=40, embargo=1, min_train_size=150),
    )

    assert set(table.index) == {ZERO_BASELINE, "hist_mean", "ridge"}
    assert np.isclose(table.loc[ZERO_BASELINE, "skill_vs_zero"], 0.0)
    assert np.isclose(table.loc[ZERO_BASELINE, "mase"], 1.0, atol=0.15)
    assert np.isfinite(table["mae"]).all()
    assert 0 < table.attrs["up_base_rate"] < 1


def test_ridge_pipeline_refits_scaler_per_fold():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 4)), columns=list("abcd"))
    y = pd.Series(rng.normal(size=300))
    model = make_ridge()
    model.fit(X.iloc[:200].to_numpy(), y.iloc[:200].to_numpy())
    first_mean = model.named_steps["scale"].mean_.copy()
    model.fit(X.iloc[:100].to_numpy(), y.iloc[:100].to_numpy())
    assert not np.allclose(first_mean, model.named_steps["scale"].mean_)
