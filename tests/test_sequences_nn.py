import numpy as np
import pandas as pd
import pytest

from luqman.evaluation.evaluate import ZERO_BASELINE, walk_forward_evaluate, walk_forward_predictions
from luqman.models.baselines import ZeroReturnBaseline
from luqman.models.sequences import make_sequences
from luqman.splits.walk_forward import WalkForwardSplitter

torch = pytest.importorskip("torch")
from luqman.models.torch_lstm import LSTMAttentionRegressor  # noqa: E402


def _toy_frame(n=200, n_features=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    X = pd.DataFrame(rng.normal(size=(n, n_features)), index=idx, columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(rng.normal(scale=0.01, size=n), index=idx, name="target")
    return X, y


def test_make_sequences_shapes_and_alignment():
    X, y = _toy_frame(n=50, n_features=4)
    X_seq, y_seq, index = make_sequences(X, y, seq_len=10)
    assert X_seq.shape == (41, 10, 4)
    assert y_seq.shape == (41,)
    assert len(index) == 41
    # sample 0's window is rows [0,10); its target is row 9's target.
    assert np.allclose(X_seq[0], X.iloc[0:10].to_numpy())
    assert np.isclose(y_seq[0], y.iloc[9])
    assert index[0] == X.index[9]
    assert index[-1] == X.index[-1]


def test_make_sequences_too_short_raises():
    X, y = _toy_frame(n=5, n_features=2)
    with pytest.raises(ValueError):
        make_sequences(X, y, seq_len=10)


def test_make_sequences_no_lookahead():
    """A sample's window must only ever contain rows up to its target row —
    truncating history after that point must not change the sample."""
    X, y = _toy_frame(n=60, n_features=3)
    X_seq_full, y_seq_full, idx_full = make_sequences(X, y, seq_len=15)
    X_seq_trunc, y_seq_trunc, idx_trunc = make_sequences(X.iloc[:40], y.iloc[:40], seq_len=15)
    shared = min(len(idx_trunc), len(idx_full))
    assert np.allclose(X_seq_full[:shared], X_seq_trunc[:shared])
    np.testing.assert_allclose(y_seq_full[:shared], y_seq_trunc[:shared])


def test_lstm_attention_fits_and_predicts_finite():
    X, y = _toy_frame(n=150, n_features=3)
    X_seq, y_seq, _ = make_sequences(X, y, seq_len=10)
    model = LSTMAttentionRegressor(
        n_features=3, hidden_size=8, max_epochs=3, patience=2, batch_size=16, seed=1
    )
    model.fit(X_seq[:120], y_seq[:120])
    pred = model.predict(X_seq[120:])
    assert pred.shape == (len(X_seq) - 120,)
    assert np.isfinite(pred).all()


def test_lstm_attention_deterministic_with_seed():
    X, y = _toy_frame(n=150, n_features=3)
    X_seq, y_seq, _ = make_sequences(X, y, seq_len=10)
    preds = []
    for _ in range(2):
        model = LSTMAttentionRegressor(
            n_features=3, hidden_size=8, max_epochs=3, patience=2, batch_size=16, seed=7
        )
        model.fit(X_seq[:120], y_seq[:120])
        preds.append(model.predict(X_seq[120:]))
    np.testing.assert_allclose(preds[0], preds[1])


def test_lstm_scaler_uses_train_fold_only():
    X, y = _toy_frame(n=150, n_features=2)
    X_seq, y_seq, _ = make_sequences(X, y, seq_len=8)
    model = LSTMAttentionRegressor(n_features=2, hidden_size=4, max_epochs=2, patience=1, seed=3)
    model.fit(X_seq[:60], y_seq[:60])
    mean_60 = model.mean_.copy()
    model.fit(X_seq[:30], y_seq[:30])
    assert not np.allclose(mean_60, model.mean_)


def test_walk_forward_evaluate_accepts_sequence_array():
    X, y = _toy_frame(n=400, n_features=3)
    X_seq, y_seq, index = make_sequences(X, y, seq_len=10)
    y_seq = pd.Series(y_seq, index=index)

    factories = {
        ZERO_BASELINE: ZeroReturnBaseline,
        "lstm": lambda: LSTMAttentionRegressor(
            n_features=3, hidden_size=8, max_epochs=2, patience=1, batch_size=32, seed=5
        ),
    }
    splitter = WalkForwardSplitter(n_splits=2, test_size=40, embargo=1, min_train_size=150)
    table = walk_forward_evaluate(X_seq, y_seq, factories, splitter)
    assert set(table.index) == {ZERO_BASELINE, "lstm"}
    assert np.isfinite(table["mae"]).all()


def test_walk_forward_predictions_accepts_sequence_array():
    X, y = _toy_frame(n=400, n_features=3)
    X_seq, y_seq, index = make_sequences(X, y, seq_len=10)
    y_seq = pd.Series(y_seq, index=index)
    splitter = WalkForwardSplitter(n_splits=2, test_size=40, embargo=1, min_train_size=150)
    preds = walk_forward_predictions(X_seq, y_seq, ZeroReturnBaseline, splitter)
    assert len(preds) == 80
    assert preds.index.equals(y_seq.index[-80:])
