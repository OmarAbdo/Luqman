import numpy as np
import pandas as pd
import pytest

from luqman.backtest.engine import backtest_signals
from luqman.evaluation.evaluate import walk_forward_predictions
from luqman.models.baselines import HistoricalMeanBaseline
from luqman.splits.walk_forward import WalkForwardSplitter


@pytest.fixture
def returns():
    values = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03]
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=6))


def test_flat_strategy_earns_nothing(returns):
    result = backtest_signals(returns, pd.Series(0.0, index=returns.index), cost_bps=5)
    assert result.metrics["total_return"] == 0.0
    assert result.metrics["n_trades"] == 0


def test_oracle_long_flat_no_costs(returns):
    oracle = returns.copy()
    result = backtest_signals(returns, oracle, cost_bps=0)
    assert np.isclose(result.metrics["total_return"], 0.06)  # sum of up moves
    assert result.metrics["max_drawdown"] == 0.0


def test_oracle_long_short_doubles_edge(returns):
    oracle = returns.copy()
    result = backtest_signals(returns, oracle, cost_bps=0, allow_short=True)
    assert np.isclose(result.metrics["total_return"], float(np.abs(returns).sum()))


def test_costs_reduce_pnl(returns):
    oracle = returns.copy()
    free = backtest_signals(returns, oracle, cost_bps=0)
    costed = backtest_signals(returns, oracle, cost_bps=50)
    assert costed.metrics["total_return"] < free.metrics["total_return"]
    assert costed.metrics["n_trades"] == free.metrics["n_trades"] > 0


def test_always_long_matches_buy_hold_minus_entry_cost(returns):
    always_up = pd.Series(1.0, index=returns.index)
    result = backtest_signals(returns, always_up, cost_bps=10)
    expected = returns.sum() - 0.001  # one entry trade at 10 bps
    assert np.isclose(result.metrics["total_return"], expected)
    assert np.isclose(result.metrics["buy_hold_return"], returns.sum())


def test_misaligned_indices_are_intersected(returns):
    preds = pd.Series(1.0, index=returns.index[2:])
    result = backtest_signals(returns, preds, cost_bps=0)
    assert result.metrics["n_periods"] == 4
    assert np.isclose(result.metrics["total_return"], returns.iloc[2:].sum())


def test_walk_forward_predictions_cover_test_rows_only(ohlcv):
    from luqman.features.features import build_features
    from luqman.features.targets import forward_log_return

    feats = build_features(ohlcv)
    target = forward_log_return(ohlcv["close"])
    aligned = feats.join(target, how="inner").dropna()
    X, y = aligned[feats.columns], aligned[target.name]

    splitter = WalkForwardSplitter(n_splits=3, test_size=40, embargo=1, min_train_size=150)
    preds = walk_forward_predictions(X, y, HistoricalMeanBaseline, splitter)

    assert len(preds) == 120
    assert preds.index.equals(X.index[-120:])
    assert preds.index.is_monotonic_increasing
