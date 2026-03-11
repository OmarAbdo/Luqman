import numpy as np
import pandas as pd

from luqman.features.targets import direction, forward_log_return


def test_alignment_hand_computed():
    close = pd.Series(
        [100.0, 110.0, 99.0, 99.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    target = forward_log_return(close, horizon=1)
    assert np.isclose(target.iloc[0], np.log(110.0 / 100.0))
    assert np.isclose(target.iloc[1], np.log(99.0 / 110.0))
    assert np.isclose(target.iloc[2], 0.0)
    assert np.isnan(target.iloc[3])


def test_horizon_tail_is_nan(ohlcv):
    target = forward_log_return(ohlcv["close"], horizon=5)
    assert target.iloc[-5:].isna().all()
    assert target.iloc[:-5].notna().all()


def test_direction_labels():
    close = pd.Series(
        [100.0, 110.0, 99.0, 99.0],
        index=pd.bdate_range("2024-01-01", periods=4),
    )
    label = direction(forward_log_return(close))
    assert list(label.iloc[:3]) == [1.0, -1.0, 0.0]
