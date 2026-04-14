import numpy as np
import pandas as pd
import pytest

from luqman.features.registry import REGISTRY, admitted, build_registered_features
from luqman.features.vol_targets import volatility_target


def test_vol_cluster_is_registered_as_volatility_target():
    assert "vol_cluster" in REGISTRY
    feat = REGISTRY["vol_cluster"]
    assert feat.target_kind == "volatility"
    assert feat.category == "technical"
    assert feat.horizon == 5
    assert feat.proof_run


def test_no_return_direction_features_registered_yet():
    """Wave 1 rejected every direction candidate — the registry must not
    silently contain one until a future wave admits it."""
    assert admitted(target_kind="return") == []


def test_admitted_filters_by_category_and_kind():
    vol_features = admitted(target_kind="volatility")
    assert len(vol_features) == 1
    assert vol_features[0].name == "vol_cluster"
    assert admitted(category="nonexistent") == []


def test_build_registered_features_matches_direct_call(ohlcv):
    built = build_registered_features(ohlcv, target_kind="volatility")
    assert "vol_cluster" in built.columns
    assert len(built) == len(ohlcv)


def test_build_registered_features_raises_for_unregistered_kind(ohlcv):
    with pytest.raises(ValueError):
        build_registered_features(ohlcv, target_kind="return")


def test_volatility_target_matches_future_realized_vol(ohlcv):
    from luqman.features.technical.candidates import future_realized_vol

    target = volatility_target(ohlcv["close"], horizon=5, window=5)
    expected = future_realized_vol(ohlcv["close"], horizon=5, window=5)
    pd.testing.assert_series_equal(target, expected)
    assert (target.dropna() >= 0).all()
