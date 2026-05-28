from luqman.features.registry import CROSS_SECTIONAL_REGISTRY, admitted_cross_sectional


def test_momentum_registered_cross_sectionally():
    assert "momentum_12_1_xs" in CROSS_SECTIONAL_REGISTRY
    factor = CROSS_SECTIONAL_REGISTRY["momentum_12_1_xs"]
    assert factor.horizon == 21
    assert factor.category == "technical"
    assert factor.proof_run


def test_admitted_cross_sectional_filters_by_category():
    assert len(admitted_cross_sectional()) == 1
    assert len(admitted_cross_sectional(category="technical")) == 1
    assert admitted_cross_sectional(category="nonexistent") == []


def test_momentum_panel_builder_shape(ohlcv):
    factor = CROSS_SECTIONAL_REGISTRY["momentum_12_1_xs"]
    panel = factor.build({"A": ohlcv, "B": ohlcv})
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == len(ohlcv)
