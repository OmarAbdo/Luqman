from luqman.features.registry import (
    CROSS_SECTIONAL_REGISTRY,
    REJECTED_CROSS_SECTIONAL_CANDIDATES,
    admitted_cross_sectional,
)


def test_momentum_registered_cross_sectionally():
    assert "momentum_12_1_xs" in CROSS_SECTIONAL_REGISTRY
    factor = CROSS_SECTIONAL_REGISTRY["momentum_12_1_xs"]
    assert factor.horizon == 21
    assert factor.category == "technical"
    assert factor.proof_run


def test_residual_momentum_registered_cross_sectionally():
    """Wave 3's one admission (dataset-plan.md Section 10)."""
    assert "residual_momentum_xs" in CROSS_SECTIONAL_REGISTRY
    factor = CROSS_SECTIONAL_REGISTRY["residual_momentum_xs"]
    assert factor.horizon == 21
    assert factor.proof_run == "wave3-residual-2026-07-21"


def test_admitted_cross_sectional_filters_by_category():
    all_admitted = admitted_cross_sectional()
    assert len(all_admitted) == len(CROSS_SECTIONAL_REGISTRY)
    assert admitted_cross_sectional(category="technical") == all_admitted
    assert admitted_cross_sectional(category="nonexistent") == []


def test_rejected_candidates_never_leak_into_admitted():
    """The registry's core safety property: a rejected candidate must not be
    reachable through the admitted-factor API, however good its raw statistics
    looked (idiosyncratic_vol_xs scored a +1.39 net Sharpe and is still out)."""
    admitted_names = {f.name for f in admitted_cross_sectional()}
    rejected_names = set(REJECTED_CROSS_SECTIONAL_CANDIDATES)
    assert admitted_names.isdisjoint(rejected_names)
    assert "idiosyncratic_vol_xs" in rejected_names


def test_momentum_panel_builder_shape(ohlcv):
    factor = CROSS_SECTIONAL_REGISTRY["momentum_12_1_xs"]
    panel = factor.build({"A": ohlcv, "B": ohlcv})
    assert list(panel.columns) == ["A", "B"]
    assert len(panel) == len(ohlcv)
