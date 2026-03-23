import numpy as np
import pandas as pd

from luqman.evaluation.feature_audit import audit_features, information_coefficient


def test_ic_perfect_monotonic_relationship():
    x = pd.Series(np.arange(100, dtype=float))
    y = x * 2 + 1  # perfectly monotonic -> Spearman rho = 1
    assert np.isclose(information_coefficient(x, y), 1.0)


def test_ic_inverse_relationship():
    x = pd.Series(np.arange(100, dtype=float))
    y = -x
    assert np.isclose(information_coefficient(x, y), -1.0)


def test_ic_pure_noise_near_zero():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=2000))
    y = pd.Series(rng.normal(size=2000))
    assert abs(information_coefficient(x, y)) < 0.05


def test_ic_nan_for_constant_feature():
    x = pd.Series(np.ones(50))
    y = pd.Series(np.arange(50, dtype=float))
    assert np.isnan(information_coefficient(x, y))


def test_audit_features_ranks_real_signal_above_noise():
    rng = np.random.default_rng(1)
    n = 500
    idx = pd.bdate_range("2022-01-03", periods=n)
    y = pd.Series(rng.normal(size=n), index=idx, name="target")
    X = pd.DataFrame(
        {
            "signal": y * 0.8 + rng.normal(scale=0.3, size=n),  # genuinely correlated
            "noise": rng.normal(size=n),  # unrelated
        },
        index=idx,
    )
    table = audit_features(X, y, n_splits=5)
    assert table.index[0] == "signal"
    assert abs(table.loc["signal", "ic_full"]) > abs(table.loc["noise", "ic_full"])
    assert table.loc["signal", "ic_sign_consistency"] > table.loc["noise", "ic_sign_consistency"]


def test_audit_features_handles_short_blocks():
    rng = np.random.default_rng(2)
    n = 20
    idx = pd.bdate_range("2022-01-03", periods=n)
    X = pd.DataFrame({"f": rng.normal(size=n)}, index=idx)
    y = pd.Series(rng.normal(size=n), index=idx, name="target")
    table = audit_features(X, y, n_splits=5)
    assert "f" in table.index
