import numpy as np
import pandas as pd

from luqman.evaluation.feature_proof import (
    CONSISTENCY_THRESHOLD,
    IC_THRESHOLD,
    MIN_TICKERS_PASSING,
    PROOF_TICKERS,
    marginal_contribution,
    prove_feature,
)


def _correlated_pair(n, seed, strength=0.9):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    y = pd.Series(rng.normal(size=n), index=idx, name="target")
    x = (y * strength + rng.normal(scale=0.2, size=n)).rename("feat")
    return x, y


def test_prove_feature_admits_strong_consistent_signal():
    data = {t: _correlated_pair(400, seed=i) for i, t in enumerate(PROOF_TICKERS)}
    proof = prove_feature("feat", "synthetic strong correlation", horizon=1, per_ticker_data=data)
    assert proof.n_tickers_passing_ic >= MIN_TICKERS_PASSING
    assert proof.verdict == "ADMITTED"
    assert "ADMITTED" in proof.summary()


def test_prove_feature_rejects_pure_noise():
    def _noise_pair(n, seed):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2022-01-03", periods=n)
        x = pd.Series(rng.normal(size=n), index=idx, name="feat")
        y = pd.Series(rng.normal(size=n), index=idx, name="target")
        return x, y

    data = {t: _noise_pair(400, seed=100 + i) for i, t in enumerate(PROOF_TICKERS)}
    proof = prove_feature("feat", "synthetic noise", horizon=1, per_ticker_data=data)
    assert proof.n_tickers_passing_ic < MIN_TICKERS_PASSING
    assert proof.verdict == "REJECTED"


def test_prove_feature_thresholds_match_dataset_plan():
    assert IC_THRESHOLD == 0.08
    assert CONSISTENCY_THRESHOLD == 0.70
    assert MIN_TICKERS_PASSING == 6
    assert len(PROOF_TICKERS) == 8


def test_marginal_contribution_positive_for_informative_addition():
    rng = np.random.default_rng(5)
    n = 500
    idx = pd.bdate_range("2022-01-03", periods=n)
    y = pd.Series(rng.normal(scale=0.01, size=n), index=idx, name="target")
    baseline = pd.DataFrame({"noise": rng.normal(size=n)}, index=idx)
    candidate = (y * 0.9 + rng.normal(scale=0.002, size=n)).rename("candidate")

    delta = marginal_contribution(baseline, candidate, y)
    assert delta > 0


def test_marginal_contribution_first_feature_no_baseline():
    rng = np.random.default_rng(6)
    n = 400
    idx = pd.bdate_range("2022-01-03", periods=n)
    y = pd.Series(rng.normal(scale=0.01, size=n), index=idx, name="target")
    candidate = (y * 0.9 + rng.normal(scale=0.002, size=n)).rename("candidate")
    empty_baseline = pd.DataFrame(index=idx)

    delta = marginal_contribution(empty_baseline, candidate, y)
    assert np.isfinite(delta)
