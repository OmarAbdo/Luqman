"""Tests for the factor-residual machinery.

The residual features are the foundation of the whole Wave 3 candidate set, so
the regression itself is checked against hand-constructed cases where the
correct answer is known analytically, not merely for plausible-looking output.
"""

import numpy as np
import pandas as pd
import pytest

from luqman.data.fama_french import parse_ff_daily_csv
from luqman.features.residual import (
    amihud_illiquidity,
    idiosyncratic_vol,
    long_term_reversal,
    residual_momentum,
    residual_return_series,
    residual_reversal,
    rolling_ols_residuals,
)


@pytest.fixture
def factors():
    dates = pd.bdate_range("2015-01-01", periods=600, name="date")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.0004, 0.01, len(dates)),
            "smb": rng.normal(0.0, 0.005, len(dates)),
            "hml": rng.normal(0.0, 0.005, len(dates)),
            "rf": np.full(len(dates), 0.00008),
        },
        index=dates,
    )


def test_exact_linear_relationship_leaves_zero_residual(factors):
    """If y is EXACTLY a linear function of the factors, every residual must be
    ~0. This is the strongest available correctness check on the regression."""
    y = (
        0.002
        + 1.4 * factors["mkt_rf"]
        - 0.6 * factors["smb"]
        + 0.3 * factors["hml"]
    )
    y.name = "y"
    resid = rolling_ols_residuals(y, factors[["mkt_rf", "smb", "hml"]], window=252)
    assert np.abs(resid.dropna()).max() < 1e-10


def test_residual_is_orthogonal_to_factors_in_window(factors):
    """OLS residuals must be uncorrelated with the regressors over the fitting
    window — the defining property of a least-squares fit."""
    rng = np.random.default_rng(1)
    y = 1.1 * factors["mkt_rf"] + rng.normal(0, 0.01, len(factors))
    y.name = "y"
    resid = rolling_ols_residuals(y, factors[["mkt_rf", "smb", "hml"]], window=252)
    joined = pd.concat([resid, factors["mkt_rf"]], axis=1).dropna()
    # Residual retains only the idiosyncratic part, so its correlation with the
    # market factor should be far below the raw series' correlation.
    raw_corr = abs(y.corr(factors["mkt_rf"]))
    resid_corr = abs(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
    assert resid_corr < raw_corr / 2


def test_no_lookahead_in_rolling_regression(factors):
    """Residuals for early dates must not change when later data is appended —
    the same no-lookahead property enforced on every other feature."""
    rng = np.random.default_rng(2)
    y = pd.Series(rng.normal(0, 0.01, len(factors)), index=factors.index, name="y")
    x = factors[["mkt_rf", "smb", "hml"]]

    full = rolling_ols_residuals(y, x, window=252)
    truncated = rolling_ols_residuals(y.iloc[:400], x.iloc[:400], window=252)
    pd.testing.assert_series_equal(full.iloc[:400], truncated)


def test_warmup_produces_nan(factors):
    y = pd.Series(np.random.default_rng(3).normal(0, 0.01, len(factors)), index=factors.index, name="y")
    resid = rolling_ols_residuals(y, factors[["mkt_rf", "smb", "hml"]], window=252, min_periods=126)
    assert resid.iloc[:100].isna().all()
    assert resid.iloc[300:].notna().all()


def test_residual_return_series_end_to_end(factors):
    close = pd.Series(
        100 * np.exp(np.cumsum(np.random.default_rng(4).normal(0.0003, 0.01, len(factors)))),
        index=factors.index,
        name="close",
    )
    resid = residual_return_series(close, factors, window=252)
    assert resid.notna().sum() > 200
    assert resid.dropna().abs().max() < 0.5  # daily residual returns stay sane


def test_residual_momentum_skips_recent_month():
    """The 12-1 construction must exclude the most recent `skip` days: a spike
    inside the skip window must not move the feature."""
    idx = pd.bdate_range("2015-01-01", periods=400, name="date")
    resid = pd.Series(0.0, index=idx, name="residual_return")
    base = residual_momentum(resid, lookback=252, skip=21)

    spiked = resid.copy()
    spiked.iloc[-5] = 10.0  # inside the skipped recent window
    after = residual_momentum(spiked, lookback=252, skip=21)

    pd.testing.assert_series_equal(base, after)


def test_residual_reversal_sign_convention():
    """A name that recently UNDERperformed its factor exposure (negative
    residuals) should get a POSITIVE feature value, meaning 'expected to rise'."""
    idx = pd.bdate_range("2015-01-01", periods=50, name="date")
    resid = pd.Series(-0.01, index=idx, name="residual_return")
    assert residual_reversal(resid, window=5).dropna().gt(0).all()


def test_long_term_reversal_sign_convention():
    """A past loser must score positive (expected to revert upward)."""
    idx = pd.bdate_range("2000-01-01", periods=1500, name="date")
    falling = pd.Series(np.linspace(200, 100, len(idx)), index=idx, name="close")
    assert long_term_reversal(falling, lookback=1260, skip=252).dropna().gt(0).all()

    rising = pd.Series(np.linspace(100, 200, len(idx)), index=idx, name="close")
    assert long_term_reversal(rising, lookback=1260, skip=252).dropna().lt(0).all()


def test_idiosyncratic_vol_is_nonnegative(factors):
    resid = pd.Series(np.random.default_rng(5).normal(0, 0.01, len(factors)), index=factors.index)
    ivol = idiosyncratic_vol(resid, window=20)
    assert ivol.dropna().ge(0).all()


def test_amihud_higher_for_thinner_volume():
    """The same price path traded on lower volume must score as MORE illiquid."""
    idx = pd.bdate_range("2015-01-01", periods=100, name="date")
    close = pd.Series(100 + np.arange(len(idx)) * 0.1, index=idx)
    liquid = amihud_illiquidity(close, pd.Series(1e7, index=idx))
    thin = amihud_illiquidity(close, pd.Series(1e5, index=idx))
    assert thin.dropna().mean() > liquid.dropna().mean()


def test_parse_ff_csv_skips_header_and_trailing_sections():
    """The French library ships a prose header and sometimes trailing annual
    blocks; the parser must take only the daily rows and convert from percent."""
    text = (
        "This file was created by CMPT_ME_BEME_RETS using the 202601 CRSP database.\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19260701,0.09,-0.25,-0.27,0.01\n"
        "19260702,0.45,-0.33,-0.06,0.01\n"
        "\n"
        "Annual Factors: January-December\n"
        "1927,29.47,-2.46,-3.75,3.12\n"
    )
    df = parse_ff_daily_csv(text)
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("1926-07-01")
    assert df["mkt_rf"].iloc[0] == pytest.approx(0.0009)  # percent -> decimal
