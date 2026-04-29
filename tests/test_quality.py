import numpy as np
import pandas as pd

from luqman.features.quality import build_gross_profitability_feature, gross_profitability

DAYS = pd.bdate_range("2020-01-02", periods=1500)  # ~6 trading years


def _gp_inputs(years, gross_profits, total_assets):
    return pd.DataFrame(
        {
            "fiscal_year_end": pd.to_datetime([f"{y}-09-30" for y in years]),
            "gross_profit": gross_profits,
            "total_assets": total_assets,
        }
    )


def test_gross_profitability_formula():
    inputs = _gp_inputs([2023], [200.0], [1000.0])
    result = gross_profitability(inputs)
    assert np.isclose(result.iloc[0], 0.20)


def test_gross_profitability_higher_for_more_profitable_per_asset():
    inputs = _gp_inputs([2023, 2023], [200.0, 200.0], [1000.0, 2000.0])
    # Same gross profit, different asset base -> different profitability
    result = gross_profitability(inputs)
    assert result.iloc[0] > result.iloc[1]


def test_build_feature_availability_delay():
    years = [2019, 2020, 2021]
    inputs = _gp_inputs(years, gross_profits=[100.0, 150.0, 400.0], total_assets=[1000.0, 1000.0, 1000.0])
    feat = build_gross_profitability_feature(inputs, DAYS, filing_lag_days=90)
    # 2021-09-30 + 90 days ~= 2021-12-29; the big jump to 0.40 must not be
    # visible before that, and must be visible after.
    before_filing = feat.loc[:"2021-12-28"]
    after_filing = feat.loc["2021-12-30":]
    assert (before_filing.dropna() != 0.40).all()
    assert (after_filing.dropna() == 0.40).any()


def test_build_feature_no_lookahead():
    years = [2019, 2020, 2021]
    inputs = _gp_inputs(years, gross_profits=[100.0, 150.0, 900.0], total_assets=[1000.0, 1000.0, 1000.0])
    full = build_gross_profitability_feature(inputs, DAYS)
    truncated = build_gross_profitability_feature(inputs.iloc[:2], DAYS)
    cutoff = pd.Timestamp("2021-06-01")
    pd.testing.assert_series_equal(full.loc[:cutoff], truncated.loc[:cutoff], check_names=False)


def test_build_feature_carries_forward_between_filings():
    years = [2019, 2020]
    inputs = _gp_inputs(years, gross_profits=[100.0, 150.0], total_assets=[1000.0, 1000.0])
    feat = build_gross_profitability_feature(inputs, DAYS, filing_lag_days=90)
    mid_year = feat.loc["2020-06-01"]
    assert np.isclose(mid_year, 0.10)  # still the 2019 figure, carried forward


def test_empty_inputs_all_nan():
    feat = build_gross_profitability_feature(
        pd.DataFrame(columns=["fiscal_year_end", "gross_profit", "total_assets"]), DAYS
    )
    assert feat.isna().all()


def test_nan_before_first_filing():
    inputs = _gp_inputs([2023], [200.0], [1000.0])
    feat = build_gross_profitability_feature(inputs, DAYS, filing_lag_days=90)
    assert feat.loc[:"2023-01-01"].isna().all()
