import numpy as np
import pandas as pd

from ..data.asof import NY_TZ, effective_trading_day

GROSS_PROFITABILITY_MECHANISM = (
    "Gross profitability (Novy-Marx 2013): gross profit / total assets "
    "predicts returns with power comparable to value, replicated "
    "internationally (19 developed markets 1990-2009; Dimensional, 15 "
    "European markets 1982-2014, ~3.6%/yr premium). A quality/profitability "
    "factor, independent of value or momentum, not a value-trap filter alone."
)

VALUE_QUALITY_MECHANISM = (
    "Value + Quality composite: Section 8a found raw trailing earnings "
    "yield fails because value and momentum are structurally negatively "
    "correlated (Asness/Moskowitz/Pedersen 2013, 'Value and Momentum "
    "Everywhere') -- cheap growth-momentum winners look 'expensive' by E/P "
    "and real distressed names look 'cheap'. Research on out-of-sample "
    "Piotroski F-Score replication (post-2013 data) found the literal "
    "9-criterion checklist has DECAYED and often loses money -- so this is "
    "NOT a checklist score. Instead: within stocks with above-median "
    "earnings yield (the cheap half), rank by gross profitability "
    "(Novy-Marx quality), keeping only cheap AND profitable names as value "
    "candidates -- the documented fix (Alpha Architect, JPM Quant 2022 "
    "Pani & Fabozzi) is a value+quality/momentum blend, not raw value."
)


def gross_profitability(gp_inputs: pd.DataFrame) -> pd.Series:
    """Novy-Marx gross profitability per fiscal year: gross_profit / total_assets."""
    return (gp_inputs["gross_profit"] / gp_inputs["total_assets"]).rename("gross_profitability")


def build_gross_profitability_feature(
    gp_inputs: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    filing_lag_days: int = 90,
) -> pd.Series:
    """Daily gross-profitability feature, available starting the trading day
    after fiscal-year-end + `filing_lag_days` (conservative approximation of
    the standard 10-K filing deadline; FMP's statement endpoints give no
    explicit filing/acceptance date), carried forward until the next annual
    figure.
    """
    if gp_inputs.empty:
        return pd.Series(np.nan, index=daily_index, name="gross_profitability")

    gp_inputs = gp_inputs.sort_values("fiscal_year_end").reset_index(drop=True)
    values = gross_profitability(gp_inputs)

    available_eod = pd.Series(
        [
            pd.Timestamp(f"{(d + pd.Timedelta(days=filing_lag_days)).date()}T23:59")
            .tz_localize(NY_TZ)
            .tz_convert("UTC")
            for d in gp_inputs["fiscal_year_end"]
        ]
    )
    day = effective_trading_day(available_eod, daily_index)
    tagged = pd.DataFrame({"day": day.to_numpy(), "gross_profitability": values.to_numpy()}).dropna(subset=["day"])
    tagged = tagged.sort_values("day").groupby("day").last()

    return tagged["gross_profitability"].reindex(daily_index).ffill().rename("gross_profitability")
