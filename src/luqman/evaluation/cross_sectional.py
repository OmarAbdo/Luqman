from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrossSectionalResult:
    factor: str
    n_tickers: int
    n_days: int
    mean_daily_long_short_return: float
    annualized_long_short_return: float
    annualized_sharpe: float
    pct_days_positive: float
    ic_mean: float  # cross-sectional Spearman IC, averaged over days
    ic_positive_rate: float

    @property
    def verdict(self) -> str:
        # A genuine cross-sectional factor should show a positive mean spread
        # with reasonable consistency (not just a few lucky days) and a
        # positive average IC. This is a real, standard factor-research bar,
        # not a coin-flip threshold.
        return (
            "ADMITTED"
            if (self.annualized_sharpe > 0.3 and self.ic_positive_rate > 0.52 and self.ic_mean > 0.01)
            else "REJECTED"
        )

    def summary(self) -> str:
        return (
            f"=== {self.factor} (cross-sectional, {self.n_tickers} tickers, {self.n_days} days) — {self.verdict} ===\n"
            f"mean daily long-short return: {self.mean_daily_long_short_return:+.5f}\n"
            f"annualized long-short return: {self.annualized_long_short_return:+.4f}\n"
            f"annualized Sharpe: {self.annualized_sharpe:+.3f}\n"
            f"pct days long-short return positive: {self.pct_days_positive:.3f}\n"
            f"mean cross-sectional IC: {self.ic_mean:+.4f}, IC positive rate: {self.ic_positive_rate:.3f}"
        )


def evaluate_cross_sectional_factor(
    factor_panel: pd.DataFrame,
    forward_return_panel: pd.DataFrame,
    factor_name: str,
    quantile: float = 0.2,
    min_names_per_day: int = 10,
) -> CrossSectionalResult:
    """Standard long-short decile-style factor test: each day, rank all
    tickers by the factor, go long the top `quantile` fraction and short the
    bottom `quantile` fraction (equal-weighted), and measure the next-period
    spread return. This is how momentum/PEAD/relative-strength are actually
    validated in the literature — a materially different (and, per prior
    single-name IC failures, potentially more appropriate) test than a
    single ticker's own time series.

    Both panels must be wide DataFrames: index = date, columns = tickers.
    `forward_return_panel` must already be the forward return at the same
    horizon the factor is meant to predict, computed by the caller.
    """
    common_dates = factor_panel.index.intersection(forward_return_panel.index)
    factor_panel = factor_panel.loc[common_dates]
    forward_return_panel = forward_return_panel.loc[common_dates]

    daily_spread = []
    daily_ic = []
    for date in common_dates:
        factors = factor_panel.loc[date].dropna()
        returns = forward_return_panel.loc[date].reindex(factors.index).dropna()
        factors = factors.reindex(returns.index)
        if len(factors) < min_names_per_day:
            continue

        ic = factors.corr(returns, method="spearman")
        if pd.notna(ic):
            daily_ic.append(ic)

        ranked = factors.sort_values()
        n = len(ranked)
        cut = max(1, int(n * quantile))
        short_names = ranked.index[:cut]
        long_names = ranked.index[-cut:]
        spread = returns.loc[long_names].mean() - returns.loc[short_names].mean()
        if pd.notna(spread):
            daily_spread.append(spread)

    spread_arr = np.array(daily_spread)
    ic_arr = np.array(daily_ic)
    mean_spread = float(np.mean(spread_arr)) if len(spread_arr) else float("nan")
    std_spread = float(np.std(spread_arr, ddof=1)) if len(spread_arr) > 1 else float("nan")
    sharpe = mean_spread / std_spread * np.sqrt(252) if std_spread and std_spread > 0 else float("nan")

    return CrossSectionalResult(
        factor=factor_name,
        n_tickers=factor_panel.shape[1],
        n_days=len(spread_arr),
        mean_daily_long_short_return=mean_spread,
        annualized_long_short_return=mean_spread * 252 if pd.notna(mean_spread) else float("nan"),
        annualized_sharpe=float(sharpe) if pd.notna(sharpe) else float("nan"),
        pct_days_positive=float(np.mean(spread_arr > 0)) if len(spread_arr) else float("nan"),
        ic_mean=float(np.mean(ic_arr)) if len(ic_arr) else float("nan"),
        ic_positive_rate=float(np.mean(ic_arr > 0)) if len(ic_arr) else float("nan"),
    )
