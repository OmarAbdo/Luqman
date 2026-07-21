"""Structural validation for daily price data.

Ingest is gated on these rules -- monotonic unique dates, positive prices,
coherent OHLC relationships, non-negative volume -- plus a business-day gap
audit that surfaces missing sessions.

The rationale is that the original system silently tolerated malformed data and
the resulting problems were only discovered much later, as inexplicable model
behaviour. Validation failures here stop the pipeline instead.
"""

from dataclasses import dataclass, field

import pandas as pd

from .fetcher import OHLCV_COLUMNS

MAX_PLAUSIBLE_GAP_DAYS = 10


@dataclass
class ValidationReport:
    n_rows: int = 0
    issues: list[str] = field(default_factory=list)
    missing_business_days: int = 0
    largest_gap_days: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def summary(self) -> str:
        status = "OK" if self.ok else "FAILED"
        lines = [
            f"Validation {status}: {self.n_rows} rows, "
            f"{self.missing_business_days} missing business days, "
            f"largest gap {self.largest_gap_days} calendar days"
        ]
        lines += [f"  - {issue}" for issue in self.issues]
        return "\n".join(lines)


def validate_daily_ohlcv(df: pd.DataFrame) -> ValidationReport:
    """Structural checks for a canonical daily OHLCV frame. Data errors are caught
    here by rule — never 'fixed' downstream by imputation or clipping."""
    report = ValidationReport(n_rows=len(df))

    if df.empty:
        report.issues.append("frame is empty")
        return report

    missing_cols = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing_cols:
        report.issues.append(f"missing columns: {missing_cols}")
        return report

    if not isinstance(df.index, pd.DatetimeIndex):
        report.issues.append("index is not a DatetimeIndex")
        return report
    if not df.index.is_monotonic_increasing:
        report.issues.append("index is not sorted ascending")
    if df.index.has_duplicates:
        report.issues.append("index has duplicate dates")

    if df[OHLCV_COLUMNS].isna().any().any():
        report.issues.append("NaN values present in OHLCV columns")

    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            report.issues.append(f"non-positive prices in '{col}'")
    if (df["volume"] < 0).any():
        report.issues.append("negative volume")

    # Split/dividend adjustment arithmetic (auto_adjust) can leave sub-cent
    # float noise (~1e-12 scale) between a bar's high/low and its open/close
    # even when they are economically identical; a relative tolerance avoids
    # flagging that noise as a real OHLC violation while still catching
    # genuine coherence errors, which are orders of magnitude larger.
    tol = 1e-9
    body_high = df[["open", "close", "low"]].max(axis=1)
    body_low = df[["open", "close", "high"]].min(axis=1)
    if (df["high"] < body_high * (1 - tol)).any():
        report.issues.append("bars where high < max(open, close, low)")
    if (df["low"] > body_low * (1 + tol)).any():
        report.issues.append("bars where low > min(open, close, high)")

    if len(df) >= 2 and isinstance(df.index, pd.DatetimeIndex):
        bdays = pd.bdate_range(df.index[0], df.index[-1])
        report.missing_business_days = len(bdays.difference(df.index))
        report.largest_gap_days = int(df.index.to_series().diff().dt.days.max())
        if report.largest_gap_days > MAX_PLAUSIBLE_GAP_DAYS:
            report.issues.append(
                f"gap of {report.largest_gap_days} calendar days exceeds "
                f"{MAX_PLAUSIBLE_GAP_DAYS} (silent data hole?)"
            )

    return report
