"""Daily Fama-French factor returns, used to residualize raw signals.

Why this module exists: several classic signals fail as raw constructions
because they carry unintended exposure to well-known factors. Two of this
project's own rejections were traced to exactly that -- trailing earnings
yield turned out to be an inverted momentum bet, and gross profitability was
distorted by the balance-sheet structure of financials. The documented fix is
to regress a return series on known factors and work with the residual, which
is what `features/residual.py` does with the data fetched here.

Source: the Kenneth R. French data library (Dartmouth), which publishes the
canonical factor series as free zipped CSVs. We use the daily 3-factor file
(Mkt-RF, SMB, HML, RF).

Availability/leakage note: these are published factor returns for a given
trading day, computed from that day's cross-section. A residual for day t is
therefore only knowable after t's close, exactly like the raw return it comes
from -- so it is used the same way (as a feature describing the past, never
shifted forward). No look-ahead is introduced by the factors themselves.
"""

from __future__ import annotations

import io
import zipfile

import pandas as pd

FF3_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

FACTOR_COLUMNS = ["mkt_rf", "smb", "hml"]


def parse_ff_daily_csv(text: str) -> pd.DataFrame:
    """Parse the French library's daily 3-factor CSV.

    The file has a prose header of variable length, then a block of rows keyed
    by YYYYMMDD, then sometimes trailing copyright/annual sections. We locate
    the data block by scanning for the first line whose first field is an
    8-digit date and stopping at the first line that is not, which is robust to
    the header length changing between vintages.

    Values are published in percent; we convert to decimal returns so they are
    directly comparable to the log returns used everywhere else in the package.
    """
    rows = []
    started = False
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        is_data = len(fields) >= 5 and len(fields[0]) == 8 and fields[0].isdigit()
        if is_data:
            started = True
            rows.append(fields[:5])
        elif started:
            break  # end of the daily block; ignore any trailing sections

    if not rows:
        raise RuntimeError("No daily factor rows found in Fama-French CSV")

    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "smb", "hml", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float)
    return df / 100.0  # published as percent


def fetch_ff_daily_factors() -> pd.DataFrame:
    """Download and parse the daily Fama-French 3-factor series.

    Returns a frame indexed by date with columns mkt_rf, smb, hml, rf as
    decimal daily returns. Fails loudly rather than returning a partial frame.
    """
    import requests

    response = requests.get(FF3_DAILY_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("latin-1")

    return parse_ff_daily_csv(text)
