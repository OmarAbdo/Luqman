"""
alpha_vantage_20yr_intraday.py

Fetches up to 20 years of 5-minute intraday data from Alpha Vantage's updated
TIME_SERIES_INTRADAY endpoint using month-by-month queries.

Features:
---------
1) Uses the optional 'month=YYYY-MM' parameter to request older historical data
   in full CSV format for each month.
2) Loops from a user-defined start month (e.g., 2000-01) to an end month (e.g., 2019-12),
   respecting the free-tier rate limit (5 calls/min) by sleeping 15 seconds after each call.
3) Handles Alpha Vantage's varying column names ('time' vs. 'timestamp').
4) Saves the final merged DataFrame to a CSV file.

Limitations:
------------
- Data for each month is updated once per day on the free plan.
- Realtime or 15-min delayed data requires premium.
- The script can easily take an hour or more if you go for multiple decades of data.

Usage Example:
--------------
1) pip install requests pandas python-dotenv
2) Put your ALPHA_VANTAGE_API_KEY in a .env file or as an environment variable.
3) python alpha_vantage_20yr_intraday.py
"""

import os
import time
import requests
import pandas as pd
from io import StringIO
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class AlphaVantageIntradayFetcher:
    """
    Fetches multi-month historical intraday data from Alpha Vantage's new TIME_SERIES_INTRADAY
    endpoint using 'month=YYYY-MM'. Ideal for large multi-year data.

    Key points:
    -----------
    - Sleeps ~15s between calls to avoid free-tier rate limit (5 calls/min).
    - If you have a premium plan, you may reduce or remove the sleep.
    - Adjust 'start_yearmonth' and 'end_yearmonth' for up to 20+ years, e.g. 2000-01 to 2024-01.
    """

    def __init__(
        self,
        symbol: str,
        interval: str,
        adjusted: bool,
        extended_hours: bool,
        start_yearmonth: str,
        end_yearmonth: str,
        output_path: str,
        api_key: str,
    ):
        """
        Args:
            symbol (str): Ticker symbol, e.g. "AAPL"
            interval (str): '1min','5min','15min','30min','60min'
            adjusted (bool): True to adjust for splits/dividends, False for raw data
            extended_hours (bool): True for pre/post market data, False for regular hours only
            start_yearmonth (str): e.g. '2000-01'
            end_yearmonth (str): e.g. '2024-01'
            output_path (str): Directory to save CSV
            api_key (str): Alpha Vantage API key
        """
        self.symbol = symbol
        self.interval = interval
        self.adjusted = adjusted
        self.extended_hours = extended_hours
        self.start_yearmonth = start_yearmonth
        self.end_yearmonth = end_yearmonth
        self.output_path = output_path
        self.api_key = api_key

        # Ensure output directory
        os.makedirs(self.output_path, exist_ok=True)

        # Parse start/end months into datetime objects
        self.start_dt = datetime.strptime(self.start_yearmonth, "%Y-%m")
        self.end_dt = datetime.strptime(self.end_yearmonth, "%Y-%m")

    def _fetch_one_month(self, yearmonth: str) -> pd.DataFrame:
        """
        Fetches one month (YYYY-MM) of intraday data (CSV).

        Args:
            yearmonth (str): e.g. "2009-01"

        Returns:
            pd.DataFrame with columns [timestamp, open, high, low, close, volume]
        """
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": self.symbol,
            "interval": self.interval,
            "month": yearmonth,
            "outputsize": "full",  # Entire month
            "adjusted": str(self.adjusted).lower(),
            "extended_hours": str(self.extended_hours).lower(),
            "datatype": "csv",
            "apikey": self.api_key,
        }

        print(f"[DEBUG] Fetching {self.symbol} {self.interval} for {yearmonth} ...")
        resp = requests.get(base_url, params=params)
        if resp.status_code != 200:
            print(f"[ERROR] HTTP {resp.status_code}: {resp.text}")
            return pd.DataFrame()

        data_str = resp.text
        if (
            "Error Message" in data_str
            or "Thank you for using Alpha Vantage" in data_str
        ):
            print(
                f"[WARNING] Possibly no data or usage limit issue for {yearmonth}. Skipping."
            )
            return pd.DataFrame()

        # Parse CSV
        df = pd.read_csv(StringIO(data_str))
        # Typically columns might be: "time"/"timestamp", "open", "high", "low", "close", "volume"

        # Make columns lowercase to standardize
        df.columns = [c.lower() for c in df.columns]

        # Accept either "time" or "timestamp"
        if "time" in df.columns:
            df.rename(columns={"time": "timestamp"}, inplace=True)
        elif "timestamp" not in df.columns:
            print(
                f"[WARNING] No 'time'/'timestamp' column in {yearmonth}. Columns: {list(df.columns)}."
            )
            return pd.DataFrame()

        # Convert to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if df["timestamp"].isna().all():
            print(f"[WARNING] 'timestamp' invalid in {yearmonth}, skipping.")
            return pd.DataFrame()

        # Sort ascending by time
        df.sort_values(by="timestamp", inplace=True)
        print(f"[DEBUG] Got {len(df)} rows for {yearmonth}.")

        return df

    def fetch_all_months(self) -> pd.DataFrame:
        """
        Fetches each month from start_dt to end_dt inclusive.

        Returns:
            Merged DataFrame, sorted by timestamp ascending.
        """
        all_data = pd.DataFrame()

        current_dt = self.start_dt
        while current_dt <= self.end_dt:
            ym_str = current_dt.strftime("%Y-%m")

            # Fetch that month
            df_month = self._fetch_one_month(ym_str)
            if not df_month.empty:
                if all_data.empty:
                    all_data = df_month
                else:
                    all_data = pd.concat([all_data, df_month], ignore_index=True)

            # Sleep to respect free-tier 5 calls/min
            time.sleep(15)

            # Increment 1 month
            year = current_dt.year
            month = current_dt.month
            if month == 12:
                year += 1
                month = 1
            else:
                month += 1
            current_dt = datetime(year, month, 1)

        # Final sort and deduplicate
        if not all_data.empty:
            all_data.sort_values(by="timestamp", inplace=True)
            all_data.drop_duplicates(subset=["timestamp"], inplace=True)

        return all_data

    def save_to_csv(self, df: pd.DataFrame, filename: str):
        """
        Saves DataFrame to CSV in output_path.

        Args:
            df (pd.DataFrame): Data to save
            filename (str): Base CSV filename (no extension)
        """
        if df.empty:
            print(f"[INFO] No data to save for {filename}.")
            return
        out_file = os.path.join(self.output_path, f"{filename}.csv")
        df.to_csv(out_file, index=False)
        print(f"[INFO] Saved {len(df)} rows to {out_file}")


if __name__ == "__main__":
    # Load from .env or just replace with your actual API key
    api_key = "VSTSPYFTOA4PTD73"

    # Example usage: 20 years of 5-min data for AAPL, from 2003-01 to 2023-01
    symbol = "AAPL"
    interval = "5min"
    adjusted = True
    extended_hours = True
    start_yearmonth = "2014-07"
    end_yearmonth = "2016-07"
    output_path = "data"

    fetcher = AlphaVantageIntradayFetcher(
        symbol=symbol,
        interval=interval,
        adjusted=adjusted,
        extended_hours=extended_hours,
        start_yearmonth=start_yearmonth,
        end_yearmonth=end_yearmonth,
        output_path=output_path,
        api_key=api_key,
    )

    # Fetch all months in the range
    all_data = fetcher.fetch_all_months()

    # Save to CSV if we got any data
    fetcher.save_to_csv(
        all_data, f"{symbol}_{interval}_{start_yearmonth}_{end_yearmonth}"
    )
