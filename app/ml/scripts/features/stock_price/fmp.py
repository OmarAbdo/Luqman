import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class StockPriceFetcher:
    """
    A class responsible for fetching historical stock data for a specified ticker
    using Financial Modeling Prep API, in 2-month increments, up to 'pagination' times
    (often ~5 years for 5-minute data).

    Non-Technical Explanation:
    --------------------------
    This class connects to an online financial data provider (Financial Modeling Prep),
    asking for 60 days of data at a time. We move backward in time, chunk by chunk,
    until we've asked for as many 2-month periods as the user wants (pagination).
    Finally, we glue all those chunks together and save them to a file.

    Technical Explanation:
    ----------------------
    - We set `start_date` and `end_date` to define a 2-month (60-day) window.
    - We call FMP's `historical-chart/{interval}/{ticker}` with query params `from=...&to=...`.
    - Because 5-minute data is large, we loop `pagination` times, each time moving
      `end_date` back by 1 day to avoid overlap or missing data in the next chunk.
    - We merge the data frames, sort by ascending timestamp, drop duplicates if needed, and save.
    """

    def __init__(self, ticker, interval, pagination, output_path, api_key):
        """
        Initialize the StockPriceFetcher class.

        Args:
            ticker (str): The stock ticker to fetch data for (e.g., 'AAPL').
            interval (str): The interval to fetch data for (e.g., '5min', '15min', etc.).
            pagination (int): Number of 2-month pages to fetch (e.g., 30 for ~5 years).
            output_path (str): Path for saving CSVs.
            api_key (str): Your Financial Modeling Prep API key.
        """
        self.ticker = ticker
        self.interval = interval
        self.pagination = pagination
        self.output_path = output_path
        self.api_key = api_key

        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """
        Ensures the output directory exists. If not, creates it.

        We need a folder for saving our final CSV. This code checks if the folder
        exists and creates it if it's missing.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def fetch_stock_data(self, start_date, end_date):
        """
        Fetch data for the given ticker at the specified interval using FMP, within a date range.

        This function hits the FMP API, asking for data between 'start_date' and 'end_date'.
        Then it cleans and returns the data as a table.

        Args:
            start_date (datetime): The start date for the data fetch (inclusive).
            end_date (datetime): The end date for the data fetch (inclusive).

        Returns:
            pd.DataFrame: Contains columns: timestamp, open, high, low, close, volume
        """
        base_url = f"https://financialmodelingprep.com/api/v3/historical-chart/{self.interval}/{self.ticker}"
        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "apikey": self.api_key,
        }

        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching data: {response.status_code} - {response.text}")
            return pd.DataFrame()

        # Convert the API response (JSON list of bars) into a DataFrame
        data = pd.DataFrame(response.json())
        if data.empty:
            print(
                f"No data in response for {self.ticker} from {start_date} to {end_date}"
            )
            return pd.DataFrame()

        # Convert 'date' to datetime and rename columns
        data["timestamp"] = pd.to_datetime(data["date"])
        data.drop(columns=["date"], inplace=True, errors="ignore")
        data.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            },
            inplace=True,
        )

        # Sort from oldest to newest so that final data merges in ascending order
        data.sort_values(by="timestamp", inplace=True)

        print(
            f"Fetched {len(data)} rows for {self.ticker}, Interval={self.interval}, "
            f"From={start_date.date()} To={end_date.date()}"
        )
        return data

    def save_data_to_csv(self, data, filename):
        """
        Saves the final (merged) DataFrame to a CSV file.

        We have a table (DataFrame) of all our stock data. This function writes it to a file.

        Args:
            data (pd.DataFrame): The combined historical stock data we want to save.
            filename (str): The base filename (no .csv extension).
        """
        if data.empty:
            print(f"No data to save for {filename}.")
            return

        file_path = os.path.join(self.output_path, f"{filename}.csv")
        data.to_csv(file_path, index=False)
        print(f"Saved merged data to {file_path} (rows={len(data)})")

    def fetch_paginated_data(self):
        """
        Fetches ~5 years of 5-minute data in 2-month blocks, merges all of it into a single DataFrame, and saves it.

        We'll do the same process 30 times (pagination=30 by default), each time grabbing 60 days of data.
        Then we move 60 days further into the past, repeating until we've got around 5 years of data or run out.
        """
        # We'll keep everything in a single DataFrame
        all_data = pd.DataFrame()

        # Start with "today" as the end, and go back 60 days for the first chunk
        end_date = datetime.now()
        start_date = end_date - timedelta(days=60)

        for page in range(1, self.pagination + 1):
            print(f"\n--- Fetching Page {page}/{self.pagination} ---")
            print(f"Requesting data from {start_date.date()} to {end_date.date()} ...")

            chunk_df = self.fetch_stock_data(start_date, end_date)
            if chunk_df.empty:
                print(
                    "No data returned; likely we've reached the earliest available data."
                )
                break

            # Append chunk to our big DataFrame
            if all_data.empty:
                all_data = chunk_df
            else:
                # Concatenate and optionally drop duplicates on 'timestamp' if needed
                all_data = pd.concat([all_data, chunk_df], ignore_index=True)
                # all_data.drop_duplicates(subset="timestamp", inplace=True)  # If you see duplicates

            # Move our 2-month window ~60 days further into the past
            # Shift end_date by 1 day to avoid the boundary from the prior chunk
            end_date = start_date - timedelta(days=1)
            start_date = end_date - timedelta(days=60)

            # Sleep a bit to avoid rate limits (you can remove or adjust as needed)
            time.sleep(1)

        # Sort and optionally drop duplicates
        if not all_data.empty:
            all_data.sort_values(by="timestamp", inplace=True)

        # Save final merged data
        self.save_data_to_csv(all_data, f"{self.ticker}_{self.interval}_raw")


# Example usage:
if __name__ == "__main__":
    # Load from .env if available; otherwise, hardcode for testing
    ticker = os.getenv("TICKER", "AAPL")
    interval = "5min"  # e.g., '1min', '5min', '15min', '30min', '1hour'
    pagination = 30  # 30 chunks x 2 months each = ~5 years
    output_path = f"app/ml/data/{ticker}/stock"
    api_key = "hOpyd96KeA4y9jBSh1VV8J7c7g9RzwVH"  # Replace with your actual key

    fetcher = StockPriceFetcher(ticker, interval, pagination, output_path, api_key)
    fetcher.fetch_paginated_data()
