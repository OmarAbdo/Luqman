import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from io import StringIO
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class StockPriceFetcher:
    """
    A class responsible for fetching historical stock data for a specified ticker using Financial Modeling Prep API.

    This class allows users to:
    1. Fetch data for a specific stock ticker for different time intervals (15 minutes, 1 hour, daily).
    2. Fetch historical data in pages, where each page represents a backward time interval until a user-defined limit is reached.
    3. Pass all parameters dynamically, such as ticker, interval, and pagination.
    """

    def __init__(self, ticker, interval, pagination, output_path, api_key):
        """
        Initializes the StockPriceFetcher class.

        Args:
            ticker (str): The stock ticker to fetch data for (e.g., 'SAP').
            interval (str): The interval to fetch data for (e.g., '15min', '1hour', 'daily').
            pagination (int): Number of pages to fetch data for, each representing the maximum possible period for the interval.
            output_path (str): The base path for saving the fetched data.
            api_key (str): The API key for accessing Financial Modeling Prep.
        """
        self.ticker = ticker
        self.interval = interval
        self.pagination = pagination
        self.output_path = output_path
        self.api_key = api_key
        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """
        Ensure the necessary directories exist for storing data.
        """
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def fetch_stock_data(self, start_date, end_date):
        """
        Fetches data for the given ticker at the specified interval using Financial Modeling Prep API, within a date range.

        Args:
            start_date (datetime): The start date for the data fetch.
            end_date (datetime): The end date for the data fetch.

        Returns:
            DataFrame: A DataFrame containing the historical stock data.
        """
        base_url = f"https://financialmodelingprep.com/api/v3/historical-chart/{self.interval}/{self.ticker}"

        params = {
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "apikey": self.api_key,
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = pd.DataFrame(response.json())
            if not data.empty:
                data["timestamp"] = pd.to_datetime(data["date"])
                data.drop(columns=["date"], inplace=True)
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
                print(
                    f"Fetched data for {self.ticker} - Interval: {self.interval}, From: {start_date} To: {end_date}"
                )
                print(
                    f"Columns in the fetched data: {data.columns.tolist()}"
                )  # Print header row for debugging purposes
                return data
        else:
            print(
                f"Error fetching data for {self.ticker}: {response.status_code} - {response.text}"
            )
        return pd.DataFrame()

    def save_data_to_csv(self, data, filename):
        """
        Saves the fetched data to a CSV file.

        Args:
            data (DataFrame): The DataFrame to save.
            filename (str): The name of the file to save the data in.
        """
        if not data.empty:
            file_path = os.path.join(self.output_path, f"{filename}.csv")
            data.to_csv(file_path, index=False)
            print(f"Saved {file_path}")
        else:
            print(f"No data available to save for {filename}")

    def fetch_paginated_data(self):
        """
        Fetches the historical stock data in a paginated manner, merging all fetched pages into a single DataFrame.
        """
        all_data = pd.DataFrame()
        current_end = datetime.now()
        start_date = current_end - timedelta(days=60)  # Start with a 2-month period
        end_date = current_end

        for page in range(self.pagination):
            print(
                f"Fetching page {page + 1} for {self.interval} interval from {start_date} to {end_date}..."
            )

            data = self.fetch_stock_data(start_date, end_date)
            if not data.empty:
                all_data = pd.concat([all_data, data]) if not all_data.empty else data

            # Move back in time to fetch the next 2-month period
            end_date = start_date
            start_date -= timedelta(days=60)

        # Sort the data from oldest to newest
        if not all_data.empty:
            print(
                f"Final columns in merged data: {all_data.columns.tolist()}"
            )  # Print header row for debugging purposes
            all_data.sort_values(by="timestamp", inplace=True)
        self.save_data_to_csv(all_data, f"{self.ticker}_{self.interval}_raw")


# Example usage:
if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    interval = "5min"  # Options: '1min', '5min', '15min', '30min', '1hour', '4hour'
    pagination = 30  # Fetch 30 pages of historical data (approx. 5 years)
    output_path = "app/ml/data/SAP/stock"
    api_key = "hOpyd96KeA4y9jBSh1VV8J7c7g9RzwVH"  # Replace with your actual Financial Modeling Prep API key

    fetcher = StockPriceFetcher(ticker, interval, pagination, output_path, api_key)
    fetcher.fetch_paginated_data()
