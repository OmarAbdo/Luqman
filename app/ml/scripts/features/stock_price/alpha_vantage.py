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
    A class responsible for fetching historical stock data for a specified ticker using Alpha Vantage API.

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
            interval (str): The interval to fetch data for (e.g., '15min', '60min', 'daily').
            pagination (int): Number of pages to fetch data for, each representing the maximum possible period for the interval.
            output_path (str): The base path for saving the fetched data.
            api_key (str): The API key for accessing Alpha Vantage.
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

    def fetch_stock_data(self, month_param):
        """
        Fetches data for the given ticker at the specified interval using Alpha Vantage API.

        Args:
            month_param (str): The month parameter for fetching data (e.g., '2022-11').

        Returns:
            DataFrame: A DataFrame containing the historical stock data.
        """
        base_url = "https://www.alphavantage.co/query"
        function = "TIME_SERIES_INTRADAY"

        params = {
            "function": function,
            "symbol": self.ticker,
            "interval": self.interval,
            "month": month_param,
            "outputsize": "full",
            "apikey": self.api_key,
            "datatype": "csv",
        }

        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            print(
                f"Fetched data for {self.ticker} - Interval: {self.interval}, Month: {month_param}"
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

        for page in range(self.pagination):
            start_date = self.get_start_date_for_interval(current_end)
            print(
                f"Fetching page {page + 1} for {self.interval} interval from {start_date} to {current_end}..."
            )

            # Fetching data for each month between start_date and current_end
            current_date = current_end
            while current_date >= start_date:
                month_param = current_date.strftime("%Y-%m")
                data = self.fetch_stock_data(month_param)
                if not data.empty:
                    all_data = (
                        pd.concat([all_data, data]) if not all_data.empty else data
                    )
                current_date -= timedelta(days=30)  # Move one month back

            current_end = start_date

        # Sort the data from oldest to newest
        if not all_data.empty:
            print(
                f"Final columns in merged data: {all_data.columns.tolist()}"
            )  # Print header row for debugging purposes
            all_data.sort_values(by=all_data.columns[0], inplace=True)
        self.save_data_to_csv(all_data, f"{self.ticker}_{self.interval}_paginated")

    def get_start_date_for_interval(self, end_date):
        """
        Determines the start date for the next fetch based on the interval and the given end date.

        Args:
            end_date (datetime): The end date for the current fetch.

        Returns:
            datetime: The start date for the next fetch.
        """
        if self.interval == "15min":
            return end_date - timedelta(days=30)
        elif self.interval == "60min":
            return end_date - timedelta(days=730)
        elif self.interval == "daily":
            return end_date - timedelta(days=1825)
        else:
            raise ValueError(f"Unsupported interval: {self.interval}")


# Example usage:
if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    interval = "60min"
    pagination = 1  # Fetch 5 pages of historical data
    output_path = f"app/ml/data/{ticker}/stock"
    api_key = "55I7F5OKXT1668P5"  # Replace with your actual Alpha Vantage API key

    fetcher = StockPriceFetcher(ticker, interval, pagination, output_path, api_key)
    fetcher.fetch_paginated_data()

# Alpha vantage API keys
# AOOE7AD9CPPFTQHH
# 55I7F5OKXT1668P5
# FJ3IBD9KR7HDKVOG
