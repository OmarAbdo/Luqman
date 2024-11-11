# scripts/data_fetch.py
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class StockPrice:
    """
    !WARNING: THIS IMPLEMENTATION IS DEPRECATED
    
    A class responsible for fetching stock, competitor, and market index data
    for LSTM model training and technical analysis, following SOLID principles.

    The StockPrice class is currently designed to work with yfinance, which is known to have limitations in terms of data availability, especially for short-term, high-frequency data. Despite its limitations, we chose yfinance because it provides easy access to historical financial data with minimal setup, which allows us to rapidly iterate during the early stages of development. This decision enables us to focus on prototyping the core functionalities, while acknowledging that yfinance may be replaced in the future by a more reliable data provider.

    Our approach includes a "3D analysis," which refers to the combination of multiple time intervals (15 minutes, 1 hour, and daily) for capturing short-term and longer-term market trends. This method is intended to provide a comprehensive perspective for short-term, low-risk trading. The goal is to understand not only the price movement of individual stocks but also how they behave in the context of their sector and broader market indices.

    During this phase, we are using yfinance's existing options for data retrieval. For the 15-minute interval, the API supports a maximum period of 1 month, which is a limitation for the desired 60-day data. In the future, we plan to implement a workaround that involves fetching the current 1-month data and combining it with data from the previous month, effectively constructing a 2-month dataset.

    Additionally, the long-term plan involves building a more scalable architecture, including replacing yfinance with multiple data sources and building adapters to handle those APIs, aligning with clean architecture principles. These adapters will help convert various API responses to our internal domain models, providing flexibility and reliability. This approach will also ensure that the system can handle changes in data providers seamlessly, allowing for better fault tolerance and reduced downtime.

    The unit tests under app/ml/tests ensure the reliability of our data fetching, storage, and handling. These tests also serve as documentation of expected behavior and provide confidence in future refactoring efforts.
    """

    FETCH_INTERVAL = 15  # Frequency to fetch data (in minutes)
    TECH_SECTOR_ETF = "XLK"  # Technology Select Sector SPDR Fund
    TARGET_TICKER = os.getenv("TICKER")  # Main target ticker for analysis
    COMPETITORS = ["MSFT", "GOOG", "AMZN"]  # Competitor tickers for analysis
    MARKET_INDEXES = ["^GSPC", "^IXIC"]  # Major market index tickers
    HOURLY_LIMIT_PERIOD = "2y"  # Limit for hourly data fetching (maximum period)
    # [TODO] Period '60d' is invalid, must be one of ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
    # [TODO] however, the api can fetch data for 60 days, even though there's no '60d' or 2mo options
    # [TODO] so we need to find a work around to fetch the current 1mo and then the data of the previous month to get the 2mo data
    MINUTE_LIMIT_PERIOD = "1mo"  # Limit for 15-minute data fetching (maximum period)

    def __init__(self):
        self.base_data_path = os.path.join("app/ml/data/", self.TARGET_TICKER, "stock")
        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """
        Ensure the necessary directories exist for storing data.
        """
        if not os.path.exists(self.base_data_path):
            os.makedirs(self.base_data_path)

    def fetch_stock_data(self, ticker, period="max", interval="1d"):
        """
        Fetches data for a given ticker symbol, at a specified interval.
        """
        stock = yf.Ticker(ticker)
        try:
            data = stock.history(period=period, interval=interval)
            print(f"Fetched data for {ticker} - Interval: {interval}, Period: {period}")
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def save_data_to_csv(self, data, filename):
        """
        Saves the fetched data to a CSV file.
        """
        if not data.empty:
            file_path = os.path.join(self.base_data_path, f"{filename}.csv")
            data.to_csv(file_path, index=True)
            print(f"Saved {file_path}")
        else:
            print(f"No data available to save for {filename}")

    def fetch_target_stock_data(self):
        """
        Fetches data for the main target stock (SAP) for different timeframes.
        """
        intervals = [
            ("15m", self.MINUTE_LIMIT_PERIOD),  # 15-minute interval for last 1 month
            ("1h", self.HOURLY_LIMIT_PERIOD),  # Hourly interval for last 2 years
            ("1d", "5y"),  # Daily interval for 5 years
        ]

        for interval, period in intervals:
            print(f"Fetching {interval} data for {self.TARGET_TICKER}...")
            data = self.fetch_stock_data(
                self.TARGET_TICKER, period=period, interval=interval
            )
            self.save_data_to_csv(data, f"{self.TARGET_TICKER}_{interval}_{period}")
            time.sleep(1)  # Avoid rate limiting

    def fetch_sector_and_market_data(self):
        """
        Fetches data for competitors, sector ETFs, and market indices.
        """
        all_tickers = self.COMPETITORS + [self.TECH_SECTOR_ETF] + self.MARKET_INDEXES
        for ticker in all_tickers:
            print(f"Fetching daily data for {ticker}...")
            data = self.fetch_stock_data(ticker, period="5y", interval="1d")
            self.save_data_to_csv(data, f"{ticker}_5y_1d")
            time.sleep(1)  # Avoid rate limiting

    def fetch_all_data(self):
        """
        Fetches all required data for analysis and stores it as CSV files.
        """
        self.fetch_target_stock_data()
        self.fetch_sector_and_market_data()


if __name__ == "__main__":
    data_fetcher = StockPrice()
    data_fetcher.fetch_all_data()
