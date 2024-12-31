# tests/test_data_fetch.py
import unittest
import pandas as pd
from app.luqman.data_fetch import StockPrice
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class TestStockPrice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_fetcher = StockPrice()
        cls.test_data_path = cls.data_fetcher.base_data_path
        cls.sample_ticker = os.getenv("TICKER")

    def test_ensure_directories_exist(self):
        """
        Test that the data directory is created correctly.
        """
        self.data_fetcher.ensure_directories_exist()
        self.assertTrue(os.path.exists(self.test_data_path))

    def test_fetch_stock_data(self):
        """
        Test fetching stock data for a valid ticker.
        """
        data = self.data_fetcher.fetch_stock_data(
            self.sample_ticker, period="5d", interval="1d"
        )
        self.assertIsInstance(data, pd.DataFrame)
        self.assertFalse(
            data.empty, "Data should not be empty for a valid ticker and period."
        )

    def test_save_data_to_csv(self):
        """
        Test saving fetched data to a CSV file.
        """
        data = self.data_fetcher.fetch_stock_data(
            self.sample_ticker, period="5d", interval="1d"
        )
        test_filename = "test_data"
        self.data_fetcher.save_data_to_csv(data, test_filename)
        expected_file_path = os.path.join(self.test_data_path, f"{test_filename}.csv")
        self.assertTrue(os.path.exists(expected_file_path))
        os.remove(expected_file_path)  # Clean up after the test

    def test_fetch_target_stock_data(self):
        """
        Test fetching target stock data for different intervals.
        """
        self.data_fetcher.fetch_target_stock_data()
        for interval, period in [
            ("15m", self.data_fetcher.MINUTE_LIMIT_PERIOD),
            ("1h", self.data_fetcher.HOURLY_LIMIT_PERIOD),
            ("1d", "5y"),
        ]:
            expected_file_path = os.path.join(
                self.test_data_path, f"{self.sample_ticker}_{interval}_{period}.csv"
            )
            self.assertTrue(os.path.exists(expected_file_path))
            os.remove(expected_file_path)  # Clean up after the test

    def test_fetch_sector_and_market_data(self):
        """
        Test fetching data for competitors, sector ETFs, and market indices.
        """
        self.data_fetcher.fetch_sector_and_market_data()
        tickers = (
            self.data_fetcher.COMPETITORS
            + [self.data_fetcher.TECH_SECTOR_ETF]
            + self.data_fetcher.MARKET_INDEXES
        )
        for ticker in tickers:
            expected_file_path = os.path.join(
                self.test_data_path, f"{ticker}_5y_1d.csv"
            )
            self.assertTrue(os.path.exists(expected_file_path))
            os.remove(expected_file_path)  # Clean up after the test


if __name__ == "__main__":
    unittest.main()
