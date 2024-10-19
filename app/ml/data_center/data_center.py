# data_center/data_center.py

from data_center.data_fetchers.yahoo_fetcher import YahooFetcher
from data_center.data_fetchers.alpha_vantage_fetcher import AlphaVantageFetcher
from data_center.cache.cache_manager import CacheManager
from data_center.mappers.data_mapper import DataMapper


class DataCenter:
    def __init__(self):
        """
        Initialize the DataCenter class.
        This class manages the entire data flow, including fetching, caching, and mapping.
        """
        self.yahoo_fetcher = YahooFetcher()
        self.alpha_vantage_fetcher = AlphaVantageFetcher(
            api_key="YOUR_ALPHA_VANTAGE_API_KEY"
        )
        self.cache_manager = CacheManager()
        self.data_mapper = DataMapper()

    def get_financial_data(self, ticker, source="yahoo"):
        """
        Get financial data for the given ticker from the specified source.

        :param ticker: Stock ticker symbol
        :param source: Data source ("yahoo" or "alpha_vantage")
        :return: Financial data in standardized format
        """
        # Check cache first
        cached_data = self.cache_manager.load_from_cache(f"{ticker}_{source}.csv")
        if cached_data is not None:
            return cached_data

        # Fetch data from the specified source
        if source == "yahoo":
            raw_data = self.yahoo_fetcher.fetch_financial_data(ticker)
            mapped_data = self.data_mapper.map_yahoo_data(raw_data)
        elif source == "alpha_vantage":
            raw_data = self.alpha_vantage_fetcher.fetch_financial_data(ticker)
            mapped_data = self.data_mapper.map_alpha_vantage_data(raw_data)
        else:
            raise ValueError("Invalid data source")

        # Save mapped data to cache and return it
        self.cache_manager.save_to_cache(mapped_data, f"{ticker}_{source}.csv")
        return mapped_data
