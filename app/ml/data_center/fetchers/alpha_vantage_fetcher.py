# data_center/data_fetchers/alpha_vantage_fetcher.py

import requests


class AlphaVantageFetcher:
    def __init__(self, api_key):
        """
        Initialize the AlphaVantageFetcher class with an API key.

        :param api_key: API key for Alpha Vantage
        """
        self.api_key = api_key

    def fetch_financial_data(self, ticker):
        """
        Fetch financial data for the given ticker symbol from Alpha Vantage.

        :param ticker: Stock ticker symbol
        :return: Financial data in raw format
        """
        try:
            base_url = "https://www.alphavantage.co/query"
            params = {"function": "OVERVIEW", "symbol": ticker, "apikey": self.api_key}
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data for {ticker} from Alpha Vantage: {e}")
            return None
