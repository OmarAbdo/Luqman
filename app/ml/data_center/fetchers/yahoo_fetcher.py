# data_center/data_fetchers/yahoo_fetcher.py

import yfinance as yf


class YahooFetcher:
    def __init__(self):
        """
        Initialize the YahooFetcher class.
        This class is responsible for fetching data from Yahoo Finance API.
        """
        pass

    def fetch_financial_data(self, ticker):
        """
        Fetch financial data for the given ticker symbol.

        :param ticker: Stock ticker symbol
        :return: Financial data in raw format
        """
        try:
            stock = yf.Ticker(ticker)
            financial_data = {
                "income_statement": stock.financials,
                "balance_sheet": stock.balance_sheet,
                "cash_flow": stock.cashflow,
                "summary": stock.info,
            }
            return financial_data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
