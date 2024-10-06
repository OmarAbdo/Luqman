import yfinance as yf
from datetime import datetime, timedelta


def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    historical_data = stock.history(period="max", interval="1d")
    # convert historical_data to a list of dictionaries
    return historical_data.reset_index().to_dict("records")
