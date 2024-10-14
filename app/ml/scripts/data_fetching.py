# scripts/data_fetching.py
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker, period="60d", interval="1h"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    return data


def save_data_to_csv(data, filename):
    data.to_csv(f"data/{filename}.csv", index=True)


if __name__ == "__main__":
    data = fetch_stock_data("AAPL")
    save_data_to_csv(data, "aapl_hourly_data")
