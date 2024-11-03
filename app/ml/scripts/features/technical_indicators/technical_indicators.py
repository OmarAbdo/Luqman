# scripts/features/technical_indicators.py
import pandas as pd
import os
import glob


class TechnicalIndicators:
    """
    A class to calculate and add technical indicators to the dataset.

    Technical indicators provide insights into price trends, volatility, and momentum, which are crucial for
    short-term trading strategies. This class includes common indicators like Moving Averages, RSI, Bollinger Bands,
    and MACD.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def add_indicators(self):
        """
        Add technical indicators to the dataset.
        """
        # Moving Averages
        self.data["MA_10"] = self.data["Close"].rolling(window=10).mean()
        self.data["MA_50"] = self.data["Close"].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = self.data["Close"].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.data["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        self.data["BB_MA"] = self.data["Close"].rolling(window=20).mean()
        self.data["BB_std"] = self.data["Close"].rolling(window=20).std()
        self.data["BB_upper"] = self.data["BB_MA"] + (self.data["BB_std"] * 2)
        self.data["BB_lower"] = self.data["BB_MA"] - (self.data["BB_std"] * 2)

        # Moving Average Convergence Divergence (MACD)
        self.data["EMA_12"] = self.data["Close"].ewm(span=12, adjust=False).mean()
        self.data["EMA_26"] = self.data["Close"].ewm(span=26, adjust=False).mean()
        self.data["MACD"] = self.data["EMA_12"] - self.data["EMA_26"]
        self.data["Signal_Line"] = self.data["MACD"].ewm(span=9, adjust=False).mean()

        return self.data

    def save_to_csv(self, ticker, interval):
        """
        Save the dataset with technical indicators to a CSV file.

        :param ticker: The stock ticker symbol.
        :param interval: The time interval (e.g., '15m', '1h', '1d').
        """
        output_dir = f"app/ml/data/{ticker}/technical_indicators"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"technical_indicators_{interval}.csv")
        self.data.to_csv(output_path)
        print(f"Technical indicators saved to {output_path}")


if __name__ == "__main__":
    intervals = ["15m", "1h", "1d"]
    ticker = "AAPL"

    for interval in intervals:
        # Load the corresponding data file for each interval
        file_pattern = f"app/ml/data/{ticker}/stock/{ticker}_{interval}_*.csv"
        file_list = glob.glob(file_pattern)
        if file_list:
            file_path = file_list[0]  # Take the first matching file
            data = pd.read_csv(file_path, parse_dates=True)
            if "Date" in data.columns:
                data.set_index("Date", inplace=True)
            technical_indicators = TechnicalIndicators(data)
            data_with_indicators = technical_indicators.add_indicators()
            technical_indicators.save_to_csv(ticker, interval)
        else:
            print(f"No data file found for interval {interval}")
