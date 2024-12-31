# components/technical_indicators.py
import pandas as pd
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class TechnicalIndicators:
    """
    A class to calculate and add technical indicators to the dataset.

    Technical indicators provide insights into price trends, volatility, and momentum, which are crucial for
    short-term trading strategies. This class includes common indicators like Moving Averages, RSI, Bollinger Bands,
    and MACD.
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = None

    def load_data(self):
        """
        Load the stock data for the given ticker.
        """
        file_path = f"app/luqman/data/{self.ticker}/stock/{self.ticker}_5min_raw.csv"
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, parse_dates=True)
            if "timestamp" in data.columns:
                data.set_index("timestamp", inplace=True)
            return data
        else:
            raise FileNotFoundError(f"Data file not found for ticker {self.ticker}")

    def add_indicators(self):
        """
        Add technical indicators to the dataset.
        """
        # Moving Averages
        self.data["MA_10"] = self.data["close"].rolling(window=10).mean()
        self.data["MA_50"] = self.data["close"].rolling(window=50).mean()

        # Relative Strength Index (RSI)
        delta = self.data["close"].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.data["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        self.data["BB_MA"] = self.data["close"].rolling(window=20).mean()
        self.data["BB_std"] = self.data["close"].rolling(window=20).std()
        self.data["BB_upper"] = self.data["BB_MA"] + (self.data["BB_std"] * 2)
        self.data["BB_lower"] = self.data["BB_MA"] - (self.data["BB_std"] * 2)

        # Moving Average Convergence Divergence (MACD)
        self.data["EMA_12"] = self.data["close"].ewm(span=12, adjust=False).mean()
        self.data["EMA_26"] = self.data["close"].ewm(span=26, adjust=False).mean()
        self.data["MACD"] = self.data["EMA_12"] - self.data["EMA_26"]
        self.data["Signal_Line"] = self.data["MACD"].ewm(span=9, adjust=False).mean()

    def save_to_csv(self):
        """
        Save the dataset with technical indicators to a CSV file.
        """
        output_path = f"app/luqman/data/{self.ticker}/stock/{self.ticker}_5min_technical_indicators.csv"
        self.data.to_csv(output_path)
        print(f"Technical indicators saved to {output_path}")

    def run(self):
        """
        Streamline the entire process of loading data, adding indicators, and saving the output.
        """
        self.data = self.load_data()
        self.add_indicators()
        self.save_to_csv()


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    technicalIndicators = TechnicalIndicators(ticker)
    technicalIndicators.run()
