# scripts/features/technical_indicators.py
import pandas as pd


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


if __name__ == "__main__":
    # Example usage
    pd.set_option("display.max_columns", None)  # Display all columns
    data = pd.read_csv(
        "app/ml/data/AAPL/AAPL_1d_5y.csv", index_col="Date", parse_dates=True
    )
    technical_indicators = TechnicalIndicators(data)
    data_with_indicators = technical_indicators.add_indicators()
    print(data_with_indicators.tail())
