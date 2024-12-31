# technical_sentiment_indicators.py

import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class TechnicalSentimentIndicators:
    """
    A class to generate technical sentiment indicators based on historical stock price and Volume data.
    This class analyzes price movement and trading Volume patterns to infer market sentiment.
    """

    def __init__(self, ticker: str):
        """
        Initialize with historical stock price data.

        :param ticker: The stock ticker symbol.
        """
        self.ticker = ticker
        self.data = None
        self.process_data()

    def load_data(self):
        """
        Load the stock data for the given ticker.
        """
        file_path = f"app/ml/data/{self.ticker}/stock/{self.ticker}_5min_technical_indicators.csv"
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, parse_dates=True)
            print(
                f"Loaded columns: {data.columns.tolist()}"
            )  # Debug: Print loaded column names
            if "timestamp" in data.columns:
                data.set_index("timestamp", inplace=True)
            return data
        else:
            raise FileNotFoundError(f"Data file not found for ticker {self.ticker}")

    def calculate_rsi_divergence(self):
        """
        Detect both regular and hidden divergence between RSI and price, which can indicate potential trend reversals or continuations.
        """
        divergence = [None] * len(self.data)
        rsi = self.data["RSI"]
        close_prices = self.data["close"]

        for i in range(1, len(rsi) - 1):
            # Regular Bullish Divergence: Price makes lower low, RSI makes higher low
            if (
                close_prices.iloc[i] < close_prices.iloc[i - 1]
                and rsi.iloc[i] > rsi.iloc[i - 1]
            ):
                divergence[i] = "Regular Bullish Divergence"
            # Regular Bearish Divergence: Price makes higher high, RSI makes lower high
            elif (
                close_prices.iloc[i] > close_prices.iloc[i - 1]
                and rsi.iloc[i] < rsi.iloc[i - 1]
            ):
                divergence[i] = "Regular Bearish Divergence"
            # Hidden Bullish Divergence: Price makes higher low, RSI makes lower low
            elif (
                close_prices.iloc[i] > close_prices.iloc[i - 1]
                and rsi.iloc[i] < rsi.iloc[i - 1]
            ):
                divergence[i] = "Hidden Bullish Divergence"
            # Hidden Bearish Divergence: Price makes lower high, RSI makes higher high
            elif (
                close_prices.iloc[i] < close_prices.iloc[i - 1]
                and rsi.iloc[i] > rsi.iloc[i - 1]
            ):
                divergence[i] = "Hidden Bearish Divergence"

        self.data["rsi_divergence"] = divergence
        return self.data["RSI"], self.data["rsi_divergence"]

    def calculate_macd_divergence(self):
        """
        Detect both regular and hidden divergence between MACD and price, which can indicate potential trend reversals or continuations.
        """
        divergence = [None] * len(self.data)
        macd = self.data["MACD"]
        close_prices = self.data["close"]

        for i in range(1, len(macd) - 1):
            # Regular Bullish Divergence: Price makes lower low, MACD makes higher low
            if (
                close_prices.iloc[i] < close_prices.iloc[i - 1]
                and macd.iloc[i] > macd.iloc[i - 1]
            ):
                divergence[i] = "Regular Bullish Divergence"
            # Regular Bearish Divergence: Price makes higher high, MACD makes lower high
            elif (
                close_prices.iloc[i] > close_prices.iloc[i - 1]
                and macd.iloc[i] < macd.iloc[i - 1]
            ):
                divergence[i] = "Regular Bearish Divergence"
            # Hidden Bullish Divergence: Price makes higher low, MACD makes lower low
            elif (
                close_prices.iloc[i] > close_prices.iloc[i - 1]
                and macd.iloc[i] < macd.iloc[i - 1]
            ):
                divergence[i] = "Hidden Bullish Divergence"
            # Hidden Bearish Divergence: Price makes lower high, MACD makes higher high
            elif (
                close_prices.iloc[i] < close_prices.iloc[i - 1]
                and macd.iloc[i] > macd.iloc[i - 1]
            ):
                divergence[i] = "Hidden Bearish Divergence"

        self.data["macd_divergence"] = divergence
        return self.data["MACD"], self.data["macd_divergence"]

    def panic_selling_detection(self):
        """
        Detect sudden drops in price accompanied by high volume, indicating panic selling.
        """
        price_change = self.data["close"].pct_change()
        avg_volume = self.data["volume"].rolling(window=20).mean()
        panic_selling = (price_change < -0.05) & (
            self.data["volume"] > avg_volume * 2
        )  # Price drop > 5% and double avg volume
        self.data["panic_selling"] = panic_selling
        return self.data[["close", "volume", "panic_selling"]]

    def buying_spree_detection(self):
        """
        Detect sharp price increases accompanied by high volume, indicating a buying spree.
        """
        price_change = self.data["close"].pct_change()
        avg_volume = self.data["volume"].rolling(window=20).mean()
        buying_spree = (price_change > 0.05) & (
            self.data["volume"] > avg_volume * 2
        )  # Price increase > 5% and double avg volume
        self.data["buying_spree"] = buying_spree
        return self.data[["close", "volume", "buying_spree"]]

    def sentiment_based_volume_analysis(self):
        """
        Analyzes volume surges and drops, attributing these changes to shifts in sentiment.
        """
        avg_volume = self.data["volume"].rolling(window=20).mean()
        self.data["high_volume"] = (
            self.data["volume"] > avg_volume * 1.5
        )  # Volume 1.5 times greater than average
        self.data["low_volume"] = (
            self.data["volume"] < avg_volume * 0.5
        )  # Volume less than half the average

        return self.data[["volume", "high_volume", "low_volume"]]

    def bearish_engulfing_pattern(self):
        """
        Detect bearish engulfing pattern which might indicate a bearish reversal.
        """
        bearish_engulfing = (
            (
                self.data["open"] > self.data["close"].shift(1)
            )  # Today's open is higher than yesterday's close
            & (
                self.data["close"] < self.data["open"]
            )  # Today's close is lower than today's open
            & (
                self.data["close"] < self.data["open"].shift(1)
            )  # Today's close is lower than yesterday's open
        )
        self.data["bearish_engulfing"] = bearish_engulfing
        return self.data[["open", "close", "bearish_engulfing"]]

    def calculate_sentiment_momentum(self, short_window=12, long_window=26):
        """
        Calculate sentiment momentum to gauge bullish or bearish strength over time.

        :param short_window: The short-term EMA period.
        :param long_window: The long-term EMA period.
        :return: A DataFrame containing sentiment momentum.
        """
        self.data["ema_short"] = (
            self.data["close"].ewm(span=short_window, adjust=False).mean()
        )
        self.data["ema_long"] = (
            self.data["close"].ewm(span=long_window, adjust=False).mean()
        )
        self.data["sentiment_momentum"] = self.data["ema_short"] - self.data["ema_long"]
        return self.data[["ema_short", "ema_long", "sentiment_momentum"]]

    def process_data(self):
        """
        Streamline the entire process of loading data, performing analysis, and saving the output.
        """
        self.data = self.load_data()
        print("Data loaded successfully.")
        self.execute_analysis()
        self.save_analysis()

    def execute_analysis(self):
        """
        Execute all sentiment-based analysis methods and return an aggregated DataFrame.
        """
        print("Calculating RSI Divergence...")
        self.calculate_rsi_divergence()
        print("RSI Divergence calculated.")

        print("Calculating MACD Divergence...")
        self.calculate_macd_divergence()
        print("MACD Divergence calculated.")

        print("Detecting Panic Selling...")
        self.panic_selling_detection()
        print("Panic Selling detection complete.")

        print("Detecting Buying Spree...")
        self.buying_spree_detection()
        print("Buying Spree detection complete.")

        print("Analyzing Volume...")
        self.sentiment_based_volume_analysis()
        print("Volume analysis complete.")

        print("Detecting Bearish Engulfing Pattern...")
        self.bearish_engulfing_pattern()
        print("Bearish Engulfing Pattern detection complete.")

        print("Calculating Sentiment Momentum...")
        self.calculate_sentiment_momentum()
        print("Sentiment Momentum calculated.")

    def save_analysis(self):
        """
        Save the analyzed data to a CSV file.
        """
        output_path = f"app/ml/data/{self.ticker}/stock/{self.ticker}_5min_technical_sentimental_indicators.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.data.to_csv(output_path, index=True)
        print(f"Analyzed data saved to {output_path}")


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    TechnicalSentimentIndicators(ticker)
