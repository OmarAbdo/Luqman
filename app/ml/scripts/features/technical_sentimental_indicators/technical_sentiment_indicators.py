# technical_sentiment_indicators.py

import pandas as pd
import numpy as np
import os
import glob


class TechnicalSentimentIndicators:
    """
    A class to generate technical sentiment indicators based on historical stock price and volume data.
    This class analyzes price movement and trading volume patterns to infer market sentiment.
    """

    def __init__(self, data):
        """
        Initialize with historical stock price data.

        :param data: A DataFrame containing columns 'Close', 'Open', 'High', 'Low', and 'Volume'.
        """
        self.data = data

    def detect_double_top(self):
        """
        Detects the 'Double Top' pattern, indicating a possible bearish reversal.
        """
        high_prices = self.data["High"]
        peaks = []
        for i in range(1, len(high_prices) - 1):
            if (
                high_prices[i] > high_prices[i - 1]
                and high_prices[i] > high_prices[i + 1]
            ):
                peaks.append(i)

        # Check for double peaks with similar height
        double_tops = []
        for i in range(len(peaks) - 1):
            if (
                abs(high_prices[peaks[i]] - high_prices[peaks[i + 1]])
                / high_prices[peaks[i]]
                < 0.02
            ):  # within 2% range
                double_tops.append((peaks[i], peaks[i + 1]))

        return double_tops

    def calculate_sentiment_rsi(self, window=14):
        """
        Calculate the Relative Strength Index (RSI) to identify overbought or oversold conditions.
        RSI > 70 is typically considered overbought, which can lead to bearish sentiment.
        RSI < 30 is considered oversold, often leading to bullish sentiment.
        """
        delta = self.data["Close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, abs(delta), 0)

        avg_gain = pd.Series(gain).rolling(window=window).mean()
        avg_loss = pd.Series(loss).rolling(window=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.data["RSI"] = rsi

        overbought = rsi > 70
        oversold = rsi < 30
        return self.data["RSI"], overbought, oversold

    def calculate_rsi_divergence(self):
        """
        Detect divergence between RSI and price, which can indicate potential trend reversals.
        """
        divergence = [None] * len(self.data)
        rsi = self.data["RSI"]
        close_prices = self.data["Close"]

        for i in range(1, len(rsi) - 1):
            # Bullish divergence: Price makes lower low, RSI makes higher low
            if close_prices[i] < close_prices[i - 1] and rsi[i] > rsi[i - 1]:
                divergence[i] = "Bullish Divergence"
            # Bearish divergence: Price makes higher high, RSI makes lower high
            elif close_prices[i] > close_prices[i - 1] and rsi[i] < rsi[i - 1]:
                divergence[i] = "Bearish Divergence"

        self.data["RSI_Divergence"] = divergence
        return self.data[["RSI", "RSI_Divergence"]]

    def panic_selling_detection(self):
        """
        Detect sudden drops in price accompanied by high volume, indicating panic selling.
        """
        price_change = self.data["Close"].pct_change()
        avg_volume = self.data["Volume"].rolling(window=20).mean()
        panic_selling = (price_change < -0.05) & (
            self.data["Volume"] > avg_volume * 2
        )  # Price drop > 5% and double avg volume
        self.data["Panic_Selling"] = panic_selling
        return self.data[["Close", "Volume", "Panic_Selling"]]

    def buying_spree_detection(self):
        """
        Detect sharp price increases accompanied by high volume, indicating a buying spree.
        """
        price_change = self.data["Close"].pct_change()
        avg_volume = self.data["Volume"].rolling(window=20).mean()
        buying_spree = (price_change > 0.05) & (
            self.data["Volume"] > avg_volume * 2
        )  # Price increase > 5% and double avg volume
        self.data["Buying_Spree"] = buying_spree
        return self.data[["Close", "Volume", "Buying_Spree"]]

    def sentiment_based_volume_analysis(self):
        """
        Analyzes volume surges and drops, attributing these changes to shifts in sentiment.
        """
        avg_volume = self.data["Volume"].rolling(window=20).mean()
        self.data["High_Volume"] = (
            self.data["Volume"] > avg_volume * 1.5
        )  # Volume 1.5 times greater than average
        self.data["Low_Volume"] = (
            self.data["Volume"] < avg_volume * 0.5
        )  # Volume less than half the average

        return self.data[["Volume", "High_Volume", "Low_Volume"]]

    def bearish_engulfing_pattern(self):
        """
        Detect bearish engulfing pattern which might indicate a bearish reversal.
        """
        bearish_engulfing = (
            (
                self.data["Open"] > self.data["Close"].shift(1)
            )  # Today's open is higher than yesterday's close
            & (
                self.data["Close"] < self.data["Open"]
            )  # Today's close is lower than today's open
            & (
                self.data["Close"] < self.data["Open"].shift(1)
            )  # Today's close is lower than yesterday's open
        )
        self.data["Bearish_Engulfing"] = bearish_engulfing
        return self.data[["Open", "Close", "Bearish_Engulfing"]]

    def calculate_sentiment_momentum(self, short_window=12, long_window=26):
        """
        Calculate sentiment momentum to gauge bullish or bearish strength over time.

        :param short_window: The short-term EMA period.
        :param long_window: The long-term EMA period.
        :return: A DataFrame containing sentiment momentum.
        """
        self.data["EMA_Short"] = (
            self.data["Close"].ewm(span=short_window, adjust=False).mean()
        )
        self.data["EMA_Long"] = (
            self.data["Close"].ewm(span=long_window, adjust=False).mean()
        )
        self.data["Sentiment_Momentum"] = self.data["EMA_Short"] - self.data["EMA_Long"]
        return self.data[["EMA_Short", "EMA_Long", "Sentiment_Momentum"]]

    def execute_analysis(self):
        """
        Execute all sentiment-based analysis methods and return an aggregated DataFrame.
        """
        self.calculate_sentiment_rsi()
        self.calculate_rsi_divergence()
        self.panic_selling_detection()
        self.buying_spree_detection()
        self.sentiment_based_volume_analysis()
        self.bearish_engulfing_pattern()
        self.calculate_sentiment_momentum()

        return self.data

    def save_analysis(self, ticker, interval):
        """
        Save the analyzed data to a CSV file.

        :param ticker: The stock ticker symbol.
        :param interval: The time interval (e.g., '15m', '1h', '1d').
        """
        path = (
            f"app/ml/data/{ticker}/technical_sentimental/analyzed_data_{interval}.csv"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.data.to_csv(path, index=True)
        print(f"Analyzed data saved to {path}")


# Example usage
if __name__ == "__main__":
    intervals = ["15m", "1h", "1d"]
    ticker = "AAPL"

    for interval in intervals:
        # Load the corresponding data file for each interval
        file_pattern = f"app/ml/data/{ticker}/stock/{ticker}_{interval}_*.csv"
        file_list = glob.glob(file_pattern)
        if file_list:
            file_path = file_list[0]  # Take the first matching file
            df = pd.read_csv(file_path)
            tsi = TechnicalSentimentIndicators(df)
            analyzed_data = tsi.execute_analysis()
            print(analyzed_data)
            tsi.save_analysis(ticker, interval)
        else:
            print(f"No data file found for interval {interval}")
