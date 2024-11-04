# Refactored version of scripts/features/technical_sentiment_indicators.py to work with merged 3D stock data
import pandas as pd
import os
import numpy as np


class TechnicalSentimentIndicators3D:
    """
    A class to generate technical sentiment indicators based on historical stock price and volume data for merged 3D datasets.
    This class analyzes price movement and trading volume patterns to infer market sentiment.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with historical stock price data.

        :param data: A DataFrame containing columns for multiple time intervals merged into one.
        """
        self.data = data

    def add_sentiment_indicators(self):
        """
        Add technical sentiment indicators to the dataset for each interval.
        """
        intervals = ["15min", "1hour", "4hour"]
        for interval in intervals:
            # Detect Double Top Pattern
            self.data[f"Double_Top_{interval}"] = self.detect_double_top(interval)

            # Relative Strength Index (RSI)
            delta = self.data[f"close_{interval}"].diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, abs(delta), 0)
            avg_gain = pd.Series(gain).rolling(window=14).mean()
            avg_loss = pd.Series(loss).rolling(window=14).mean()
            rs = avg_gain / avg_loss
            self.data[f"RSI_{interval}"] = 100 - (100 / (1 + rs))

            # RSI Divergence
            self.data[f"RSI_Divergence_{interval}"] = self.calculate_rsi_divergence(
                interval
            )

            # Moving Average Convergence Divergence (MACD)
            self.data[f"EMA_Fast_{interval}"] = (
                self.data[f"close_{interval}"].ewm(span=12, adjust=False).mean()
            )
            self.data[f"EMA_Slow_{interval}"] = (
                self.data[f"close_{interval}"].ewm(span=26, adjust=False).mean()
            )
            self.data[f"MACD_{interval}"] = (
                self.data[f"EMA_Fast_{interval}"] - self.data[f"EMA_Slow_{interval}"]
            )
            self.data[f"MACD_Signal_{interval}"] = (
                self.data[f"MACD_{interval}"].ewm(span=9, adjust=False).mean()
            )
            self.data[f"MACD_Divergence_{interval}"] = self.calculate_macd_divergence(
                interval
            )

            # Panic Selling Detection
            self.data[f"Panic_Selling_{interval}"] = self.panic_selling_detection(
                interval
            )

            # Buying Spree Detection
            self.data[f"Buying_Spree_{interval}"] = self.buying_spree_detection(
                interval
            )

            # Sentiment-Based Volume Analysis
            avg_volume = self.data[f"volume_{interval}"].rolling(window=20).mean()
            self.data[f"High_Volume_{interval}"] = (
                self.data[f"volume_{interval}"] > avg_volume * 1.5
            )
            self.data[f"Low_Volume_{interval}"] = (
                self.data[f"volume_{interval}"] < avg_volume * 0.5
            )

            # Bearish Engulfing Pattern
            self.data[f"Bearish_Engulfing_{interval}"] = self.bearish_engulfing_pattern(
                interval
            )

            # Sentiment Momentum
            self.data[f"EMA_Short_{interval}"] = (
                self.data[f"close_{interval}"].ewm(span=12, adjust=False).mean()
            )
            self.data[f"EMA_Long_{interval}"] = (
                self.data[f"close_{interval}"].ewm(span=26, adjust=False).mean()
            )
            self.data[f"Sentiment_Momentum_{interval}"] = (
                self.data[f"EMA_Short_{interval}"] - self.data[f"EMA_Long_{interval}"]
            )

        return self.data

    def detect_double_top(self, interval):
        """
        Detects the 'Double Top' pattern for a given interval, indicating a possible bearish reversal.
        """
        high_prices = self.data[f"high_{interval}"]
        peaks = []
        for i in range(1, len(high_prices) - 1):
            if (
                high_prices[i] > high_prices[i - 1]
                and high_prices[i] > high_prices[i + 1]
            ):
                peaks.append(i)

        double_tops = [0] * len(high_prices)
        for i in range(len(peaks) - 1):
            if (
                abs(high_prices[peaks[i]] - high_prices[peaks[i + 1]])
                / high_prices[peaks[i]]
                < 0.02
            ):
                double_tops[peaks[i]] = 1
                double_tops[peaks[i + 1]] = 1

        return pd.Series(double_tops, index=self.data.index)

    def calculate_rsi_divergence(self, interval):
        """
        Detect both regular and hidden divergence between RSI and price for a given interval.
        """
        divergence = [None] * len(self.data)
        rsi = self.data[f"RSI_{interval}"]
        close_prices = self.data[f"close_{interval}"]

        for i in range(1, len(rsi) - 1):
            if close_prices[i] < close_prices[i - 1] and rsi[i] > rsi[i - 1]:
                divergence[i] = "Regular Bullish Divergence"
            elif close_prices[i] > close_prices[i - 1] and rsi[i] < rsi[i - 1]:
                divergence[i] = "Regular Bearish Divergence"

        return pd.Series(divergence, index=self.data.index)

    def calculate_macd_divergence(self, interval):
        """
        Detect both regular and hidden divergence between MACD and price for a given interval.
        """
        divergence = [None] * len(self.data)
        macd = self.data[f"MACD_{interval}"]
        close_prices = self.data[f"close_{interval}"]

        for i in range(1, len(macd) - 1):
            if close_prices[i] < close_prices[i - 1] and macd[i] > macd[i - 1]:
                divergence[i] = "Regular Bullish Divergence"
            elif close_prices[i] > close_prices[i - 1] and macd[i] < macd[i - 1]:
                divergence[i] = "Regular Bearish Divergence"

        return pd.Series(divergence, index=self.data.index)

    def panic_selling_detection(self, interval):
        """
        Detect sudden drops in price accompanied by high volume, indicating panic selling.
        """
        price_change = self.data[f"close_{interval}"].pct_change()
        avg_volume = self.data[f"volume_{interval}"].rolling(window=20).mean()
        panic_selling = (price_change < -0.05) & (
            self.data[f"volume_{interval}"] > avg_volume * 2
        )
        return pd.Series(panic_selling, index=self.data.index)

    def buying_spree_detection(self, interval):
        """
        Detect sharp price increases accompanied by high volume, indicating a buying spree.
        """
        price_change = self.data[f"close_{interval}"].pct_change()
        avg_volume = self.data[f"volume_{interval}"].rolling(window=20).mean()
        buying_spree = (price_change > 0.05) & (
            self.data[f"volume_{interval}"] > avg_volume * 2
        )
        return pd.Series(buying_spree, index=self.data.index)

    def bearish_engulfing_pattern(self, interval):
        """
        Detect bearish engulfing pattern which might indicate a bearish reversal.
        """
        bearish_engulfing = (
            (self.data[f"open_{interval}"] > self.data[f"close_{interval}"].shift(1))
            & (self.data[f"close_{interval}"] < self.data[f"open_{interval}"])
            & (self.data[f"close_{interval}"] < self.data[f"open_{interval}"].shift(1))
        )
        return pd.Series(bearish_engulfing, index=self.data.index)

    def save_to_csv(self, ticker):
        """
        Save the dataset with technical sentiment indicators to a CSV file.

        :param ticker: The stock ticker symbol.
        """
        output_dir = f"app/ml/data/{ticker}/technical_sentimental"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir, f"technical_sentimental_merged_3d.csv"
        )
        self.data.to_csv(output_path, index=False)
        print(f"Technical sentiment indicators saved to {output_path}")


if __name__ == "__main__":
    ticker = "AAPL"
    file_path = "app/ml/data/AAPL/stock/merged_3d_stock_data.csv"

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, parse_dates=["timestamp"])
        technical_sentiment_indicators_3d = TechnicalSentimentIndicators3D(data)
        data_with_sentiment = (
            technical_sentiment_indicators_3d.add_sentiment_indicators()
        )
        technical_sentiment_indicators_3d.save_to_csv(ticker)
    else:
        print(f"No merged data file found for {ticker}")
