# Refactored version of scripts/features/technical_indicators.py to work with merged 3D stock data
import pandas as pd
import os


class TechnicalIndicators3D:
    """
    A class to calculate and add technical indicators to a 3D dataset.

    This class processes data that has multiple time intervals merged into a single DataFrame,
    allowing for consistent feature extraction and comparison across different time frames.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def add_indicators(self):
        """
        Add technical indicators to the dataset for each interval.
        """
        intervals = ["15min", "1hour", "4hour"]
        for interval in intervals:
            # Moving Averages
            self.data[f"MA_10_{interval}"] = (
                self.data[f"close_{interval}"].rolling(window=10).mean()
            )
            self.data[f"MA_50_{interval}"] = (
                self.data[f"close_{interval}"].rolling(window=50).mean()
            )

            # Relative Strength Index (RSI)
            delta = self.data[f"close_{interval}"].diff(1)
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            self.data[f"RSI_{interval}"] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            self.data[f"BB_MA_{interval}"] = (
                self.data[f"close_{interval}"].rolling(window=20).mean()
            )
            self.data[f"BB_std_{interval}"] = (
                self.data[f"close_{interval}"].rolling(window=20).std()
            )
            self.data[f"BB_upper_{interval}"] = self.data[f"BB_MA_{interval}"] + (
                self.data[f"BB_std_{interval}"] * 2
            )
            self.data[f"BB_lower_{interval}"] = self.data[f"BB_MA_{interval}"] - (
                self.data[f"BB_std_{interval}"] * 2
            )

            # Moving Average Convergence Divergence (MACD)
            self.data[f"EMA_12_{interval}"] = (
                self.data[f"close_{interval}"].ewm(span=12, adjust=False).mean()
            )
            self.data[f"EMA_26_{interval}"] = (
                self.data[f"close_{interval}"].ewm(span=26, adjust=False).mean()
            )
            self.data[f"MACD_{interval}"] = (
                self.data[f"EMA_12_{interval}"] - self.data[f"EMA_26_{interval}"]
            )
            self.data[f"Signal_Line_{interval}"] = (
                self.data[f"MACD_{interval}"].ewm(span=9, adjust=False).mean()
            )

            # Average True Range (ATR)
            high_low = self.data[f"high_{interval}"] - self.data[f"low_{interval}"]
            high_close = (
                self.data[f"high_{interval}"] - self.data[f"close_{interval}"].shift()
            ).abs()
            low_close = (
                self.data[f"low_{interval}"] - self.data[f"close_{interval}"].shift()
            ).abs()
            true_range = high_low.combine(high_close, max).combine(low_close, max)
            self.data[f"ATR_{interval}"] = true_range.rolling(window=14).mean()

            # Stochastic Oscillator
            low_min = self.data[f"low_{interval}"].rolling(window=14).min()
            high_max = self.data[f"high_{interval}"].rolling(window=14).max()
            self.data[f"Stochastic_{interval}"] = 100 * (
                (self.data[f"close_{interval}"] - low_min) / (high_max - low_min)
            )

            # On-Balance Volume (OBV)
            self.data[f"OBV_{interval}"] = (
                self.data[f"volume_{interval}"].where(
                    self.data[f"close_{interval}"].diff() > 0,
                    -self.data[f"volume_{interval}"],
                )
            ).cumsum()

        return self.data

    def save_to_csv(self, ticker):
        """
        Save the dataset with technical indicators to a CSV file.

        :param ticker: The stock ticker symbol.
        """
        output_dir = f"app/ml/data/{ticker}/technical_indicators"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"technical_indicators_merged_3d.csv")
        self.data.to_csv(output_path, index=False)
        print(f"Technical indicators saved to {output_path}")


if __name__ == "__main__":
    ticker = "AAPL"
    file_path = "app/ml/data/AAPL/stock/merged_3d_stock_data.csv"

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, parse_dates=["timestamp"])
        technical_indicators_3d = TechnicalIndicators3D(data)
        data_with_indicators = technical_indicators_3d.add_indicators()
        technical_indicators_3d.save_to_csv(ticker)
    else:
        print(f"No merged data file found for {ticker}")
