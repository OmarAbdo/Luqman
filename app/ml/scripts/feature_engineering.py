# scripts/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineer:
    """
    A class to handle feature engineering for stock data, including adding technical indicators,
    normalizing data, preparing sequences for LSTM models, and handling outliers.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_technical_indicators(self):
        """
        Add technical indicators to the stock data, such as Moving Average, RSI, and Bollinger Bands.
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

        return self

    def handle_outliers(self, z_threshold=3):
        """
        Handle outliers by capping values beyond a certain Z-score threshold.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            z_scores = (self.data[col] - mean) / std
            self.data[col] = np.where(
                z_scores > z_threshold,
                mean + z_threshold * std,
                np.where(
                    z_scores < -z_threshold, mean - z_threshold * std, self.data[col]
                ),
            )
        return self

    def handle_missing_values(self):
        """
        Handle missing values by forward-filling.
        """
        self.data.fillna(method="ffill", inplace=True)
        self.data.fillna(method="bfill", inplace=True)
        return self

    def normalize_data(self):
        """
        Normalize the data using Min-Max scaling.
        """
        columns_to_scale = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "MA_10",
            "MA_50",
            "RSI",
            "BB_MA",
            "BB_upper",
            "BB_lower",
            "MACD",
            "Signal_Line",
        ]
        self.data[columns_to_scale] = self.scaler.fit_transform(
            self.data[columns_to_scale]
        )
        return self

    def create_sequences(self, sequence_length=60):
        """
        Create sequences of data for LSTM training.
        """
        sequences = []
        targets = []
        data_array = self.data.values

        for i in range(len(data_array) - sequence_length):
            sequences.append(data_array[i : i + sequence_length])
            targets.append(
                data_array[i + sequence_length, 3]
            )  # Assuming 'Close' is the 4th column

        return np.array(sequences), np.array(targets)


if __name__ == "__main__":
    # Example usage
    data = pd.read_csv(
        "app/ml/data/AAPL/AAPL_1d_5y.csv", index_col="Date", parse_dates=True
    )
    feature_engineer = FeatureEngineer(data)
    feature_engineer.add_technical_indicators().handle_outliers().handle_missing_values().normalize_data()
    sequences, targets = feature_engineer.create_sequences()
    print(f"Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
