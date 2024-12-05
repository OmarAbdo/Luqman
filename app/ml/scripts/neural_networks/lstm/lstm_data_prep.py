import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class LSTMDataPreparer:
    def __init__(
        self,
        ticker,
        sequence_length=30,
        output_dir="app/ml/data/{ticker}/stock/lstm_ready/",
        sample_rate=0.1,
    ):
        self.ticker = ticker
        self.input_file = (
            f"app/ml/data/{self.ticker}/stock/{self.ticker}_5min_standardized.csv"
        )
        self.sequence_length = sequence_length
        self.output_dir = output_dir.format(ticker=self.ticker)
        self.sample_rate = sample_rate
        self.data = None
        self.X = None
        self.y = None
        self.timestamps = None  # Store timestamps separately
        self.prepare_data()

    def load_data(self):
        # Load the dataset
        self.data = pd.read_csv(self.input_file, low_memory=False)
        # Keep timestamps separately for later use
        self.timestamps = self.data["timestamp"].values
        # Drop the timestamp column from the features
        self.data.drop(columns=["timestamp"], inplace=True)
        return self

    def sample_data(self):
        if self.sample_rate < 1.0:
            self.data = self.data.sample(
                frac=self.sample_rate, random_state=42
            ).reset_index(drop=True)
            self.timestamps = self.timestamps[: len(self.data)]  # Adjust timestamps
        return self

    def prepare_sequences(self):
        data_values = self.data.values
        close_index = list(self.data.columns).index("close")
        X, y = [], []
        for i in range(len(data_values) - self.sequence_length):
            X.append(data_values[i : i + self.sequence_length])
            y.append(data_values[i + self.sequence_length][close_index])
        self.X, self.y = np.array(X), np.array(y)
        self.timestamps = self.timestamps[self.sequence_length :]  # Align timestamps
        return self

    def save_data(self):
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(os.path.join(self.output_dir, "X.npy"), self.X)
        np.save(os.path.join(self.output_dir, "y.npy"), self.y)
        np.save(
            os.path.join(self.output_dir, "timestamps.npy"), self.timestamps
        )  # Save timestamps
        print(f"Sequences saved to {self.output_dir}")

    def split_data(self, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def prepare_data(self):
        self.load_data().sample_data().prepare_sequences().save_data()


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    preparer = LSTMDataPreparer(ticker, sequence_length=780, sample_rate=1.0)
    X_train, X_test, y_train, y_test = preparer.split_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
