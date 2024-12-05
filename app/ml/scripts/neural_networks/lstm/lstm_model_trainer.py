import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Bidirectional,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
import joblib  # To load scalers
import json  # To load scaler info
import pandas as pd  # To handle timestamps
from sklearn.model_selection import train_test_split  # For splitting data

# Load the .env file
load_dotenv()


class LSTMModelTrainer:
    def __init__(self, ticker, sample_fraction=1.0):
        self.ticker = ticker
        self.X_file = f"app/ml/data/{ticker}/stock/lstm_ready/X.npy"
        self.y_file = f"app/ml/data/{ticker}/stock/lstm_ready/y.npy"
        self.timestamps_file = f"app/ml/data/{ticker}/stock/lstm_ready/timestamps.npy"
        self.sample_fraction = sample_fraction

        # Load data
        self.X = np.load(self.X_file)
        self.y = np.load(self.y_file)
        self.timestamps = np.load(self.timestamps_file, allow_pickle=True)

        # Handle sampling if sample_fraction < 1.0
        if self.sample_fraction < 1.0:
            sample_size = int(len(self.X) * self.sample_fraction)
            self.X = self.X[:sample_size]
            self.y = self.y[:sample_size]
            self.timestamps = self.timestamps[:sample_size]

        # Split data (including timestamps)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.timestamps_train,
            self.timestamps_test,
        ) = train_test_split(
            self.X, self.y, self.timestamps, test_size=0.2, random_state=42
        )

        self.handle_missing_values()

        # Load scaler for 'close' price
        self.load_close_scaler()

        self.model = None
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        self.calculate_accuracies()
        self.predict_and_plot()

    def handle_missing_values(self):
        # Replace NaNs in the dataset using mean value
        self.X_train = np.nan_to_num(self.X_train, nan=np.nanmean(self.X_train))
        self.X_test = np.nan_to_num(self.X_test, nan=np.nanmean(self.X_test))

    def load_close_scaler(self):
        """Loads the scaler used for the 'close' price during data standardization."""
        scaler_directory = f"app/ml/data/{self.ticker}/stock/scalers/"
        scaler_info_file = os.path.join(scaler_directory, "scaler_info.json")
        with open(scaler_info_file, "r") as f:
            scalers_info = json.load(f)

        # Get the scaler information for 'close'
        close_scaler_info = scalers_info.get("close")
        if close_scaler_info:
            scaler_file = close_scaler_info["scaler_file"]
            self.close_scaler_method = close_scaler_info["method"]
            self.close_scaler = joblib.load(os.path.join(scaler_directory, scaler_file))
        else:
            raise ValueError("Scaler for 'close' not found.")

    # The commented hyperparameters are (reducing) the quality of the prediction.
    # Probably they could be more useful with future bigger and more complex datasets
    def build_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(
            #Bidirectional(
                LSTM(
                    128,
                    activation="tanh",
                    return_sequences=True,
                    input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                    kernel_regularizer=l2(0.001),
                )
            #)
        )
        # self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(
            # Bidirectional(
                LSTM(
                    64,
                    activation="tanh",
                    return_sequences=False,
                )
            # )
        )
        # self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mse")

    def train_model(self, epochs=1, batch_size=16):
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=1,
        )

    def evaluate_model(self):
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {loss}")

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = f"app/ml/models/{self.ticker}_lstm_model.keras"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def inverse_transform(self, data):
        """Inverse transforms the data using the close price scaler."""
        data = data.reshape(-1, 1)
        if self.close_scaler_method in ["minmax", "standard"]:
            return self.close_scaler.inverse_transform(data).flatten()
        elif self.close_scaler_method == "log_scaling":
            return np.expm1(data).flatten()
        else:
            raise ValueError("Unknown scaling method for 'close'.")

    def calculate_accuracies(self):
        predictions = self.model.predict(self.X_test)

        # Inverse transform predictions and actual values
        y_test_descaled = self.inverse_transform(self.y_test)
        predictions_descaled = self.inverse_transform(predictions)

        mse = np.mean((y_test_descaled - predictions_descaled) ** 2)
        print(f"Mean Squared Error: {mse:.2f}")

        actual_directions = np.diff(y_test_descaled) > 0
        predicted_directions = np.diff(predictions_descaled) > 0

        correct_directions = np.sum(actual_directions == predicted_directions)
        directional_accuracy = correct_directions / len(actual_directions) * 100

        print(f"Directional Accuracy: {directional_accuracy:.2f}%")

    def predict_and_plot(self):
        predictions = self.model.predict(self.X_test)

        # Inverse transform predictions and actual values
        y_test_descaled = self.inverse_transform(self.y_test)
        predictions_descaled = self.inverse_transform(predictions)

        # Convert timestamps to datetime objects
        timestamps = pd.to_datetime(self.timestamps_test)

        plt.figure(figsize=(15, 8))
        sns.lineplot(
            x=timestamps,
            y=y_test_descaled,
            label="Actual Values",
            color="blue",
        )
        sns.lineplot(
            x=timestamps,
            y=predictions_descaled,
            label="Predicted Values",
            color="red",
        )
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.title("Actual vs Predicted Close Price (Line Chart)")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    trainer = LSTMModelTrainer(ticker, sample_fraction=1.0)
