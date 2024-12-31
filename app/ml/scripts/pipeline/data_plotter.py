# File: DataPlotter.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Description:
# The DataPlotter class is designed to visualize the actual vs. predicted close prices.
# It loads the scaler information to inverse transform the scaled data back to its original scale 
# and plots the results using matplotlib and seaborn.

class DataPlotter:
    """Class responsible for plotting de-scaled features."""

    def __init__(self, scaler_directory: str):
        """
        Initializes the DataPlotter.

        Args:
            scaler_directory (str): Path to the directory where scalers are saved.
        """
        self.scaler_directory = scaler_directory
        self.scalers = {}
        self.load_scalers()

    def load_scalers(self):
        """
        Loads the scalers from the scaler directory.
        """
        scaler_info_file = os.path.join(self.scaler_directory, "scaler_info.json")
        if not os.path.exists(scaler_info_file):
            raise FileNotFoundError(
                f"The scaler info file {scaler_info_file} does not exist."
            )

        with open(scaler_info_file, "r") as f:
            scaler_info = json.load(f)

        self.scalers = scaler_info

        # Load feature scaler
        feature_scaler = joblib.load(
            os.path.join(self.scaler_directory, scaler_info["features"]["scaler_file"])
        )
        # Load target scaler
        target_column = [key for key in scaler_info.keys() if key != "features"][0]
        target_scaler = joblib.load(
            os.path.join(
                self.scaler_directory, scaler_info[target_column]["scaler_file"]
            )
        )

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.target_column = target_column

        print("Scalers loaded successfully.")

    def inverse_transform(self, data: np.ndarray):
        """
        Inverse transforms the scaled target data back to original scale.

        Args:
            data (np.ndarray): Scaled target data.

        Returns:
            np.ndarray: De-scaled target data.
        """
        data = data.reshape(-1, 1)
        method = self.scalers[self.target_column]["method"]
        if method == "standard":
            return self.target_scaler.inverse_transform(data).flatten()
        elif method == "minmax":
            return self.target_scaler.inverse_transform(data).flatten()
        elif method == "log_scaling":
            return np.expm1(data).flatten()
        else:
            raise ValueError("Unknown scaling method for target.")

    def plot_close_price(
        self,
        y_test_scaled: np.ndarray,
        predictions_scaled: np.ndarray,
        timestamps: np.ndarray,
    ):
        """
        Plots the actual vs predicted close prices.

        Args:
            y_test_scaled (np.ndarray): Scaled actual close prices.
            predictions_scaled (np.ndarray): Scaled predicted close prices.
            timestamps (np.ndarray): Timestamps corresponding to the test data.
        """
        y_test = self.inverse_transform(y_test_scaled)
        predictions = self.inverse_transform(predictions_scaled)

        # Convert timestamps to datetime objects
        timestamps = pd.to_datetime(timestamps)

        plt.figure(figsize=(15, 8))
        sns.lineplot(
            x=timestamps,
            y=y_test,
            label="Actual Close Price",
            color="blue",
        )
        sns.lineplot(
            x=timestamps,
            y=predictions,
            label="Predicted Close Price",
            color="red",
        )
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.title("Actual vs Predicted Close Price (Line Chart)")
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Example usage:
    # Ensure that you have 'X_test.npy', 'y_test.npy', 'timestamps_test.npy', and a trained model.
    import tensorflow as tf

    # Define paths
    ticker = os.getenv("TICKER")
    if not ticker:
        raise ValueError("TICKER environment variable not set.")

    scaler_directory = f"app/ml/data/{ticker}/stock/scalers/"
    output_dir = f"app/ml/data/{ticker}/stock/lstm_ready/"
    model_path = f"app/ml/models/{ticker}_lstm_model.keras"

    # Load scalers and plotter
    plotter = DataPlotter(scaler_directory=scaler_directory)

    # Load test data
    X_test = np.load(os.path.join(output_dir, "X_test.npy"))
    y_test = np.load(os.path.join(output_dir, "y_test.npy"))
    timestamps_test = np.load(
        os.path.join(output_dir, "timestamps_test.npy"), allow_pickle=True
    )

    # Load the trained model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The model file {model_path} does not exist.")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")

    # Make predictions
    predictions_scaled = model.predict(X_test)
    print("Predictions made successfully.")

    # Plot actual vs predicted close prices
    plotter.plot_close_price(y_test, predictions_scaled, timestamps_test)
