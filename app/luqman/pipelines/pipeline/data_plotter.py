# File: DataPlotter.py

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
        Expects a 'scaler_info.json' file detailing the scaler files and methods.
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
        if method in ["standard", "minmax"]:
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
        num_points: int = None,
    ):
        """
        Plots the actual vs predicted close prices.

        Args:
            y_test_scaled (np.ndarray): Scaled actual close prices.
            predictions_scaled (np.ndarray): Scaled predicted close prices.
            timestamps (np.ndarray): Timestamps corresponding to the test data.
            num_points (int, optional): Number of points to plot. Defaults to all.
        """
        # Inverse transform
        y_test = self.inverse_transform(y_test_scaled)
        predictions = self.inverse_transform(predictions_scaled)

        # Convert timestamps to datetime objects
        timestamps = pd.to_datetime(timestamps)

        # Determine number of points to plot
        if num_points is None or num_points > len(timestamps):
            num_points = len(timestamps)
        else:
            num_points = num_points

        # Slice the data
        timestamps_plot = timestamps[:num_points]
        y_test_plot = y_test[:num_points]
        predictions_plot = predictions[:num_points]

        plt.figure(figsize=(15, 8))
        sns.lineplot(
            x=timestamps_plot,
            y=y_test_plot,
            label="Actual Close Price",
            color="blue",
        )
        sns.lineplot(
            x=timestamps_plot,
            y=predictions_plot,
            label="Predicted Close Price",
            color="red",
        )
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.title("Actual vs Predicted Close Price (Informer)")
        plt.legend()
        plt.grid(True)
        plt.show()
