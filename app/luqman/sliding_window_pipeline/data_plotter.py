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
        y_test_scaled: np.ndarray = None,
        predictions_scaled: np.ndarray = None,
        timestamps: np.ndarray = None,
        # New optional arguments:
        actual_unscaled: np.ndarray = None,
        preds_unscaled: np.ndarray = None,
        title: str = "Actual vs Predicted Close Price (Line Chart)",
    ):
        """
        Plots the actual vs predicted close prices, supporting both scaled and unscaled data.
        If only predictions are provided, it plots only the predictions.

        Args:
            y_test_scaled (np.ndarray, optional): Scaled actual close prices.
            predictions_scaled (np.ndarray, optional): Scaled predicted close prices.
            timestamps (np.ndarray, optional): Timestamps corresponding to the test data.
            actual_unscaled (np.ndarray, optional): Unscaled actual close prices.
            preds_unscaled (np.ndarray, optional): Unscaled predicted close prices.
            title (str, optional): Plot title. Defaults to "Actual vs Predicted Close Price (Line Chart)".
        """
        if timestamps is None:
            raise ValueError("Timestamps are required to plot the data.")

        # Determine actual values
        if actual_unscaled is not None:
            actual = actual_unscaled
        elif y_test_scaled is not None:
            actual = self.inverse_transform(y_test_scaled)
        else:
            actual = None

        # Determine predicted values
        if preds_unscaled is not None:
            preds = preds_unscaled
        elif predictions_scaled is not None:
            preds = self.inverse_transform(predictions_scaled)
        else:
            preds = None

        # Initialize the plot
        plt.figure(figsize=(15, 8))

        # Plot actual vs predicted if both are available
        if actual is not None and preds is not None:
            sns.lineplot(
                x=timestamps, y=actual, label="Actual Close Price", color="blue"
            )
            sns.lineplot(
                x=timestamps, y=preds, label="Predicted Close Price", color="red"
            )
        elif preds is not None:
            # Plot only predictions
            sns.lineplot(
                x=timestamps, y=preds, label="Predicted Close Price", color="red"
            )
        elif actual is not None:
            # Plot only actuals (less common in your use-case)
            sns.lineplot(
                x=timestamps, y=actual, label="Actual Close Price", color="blue"
            )
        else:
            raise ValueError("No data provided for plotting.")

        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
