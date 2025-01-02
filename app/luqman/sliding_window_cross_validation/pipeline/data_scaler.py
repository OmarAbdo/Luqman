# File: DataScaler.py

import pandas as pd
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# The DataScaler class is responsible for scaling the training and testing data.
# It uses MinMaxScaler for feature columns and StandardScaler for the target column (close).
# The scalers are saved to disk to ensure consistent scaling during model evaluation and prediction.
class DataScaler:
    """Class responsible for scaling data."""

    def __init__(self, scaler_directory: str):
        """
        Initializes the DataScaler with the directory to save scaler files.

        Args:
            scaler_directory (str): Path to the directory where scalers will be saved.
        """
        self.scaler_directory = scaler_directory
        os.makedirs(self.scaler_directory, exist_ok=True)
        self.scalers = {}

    def scale_train_test(
        self, train_data: pd.DataFrame, test_data: pd.DataFrame, target_column: str
    ):
        """
        Scales the training and testing data while preserving non-scalable columns like timestamps.

        - Scales features using MinMaxScaler.
        - Scales the target column using StandardScaler.
        - Ensures non-numeric columns are excluded from the scaling process.

        Args:
            train_data (pd.DataFrame): The training DataFrame.
            test_data (pd.DataFrame): The testing DataFrame.
            target_column (str): The name of the target column to scale.

        Returns:
            tuple: Scaled training and testing DataFrames.
        """
        # Initialize scalers
        feature_scaler = MinMaxScaler()
        target_scaler = StandardScaler()

        # Identify feature columns (exclude target and non-numeric columns)
        feature_columns = train_data.select_dtypes(include=[np.number]).columns.drop(
            target_column
        )

        # Scale features
        train_data_scaled = train_data.copy()
        test_data_scaled = test_data.copy()

        train_data_scaled[feature_columns] = feature_scaler.fit_transform(
            train_data[feature_columns]
        )
        test_data_scaled[feature_columns] = feature_scaler.transform(
            test_data[feature_columns]
        )

        # Scale target column
        train_data_scaled[[target_column]] = target_scaler.fit_transform(
            train_data[[target_column]]
        )
        test_data_scaled[[target_column]] = target_scaler.transform(
            test_data[[target_column]]
        )

        # Save scalers
        joblib.dump(
            feature_scaler,
            os.path.join(self.scaler_directory, "feature_minmax_scaler.pkl"),
        )
        joblib.dump(
            target_scaler,
            os.path.join(self.scaler_directory, "target_standard_scaler.pkl"),
        )

        # Update scaler info
        self.scalers = {
            "features": {
                "method": "minmax",
                "scaler_file": "feature_minmax_scaler.pkl",
            },
            target_column: {
                "method": "standard",
                "scaler_file": "target_standard_scaler.pkl",
            },
        }

        # Save scaler info to JSON
        with open(os.path.join(self.scaler_directory, "scaler_info.json"), "w") as f:
            json.dump(self.scalers, f)

        print("Data scaling completed and scalers saved.")
        return train_data_scaled, test_data_scaled

    def load_scalers(self):
        """
        Loads the scalers from the scaler directory.

        Returns:
            tuple: Loaded feature scaler and target scaler.

        Raises:
            FileNotFoundError: If the scaler info file does not exist.
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

        print("Scalers loaded successfully.")
        return feature_scaler, target_scaler
