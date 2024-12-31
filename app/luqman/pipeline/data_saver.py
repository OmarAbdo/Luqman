# File: DataSaver.py

import numpy as np
import os

# Description:
# The DataSaver class handles the saving of processed data, 
# including the input sequences (X_train, X_test) and their 
# corresponding targets (y_train, y_test), along with their respective timestamps. 
# This ensures that the prepared data is stored systematically for future model training and evaluation.
class DataSaver:
    """Class responsible for saving processed data and scaler information."""

    def __init__(self, output_dir: str):
        """
        Initializes the DataSaver.

        Args:
            output_dir (str): Path to the directory where processed data will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_numpy_arrays(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        timestamps_train: np.ndarray,
        timestamps_test: np.ndarray,
    ):
        """
        Saves the numpy arrays to the output directory.

        Args:
            X_train (np.ndarray): Training input sequences.
            X_test (np.ndarray): Testing input sequences.
            y_train (np.ndarray): Training target values.
            y_test (np.ndarray): Testing target values.
            timestamps_train (np.ndarray): Timestamps corresponding to training sequences.
            timestamps_test (np.ndarray): Timestamps corresponding to testing sequences.
        """
        np.save(os.path.join(self.output_dir, "X_train.npy"), X_train)
        np.save(os.path.join(self.output_dir, "X_test.npy"), X_test)
        np.save(os.path.join(self.output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(self.output_dir, "y_test.npy"), y_test)
        np.save(os.path.join(self.output_dir, "timestamps_train.npy"), timestamps_train)
        np.save(os.path.join(self.output_dir, "timestamps_test.npy"), timestamps_test)
        print(f"Sequences and split data saved to {self.output_dir}")
