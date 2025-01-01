# File: DataSaver.py

import os
import numpy as np

# Description:
# The DataSaver class is responsible for saving the processed NumPy arrays
# into the specified output directory. It organizes the data based on the split method.


class DataSaver:
    """Class responsible for saving processed data as NumPy arrays."""

    def __init__(self, output_dir: str):
        """
        Initializes the DataSaver.

        Args:
            output_dir (str): Directory where processed data will be saved.
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
        split_name: str = "chronological",
    ):
        """
        Saves the NumPy arrays to disk.

        Args:
            X_train (np.ndarray): Training input sequences.
            X_test (np.ndarray): Testing input sequences.
            y_train (np.ndarray): Training targets.
            y_test (np.ndarray): Testing targets.
            timestamps_train (np.ndarray): Timestamps for training data.
            timestamps_test (np.ndarray): Timestamps for testing data.
            split_name (str, optional): Identifier for the split. Defaults to "chronological".
        """
        # Create a subdirectory for the split
        split_dir = os.path.join(self.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        # Save the arrays
        np.save(os.path.join(split_dir, "X_train.npy"), X_train)
        np.save(os.path.join(split_dir, "X_test.npy"), X_test)
        np.save(os.path.join(split_dir, "y_train.npy"), y_train)
        np.save(os.path.join(split_dir, "y_test.npy"), y_test)
        np.save(os.path.join(split_dir, "timestamps_train.npy"), timestamps_train)
        np.save(os.path.join(split_dir, "timestamps_test.npy"), timestamps_test)

        print(f"Saved processed data for split '{split_name}' in '{split_dir}'.")
