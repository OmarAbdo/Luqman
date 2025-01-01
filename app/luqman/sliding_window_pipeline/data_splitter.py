# File: DataSplitter.py

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

# Description:
# The DataSplitter class is responsible for splitting the cleaned and engineered data into
# training and testing sets using the sliding window cross-validation method.
# This ensures that the temporal integrity of the data is maintained, preventing any look-ahead bias.


class DataSplitter:
    """Class responsible for splitting data using sliding window cross-validation."""

    def __init__(self, n_splits=5, max_train_size=None):
        """
        Initializes the DataSplitter.

        Args:
            n_splits (int, optional): Number of splits for sliding window method. Defaults to 5.
            max_train_size (int, optional): Maximum size for the training set in sliding window.
                                            Defaults to None (use all available data up to the split point).
        """
        self.n_splits = n_splits
        self.max_train_size = max_train_size

    def split(self, data: pd.DataFrame):
        """
        Splits the data into multiple training and testing sets using sliding window cross-validation.

        Args:
            data (pd.DataFrame): The DataFrame to split.

        Returns:
            list of tuples: Each tuple contains (train_data, test_data) for a split.
        """
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits, max_train_size=self.max_train_size
        )
        splits = []
        for fold, (train_index, test_index) in enumerate(tscv.split(data), start=1):
            train_data = data.iloc[train_index].reset_index(drop=True)
            test_data = data.iloc[test_index].reset_index(drop=True)
            print(
                f"Sliding Window Split {fold}: {train_data.shape[0]} training and {test_data.shape[0]} testing samples."
            )
            splits.append((train_data, test_data))
        return splits
