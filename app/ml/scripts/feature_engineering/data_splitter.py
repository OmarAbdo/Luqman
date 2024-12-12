# File: DataSplitter.py

import pandas as pd

# Description:
# The DataSplitter class is responsible for splitting the cleaned and engineered data into 
# training and testing sets based on chronological order. 
# This ensures that the temporal integrity of the data is maintained, preventing any look-ahead bias.
class DataSplitter:
    """Class responsible for splitting data chronologically into training and testing sets."""

    def __init__(self, test_size=0.2):
        """
        Initializes the DataSplitter.

        Args:
            test_size (float, optional): Proportion of data to be used for testing. Defaults to 0.2.
        """
        self.test_size = test_size

    def split(self, data: pd.DataFrame):
        """
        Splits the data into training and testing sets based on chronological order.

        Args:
            data (pd.DataFrame): The DataFrame to split.

        Returns:
            tuple: Tuple containing training and testing DataFrames.
        """
        split_index = int(len(data) * (1 - self.test_size))
        train_data = data.iloc[:split_index].reset_index(drop=True)
        test_data = data.iloc[split_index:].reset_index(drop=True)
        print(
            f"Data split into {train_data.shape[0]} training and {test_data.shape[0]} testing samples."
        )
        return train_data, test_data
