# File: DataLoader.py

import pandas as pd
import os


# The DataLoader class is responsible for loading the raw CSV data into a pandas DataFrame.
# It ensures that the data is sorted chronologically based on the timestamp column and resets the index for consistency.
class DataLoader:
    """Class responsible for loading data."""

    def __init__(self, input_file: str):
        """
        Initializes the DataLoader with the path to the input CSV file.

        Args:
            input_file (str): Path to the CSV file containing the data.
        """
        self.input_file = input_file

    def load_data(self) -> pd.DataFrame:
        """
        Loads the CSV file into a pandas DataFrame, sorts it chronologically,
        and resets the index.

        Returns:
            pd.DataFrame: The loaded and sorted DataFrame.

        Raises:
            FileNotFoundError: If the input file does not exist.
        """
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"The file {self.input_file} does not exist.")

        data = pd.read_csv(self.input_file, low_memory=False, parse_dates=["timestamp"])
        data.sort_values("timestamp", inplace=True)
        data.reset_index(drop=True, inplace=True)
        print(f"Data loaded successfully with shape: {data.shape}")
        return data
