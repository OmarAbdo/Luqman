# data_center/cache/cache_manager.py

import os
import pandas as pd


class CacheManager:
    def __init__(self, cache_directory="data_center/cache/csv_cache"):
        """
        Initialize the CacheManager class.

        :param cache_directory: Directory path to store cache files
        """
        self.cache_directory = cache_directory
        if not os.path.exists(cache_directory):
            os.makedirs(cache_directory)

    def load_from_cache(self, filename):
        """
        Load data from the CSV cache file if it exists.

        :param filename: Name of the CSV file to load
        :return: DataFrame or None if file doesn't exist
        """
        file_path = os.path.join(self.cache_directory, filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        return None

    def save_to_cache(self, data, filename):
        """
        Save data to a CSV cache file.

        :param data: Data to be saved (DataFrame)
        :param filename: Name of the CSV file
        """
        file_path = os.path.join(self.cache_directory, filename)
        data.to_csv(file_path, index=False)
