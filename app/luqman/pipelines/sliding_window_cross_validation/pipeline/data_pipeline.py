# File: DataPipeline.py

import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineer import FeatureEngineer
from data_scaler import DataScaler
from sequence_preparer import SequencePreparer
from data_saver import DataSaver
from data_splitter import DataSplitter

load_dotenv()

# Description:
# The DataPipeline class orchestrates the entire data processing workflow by
# sequentially executing each of the smaller classes. It ensures that data is loaded,
# cleaned, engineered, split using sliding window cross-validation, scaled, sequenced,
# and saved in the correct order, maintaining the integrity of the time-series data.
# All operations related to other split methods have been removed for exclusivity to sliding window.


class DataPipeline:
    """Main class orchestrating the data processing pipeline."""

    def __init__(
        self,
        ticker: str,
        sequence_length: int = 60,
        sample_rate: float = 1.0,
        n_splits: int = 5,
        max_train_size: int = None,
        max_workers: int = 2,  # Number of parallel processes
    ):
        """
        Initializes the DataPipeline.

        Args:
            ticker (str): Stock ticker symbol.
            sequence_length (int, optional): Number of past time steps for each sequence. Defaults to 60.
            sample_rate (float, optional): Fraction of data to use. Defaults to 1.0 (use all data).
            n_splits (int, optional): Number of splits for sliding window method. Defaults to 5.
            max_train_size (int, optional): Maximum training size for each split in sliding window. Defaults to None.
            max_workers (int, optional): Maximum number of parallel processes. Defaults to 2.
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.max_workers = max_workers

        # Define file paths
        self.input_file = f"app/luqman/data/{self.ticker}/stock/{self.ticker}_5min_technical_sentimental_indicators.csv"
        self.scaler_directory = f"app/luqman/data/{self.ticker}/stock/scalers/"
        self.output_dir = f"app/luqman/data/{self.ticker}/stock/lstm_ready/"

        # Initialize classes
        self.loader = DataLoader(self.input_file)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.splitter = DataSplitter(
            n_splits=self.n_splits, max_train_size=self.max_train_size
        )
        self.scaler = DataScaler(self.scaler_directory)
        self.preparer = SequencePreparer(
            sequence_length=self.sequence_length,
            target_column="close",
            sample_rate=self.sample_rate,
        )
        self.saver = DataSaver(self.output_dir)

    def process_split(self, split_config):
        """
        Processes a single split: scaling, sequencing, and saving.

        Args:
            split_config (dict): Dictionary containing 'split_name', 'train_data', and 'test_data'.
        """
        split_name = split_config["split_name"]
        train_data = split_config["train_data"]
        test_data = split_config["test_data"]

        print(f"\nProcessing split: {split_name}")

        # Step 1: Scale Data
        print(f"[{split_name}] Scaling data...")
        train_data_scaled, test_data_scaled = self.scaler.scale_train_test(
            train_data, test_data, target_column="close"
        )
        # Reattach timestamp to the scaled datasets
        train_data_scaled["timestamp"] = train_data["timestamp"].values
        test_data_scaled["timestamp"] = test_data["timestamp"].values
        print(f"[{split_name}] Data scaling completed.")

        # Step 2: Prepare Sequences
        print(f"[{split_name}] Preparing sequences for LSTM...")
        X_train, y_train = self.preparer.create_sequences(train_data_scaled)
        X_test, y_test = self.preparer.create_sequences(test_data_scaled)
        print(f"[{split_name}] Sequence preparation completed.")

        # Step 3: Extract Timestamps
        print(f"[{split_name}] Extracting timestamps for sequences...")
        timestamps_train = (
            train_data_scaled["timestamp"].iloc[self.sequence_length :].values
        )
        timestamps_test = (
            test_data_scaled["timestamp"].iloc[self.sequence_length :].values
        )
        print(f"[{split_name}] Timestamps extracted.")

        # Step 4: Save Processed Data
        print(f"[{split_name}] Saving processed data...")
        self.saver.save_numpy_arrays(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            timestamps_train=timestamps_train,
            timestamps_test=timestamps_test,
            split_name=split_name,
        )
        print(f"[{split_name}] Data saved successfully.")

    def run(self):
        """Executes the data processing pipeline in parallel for all sliding window splits."""
        # Step 1: Load Data
        print("Loading data...")
        data = self.loader.load_data()
        print(f"Data loaded with shape: {data.shape}")

        # Step 2: Clean Data
        print("Cleaning data...")
        numeric_cols, boolean_cols, categorical_cols, datetime_cols = (
            self.cleaner.identify_columns(data)
        )
        data = self.cleaner.handle_missing_values(
            data, numeric_cols, boolean_cols, categorical_cols
        )
        data = self.cleaner.handle_outliers(data, numeric_cols)
        print("Data cleaning completed.")

        # Step 3: Feature Engineering
        print("Performing feature engineering...")
        data = self.engineer.handle_datetime(data, "timestamp")
        if boolean_cols:
            data = self.engineer.convert_boolean(data, boolean_cols)
        if categorical_cols:
            data = self.engineer.one_hot_encode(data, categorical_cols)
        print("Feature engineering completed.")

        # Step 4: Generate Splits
        print("Generating sliding window train-test splits...")
        splits = self.splitter.split(data)
        print(f"Total splits generated: {len(splits)}")

        # Step 5: Prepare Split Configurations
        split_configs = []
        for idx, (train, test) in enumerate(splits, start=1):
            split_name = f"sliding_window_split_{idx}"
            split_configs.append(
                {"split_name": split_name, "train_data": train, "test_data": test}
            )

        # Step 6: Process Splits in Parallel
        print(f"Processing splits in parallel with max_workers={self.max_workers}...")
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.process_split, split) for split in split_configs
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred during split processing: {e}")

        print(
            "\nAll sliding window data splits have been processed and saved successfully."
        )


if __name__ == "__main__":
    # Example usage:
    # Ensure that the TICKER environment variable is set, or replace os.getenv("TICKER") with a specific ticker string.

    ticker = os.getenv("TICKER")
    if not ticker:
        raise ValueError(
            "TICKER environment variable not set. Please set it in your .env file or environment."
        )

    # Initialize DataPipeline with sliding window parameters
    pipeline = DataPipeline(
        ticker=ticker,
        sequence_length=90,  # Adjust as needed
        sample_rate=1.0,  # Use full dataset; adjust if sampling is needed
        n_splits=6,  # Number of sliding window splits
        max_train_size=None,  # Use all data up to the split point
        max_workers=2,  # Adjust based on your CPU cores
    )
    pipeline.run()
