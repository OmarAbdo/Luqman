# File: DataPipeline.py

import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd

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
# cleaned, engineered, split, scaled, sequenced, and saved in the correct order,
# maintaining the integrity of the time-series data without any randomization.


class DataPipeline:
    """Main class orchestrating the data processing pipeline."""

    def __init__(
        self,
        ticker: str,
        sequence_length: int = 60,
        test_size: float = 0.2,
        sample_rate: float = 1.0,
    ):
        """
        Initializes the DataPipeline.

        Args:
            ticker (str): Stock ticker symbol.
            sequence_length (int, optional): Number of past time steps for each sequence. Defaults to 60.
            test_size (float, optional): Proportion of data to be used for testing. Defaults to 0.2.
            sample_rate (float, optional): Fraction of data to use. Defaults to 1.0 (use all data).
        """
        self.ticker = ticker
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.sample_rate = sample_rate

        # Define file paths
        self.input_file = f"app/luqman/data/{self.ticker}/stock/{self.ticker}_5min_technical_sentimental_indicators.csv"
        self.scaler_directory = f"app/luqman/data/{self.ticker}/stock/scalers/"
        self.output_dir = f"app/luqman/data/{self.ticker}/stock/lstm_ready/"

        # Initialize classes
        self.loader = DataLoader(self.input_file)
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.splitter = DataSplitter(test_size=self.test_size)
        self.scaler = DataScaler(self.scaler_directory)
        self.preparer = SequencePreparer(
            sequence_length=self.sequence_length,
            target_column="close",
            sample_rate=self.sample_rate,
        )
        self.saver = DataSaver(self.output_dir)

    def run(self):
        """Executes the data processing pipeline."""
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

        # Step 4: Split Data
        print("Splitting data into training and testing sets...")
        train_data, test_data = self.splitter.split(data)
        print(f"Training data shape: {train_data.shape}")
        print(f"Testing data shape: {test_data.shape}")

        # Step 5: Scale Data
        print("Scaling data...")
        train_data_scaled, test_data_scaled = self.scaler.scale_train_test(
            train_data, test_data, target_column="close"
        )
        # Reattach timestamp to the scaled datasets
        train_data_scaled["timestamp"] = train_data["timestamp"].values
        test_data_scaled["timestamp"] = test_data["timestamp"].values
        print("Data scaling completed.")

        # Step 6: Prepare Sequences
        print("Preparing sequences for LSTM...")
        X_train, y_train = self.preparer.create_sequences(train_data_scaled)
        X_test, y_test = self.preparer.create_sequences(test_data_scaled)
        print("Sequence preparation completed.")

        # Step 7: Extract Timestamps
        print("Extracting timestamps for sequences...")
        # Align timestamps after sequence preparation
        timestamps_train = (
            train_data_scaled["timestamp"].iloc[self.sequence_length :].values
        )
        timestamps_test = (
            test_data_scaled["timestamp"].iloc[self.sequence_length :].values
        )
        print(f"Timestamps for training sequences: {timestamps_train.shape}")
        print(f"Timestamps for testing sequences: {timestamps_test.shape}")

        # Step 8: Save Processed Data
        print("Saving processed data...")
        self.saver.save_numpy_arrays(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            timestamps_train=timestamps_train,
            timestamps_test=timestamps_test,
        )
        print("Data processing pipeline completed successfully.")


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    if not ticker:
        raise ValueError("TICKER environment variable not set.")

    pipeline = DataPipeline(
        ticker=ticker,
        sequence_length=90,  # default is 60
        test_size=0.2,  # 80% train, 20% test
        sample_rate=1,  # Use full dataset; adjust if sampling is needed
    )
    pipeline.run()
