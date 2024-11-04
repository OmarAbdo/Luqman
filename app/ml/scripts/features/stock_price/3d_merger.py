import pandas as pd
import os
from datetime import datetime


class StockDataMerger:
    def __init__(self, file_paths, output_path):
        """
        Initializes the StockDataMerger class.

        Args:
            file_paths (list): List of paths to the CSV files to merge.
            output_path (str): The path to save the merged CSV file.
        """
        self.file_paths = file_paths
        self.output_path = output_path
        self.dataframes = []

    def load_dataframes(self):
        """
        Loads all CSV files into pandas DataFrames.
        """
        for file_path in self.file_paths:
            if os.path.exists(file_path):
                interval = self.get_interval_from_filename(file_path)
                df = pd.read_csv(file_path, parse_dates=["timestamp"])
                df.rename(
                    columns=lambda x: f"{x}_{interval}" if x != "timestamp" else x,
                    inplace=True,
                )
                self.dataframes.append(df)
                print(f"Loaded data from {file_path} with {len(df)} rows.")
            else:
                print(f"File not found: {file_path}")

    def get_interval_from_filename(self, file_path):
        """
        Extracts the interval from the filename.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            str: The interval extracted from the filename (e.g., '15min', '1hour', '4hour').
        """
        return os.path.basename(file_path).split("_")[1]

    def merge_dataframes(self):
        """
        Merges all loaded DataFrames on the 'timestamp' column, considering only overlapping date ranges.

        Returns:
            DataFrame: The merged DataFrame containing all intervals.
        """
        if not self.dataframes:
            raise ValueError("No dataframes loaded. Please load dataframes first.")

        # Determine the common date range across all dataframes
        min_timestamp = max(df["timestamp"].min() for df in self.dataframes)
        max_timestamp = min(df["timestamp"].max() for df in self.dataframes)

        # Filter dataframes to the common date range
        filtered_dataframes = [
            df[(df["timestamp"] >= min_timestamp) & (df["timestamp"] <= max_timestamp)]
            for df in self.dataframes
        ]

        # Merge the filtered dataframes
        merged_df = filtered_dataframes[0]
        for df in filtered_dataframes[1:]:
            merged_df = merged_df.merge(df, on="timestamp", how="outer")

        # Sort by timestamp and fill missing values
        merged_df.sort_values(by="timestamp", inplace=True)
        merged_df.fillna(method="ffill", inplace=True)
        merged_df.fillna(method="bfill", inplace=True)
        print("Successfully merged all dataframes with consistent date range.")
        return merged_df

    def save_merged_dataframe(self, merged_df):
        """
        Saves the merged DataFrame to a CSV file.

        Args:
            merged_df (DataFrame): The DataFrame to save.
        """
        if not merged_df.empty:
            merged_file_path = os.path.join(
                self.output_path, "merged_3d_stock_data.csv"
            )
            merged_df.to_csv(merged_file_path, index=False)
            print(f"Merged data saved to {merged_file_path}")
        else:
            print("Merged DataFrame is empty. Nothing to save.")

    def execute(self):
        """
        Executes the merging process: loading data, merging them, and saving the result.
        """
        self.load_dataframes()
        merged_df = self.merge_dataframes()
        self.save_merged_dataframe(merged_df)


# Example usage
if __name__ == "__main__":
    file_paths = [
        "app/ml/data/AAPL/stock/AAPL_15min_paginated.csv",
        "app/ml/data/AAPL/stock/AAPL_1hour_paginated.csv",
        "app/ml/data/AAPL/stock/AAPL_4hour_paginated.csv",
    ]
    output_path = "app/ml/data/AAPL/stock/"

    merger = StockDataMerger(file_paths, output_path)
    merger.execute()
