"""
csv_data_merger.py

Utility class to merge multiple CSV chunk files into a single CSV file.

Usage Example:
--------------
1. Ensure all chunk CSV files are located under app/luqman/data/[ticker]/stock/chunks/.
2. Update the 'ticker' and 'interval' variables in the example script as needed.
3. Run the script:
       python csv_data_merger.py
"""

import os
import pandas as pd
from pathlib import Path
import logging
from typing import List


class CSVDataMerger:
    """
    Utility class to merge multiple CSV chunk files into a single CSV file.

    Attributes:
        ticker (str): The stock ticker symbol, e.g., "AAPL".
        interval (str): The data interval, e.g., "5min".
        base_dir (str): The base directory where data is stored, e.g., "app/luqman/data".
        stock_dir (Path): Path object pointing to the stock's directory.
        chunks_dir (Path): Path object pointing to the 'chunks' directory.
        output_file (Path): Path object pointing to the merged CSV file.
    """

    def __init__(self, ticker: str, interval: str, base_dir: str = "app/luqman/data"):
        """
        Initializes the CSVDataMerger with ticker, interval, and base directory.

        Args:
            ticker (str): The stock ticker symbol, e.g., "AAPL".
            interval (str): The data interval, e.g., "5min".
            base_dir (str, optional): The base directory path. Defaults to "app/luqman/data".
        """
        self.ticker = ticker
        self.interval = interval
        self.base_dir = Path(base_dir)
        self.stock_dir = self.base_dir / self.ticker / "stock"
        self.chunks_dir = self.stock_dir / "chunks"
        self.output_file = self.stock_dir / f"{self.ticker}_{self.interval}_raw.csv"

        # Ensure chunks directory exists
        if not self.chunks_dir.exists():
            raise FileNotFoundError(
                f"Chunks directory does not exist: {self.chunks_dir}"
            )

        # Setup logging
        log_file = self.stock_dir / "merger_log.log"
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logging.info("CSVDataMerger initialized.")

    def merge_chunks(self):
        """
        Merges all CSV files in the chunks directory into a single CSV file.
        """
        csv_pattern = f"{self.ticker}_{self.interval}_*.csv"
        csv_files = list(self.chunks_dir.glob(csv_pattern))
        if not csv_files:
            logging.error(f"No CSV chunk files found in {self.chunks_dir}")
            print(f"[ERROR] No CSV chunk files found in {self.chunks_dir}")
            raise FileNotFoundError(f"No CSV chunk files found in {self.chunks_dir}")

        logging.info(f"Found {len(csv_files)} chunk files to merge.")
        print(f"[INFO] Found {len(csv_files)} chunk files to merge.")

        df_list: List[pd.DataFrame] = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                df_list.append(df)
                logging.info(f"Loaded {file} with {len(df)} rows.")
                print(f"[DEBUG] Loaded {file} with {len(df)} rows.")
            except Exception as e:
                logging.error(f"Failed to read {file}: {e}")
                print(f"[ERROR] Failed to read {file}: {e}")

        if not df_list:
            logging.error("No valid CSV files to merge after reading chunks.")
            print("[ERROR] No valid CSV files to merge after reading chunks.")
            raise ValueError("No valid CSV files to merge after reading chunks.")

        # Concatenate all DataFrames
        merged_df = pd.concat(df_list, ignore_index=True)
        logging.info(
            f"Concatenated all chunks into a single DataFrame with {len(merged_df)} rows."
        )
        print(
            f"[DEBUG] Concatenated all chunks into a single DataFrame with {len(merged_df)} rows."
        )

        # Convert 'timestamp' to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(merged_df["timestamp"]):
            merged_df["timestamp"] = pd.to_datetime(
                merged_df["timestamp"], errors="coerce"
            )
            logging.info("Converted 'timestamp' column to datetime.")
            print("[DEBUG] Converted 'timestamp' column to datetime.")

        # Drop rows with invalid timestamps
        initial_len = len(merged_df)
        merged_df.dropna(subset=["timestamp"], inplace=True)
        final_len = len(merged_df)
        if final_len < initial_len:
            dropped = initial_len - final_len
            logging.warning(f"Dropped {dropped} rows with invalid timestamps.")
            print(f"[WARNING] Dropped {dropped} rows with invalid timestamps.")

        # Sort by 'timestamp' ascending
        merged_df.sort_values(by="timestamp", inplace=True)
        logging.info("Sorted merged DataFrame by 'timestamp' in ascending order.")
        print("[DEBUG] Sorted merged DataFrame by 'timestamp' in ascending order.")

        # Drop duplicate timestamps
        duplicates = merged_df.duplicated(subset=["timestamp"]).sum()
        if duplicates > 0:
            merged_df.drop_duplicates(subset=["timestamp"], inplace=True)
            logging.warning(
                f"Dropped {duplicates} duplicate rows based on 'timestamp'."
            )
            print(
                f"[WARNING] Dropped {duplicates} duplicate rows based on 'timestamp'."
            )

        # Save merged DataFrame to CSV
        try:
            merged_df.to_csv(self.output_file, index=False)
            logging.info(
                f"Saved merged data to {self.output_file} with {len(merged_df)} rows."
            )
            print(
                f"[INFO] Saved merged data to {self.output_file} with {len(merged_df)} rows."
            )
        except Exception as e:
            logging.error(f"Failed to save merged data to {self.output_file}: {e}")
            print(f"[ERROR] Failed to save merged data to {self.output_file}: {e}")

    def clear_chunks(self):
        """
        Deletes all chunk CSV files after successful merging.
        """
        csv_pattern = f"{self.ticker}_{self.interval}_*.csv"
        csv_files = list(self.chunks_dir.glob(csv_pattern))
        for file in csv_files:
            try:
                os.remove(file)
                logging.info(f"Deleted chunk file {file}.")
                print(f"[INFO] Deleted chunk file {file}.")
            except Exception as e:
                logging.error(f"Failed to delete {file}: {e}")
                print(f"[ERROR] Failed to delete {file}: {e}")


# ------------------------- Example Usage ------------------------- #


def example_merge():
    """
    Example script to merge CSV chunks for a specific ticker and interval.
    """
    # Define parameters
    ticker = "AAPL"  # Replace with your target ticker
    interval = "5min"  # Replace with your data interval (e.g., '1min', '5min', etc.)
    base_dir = "app/luqman/data"  # Base directory where data is stored

    # Initialize the merger
    try:
        merger = CSVDataMerger(ticker=ticker, interval=interval, base_dir=base_dir)
    except FileNotFoundError as fe:
        print(f"[FATAL ERROR] {fe}")
        return
    except Exception as e:
        print(f"[FATAL ERROR] An unexpected error occurred: {e}")
        return

    # Perform the merge
    try:
        merger.merge_chunks()
        print(f"[SUCCESS] Merged data saved to {merger.output_file}")
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        return

    # Optional: Clear chunk files after merging
    # Uncomment the following line to delete chunk files after merging
    # merger.clear_chunks()


if __name__ == "__main__":
    example_merge()
