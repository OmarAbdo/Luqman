import pandas as pd
import os


class StockDataHandler:
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2
        self.ticker = self.extract_ticker()
        self.output_file = (
            f"app/ml/data_processed/{self.ticker}/stock/processed_data.csv"
        )
        self.merged_data = None

    def extract_ticker(self):
        """Extracts the ticker name from the file path."""
        # Extract the ticker assuming it's the parent folder of 'technical_indicators' or similar
        try:
            parts = os.path.normpath(self.file1).split(os.sep)
            if "technical_indicators" in parts:
                return parts[parts.index("technical_indicators") - 1]
            elif "technical_sentimental" in parts:
                return parts[parts.index("technical_sentimental") - 1]
            else:
                return parts[4]  # Fallback, adjust if needed
        except IndexError:
            raise ValueError(
                "Unable to extract ticker from the provided file path. Check the file path structure."
            )  # Adjust index if path structure changes
        except IndexError:
            raise ValueError(
                "Unable to extract ticker from the provided file path. Check the file path structure."
            )

    def load_data(self):
        """Loads the two CSV files into DataFrames."""
        # Load CSV files into dataframes
        self.df1 = pd.read_csv(self.file1, index_col=0)
        self.df2 = pd.read_csv(self.file2, index_col=0)
        return self

    def merge_data(self):
        """Merges the two dataframes based on the index column."""
        # Merge the dataframes on the index column
        self.merged_data = pd.merge(
            self.df1, self.df2, left_index=True, right_index=True, how="inner"
        )
        # Remove duplicate columns by selecting only one version of columns with '_x' or '_y' suffix
        for column in self.merged_data.columns:
            if column.endswith("_x") or column.endswith("_y"):
                base_column = column[:-2]
                if base_column in self.merged_data.columns:
                    self.merged_data.drop(column, axis=1, inplace=True)
                else:
                    self.merged_data.rename(columns={column: base_column}, inplace=True)
        return self

    def save_data(self):
        """Saves the merged DataFrame to a CSV file."""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        self.merged_data.to_csv(self.output_file)


# Example usage:
if __name__ == "__main__":
    technical_file = "app/ml/data/AAPL/technical_indicators/technical_indicators_1h.csv"
    technical_sentiment_file = (
        "app/ml/data/AAPL/technical_sentimental/analyzed_data_1h.csv"
    )

    merger = StockDataHandler(technical_file, technical_sentiment_file)
    merger.load_data().merge_data().save_data()

    print(f"Merged data saved to {merger.output_file}")
