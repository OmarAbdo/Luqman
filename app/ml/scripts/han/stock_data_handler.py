import pandas as pd


class StockDataHandler:
    def __init__(self, file1, file2, output_file):
        self.file1 = file1
        self.file2 = file2
        self.output_file = output_file
        self.merged_data = None

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
        # Remove duplicate columns by renaming them and keeping only one
        self.merged_data = self.merged_data.loc[
            :, ~self.merged_data.columns.duplicated()
        ].rename(columns=lambda x: x.replace("_x", "").replace("_y", ""))
        return self

    def save_data(self):
        """Saves the merged DataFrame to a CSV file."""
        self.merged_data.to_csv(self.output_file)


# Example usage:
if __name__ == "__main__":
    technical_file = "app/ml/data/AAPL/technical_indicators/technical_indicators_1h.csv"
    technical_sentiment_file = (
        "app/ml/data/AAPL/technical_sentimental/analyzed_data_1h.csv"
    )
    output_file = "app/ml/data_processed/processed_data.csv"

    merger = StockDataHandler(technical_file, technical_sentiment_file, output_file)
    merger.load_data().merge_data().save_data()

    print(f"Merged data saved to {output_file}")
