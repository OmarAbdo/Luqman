import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class DataStandardizer:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None
        self.numeric_columns = []
        self.boolean_columns = []
        self.categorical_columns = []
        self.datetime_column = "Datetime"
        self.standardized_data = None

    def load_data(self):
        """Loads the CSV file into a DataFrame."""
        self.data = pd.read_csv(self.input_file)
        return self

    def identify_columns(self):
        """Identifies columns by type: numeric, boolean, categorical, and datetime."""
        for column in self.data.columns:
            if (
                self.data[column].dtype in ["int64", "float64"]
                and column != self.datetime_column
            ):
                self.numeric_columns.append(column)
            elif self.data[column].dtype == "bool":
                self.boolean_columns.append(column)
            elif self.data[column].dtype == "object" and column != self.datetime_column:
                self.categorical_columns.append(column)
        return self

    def standardize_numeric(self):
        """Normalizes numeric fields using Min-Max normalization."""
        scaler = MinMaxScaler()
        self.data[self.numeric_columns] = scaler.fit_transform(
            self.data[self.numeric_columns]
        )
        return self

    def convert_boolean(self):
        """Converts boolean fields to binary (0 or 1)."""
        self.data[self.boolean_columns] = self.data[self.boolean_columns].astype(int)
        return self

    def one_hot_encode_categorical(self):
        """Applies one-hot encoding to categorical fields."""
        for column in self.categorical_columns:
            dummies = pd.get_dummies(self.data[column], prefix=column)
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data.drop(column, axis=1, inplace=True)
        return self

    def handle_datetime(self):
        """Extracts datetime components and normalizes them."""
        if self.datetime_column in self.data.columns:
            datetime_series = pd.to_datetime(self.data[self.datetime_column], utc=True)
            self.data["Year"] = datetime_series.dt.year
            self.data["Month"] = datetime_series.dt.month / 12.0
            self.data["Day"] = datetime_series.dt.day / 31.0
            self.data["Hour"] = datetime_series.dt.hour / 23.0
            self.data["Minute"] = datetime_series.dt.minute / 59.0
            self.data["DayOfWeek"] = datetime_series.dt.weekday
            day_of_week_dummies = pd.get_dummies(
                self.data["DayOfWeek"], prefix="DayOfWeek"
            )
            self.data = pd.concat([self.data, day_of_week_dummies], axis=1)
            self.data.drop([self.datetime_column, "DayOfWeek"], axis=1, inplace=True)
        return self

    def save_standardized_data(self):
        """Saves the standardized data to a CSV file."""
        self.data.to_csv(self.output_file, index=False)

    def standardize_data(self):
        """Runs the full standardization process."""
        return (
            self.load_data()
            .identify_columns()
            .standardize_numeric()
            .convert_boolean()
            .one_hot_encode_categorical()
            .handle_datetime()
            .save_standardized_data()
        )


# Example usage:
if __name__ == "__main__":
    input_file = "app/ml/data_processed/processed_data.csv"
    output_file = "app/ml/data_processed/standardized_data.csv"

    standardizer = DataStandardizer(input_file, output_file)
    standardizer.standardize_data()

    print(f"Standardized data saved to {output_file}")
