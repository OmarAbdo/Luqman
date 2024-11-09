import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import os
from dotenv import load_dotenv

load_dotenv()
"""""
DataStandardizer Class

This class is responsible for standardizing a dataset containing technical and sentimental indicators for a given stock ticker. The purpose of this standardization is to prepare the data for training machine learning models, specifically LSTMs, by handling the various types of data appropriately. Below is an overview of the design decisions and thought process behind the methods in this class:

1. **Data Type Identification**: Columns are classified into numeric, boolean, categorical, and datetime types to apply appropriate preprocessing methods for each.

2. **Handling Missing Values**: Missing values are imputed based on the type of data:
   - Numeric columns are imputed with the mean value.
   - Categorical columns are imputed with the most frequent value.
   - Boolean columns are filled with `0` (assuming False).

3. **Outlier Handling**: Outliers are capped using the Interquartile Range (IQR) method to reduce their impact on model training. This prevents extreme values from skewing the training process.

4. **Standardization**:
   - **Numeric Columns**: Min-Max scaling is applied to normalize values between `0` and `1`, except for key indicators like MACD, Signal Line, RSI, and others that convey critical information.
   - **Key Indicators (MACD, RSI, etc.)**: Z-score normalization is used for key indicators to retain their relative importance and thresholds. These indicators often have thresholds (e.g., MACD > 0.7) that are meaningful for market interpretation, and Z-score scaling helps maintain these relationships.
   - **Logarithmic Scaling for Volume**: Logarithmic scaling (`log(1 + x)`) is applied to the volume column to handle large differences in volume while preserving relative changes.

5. **Replacing Misleading Zeros**: Many indicators like MACD, RSI, etc., might initially be `0` due to insufficient historical data for calculation. These `0` values can mislead the LSTM during training, so they are replaced with `NaN` to indicate missing data.

6. **Datetime Handling**: Datetime components are extracted and normalized. The `Year` column is normalized using a fixed range (`2000-2050`) to retain temporal context while making it suitable for machine learning models.

7. **One-Hot Encoding for Categorical Columns**: Categorical values with fewer unique values are one-hot encoded, while other categorical signals are converted to numerical labels.

8. **Order Preservation**: The original row order is preserved throughout the process. This is crucial for time series data, such as stock prices, where historical order directly impacts model training.

These design choices ensure that the data is effectively transformed for machine learning, maintaining important relationships and making the dataset more suitable for LSTM models.
"""
class DataStandardizer:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.input_file = f"app/ml/data/{self.ticker}/stock/{self.ticker}_5min_technical_sentimental_indicators.csv"
        self.output_file = (
            f"app/ml/data/{self.ticker}/stock/{self.ticker}_5min_standardized.csv"
        )
        self.data = None
        self.numeric_columns = []
        self.boolean_columns = []
        self.categorical_columns = []
        self.datetime_column = "timestamp"
        self.standardized_data = None
        self.standardize_data()

    def load_data(self):
        """Loads the CSV file into a DataFrame."""
        self.data = pd.read_csv(self.input_file, low_memory=False)
        return self

    def identify_columns(self):
        """Identifies columns by type: numeric, boolean, categorical, and datetime."""
        for column in self.data.columns:
            if column == self.datetime_column:
                continue
            if self.data[column].dtype in ["int64", "float64"]:
                self.numeric_columns.append(column)
            elif self.data[column].dtype == "bool":
                self.boolean_columns.append(column)
            elif self.data[column].dtype == "object":
                self.categorical_columns.append(column)
        return self

    def handle_missing_values(self):
        """Handles missing values by imputing them."""
        # Impute numeric columns with the mean
        imputer_numeric = SimpleImputer(strategy="mean")
        self.data[self.numeric_columns] = imputer_numeric.fit_transform(
            self.data[self.numeric_columns]
        )

        # Impute categorical columns with the most frequent value
        imputer_categorical = SimpleImputer(strategy="most_frequent")
        self.data[self.categorical_columns] = imputer_categorical.fit_transform(
            self.data[self.categorical_columns]
        )

        # For boolean columns, fill missing values with 0 (assuming False)
        self.data[self.boolean_columns] = self.data[self.boolean_columns].fillna(0)
        return self

    def handle_outliers(self):
        """Handles outliers in numeric columns using the IQR method."""
        for column in self.numeric_columns:
            Q1 = self.data[column].quantile(0.25)
            Q3 = self.data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Cap the outliers to the lower and upper bounds
            self.data[column] = np.where(
                self.data[column] < lower_bound, lower_bound, self.data[column]
            )
            self.data[column] = np.where(
                self.data[column] > upper_bound, upper_bound, self.data[column]
            )
        return self

    def standardize_numeric(self):
        """Normalizes numeric fields using Min-Max normalization, except key indicators."""
        scaler = MinMaxScaler()
        # Define columns to use Min-Max scaling (excluding key indicators)
        columns_to_min_max = [
            col
            for col in self.numeric_columns
            if col
            not in [
                "MACD",
                "Signal_Line",
                "RSI",
                "EMA_12",
                "EMA_26",
                "sentiment_momentum",
            ]
        ]
        self.data[columns_to_min_max] = scaler.fit_transform(
            self.data[columns_to_min_max]
        )
        return self

    def standardize_key_indicators(self):
        """Standardizes key indicators using Z-score normalization."""
        key_indicators = [
            "MACD",
            "Signal_Line",
            "RSI",
            "EMA_12",
            "EMA_26",
            "sentiment_momentum",
        ]
        scaler = StandardScaler()
        for column in key_indicators:
            if column in self.data.columns:
                self.data[column] = scaler.fit_transform(self.data[[column]])
        return self

    def convert_boolean(self):
        """Converts boolean fields to binary (0 or 1)."""
        self.data[self.boolean_columns] = self.data[self.boolean_columns].astype(int)
        return self

    def one_hot_encode_categorical(self):
        """Applies one-hot encoding to categorical fields, including textual signals."""
        for column in self.categorical_columns:
            if (
                self.data[column].nunique() < 10
            ):  # For categorical with few unique values
                dummies = pd.get_dummies(self.data[column], prefix=column).astype(int)
                self.data = pd.concat([self.data, dummies], axis=1)
                self.data.drop(column, axis=1, inplace=True)
            else:
                # For textual signals like "Regular Bearish Divergence", convert to numerical labels
                self.data[column] = self.data[column].astype("category").cat.codes
        return self

    def handle_datetime(self):
        """Extracts datetime components and normalizes them."""
        if self.datetime_column in self.data.columns:
            datetime_series = pd.to_datetime(self.data[self.datetime_column], utc=True)
            self.data["Year"] = (datetime_series.dt.year - 2000) / (
                2050 - 2000
            )  # Normalizing year
            self.data["Month"] = datetime_series.dt.month / 12.0
            self.data["Day"] = datetime_series.dt.day / 31.0
            self.data["Hour"] = datetime_series.dt.hour / 23.0
            self.data["Minute"] = datetime_series.dt.minute / 59.0
            self.data["DayOfWeek"] = datetime_series.dt.weekday
            day_of_week_dummies = pd.get_dummies(
                self.data["DayOfWeek"], prefix="DayOfWeek"
            ).astype(int)
            self.data = pd.concat([self.data, day_of_week_dummies], axis=1)
            self.data.drop([self.datetime_column, "DayOfWeek"], axis=1, inplace=True)
        return self

    def handle_zeros_for_indicators(self):
        """Replaces misleading zeros in calculated indicator columns with NaN."""
        indicator_columns = [
            "MA_10",
            "MA_50",
            "BB_MA",
            "BB_std",
            "BB_upper",
            "BB_lower",
            "EMA_12",
            "EMA_26",
            "MACD",
            "Signal_Line",
            "sentiment_momentum",
        ]
        for column in indicator_columns:
            if column in self.data.columns:
                self.data[column] = self.data[column].replace(0, np.nan)
        return self

    def apply_log_scaling(self):
        """Applies logarithmic scaling to the volume column."""
        if "volume" in self.data.columns:
            self.data["volume"] = self.data["volume"].apply(
                lambda x: np.log1p(x)
            )  # log(1 + x) to handle zero values
        return self

    def save_standardized_data(self):
        """Saves the standardized data to a CSV file."""
        self.data.to_csv(self.output_file, index=False)
        print(f"Standardized data saved to {self.output_file}")

    def standardize_data(self):
        """Runs the full standardization process."""
        return (
            self.load_data()
            .identify_columns()
            .handle_missing_values()
            .handle_outliers()
            .standardize_numeric()
            .standardize_key_indicators()
            .convert_boolean()
            .one_hot_encode_categorical()
            .handle_datetime()
            .handle_zeros_for_indicators()
            .apply_log_scaling()
            .save_standardized_data()
        )


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    DataStandardizer(ticker)
