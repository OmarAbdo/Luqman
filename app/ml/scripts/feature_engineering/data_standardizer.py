import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import os
import joblib
import json
from dotenv import load_dotenv

load_dotenv()


class DataLoader:
    """Class responsible for loading data."""

    def __init__(self, input_file: str):
        self.input_file = input_file

    def load_data(self) -> pd.DataFrame:
        """Loads the CSV file into a DataFrame."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"The file {self.input_file} does not exist.")
        data = pd.read_csv(self.input_file, low_memory=False, parse_dates=["timestamp"])
        data.reset_index(drop=False, inplace=True)  # Add index column at the beginning
        return data


class DataCleaner:
    """Class responsible for data cleaning tasks."""

    def identify_columns(self, data: pd.DataFrame):
        """Identifies columns by type."""
        datetime_columns = data.select_dtypes(
            include=["datetime64[ns]"]
        ).columns.tolist()
        numeric_columns = data.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        boolean_columns = data.select_dtypes(include=["bool"]).columns.tolist()
        categorical_columns = data.select_dtypes(include=["object"]).columns.tolist()

        # Exclude 'index' and datetime columns from numeric columns
        exclude_columns = ["index"] + datetime_columns
        numeric_columns = [col for col in numeric_columns if col not in exclude_columns]

        # Remove 'timestamp' from categorical columns if present
        if "timestamp" in categorical_columns:
            categorical_columns.remove("timestamp")

        # Debug: Print the lists of columns
        print(f"Numeric columns: {numeric_columns}")
        print(f"Boolean columns: {boolean_columns}")
        print(f"Categorical columns: {categorical_columns}")
        print(f"Datetime columns: {datetime_columns}")

        return numeric_columns, boolean_columns, categorical_columns, datetime_columns

    def handle_missing_values(
        self,
        data: pd.DataFrame,
        numeric_columns: list,
        boolean_columns: list,
        categorical_columns: list,
    ) -> pd.DataFrame:
        """Handles missing values by imputing them."""
        # Impute numeric columns with the mean
        if numeric_columns:
            imputer_numeric = SimpleImputer(strategy="mean")
            data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])

        # Fill missing values in categorical columns with 'No Divergence' or 'None'
        if categorical_columns:
            for col in categorical_columns:
                data[col] = data[col].fillna("No Divergence")

        # For boolean columns, fill missing values with False
        if boolean_columns:
            data[boolean_columns] = data[boolean_columns].fillna(False)
        return data

    def handle_outliers(
        self, data: pd.DataFrame, numeric_columns: list
    ) -> pd.DataFrame:
        """Handles outliers in numeric columns using the IQR method."""
        for column in numeric_columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[column] = data[column].clip(lower_bound, upper_bound)
        return data


class DataScaler:
    """Class responsible for scaling data."""

    def __init__(self, scaler_directory: str):
        self.scaler_directory = scaler_directory
        os.makedirs(self.scaler_directory, exist_ok=True)
        self.scalers = {}

    def apply_log_scaling(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """Applies logarithmic scaling to the specified column."""
        if (data[column] < 0).any():
            raise ValueError(
                f"Negative values found in {column}; cannot apply log scaling."
            )
        data[column] = np.log1p(data[column])  # log(1 + x)
        self.scalers[column] = {"method": "log_scaling"}
        return data

    def standardize_columns(
        self, data: pd.DataFrame, columns: list, scaler_type: str = "minmax"
    ) -> pd.DataFrame:
        """Standardizes the specified columns using the specified scaler."""
        for column in columns:
            if scaler_type == "minmax":
                scaler = MinMaxScaler()
            elif scaler_type == "standard":
                scaler = StandardScaler()
            else:
                raise ValueError("Invalid scaler_type. Choose 'minmax' or 'standard'.")
            data[[column]] = scaler.fit_transform(data[[column]])
            scaler_filename = f"{column}_{scaler_type}_scaler.pkl"
            joblib.dump(scaler, os.path.join(self.scaler_directory, scaler_filename))
            self.scalers[column] = {
                "method": scaler_type,
                "scaler_file": scaler_filename,
            }
        return data

    def save_scalers_info(self):
        """Saves the scaler information to a JSON file."""
        with open(os.path.join(self.scaler_directory, "scaler_info.json"), "w") as f:
            json.dump(self.scalers, f)


class FeatureEngineer:
    """Class responsible for feature engineering tasks."""

    def convert_boolean(
        self, data: pd.DataFrame, boolean_columns: list
    ) -> pd.DataFrame:
        """Converts boolean fields to binary (0 or 1)."""
        data[boolean_columns] = data[boolean_columns].astype(int)

        # Debug: Check unique values after conversion
        for col in boolean_columns:
            print(
                f"Unique values in '{col}' after convert_boolean: {data[col].unique()}"
            )

        return data

    def one_hot_encode(
        self, data: pd.DataFrame, categorical_columns: list
    ) -> pd.DataFrame:
        """Applies one-hot encoding to categorical fields."""
        # Only encode columns that exist in the data
        existing_columns = [col for col in categorical_columns if col in data.columns]
        if existing_columns:
            data = pd.get_dummies(data, columns=existing_columns, dummy_na=False)
            # Ensure all new columns are of numeric type
            for col in data.columns:
                if data[col].dtype == "uint8":
                    data[col] = data[col].astype(int)
                elif data[col].dtype == "bool":
                    data[col] = data[col].astype(int)
        return data

    def handle_datetime(self, data: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """Extracts datetime components and normalizes them."""
        data[datetime_column] = pd.to_datetime(data[datetime_column])
        data["Year"] = data[datetime_column].dt.year
        data["Month"] = data[datetime_column].dt.month
        data["Day"] = data[datetime_column].dt.day
        data["Hour"] = data[datetime_column].dt.hour
        data["Minute"] = data[datetime_column].dt.minute
        data["DayOfWeek"] = data[datetime_column].dt.weekday
        data.drop(datetime_column, axis=1, inplace=True)

        # Normalize the new datetime columns
        data["Year"] = (data["Year"] - data["Year"].min()) / (
            data["Year"].max() - data["Year"].min()
        )
        data["Month"] = data["Month"] / 12.0
        data["Day"] = data["Day"] / 31.0
        data["Hour"] = data["Hour"] / 23.0
        data["Minute"] = data["Minute"] / 59.0

        # One-hot encode DayOfWeek
        data = pd.get_dummies(
            data, columns=["DayOfWeek"], prefix="DayOfWeek", drop_first=True
        )
        # Ensure DayOfWeek columns are of numeric type
        for col in data.columns:
            if col.startswith("DayOfWeek_"):
                data[col] = data[col].astype(int)
        return data


class DataStandardizer:
    """Main class orchestrating the data standardization process."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.input_file = f"app/ml/data/{self.ticker}/stock/{self.ticker}_5min_technical_sentimental_indicators.csv"
        self.output_file = (
            f"app/ml/data/{self.ticker}/stock/{self.ticker}_5min_standardized.csv"
        )
        self.scaler_directory = f"app/ml/data/{self.ticker}/stock/scalers/"
        self.data = None

    def standardize_data(self):
        """Runs the full standardization process."""
        # Data Loading
        loader = DataLoader(self.input_file)
        self.data = loader.load_data()

        # Save original data for retaining timestamp later
        self.original_data = self.data.copy()

        # Data Cleaning
        cleaner = DataCleaner()
        (
            numeric_columns,
            boolean_columns,
            categorical_columns,
            datetime_columns,
        ) = cleaner.identify_columns(self.data)

        # Handle missing values
        self.data = cleaner.handle_missing_values(
            self.data, numeric_columns, boolean_columns, categorical_columns
        )

        # Handle outliers
        self.data = cleaner.handle_outliers(self.data, numeric_columns)

        # Feature Engineering
        engineer = FeatureEngineer()
        # Handle datetime
        if "timestamp" in self.data.columns:
            self.data = engineer.handle_datetime(self.data, "timestamp")
        # Convert boolean to binary
        if boolean_columns:
            self.data = engineer.convert_boolean(self.data, boolean_columns)
        # One-hot encode categorical columns
        if categorical_columns:
            self.data = engineer.one_hot_encode(self.data, categorical_columns)

        # Ensure all data columns are numeric
        self.data = self.data.apply(pd.to_numeric, errors="coerce")
        # Drop rows with any NaN values that may have been introduced
        self.data.dropna(inplace=True)

        # Data Scaling
        scaler = DataScaler(self.scaler_directory)
        # Apply log scaling to volume
        if "volume" in self.data.columns:
            self.data = scaler.apply_log_scaling(self.data, "volume")
        # Standardize numeric columns
        key_indicators = [
            "MACD",
            "Signal_Line",
            "EMA_12",
            "EMA_26",
            "sentiment_momentum",
        ]
        columns_to_minmax = [
            col
            for col in numeric_columns
            if col not in key_indicators + ["volume", "RSI"]
        ]
        if columns_to_minmax:
            self.data = scaler.standardize_columns(
                self.data, columns_to_minmax, scaler_type="minmax"
            )
        # Scale RSI separately
        if "RSI" in self.data.columns:
            self.data = scaler.standardize_columns(
                self.data, ["RSI"], scaler_type="minmax"
            )
        # Standardize key indicators
        existing_key_indicators = [
            col for col in key_indicators if col in self.data.columns
        ]
        if existing_key_indicators:
            self.data = scaler.standardize_columns(
                self.data, existing_key_indicators, scaler_type="standard"
            )
        # Save scalers info
        scaler.save_scalers_info()

        # Save the standardized data
        if "timestamp" in self.original_data.columns:
            self.data["timestamp"] = self.original_data["timestamp"]
        self.data.to_csv(self.output_file, index=False)
        print(f"Standardized data saved to {self.output_file}")

        # Save feature names
        feature_names = self.data.columns.tolist()
        with open(os.path.join(self.scaler_directory, "feature_names.json"), "w") as f:
            json.dump(feature_names, f)
        print(
            f"Feature names saved to {os.path.join(self.scaler_directory, 'feature_names.json')}"
        )


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    standardizer = DataStandardizer(ticker)
    standardizer.standardize_data()
