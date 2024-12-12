# File: FeatureEngineer.py

import pandas as pd

# The FeatureEngineer class performs various feature engineering tasks, including:

# Converting boolean columns to binary (0 or 1).
# Applying one-hot encoding to categorical columns.
# Handling datetime columns by extracting components (Year, Month, Day, Hour, Minute, DayOfWeek), normalizing them, and one-hot encoding the DayOfWeek.
class FeatureEngineer:
    """Class responsible for feature engineering tasks."""

    def convert_boolean(
        self, data: pd.DataFrame, boolean_columns: list
    ) -> pd.DataFrame:
        """
        Converts boolean fields to binary (0 or 1).

        Args:
            data (pd.DataFrame): The DataFrame to modify.
            boolean_columns (list): List of boolean columns.

        Returns:
            pd.DataFrame: The DataFrame with converted boolean columns.
        """
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
        """
        Applies one-hot encoding to categorical fields.

        Args:
            data (pd.DataFrame): The DataFrame to modify.
            categorical_columns (list): List of categorical columns.

        Returns:
            pd.DataFrame: The DataFrame with one-hot encoded categorical columns.
        """
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
            print(f"One-hot encoding applied to columns: {existing_columns}")
        else:
            print("No categorical columns found for one-hot encoding.")

        return data

    def handle_datetime(self, data: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """
        Extracts datetime components and normalizes them.

        Args:
            data (pd.DataFrame): The DataFrame to modify.
            datetime_column (str): The name of the datetime column.

        Returns:
            pd.DataFrame: The DataFrame with extracted and normalized datetime components.
        """
        data[datetime_column] = pd.to_datetime(data[datetime_column])
        data["Year"] = data[datetime_column].dt.year
        data["Month"] = data[datetime_column].dt.month
        data["Day"] = data[datetime_column].dt.day
        data["Hour"] = data[datetime_column].dt.hour
        data["Minute"] = data[datetime_column].dt.minute
        data["DayOfWeek"] = data[datetime_column].dt.weekday

        # Retain the original timestamp column for sequence extraction
        original_timestamps = data[datetime_column]

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

        # Reattach the original timestamps
        data["timestamp"] = original_timestamps

        return data
