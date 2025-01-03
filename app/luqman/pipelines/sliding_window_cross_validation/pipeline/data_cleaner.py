# File: DataCleaner.py

import pandas as pd
from sklearn.impute import SimpleImputer

# The DataCleaner class handles the cleaning of the DataFrame by:


# Identifying columns based on their data types (numeric, boolean, categorical, datetime).
# Imputing missing values for numeric, categorical, and boolean columns.
# Handling outliers in numeric columns using the Interquartile Range (IQR) method.
class DataCleaner:
    """Class responsible for data cleaning tasks."""

    def identify_columns(self, data: pd.DataFrame):
        """
        Identifies columns by their data types.

        Args:
            data (pd.DataFrame): The DataFrame to analyze.

        Returns:
            tuple: Lists of numeric, boolean, categorical, and datetime columns.
        """
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
        """
        Handles missing values by imputing them.

        Args:
            data (pd.DataFrame): The DataFrame to clean.
            numeric_columns (list): List of numeric columns.
            boolean_columns (list): List of boolean columns.
            categorical_columns (list): List of categorical columns.

        Returns:
            pd.DataFrame: The DataFrame with imputed missing values.
        """
        # Impute numeric columns with the mean
        if numeric_columns:
            imputer_numeric = SimpleImputer(strategy="mean")
            data[numeric_columns] = imputer_numeric.fit_transform(data[numeric_columns])
            print(f"Missing values in numeric columns imputed with mean.")

        # Fill missing values in categorical columns with 'No Divergence'
        if categorical_columns:
            for col in categorical_columns:
                data[col] = data[col].fillna("No Divergence")
            print(f"Missing values in categorical columns filled with 'No Divergence'.")

        # For boolean columns, fill missing values with False
        if boolean_columns:
            data[boolean_columns] = data[boolean_columns].fillna(False)
            print(f"Missing values in boolean columns filled with False.")

        return data

    def handle_outliers(
        self, data: pd.DataFrame, numeric_columns: list
    ) -> pd.DataFrame:
        """
        Handles outliers in numeric columns using the IQR method.

        Args:
            data (pd.DataFrame): The DataFrame to clean.
            numeric_columns (list): List of numeric columns.

        Returns:
            pd.DataFrame: The DataFrame with outliers handled.
        """
        for column in numeric_columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            original_outliers = data[
                (data[column] < lower_bound) | (data[column] > upper_bound)
            ][column].count()
            data[column] = data[column].clip(lower_bound, upper_bound)
            print(
                f"Handled outliers in '{column}': {original_outliers} outliers clipped."
            )

        return data
