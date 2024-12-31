import sys
import numpy as np
import pandas as pd


class SequencePreparer:
    """
    Class responsible for preparing sequences for LSTM input.

    This class takes a DataFrame with features and a target column, then:
      - Splits the features from the target.
      - Optionally drops a timestamp column if it exists (since it's not a numeric feature).
      - Creates sequences of a specified length (window) and the corresponding target values.
      - Uses a sample_rate to limit how many of these sequences get generated (useful for very large datasets).
    """

    def __init__(
        self, sequence_length: int, target_column: str, sample_rate: float = 1.0
    ):
        """
        Initializes the SequencePreparer.

        Args:
            sequence_length (int): The number of past time steps to include in each input sequence.
            target_column (str): The name of the target column to predict.
            sample_rate (float, optional): Fraction of the dataset to process (0 < sample_rate <= 1).
                                           Defaults to 1.0 (process the entire dataset).

        Raises:
            ValueError: If sample_rate is not in the range (0, 1].

        Explanation:
            - sequence_length determines how many rows form one 'window' or sequence.
            - target_column indicates which column in the DataFrame is the prediction target.
            - sample_rate gives control over how many of the total possible sequences are created
              (for down-sampling if data is very large).
        """
        self.sequence_length = sequence_length
        self.target_column = target_column

        # Validate the sample_rate to ensure it's between 0 and 1
        if not (0 < sample_rate <= 1.0):
            raise ValueError(
                "sample_rate must be a float between 0 (exclusive) and 1 (inclusive)."
            )
        self.sample_rate = sample_rate

    def create_sequences(self, data: pd.DataFrame):
        """
        Creates input sequences (X) and corresponding target values (y) from the provided DataFrame.

        Process:
            1. Verify that the target column exists in the DataFrame.
            2. Drop the target column (and 'timestamp' if present) from the features.
            3. Convert the features and target columns to NumPy arrays for faster slicing.
            4. Calculate how many sequences can be formed:
               - total_sequences = len(data) - self.sequence_length
            5. Apply the sample_rate to decide how many sequences to actually create.
            6. Build the input sequences (X) and targets (y):
               - For each i in [0 .. sampled_sequences - 1]:
                 * X[i] = feature_data[i : i + sequence_length]
                 * y[i] = target_data[i + sequence_length]
            7. Return the final NumPy arrays X and y.

        Args:
            data (pd.DataFrame): The scaled DataFrame containing the features and the target column.

        Returns:
            tuple:
                - X (np.ndarray): 3D array of shape (num_samples, sequence_length, num_features).
                                  Each slice along axis 0 is one sequence.
                - y (np.ndarray): 1D array of shape (num_samples,).
                                  Each element corresponds to the target of a sequence.

        Raises:
            KeyError: If the target column doesn't exist in the given DataFrame.
            ValueError: If the data isn't long enough to form a single sequence.

        Explanation:
            - Converting to .values (or .to_numpy()) upfront allows slicing with NumPy (faster than pandas operations).
            - We compute how many sequences are possible, then apply the sample_rate to reduce the total if desired.
            - A list comprehension is used to create a list of windows (X_list) before converting to a NumPy array.
            - Finally, the target array (y) is sliced directly without looping.
        """
        # 1. Ensure the target column exists in the data
        if self.target_column not in data.columns:
            raise KeyError(
                f"Target column '{self.target_column}' not found in the data."
            )

        # 2. Prepare feature columns (drop target and possibly 'timestamp')
        feature_columns = data.columns.drop(self.target_column)
        if "timestamp" in feature_columns:
            feature_columns = feature_columns.drop("timestamp")

        # 3. Convert feature columns and target column to NumPy arrays
        feature_data = data[feature_columns].values
        target_data = data[self.target_column].values

        # 4. Determine total number of possible sequences
        total_sequences = len(data) - self.sequence_length
        if total_sequences <= 0:
            raise ValueError("Data length must be greater than the sequence length.")

        # 5. Calculate how many sequences to process based on sample_rate
        sampled_sequences = int(total_sequences * self.sample_rate)
        sampled_sequences = max(sampled_sequences, 1)  # Ensure at least one

        print(f"Total sequences to process: {sampled_sequences}")

        # 6a. Build X using a list comprehension
        X_list = [
            feature_data[i : i + self.sequence_length] for i in range(sampled_sequences)
        ]
        X = np.array(X_list)  # Convert the list of arrays into a single 3D array

        # 6b. Build y by slicing from sequence_length onward
        y = target_data[self.sequence_length : self.sequence_length + sampled_sequences]

        # Summarize shapes
        print(
            f"Created {X.shape[0]} sequences with shape {X.shape[1:]} for inputs "
            f"and {y.shape} for targets."
        )

        # 7. Return the result
        return X, y
