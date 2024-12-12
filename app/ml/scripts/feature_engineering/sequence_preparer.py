# File: SequencePreparer.py

import numpy as np
import pandas as pd

# Description:
# The SequencePreparer class is responsible for transforming the scaled training and 
# testing datasets into sequences suitable for LSTM input. It creates input 
# sequences of a specified length (sequence_length) and their corresponding target values.
class SequencePreparer:
    """Class responsible for preparing sequences for LSTM input."""

    def __init__(self, sequence_length: int, target_column: str):
        """
        Initializes the SequencePreparer.

        Args:
            sequence_length (int): The number of past time steps to include in each input sequence.
            target_column (str): The name of the target column to predict.
        """
        self.sequence_length = sequence_length
        self.target_column = target_column

    def create_sequences(self, data: pd.DataFrame):
        """
        Creates input sequences and corresponding targets.

        Args:
            data (pd.DataFrame): The scaled DataFrame containing features and target.

        Returns:
            tuple: Tuple containing:
                - X (np.ndarray): Input sequences of shape (num_samples, sequence_length, num_features).
                - y (np.ndarray): Target values of shape (num_samples,).
        """
        feature_columns = data.columns.drop(self.target_column)        
        # Exclude 'timestamp' so it doesn't end up in the X arrays
        if 'timestamp' in feature_columns:
            feature_columns = feature_columns.drop('timestamp')
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[feature_columns].iloc[i : i + self.sequence_length].values)
            y.append(data[self.target_column].iloc[i + self.sequence_length])
        X = np.array(X)
        y = np.array(y)
        print(
            f"Created {X.shape[0]} sequences with shape {X.shape[1:]} for inputs and {y.shape} for targets."
        )
        return X, y