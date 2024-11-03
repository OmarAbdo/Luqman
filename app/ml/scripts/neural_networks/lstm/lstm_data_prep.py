import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf


class LSTMDataPreparer:
    def __init__(
        self,
        input_file,
        sequence_length=30,
        output_dir="app/ml/data_processed/AAPL/stock/lstm_ready/",
    ):
        """
        Initializes the LSTMDataPreparer class.

        Args:
            input_file (str): Path to the input standardized CSV file.
            sequence_length (int): The length of each sequence for LSTM input (default is 30).
            output_dir (str): Directory to save the processed LSTM-ready data.

        From a user perspective:
        - This initializes the class with the required input file and the desired sequence length.
        - The output directory can be specified where processed data will be saved.

        From a technical perspective:
        - The constructor initializes key attributes like input file path, sequence length, and output directory.
        - Initializes placeholders for the dataset (X and y).
        """
        self.input_file = input_file
        self.sequence_length = sequence_length
        self.output_dir = output_dir
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """
        Loads the standardized CSV file into a DataFrame.

        From a user perspective:
        - This method reads the input CSV file to be used for LSTM preparation.

        From a technical perspective:
        - Uses pandas to read the CSV and load it into a DataFrame.
        - Stores the loaded data in the `self.data` attribute.
        """
        self.data = pd.read_csv(self.input_file)
        return self

    def prepare_sequences(self):
        """
        Converts the loaded data into sequences for LSTM input.

        From a user perspective:
        - This method transforms the loaded data into input sequences that the LSTM model can use.
        - Sequences are of a specified length, and the target value is the 'Close' price.

        From a technical perspective:
        - Iterates through the dataset to create sequences of the specified length (`sequence_length`).
        - Each sequence (X) is a slice of the data, and the target value (y) is the next value's 'Close' price.
        - Stores the sequences and labels in `self.X` and `self.y` respectively.
        """
        data_values = self.data.values
        X, y = [], []
        for i in range(len(data_values) - self.sequence_length):
            X.append(data_values[i : i + self.sequence_length])
            y.append(
                data_values[i + self.sequence_length][3]
            )  # Predicting 'Close' price
        self.X, self.y = np.array(X), np.array(y)
        return self

    def split_data(self, test_size=0.2):
        """
        Splits the dataset into training and testing sets.

        Args:
            test_size (float): Proportion of the data to use for testing (default is 20%).

        From a user perspective:
        - This method splits the dataset into training and testing subsets for model training and evaluation.

        From a technical perspective:
        - Uses `train_test_split` from `sklearn` to split the sequences (`self.X`) and labels (`self.y`).
        - Returns the split data for training and testing.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def save_data(self):
        """
        Saves the prepared data as numpy arrays.

        From a user perspective:
        - This method saves the LSTM-ready input and target data as `.npy` files in the specified output directory.

        From a technical perspective:
        - Uses `numpy.save()` to save `self.X` and `self.y` arrays to the output directory.
        - Creates the output directory if it doesn't already exist.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(os.path.join(self.output_dir, "X.npy"), self.X)
        np.save(os.path.join(self.output_dir, "y.npy"), self.y)
        print(f"Sequences saved to {self.output_dir}")

    def get_data(self):
        """
        Returns the prepared sequences.

        From a user perspective:
        - This method returns the LSTM input and target data that has been prepared.

        From a technical perspective:
        - Simply returns `self.X` and `self.y` as numpy arrays.
        """
        return self.X, self.y


# Example usage:
if __name__ == "__main__":
    input_file = "app/ml/data_processed/AAPL/stock/standardized_data.csv"

    preparer = LSTMDataPreparer(input_file)
    preparer.load_data().prepare_sequences().save_data()

    X_train, X_test, y_train, y_test = preparer.split_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
