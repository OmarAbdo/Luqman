import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns


class LSTMModelTrainer:
    def __init__(
        self, X_train_file, y_train_file, X_test_file, y_test_file, sample_fraction=1.0
    ):
        """
        Initializes the LSTMModelTrainer class.

        Args:
            X_train_file (str): Path to the training input sequences (.npy file).
            y_train_file (str): Path to the training target values (.npy file).
            X_test_file (str): Path to the testing input sequences (.npy file).
            y_test_file (str): Path to the testing target values (.npy file).
            sample_fraction (float): Fraction of data to use for training and testing (default is 1.0).

        This class is responsible for building, training, and testing an LSTM model, as well as visualizing the predictions.
        """
        self.X_train = np.load(X_train_file)
        self.y_train = np.load(y_train_file)
        self.X_test = np.load(X_test_file)
        self.y_test = np.load(y_test_file)

        # Use only a fraction of the data if specified
        if sample_fraction < 1.0:
            train_size = int(len(self.X_train) * sample_fraction)
            test_size = int(len(self.X_test) * sample_fraction)
            self.X_train = self.X_train[:train_size]
            self.y_train = self.y_train[:train_size]
            self.X_test = self.X_test[:test_size]
            self.y_test = self.y_test[:test_size]

        self.sample_fraction = sample_fraction
        self.model = None

    def build_model(self):
        """
        Builds the LSTM model architecture.

        From a user perspective:
        - This method creates the structure of the LSTM model, which will be used for training.

        From a technical perspective:
        - Uses Keras' Sequential API to add LSTM, Dropout, and Dense layers.
        """
        self.model = Sequential()
        self.model.add(
            LSTM(
                50,
                activation="relu",
                input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
            )
        )
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        # Compiling the model
        self.model.compile(optimizer="adam", loss="mse")
        return self

    def train_model(self, epochs=50, batch_size=32):
        """
        Trains the LSTM model.

        Args:
            epochs (int): Number of epochs to train the model (default is 50).
            batch_size (int): Number of samples per gradient update (default is 32).

        From a user perspective:
        - This method trains the model using the training data.

        From a technical perspective:
        - Uses the `fit` method of the Keras model to train with the given training data.
        """
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=1,
        )
        return self

    def evaluate_model(self):
        """
        Evaluates the LSTM model on the test set.

        From a user perspective:
        - This method evaluates the trained model to check its performance on unseen data.

        From a technical perspective:
        - Uses the `evaluate` method from Keras to calculate the loss on the test set.
        """
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {loss}")
        return loss

    def save_model(self, model_path="app/ml/models/AAPL_lstm_model.h5"):
        """
        Saves the trained LSTM model.

        Args:
            model_path (str): Path to save the trained model (default is "app/ml/models/AAPL_lstm_model.h5").

        From a user perspective:
        - This method saves the trained model to a specified path.

        From a technical perspective:
        - Uses `model.save()` to save the model in HDF5 format.
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def calculate_accuracies(self):
        """
        Calculates both the original accuracy (mean squared error) and the directional accuracy.

        From a user perspective:
        - This method calculates both the typical accuracy and how often the model correctly predicts the direction of price movement.

        From a technical perspective:
        - Uses `model.predict()` to generate predictions and calculates both mean squared error and directional accuracy.
        """
        predictions = self.model.predict(self.X_test)

        # Original MSE-based Accuracy
        mse = np.mean((self.y_test.flatten() - predictions.flatten()) ** 2)
        print(f"Mean Squared Error: {mse:.2f}")

        # Directional Accuracy
        actual_directions = np.diff(self.y_test.flatten()) > 0
        predicted_directions = np.diff(predictions.flatten()) > 0

        correct_directions = np.sum(actual_directions == predicted_directions)
        directional_accuracy = correct_directions / len(actual_directions) * 100

        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        return mse, directional_accuracy

    def predict_and_plot(self):
        """
        Makes predictions using the test set and plots the predicted vs actual values in a user-friendly way.

        From a user perspective:
        - This method visualizes how well the model's predictions match the actual values.

        From a technical perspective:
        - Uses the trained model to predict values for the test set and plots the predicted vs actual values using Seaborn for a more readable visualization.
        """
        predictions = self.model.predict(self.X_test)

        plt.figure(figsize=(15, 8))
        sns.lineplot(
            x=range(len(self.y_test)),
            y=self.y_test.flatten(),
            label="Actual Values",
            color="blue",
        )
        sns.lineplot(
            x=range(len(predictions)),
            y=predictions.flatten(),
            label="Predicted Values",
            color="red",
        )
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.title("Actual vs Predicted Close Price (Line Chart)")
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    X_train_file = "app/ml/data_processed/AAPL/stock/lstm_ready/X.npy"
    y_train_file = "app/ml/data_processed/AAPL/stock/lstm_ready/y.npy"
    X_test_file = "app/ml/data_processed/AAPL/stock/lstm_ready/X.npy"
    y_test_file = "app/ml/data_processed/AAPL/stock/lstm_ready/y.npy"

    trainer = LSTMModelTrainer(
        X_train_file, y_train_file, X_test_file, y_test_file, sample_fraction=1
    )
    trainer.build_model().train_model().evaluate_model()
    trainer.save_model()
    trainer.calculate_accuracies()
    trainer.predict_and_plot()
