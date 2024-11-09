import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns


class LSTMModelExecutor:
    def __init__(self, model_path, X_test_file, y_test_file):
        """
        Initializes the LSTMModelExecutor class.

        Args:
            model_path (str): Path to the saved LSTM model (.h5 file).
            X_test_file (str): Path to the testing input sequences (.npy file).
            y_test_file (str): Path to the testing target values (.npy file).

        This class is responsible for loading a saved LSTM model, making predictions, and visualizing the results.
        """
        self.model_path = model_path
        self.X_test = np.load(X_test_file)
        self.y_test = np.load(y_test_file)
        self.model = None

    def load_model(self):
        """
        Loads the saved LSTM model from the specified path.

        From a user perspective:
        - This method loads a trained model so that it can be used for predictions.

        From a technical perspective:
        - Uses TensorFlow's `load_model()` function to load the model.
        """
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
        print(f"Model loaded from {self.model_path}")
        return self

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
    model_path = "app/ml/models/SAP_lstm_model.h5"
    X_test_file = "app/ml/data_processed/SAP/stock/lstm_ready/X.npy"
    y_test_file = "app/ml/data_processed/SAP/stock/lstm_ready/y.npy"

    executor = LSTMModelExecutor(model_path, X_test_file, y_test_file)
    executor.load_model()
    executor.calculate_accuracies()
    executor.predict_and_plot()
