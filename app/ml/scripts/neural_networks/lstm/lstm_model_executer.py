import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class LSTMModelExecutor:
    def __init__(self, model_path, X_test_file, y_test_file):
        self.model_path = model_path
        self.X_test = np.load(X_test_file)
        self.y_test = np.load(y_test_file)
        self.model = None
        self.sequence_length = None

        # Debug prints
        print("X_test shape:", self.X_test.shape)
        print("y_test shape:", self.y_test.shape)

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        self.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())
        self.sequence_length = self.X_test.shape[1]
        print(f"Model loaded from {self.model_path}")
        return self

    def predict_future(self, num_future_steps=30):
        """
        Predicts future values using the last known sequence.
        """
        last_sequence = self.X_test[-1].copy()  # Shape: (sequence_length, num_features)
        future_predictions = []

        for _ in range(num_future_steps):
            # Prepare input for prediction (batch_size=1)
            input_sequence = last_sequence.reshape(1, self.sequence_length, -1)
            next_pred = self.model.predict(input_sequence, verbose=0)[0]

            # If the model output is a scalar
            if next_pred.shape == ():
                next_pred = np.array([next_pred])

            future_predictions.append(next_pred[0])

            # Update the last_sequence with the new prediction
            # Shift the sequence to the left and append the new prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            # Replace the last time step with the new prediction
            # Assuming the prediction corresponds to the target feature at a specific index
            target_feature_index = 0  # Adjust this index as per your data
            last_sequence[-1, target_feature_index] = next_pred[0]

            # Optionally, keep other features constant or update them as needed
            # For now, we'll keep other features unchanged

        return np.array(future_predictions)

    def plot_with_future(self, num_future_steps=30):
        """
        Plots historical data along with future predictions using direct matplotlib plotting.
        """
        # Get predictions on test data
        historical_predictions = self.model.predict(self.X_test).flatten()
        future_predictions = self.predict_future(num_future_steps)

        # Debug prints
        print("Historical predictions shape:", historical_predictions.shape)
        print("Future predictions shape:", future_predictions.shape)

        # Create indices for plotting
        total_length = len(self.y_test) + num_future_steps
        x_indices = np.arange(total_length)

        # Combine actual and future values
        y_actual = self.y_test.flatten()
        y_combined = np.concatenate([y_actual, [np.nan] * num_future_steps])

        # Combine historical and future predictions
        y_pred = np.concatenate([historical_predictions, future_predictions])

        plt.figure(figsize=(15, 8))

        # Plot actual values
        plt.plot(x_indices[: len(y_actual)], y_actual, label="Actual", color="blue")

        # Plot historical predictions
        plt.plot(
            x_indices[: len(historical_predictions)],
            historical_predictions,
            label="Historical Predictions",
            color="green",
        )

        # Plot future predictions
        plt.plot(
            x_indices[len(historical_predictions) :],
            future_predictions,
            "--",
            label="Future Predictions",
            color="red",
        )

        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    model_path = f"app/ml/models/{ticker}_lstm_model.keras"
    X_test_file = f"app/ml/data/{ticker}/stock/lstm_ready/X.npy"
    y_test_file = f"app/ml/data/{ticker}/stock/lstm_ready/y.npy"

    executor = LSTMModelExecutor(model_path, X_test_file, y_test_file)
    executor.load_model()

    # Print shapes before plotting
    print("\nBefore calling plot_with_future:")
    print("X_test final shape:", executor.X_test.shape)
    print("y_test final shape:", executor.y_test.shape)

    try:
        executor.plot_with_future(num_future_steps=30)
    except Exception as e:
        print("\nError occurred:")
        print(e)
        print("\nError type:", type(e))
