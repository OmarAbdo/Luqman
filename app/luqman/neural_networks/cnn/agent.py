import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd

import sys

sys.path.append("D:/Freelance/Software - reborn/Github/3 Tools/Luqman")
# Importing the same DataPlotter you've shown
# Make sure the path and class name are correct for your project structure.
from app.luqman.pipeline.data_plotter import DataPlotter


class AdvancedModelAgent:
    """
    Agent class for loading a trained advanced model (Dilated CNN + LSTM + Attention),
    evaluating on test data, plotting the results via a shared DataPlotter, and optionally forecasting future steps.

    Key Features:
    - Loads model from disk.
    - Loads scaled test data and timestamps from disk.
    - Uses DataPlotter to:
      1) Inverse transform scaled predictions and actuals to original scale.
      2) Plot Actual vs. Predicted time-series using consistent scaling.
    - Computes metrics (MSE, RMSE, MAE, Directional Accuracy) on de-scaled values.
    - Supports iterative multi-step forecasting on scaled data, then inverse transforms for final outputs.
    """

    def __init__(
        self,
        ticker: str,
        data_directory: str = "app/luqman/data",
        model_directory: str = "app/luqman/models",
        model_name_template: str = "{ticker}_advanced_model.keras",
        future_steps: int = 10,
    ):
        """
        Initializes the AdvancedModelAgent with relevant paths and forecasting parameters.

        Args:
            ticker (str): Stock ticker symbol.
            data_directory (str): Base directory where data is located (containing 'lstm_ready').
            model_directory (str): Directory where the trained model is saved.
            model_name_template (str): Template for the model filename.
            future_steps (int): Number of steps to forecast in the future.
        """
        self.ticker = ticker
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.model_name_template = model_name_template.format(ticker=self.ticker)
        self.future_steps = future_steps

        # Paths
        self.lstm_ready_dir = os.path.join(
            self.data_directory, self.ticker, "stock", "lstm_ready"
        )
        self.scaler_directory = os.path.join(
            self.data_directory, self.ticker, "stock", "scalers"
        )
        self.model_path = os.path.join(self.model_directory, self.model_name_template)

        # Data placeholders
        self.X_test = None
        self.y_test = None
        self.timestamps_test = None

        # Load model
        self.model = None

        # DataPlotter for consistent inverse transforms and plotting
        self.plotter = DataPlotter(scaler_directory=self.scaler_directory)

    def load_data(self):
        """
        Loads the scaled test data and timestamps from disk,
        consistent with how the trainer code saves them.
        """
        try:
            self.X_test = np.load(os.path.join(self.lstm_ready_dir, "X_test.npy"))
            self.y_test = np.load(os.path.join(self.lstm_ready_dir, "y_test.npy"))
            self.timestamps_test = np.load(
                os.path.join(self.lstm_ready_dir, "timestamps_test.npy"),
                allow_pickle=True,
            )
            print("Test data loaded successfully.")
            print(
                f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}"
            )
            print(f"Timestamps shape: {self.timestamps_test.shape}")
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            raise

    def load_model(self):
        """
        Loads the trained model from disk.
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        try:
            self.model = load_model(self.model_path, compile=True)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def _calculate_metrics(
        self, actual_unscaled: np.ndarray, predicted_unscaled: np.ndarray
    ):
        """
        Calculate regression accuracy metrics and directional accuracy on the original (de-scaled) values.

        Args:
            actual_unscaled (np.ndarray): Actual values in original scale.
            predicted_unscaled (np.ndarray): Predicted values in original scale.

        Returns:
            dict: Dictionary of metrics: MSE, RMSE, MAE, Directional Accuracy (%).
        """
        mae = mean_absolute_error(actual_unscaled, predicted_unscaled)
        mse = mean_squared_error(actual_unscaled, predicted_unscaled)
        rmse = np.sqrt(mse)

        # Directional Accuracy
        if len(actual_unscaled) < 2:
            directional_accuracy = np.nan
            print("Not enough data points to compute Directional Accuracy.")
        else:
            actual_directions = np.sign(np.diff(actual_unscaled))
            predicted_directions = np.sign(np.diff(predicted_unscaled))
            directional_matches = actual_directions == predicted_directions
            directional_accuracy = np.mean(directional_matches) * 100

        return {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "Directional Accuracy (%)": directional_accuracy,
        }

    def run_evaluation(self, plot_results: bool = True):
        """
        Loads data, loads model, generates predictions, inverse transforms,
        calculates metrics, and optionally plots results.
        """
        self.load_data()
        self.load_model()

        # Predict on test data (scaled)
        predictions_scaled = self.model.predict(self.X_test)
        print(
            f"Inference done on test set, shape of predictions: {predictions_scaled.shape}"
        )

        # Inverse transform to original scale for metrics and interpretability
        actual_unscaled = self.plotter.inverse_transform(self.y_test)
        predicted_unscaled = self.plotter.inverse_transform(predictions_scaled)

        # Calculate metrics
        metrics = self._calculate_metrics(actual_unscaled, predicted_unscaled)
        print("Test Set Metrics (on original scale):")
        for k, v in metrics.items():
            if np.isnan(v):
                print(f"{k}: Not Computable")
            else:
                print(f"{k}: {v:.4f}")

        # Plot if requested
        if plot_results:
            # We pass the scaled versions to the DataPlotter because
            # DataPlotter does the inverse transform inside plot_close_price as well.
            # However, we've already done an inverse transform for metrics.
            # EITHER do the transform outside or inside the plotter, not both.
            #
            # If you want to rely on DataPlotter's internal inverse transforms,
            # pass the scaled arrays:
            self.plotter.plot_close_price(
                y_test_scaled=self.y_test,
                predictions_scaled=predictions_scaled,
                timestamps=self.timestamps_test,
            )

    def run_forecast(self):
        """
        Forecasts the next 'future_steps' points after the last known point in test data.
        """
        if self.model is None:
            raise ValueError(
                "Model not loaded. Call load_model() before run_forecast()."
            )
        if self.X_test is None:
            raise ValueError(
                "No test data loaded. Call load_data() before run_forecast()."
            )

        # Start from the last sequence in X_test
        last_sequence = self.X_test[-1].copy()  # shape: (sequence_length, num_features)
        future_predictions_scaled = []

        for _ in range(self.future_steps):
            current_seq = np.expand_dims(
                last_sequence, axis=0
            )  # (1, seq_len, num_features)
            next_pred_scaled = self.model.predict(current_seq, verbose=0)[0, 0]
            future_predictions_scaled.append(next_pred_scaled)

            # Shift the window
            last_sequence = np.roll(last_sequence, -1, axis=0)
            # If target is in the last column:
            last_sequence[-1, -1] = next_pred_scaled

        # Attempt to load actual future data if available
        future_actual_path = os.path.join(self.lstm_ready_dir, "y_future.npy")
        future_timestamps_path = os.path.join(
            self.lstm_ready_dir, "timestamps_future.npy"
        )

        if os.path.exists(future_actual_path) and os.path.exists(
            future_timestamps_path
        ):
            y_future_scaled = np.load(future_actual_path)
            timestamps_future = np.load(future_timestamps_path, allow_pickle=True)

            # Inverse transform
            actual_values_future = self.plotter.inverse_transform(y_future_scaled)
            predicted_values_future = self.plotter.inverse_transform(
                np.array(future_predictions_scaled)
            )

            metrics_future = self._calculate_metrics(
                actual_values_future, predicted_values_future
            )
            print("Future Forecast Metrics (on original scale):")
            for k, v in metrics_future.items():
                if np.isnan(v):
                    print(f"{k}: Not Computable")
                else:
                    print(f"{k}: {v:.4f}")

            # Plot actual vs predicted future
            self.plotter.plot_close_price(
                y_test_scaled=y_future_scaled,
                predictions_scaled=np.array(future_predictions_scaled),
                timestamps=timestamps_future,
            )
        else:
            # No future actual data, just plot predicted line as a forecast
            predicted_values_future = self.plotter.inverse_transform(
                np.array(future_predictions_scaled)
            )

            last_timestamp = pd.to_datetime(self.timestamps_test[-1])
            future_timestamps = [
                last_timestamp + pd.Timedelta(minutes=5 * (i + 1))
                for i in range(self.future_steps)
            ]
            future_timestamps = np.array(future_timestamps, dtype="datetime64[ns]")

            # We'll pass scaled arrays to the DataPlotter again
            # but create a dummy array for actual
            dummy_actual_scaled = np.zeros_like(future_predictions_scaled)

            print(
                "No actual future data available. Showing forecast only (no accuracy metrics)."
            )
            self.plotter.plot_close_price(
                y_test_scaled=dummy_actual_scaled,
                predictions_scaled=np.array(future_predictions_scaled),
                timestamps=future_timestamps,
            )

    def save_predictions(self, predictions, save_path: str = "predictions.npy"):
        """
        Saves the predictions to a NumPy file.

        Args:
            predictions (ndarray): Predicted values.
            save_path (str): Path to save the predictions.
        """
        np.save(save_path, predictions)
        print(f"Predictions saved to {save_path}")

    def save_forecasts(self, forecasts, save_path: str = "forecasts.npy"):
        """
        Saves the forecasted future steps to a NumPy file.

        Args:
            forecasts (list): Forecasted values.
            save_path (str): Path to save the forecasts.
        """
        np.save(save_path, forecasts)
        print(f"Forecasts saved to {save_path}")


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()

    ticker = os.getenv("TICKER", "AAPL")  # default to AAPL if not set
    agent = AdvancedModelAgent(
        ticker=ticker,
        data_directory="app/luqman/data",
        model_directory="app/luqman/models",
        model_name_template="{ticker}_advanced_model.keras",
        future_steps=10,
    )

    # Evaluate the model on test data
    agent.run_evaluation(plot_results=True)

    # Optionally forecast future steps
    # forecasts = agent.run_forecast()
    # agent.save_forecasts(forecasts, save_path=f"forecasts_{ticker}.npy")
