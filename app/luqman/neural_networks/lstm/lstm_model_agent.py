import os
import numpy as np
import pandas as pd
import tensorflow as tf

import sys

sys.path.append("D:/Freelance/Software - reborn/Github/3 Tools/Luqman")

from app.luqman.pipeline.data_plotter import (
    DataPlotter,
)  # Assuming DataPlotter is imported as in your provided code


class ModelAgent:
    """
    Class to evaluate a trained LSTM model on test data and to forecast future steps.
    It uses the DataPlotter class to plot results and inverse transform predictions.
    """

    def __init__(
        self,
        ticker: str,
        data_directory: str = "app/luqman/data",
        model_directory: str = "app/luqman/models",
        model_name_template: str = "{ticker}_lstm_model.keras",
        future_steps: int = 100,
    ):
        """
        Initializes the ModelTesterAndForecaster.

        Args:
            ticker (str): The stock ticker symbol.
            data_directory (str): The base directory where the data resides.
            model_directory (str): The directory where the model is saved.
            model_name_template (str): Template for the model filename.
            future_steps (int): Number of steps to forecast into the future.
        """
        self.ticker = ticker
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.model_name_template = model_name_template.format(ticker=self.ticker)
        self.future_steps = future_steps

        self.lstm_ready_dir = os.path.join(
            data_directory, self.ticker, "stock", "lstm_ready"
        )
        self.scaler_directory = os.path.join(
            data_directory, self.ticker, "stock", "scalers"
        )
        self.model_path = os.path.join(model_directory, self.model_name_template)

        # Load model
        self.model = tf.keras.models.load_model(self.model_path)

        # Load test data
        self.X_test = np.load(os.path.join(self.lstm_ready_dir, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.lstm_ready_dir, "y_test.npy"))
        self.timestamps_test = np.load(
            os.path.join(self.lstm_ready_dir, "timestamps_test.npy"), allow_pickle=True
        )

        # Initialize plotter
        self.plotter = DataPlotter(scaler_directory=self.scaler_directory)

    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray):
        """
        Calculate regression accuracy metrics and directional accuracy on the original scale.

        Args:
            actual (np.ndarray): Actual values (inverse transformed).
            predicted (np.ndarray): Predicted values (inverse transformed).

        Returns:
            dict: Dictionary of metrics: MAE, MSE, RMSE, Directional Accuracy (%).
        """
        # Calculate regression metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)

        # Calculate directional accuracy
        actual_directions = np.sign(np.diff(actual))
        predicted_directions = np.sign(np.diff(predicted))
        directional_matches = actual_directions == predicted_directions
        directional_accuracy = np.mean(directional_matches) * 100

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Directional Accuracy (%)": directional_accuracy,
        }

    def run(self):
        """
        Predict on test data, measure accuracy, and plot Actual vs. Predicted close prices.
        """
        # Predict on test data
        predictions_scaled = self.model.predict(self.X_test)

        # We have scaled predictions and scaled test targets (y_test).
        # The DataPlotter can inverse transform them for us internally (in the plot function or externally).
        # To measure accuracy, we need the inverse transformed values:
        actual_values = self.plotter.inverse_transform(self.y_test)
        predicted_values = self.plotter.inverse_transform(predictions_scaled)

        # Calculate metrics
        metrics = self._calculate_metrics(actual_values, predicted_values)
        print("Test Set Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        # Plot
        self.plotter.plot_close_price(
            y_test_scaled=self.y_test,
            predictions_scaled=predictions_scaled,
            timestamps=self.timestamps_test,
        )

    def run_forecast(self):
        """
        Forecast future steps after the last known point in test data.

        - Takes the last observed sequence from X_test.
        - Iteratively predict the next step and append it to the sequence.
        - Repeat for 'self.future_steps' times.

        If actual future data is not available, no accuracy metric is computed.
        Otherwise, if future actual values are available, measure accuracy and plot.
        """
        # Start from the last sequence in X_test
        last_sequence = self.X_test[-1]

        # We'll hold predictions scaled as we go
        future_predictions_scaled = []

        current_sequence = last_sequence

        # Iteratively forecast future steps
        for _ in range(self.future_steps):
            next_pred_scaled = self.model.predict(current_sequence[np.newaxis, ...])
            future_predictions_scaled.append(
                next_pred_scaled[0, 0]
            )  # single value predicted

            # Append predicted value to sequence and remove the oldest step
            # current_sequence shape: (sequence_length, num_features)
            # Insert predicted value into the last position of current_sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            # Assuming target is the last column or a known position:
            # Actually, we only need to replace the target column since features come from real data normally.
            # But since we are purely forecasting, we'll keep features constant or zero?
            # A better approach:
            # If we have external features they should be known or set to zero/future data. For simplicity assume only target is predicted.
            # Replace last row's target column with next_pred_scaled
            # We'll identify target_column from the shape. It's 1 since we predict a single value.
            # If multiple features are present, we have no future data for them. This is a limitation.
            # For a pure univariate scenario, this is straightforward:

            # Univariate assumption:
            # current_sequence[-1, :] = something containing predicted value
            # If the original sequences had multiple features, ideally you know what they are. For now, assume univariate model:
            current_sequence[-1, :] = next_pred_scaled

        future_predictions_scaled = np.array(future_predictions_scaled)

        # Attempt to load actual future data if available:
        # This might be future data beyond test set. If it's not available, we just plot predictions as a forecast line.
        future_actual_path = os.path.join(
            self.lstm_ready_dir, "y_future.npy"
        )  # hypothetical file
        future_timestamps_path = os.path.join(
            self.lstm_ready_dir, "timestamps_future.npy"
        )  # hypothetical file

        if os.path.exists(future_actual_path) and os.path.exists(
            future_timestamps_path
        ):
            y_future = np.load(future_actual_path)
            timestamps_future = np.load(future_timestamps_path, allow_pickle=True)

            # Inverse transform
            actual_values_future = self.plotter.inverse_transform(y_future)
            predicted_values_future = self.plotter.inverse_transform(
                future_predictions_scaled
            )

            metrics = self._calculate_metrics(
                actual_values_future, predicted_values_future
            )
            print("Future Forecast Metrics:")
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")

            # Plot actual vs predicted future
            self.plotter.plot_close_price(
                y_test_scaled=y_future,
                predictions_scaled=future_predictions_scaled,
                timestamps=timestamps_future,
            )
        else:
            # No actual future data, just plot predicted values as a forecast
            predicted_values_future = self.plotter.inverse_transform(
                future_predictions_scaled
            )
            # For plotting, we need timestamps. We'll create them as increments from the last test timestamp:
            last_timestamp = pd.to_datetime(self.timestamps_test[-1])
            future_timestamps = [
                last_timestamp + pd.Timedelta(minutes=5 * (i + 1))
                for i in range(self.future_steps)
            ]
            future_timestamps = np.array(future_timestamps, dtype="datetime64[ns]")

            # We don't have actual values, so we can call plotter in a way that just plots predicted line:
            # We'll pass predicted as both actual and predicted to visualize the forecast line on its own.
            # Or we can modify the plotter to handle missing actual gracefully. For now, we'll just plot predicted alone.
            # Create a dummy array for actual (just zeros) to satisfy the function signature:
            dummy_actual_scaled = np.zeros_like(future_predictions_scaled)

            print(
                "No actual future data available. Showing forecast only (no accuracy metrics)."
            )
            self.plotter.plot_close_price(
                y_test_scaled=dummy_actual_scaled,
                predictions_scaled=future_predictions_scaled,
                timestamps=future_timestamps,
            )


if __name__ == "__main__":
    # Set ticker symbol
    ticker = "AAPL"
    os.environ["TICKER"] = ticker  # if needed

    tester = ModelAgent(ticker=ticker)
    tester.run()


# if __name__ == "__main__":
#     # Set ticker symbol
#     ticker = "AAPL"
#     os.environ["TICKER"] = ticker  # if needed

#     forecaster = ModelAgent(ticker=ticker, future_steps=100)
#     forecaster.forecast_future()
