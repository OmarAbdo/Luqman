# File: ModelAgent.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple

import sys

sys.path.append("D:/Freelance/Software - reborn/Github/3 Tools/Luqman")

from app.luqman.sliding_window_pipeline.data_plotter import (
    DataPlotter,
)  # Assuming DataPlotter is imported as in your provided code


class ModelAgent:
    """
    Loads and evaluates trained LSTM models on each sliding window split,
    optionally performs future forecasting.
    Assumes data is already scaled and splitted by the pipeline.
    """

    def __init__(
        self,
        ticker: str,
        data_directory: str = "app/luqman/data",
        model_directory: str = "app/luqman/models",
        future_steps: int = 100,
        max_workers: int = 1,
    ):
        """
        Args:
            ticker (str): Stock ticker symbol.
            data_directory (str): Base directory where data is stored.
            model_directory (str): Directory containing trained models.
            future_steps (int): Number of steps to forecast.
            max_workers (int): Max parallel processes for evaluation.
        """
        self.ticker = ticker
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.future_steps = future_steps
        self.max_workers = max_workers

        # Path to splits
        self.lstm_ready_dir = os.path.join(
            self.data_directory, self.ticker, "stock", "lstm_ready"
        )

        # Optional: to inverse transform predictions if needed
        self.plotter = DataPlotter(
            scaler_directory=os.path.join(
                self.data_directory, self.ticker, "stock", "scalers"
            )
        )

    def _load_split_data(self, split_name: str):
        """Loads test data arrays for a given split."""
        split_dir = os.path.join(self.lstm_ready_dir, split_name)
        X_test = np.load(os.path.join(split_dir, "X_test.npy"))
        y_test = np.load(os.path.join(split_dir, "y_test.npy"))
        timestamps_test = np.load(
            os.path.join(split_dir, "timestamps_test.npy"), allow_pickle=True
        )
        return X_test, y_test, timestamps_test

    def _load_model(self, model_path: str) -> tf.keras.Model:
        """Loads a trained model from disk."""
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
        return model

    def _evaluate(self, model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray):
        """Evaluates model predictions and calculates metrics on the scaled test data."""
        predictions_scaled = model.predict(X_test)

        # Inverse transform if needed
        actual = self.plotter.inverse_transform(y_test)
        preds = self.plotter.inverse_transform(predictions_scaled)

        # Compute metrics
        mae = np.mean(np.abs(actual - preds))
        mse = np.mean((actual - preds) ** 2)
        rmse = np.sqrt(mse)

        # Directional Accuracy
        actual_dirs = np.sign(np.diff(actual))
        pred_dirs = np.sign(np.diff(preds))
        directional_accuracy = np.mean(actual_dirs == pred_dirs) * 100

        return {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "Directional Accuracy (%)": directional_accuracy,
            "actual": actual,
            "preds": preds,
        }

    def _plot_results(
        self,
        actual: np.ndarray,
        preds: np.ndarray,
        timestamps: np.ndarray,
        split_name: str,
    ):
        """Optional method: plots actual vs. predicted close prices."""
        print(f"Plotting Actual vs. Predicted for split '{split_name}'...")
        self.plotter.plot_close_price(
            y_test_scaled=None,  # we already have actual in unscaled form, so ignoring y_test_scaled
            predictions_scaled=None,  # ignoring scaled preds
            timestamps=timestamps,
            title=f"Split '{split_name}' - Actual vs. Predicted",
            actual_unscaled=actual,  # can pass unscaled actual
            preds_unscaled=preds,  # pass unscaled preds
        )

    def _forecast(
        self,
        model: tf.keras.Model,
        X_test: np.ndarray,
        timestamps: np.ndarray,
        split_name: str,
    ):
        """Generates future predictions (forecasting) from the last test sequence."""
        last_sequence = X_test[-1].copy()
        future_preds_scaled = []
        steps = self.future_steps

        for _ in range(steps):
            # Predict next step
            next_scaled = model.predict(last_sequence[np.newaxis, ...])[0, 0]
            future_preds_scaled.append(next_scaled)
            # Shift the sequence and insert the new prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, :] = next_scaled

        # Inverse transform
        future_preds = self.plotter.inverse_transform(np.array(future_preds_scaled))

        # Generate future timestamps
        last_time = pd.to_datetime(timestamps[-1])
        future_times = [
            last_time + pd.Timedelta(minutes=5 * (i + 1)) for i in range(steps)
        ]
        future_times = np.array(future_times, dtype="datetime64[ns]")

        # Simple plotting approach
        print(f"Forecasting {steps} steps ahead for split '{split_name}'...")
        self.plotter.plot_close_price(
            y_test_scaled=None,
            predictions_scaled=None,
            timestamps=future_times,
            title=f"Forecast for split '{split_name}'",
            actual_unscaled=None,
            preds_unscaled=future_preds,
        )

    def _process_model(self, model_path: str):
        """
        Loads the model, identifies its split name, evaluates, and forecasts on that split's test data.
        """
        # Extract split name from file name
        basename = os.path.basename(model_path)
        # e.g., "AAPL_lstm_model_sliding_window_split_1.keras"
        split_name = basename.replace(f"{self.ticker}_lstm_model_", "").replace(
            ".keras", ""
        )

        # Load data for this split
        X_test, y_test, timestamps_test = self._load_split_data(split_name)
        model = self._load_model(model_path)

        # Evaluate
        metrics = self._evaluate(model, X_test, y_test)
        print(f"\nMetrics for split '{split_name}':")
        for k, v in metrics.items():
            if k not in ("actual", "preds"):  # these are arrays
                print(f"  {k}: {v:.4f}")

        # Optional plot of actual vs. predicted
        self._plot_results(
            metrics["actual"], metrics["preds"], timestamps_test, split_name
        )

        # Optional forecast
        self._forecast(model, X_test, timestamps_test, split_name)

    def run(self):
        """Finds all trained models and evaluates them in parallel."""
        # Identify all model files for this ticker
        model_files = [
            os.path.join(self.model_directory, f)
            for f in os.listdir(self.model_directory)
            if f.startswith(f"{self.ticker}_lstm_model_") and f.endswith(".keras")
        ]
        print(f"Found {len(model_files)} model(s) for ticker '{self.ticker}'.")

        # Evaluate in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._process_model, mf) for mf in model_files]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error evaluating a model: {e}")

        print("Evaluation and forecasting completed for all models.")


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    ticker = os.getenv("TICKER", "AAPL")

    agent = ModelAgent(
        ticker=ticker,
        data_directory="app/luqman/data",
        model_directory="app/luqman/models",
        future_steps=50,  # e.g., forecasting 50 steps
        max_workers=2,  # parallel processes
    )
    agent.run()
