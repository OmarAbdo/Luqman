# File: LSTMModelTrainer.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Conv1D,
    MaxPooling1D,
    Input,
    Attention,
    GlobalAveragePooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from concurrent.futures import ProcessPoolExecutor, as_completed


class LSTMModelTrainer:
    """
    Trains an LSTM model on each sliding window split without needing extra
    data preprocessing or sequence preparation. Assumes X_train, y_train, X_test, and y_test
    are already saved in each split directory.
    """

    def __init__(
        self,
        ticker: str,
        data_directory: str = "app/luqman/data",
        model_directory: str = "app/luqman/models",
        units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32,
        validation_split: float = 0.1,
        patience: int = 3,
        max_workers: int = 1,  # Number of parallel processes
    ):
        """
        Args:
            ticker (str): Stock ticker symbol.
            data_directory (str): Base directory where data is stored.
            model_directory (str): Directory to save trained models.
            units (int): Number of LSTM units.
            dropout (float): Dropout rate.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of training data for validation.
            patience (int): Patience for early stopping.
            max_workers (int): Max parallel processes for split processing.
        """
        self.ticker = ticker
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.max_workers = max_workers

        # Path to where split subdirectories (sliding_window_split_1, etc.) reside
        self.lstm_ready_dir = os.path.join(
            self.data_directory, self.ticker, "stock", "lstm_ready"
        )
        os.makedirs(self.model_directory, exist_ok=True)

    def _load_split_data(self, split_dir: str):
        """Loads training/testing arrays from a given split directory."""
        X_train = np.load(os.path.join(split_dir, "X_train.npy"))
        y_train = np.load(os.path.join(split_dir, "y_train.npy"))
        X_test = np.load(os.path.join(split_dir, "X_test.npy"))
        y_test = np.load(os.path.join(split_dir, "y_test.npy"))

        split_name = os.path.basename(split_dir)
        print(f"Loaded data for '{split_name}':")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")
        return X_train, y_train, X_test, y_test, split_name

    def _build_model(self, input_shape):
        """Builds a CNN+LSTM+Attention model given the input shape."""
        inputs = Input(shape=input_shape)

        # CNN block
        x = Conv1D(
            filters=32,
            kernel_size=3,
            padding="causal",
            activation="relu",
            kernel_regularizer=l2(0.001),
        )(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Dropout(self.dropout)(x)

        # LSTM stack
        x = LSTM(self.units, return_sequences=True, kernel_regularizer=l2(0.001))(x)
        x = LSTM(self.units // 2, return_sequences=True, kernel_regularizer=l2(0.001))(
            x
        )
        x = LSTM(self.units // 4, return_sequences=True, kernel_regularizer=l2(0.001))(
            x
        )

        # Attention
        x_att = Attention()([x, x])

        # Global Average Pooling + Dropout + Dense output
        x = GlobalAveragePooling1D()(x_att)
        x = Dropout(self.dropout)(x)
        outputs = Dense(1, kernel_regularizer=l2(0.001), activation="linear")(x)

        model = Model(inputs, outputs)
        model.compile(
            loss="mean_squared_error",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["mae"],
        )
        return model

    def _train_single_split(self, split_dir: str):
        """Loads data from one split, trains, and saves the best model."""
        X_train, y_train, X_test, y_test, split_name = self._load_split_data(split_dir)
        model = self._build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        # Define callbacks
        checkpoint_path = os.path.join(
            self.model_directory, f"{self.ticker}_lstm_model_{split_name}.keras"
        )
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1,
            ),
        ]

        print(f"Training on split '{split_name}'...")
        model.fit(
            X_train,
            y_train,
            validation_split=self.validation_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        print(
            f"Training complete for '{split_name}'. Model saved as '{checkpoint_path}'."
        )

    def run(self):
        """Trains an LSTM model for each sliding window split in parallel."""
        split_dirs = [
            os.path.join(self.lstm_ready_dir, d)
            for d in os.listdir(self.lstm_ready_dir)
            if os.path.isdir(os.path.join(self.lstm_ready_dir, d))
        ]

        print(
            f"Found {len(split_dirs)} split(s) in '{self.lstm_ready_dir}' to train on."
        )

        # Process each split in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._train_single_split, s_dir) for s_dir in split_dirs
            ]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error training on a split: {e}")

        print("All splits have been processed. Training complete.")


if __name__ == "__main__":
    # Example usage
    import dotenv

    dotenv.load_dotenv()  # loads environment variables
    ticker = os.getenv("TICKER", "AAPL")

    trainer = LSTMModelTrainer(
        ticker=ticker,
        data_directory="app/luqman/data",
        model_directory="app/luqman/models",
        units=256,
        dropout=0.3,
        learning_rate=0.0005,
        epochs=2,
        batch_size=128,
        validation_split=0.1,
        patience=3,
        max_workers=2,  # Adjust based on CPU cores
    )
    trainer.run()
