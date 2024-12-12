import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


class LSTMModelTrainer:
    """
    Class responsible for building, training, evaluating and saving the LSTM model.
    """

    def __init__(
        self,
        ticker: str,
        data_directory: str = "app/ml/data",
        model_directory: str = "app/ml/models",
        model_name_template: str = "{ticker}_lstm_model.keras",
        units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        patience: int = 5,
    ):
        """
        Initializes the LSTMModelTrainer with hyperparameters and paths.

        Args:
            ticker (str): The stock ticker symbol.
            data_directory (str): Path to the directory containing the data.
            model_directory (str): Path to the directory where the model will be saved.
            model_name_template (str): Template for the model filename.
            units (int): Number of LSTM units.
            dropout (float): Dropout rate.
            learning_rate (float): Learning rate for the optimizer.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Batch size for training.
            validation_split (float): Fraction of the training data to be used as validation.
            patience (int): Patience for early stopping.
        """
        self.ticker = ticker
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.model_name_template = model_name_template.format(ticker=self.ticker)
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience

        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.timestamps_train = None
        self.timestamps_test = None

        self.lstm_ready_dir = os.path.join(
            self.data_directory, self.ticker, "stock", "lstm_ready"
        )
        self.model_path = os.path.join(self.model_directory, self.model_name_template)

        os.makedirs(self.model_directory, exist_ok=True)

    def load_data(self):
        """
        Loads the training and testing data from disk.
        """
        self.X_train = np.load(os.path.join(self.lstm_ready_dir, "X_train.npy"))
        self.X_test = np.load(os.path.join(self.lstm_ready_dir, "X_test.npy"))
        self.y_train = np.load(os.path.join(self.lstm_ready_dir, "y_train.npy"))
        self.y_test = np.load(os.path.join(self.lstm_ready_dir, "y_test.npy"))

        # Since timestamps are object arrays (datetime objects), we need allow_pickle=True
        self.timestamps_train = np.load(
            os.path.join(self.lstm_ready_dir, "timestamps_train.npy"), allow_pickle=True
        )
        self.timestamps_test = np.load(
            os.path.join(self.lstm_ready_dir, "timestamps_test.npy"), allow_pickle=True
        )

        print("Data loaded:")
        print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

    def build_model(self):
        """
        Builds the LSTM model using the specified hyperparameters.
        """
        model = Sequential()
        model.add(
            LSTM(
                self.units,
                input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                return_sequences=True,
            )
        )
        model.add(Dropout(self.dropout))
        model.add(LSTM(self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(1))  # Predicting a single value (close price)

        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])
        self.model = model
        print("Model built successfully.")
        print(self.model.summary())

    def train_model(self):
        """
        Trains the LSTM model with early stopping and model checkpointing.
        """
        if self.model is None:
            raise ValueError(
                "Model is not built. Call build_model() before train_model()."
            )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True
        )
        checkpoint_path = os.path.join(
            self.model_directory, f"{self.ticker}_best_model.keras"
        )
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )

        print("Starting training...")
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1,
        )
        print("Training completed.")
        return history

    def evaluate_model(self):
        """
        Evaluates the trained model on the test data.
        """
        if self.model is None:
            raise ValueError("Model is not built or loaded. Cannot evaluate.")

        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")
        return loss, mae

    def save_model(self):
        """
        Saves the trained model to disk.
        """
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def run(self):
        """
        Orchestrates the entire training process: load data, build model, train, evaluate, and save.
        """
        self.load_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()


if __name__ == "__main__":
    # Example usage:
    # Set the environment variable TICKER or pass it directly as a parameter.
    # E.g., ticker = "AAPL"
    ticker = os.getenv(
        "TICKER", "AAPL"
    )  # For demonstration, defaulting to AAPL if not set.

    trainer = LSTMModelTrainer(
        ticker=ticker,
        data_directory="app/ml/data",
        model_directory="app/ml/models",
        units=256,
        dropout=0.3,
        learning_rate=0.001,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        patience=3,
    )

    trainer.run()
