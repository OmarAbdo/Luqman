import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Input,
    Bidirectional,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class LSTMModelTrainer:
    def __init__(self, ticker, sample_fraction=1.0):
        self.ticker = ticker
        self.X_train_file = f"app/ml/data/{ticker}/stock/lstm_ready/X.npy"
        self.y_train_file = f"app/ml/data/{ticker}/stock/lstm_ready/y.npy"
        self.X_test_file = f"app/ml/data/{ticker}/stock/lstm_ready/X.npy"
        self.y_test_file = f"app/ml/data/{ticker}/stock/lstm_ready/y.npy"
        self.sample_fraction = sample_fraction

        self.X_train = np.load(self.X_train_file)
        self.y_train = np.load(self.y_train_file)
        self.X_test = np.load(self.X_test_file)
        self.y_test = np.load(self.y_test_file)

        if self.sample_fraction < 1.0:
            train_size = int(len(self.X_train) * self.sample_fraction)
            test_size = int(len(self.X_test) * self.sample_fraction)
            self.X_train = self.X_train[:train_size]
            self.y_train = self.y_train[:train_size]
            self.X_test = self.X_test[:test_size]
            self.y_test = self.y_test[:test_size]

        self.handle_missing_values()
        self.model = None
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.save_model()
        self.calculate_accuracies()
        self.predict_and_plot()

    def handle_missing_values(self):
        # Replace NaNs in the dataset using forward fill
        self.X_train = np.nan_to_num(self.X_train, nan=np.nanmean(self.X_train))
        self.X_test = np.nan_to_num(self.X_test, nan=np.nanmean(self.X_test))

    # The commented hyperparameters are (reducing) the quality of the prediction.
    # probably they could be more useful with future bigger and more complex datasets
    def build_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(
            Bidirectional(
                LSTM(
                    100,
                    activation="tanh",
                    # return_sequences=True,
                    # kernel_regularizer=l2(0.001),
                )
            )
        )
        # self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        # self.model.add(
        #     Bidirectional(
        #         LSTM(
        #             300,
        #             activation="tanh",
        #             return_sequences=False,
        #         )
        #     )
        # )
        # self.model.add(BatchNormalization())
        # self.model.add(Dropout(0.2))

        self.model.add(Dense(1))
        self.model.compile(optimizer="adam", loss="mse")

    def train_model(self, epochs=200, batch_size=8):
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test, self.y_test),
            verbose=1,
        )

    def evaluate_model(self):
        loss = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        print(f"Test Loss: {loss}")

    def save_model(self, model_path=None):
        if model_path is None:
            model_path = f"app/ml/models/{self.ticker}_lstm_model.keras"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def calculate_accuracies(self):
        predictions = self.model.predict(self.X_test)

        mse = np.mean((self.y_test.flatten() - predictions.flatten()) ** 2)
        print(f"Mean Squared Error: {mse:.2f}")

        actual_directions = np.diff(self.y_test.flatten()) > 0
        predicted_directions = np.diff(predictions.flatten()) > 0

        correct_directions = np.sum(actual_directions == predicted_directions)
        directional_accuracy = correct_directions / len(actual_directions) * 100

        print(f"Directional Accuracy: {directional_accuracy:.2f}%")

    def predict_and_plot(self):
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


if __name__ == "__main__":
    ticker = os.getenv("TICKER")
    trainer = LSTMModelTrainer(ticker, sample_fraction=1.0)
