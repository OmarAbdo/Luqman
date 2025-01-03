import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D,
    LSTM,
    Dense,
    Dropout,
    Input,
    Attention,
    GlobalAveragePooling1D,
    LayerNormalization,
    Add,
    Concatenate,
    MultiHeadAttention,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2


class AdvancedModelTrainer:
    def __init__(
        self,
        ticker: str,
        data_directory: str = "app/luqman/data",
        model_directory: str = "app/luqman/models",
        model_name_template: str = "{ticker}_advanced_model.keras",
        units: int = 192,
        cnn_filters: int = 128,
        dilation_rate: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 0.005,
        epochs: int = 64,
        batch_size: int = 32,
        validation_split: float = 0.15,
        patience: int = 5,
        l2_reg: float = 1e-5,
    ):
        self.ticker = ticker
        self.data_directory = data_directory
        self.model_directory = model_directory
        self.model_name_template = model_name_template.format(ticker=self.ticker)
        self.units = units
        self.cnn_filters = cnn_filters
        self.dilation_rate = dilation_rate
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.l2_reg = l2_reg

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
        self.X_train = np.load(os.path.join(self.lstm_ready_dir, "X_train.npy"))
        self.y_train = np.load(os.path.join(self.lstm_ready_dir, "y_train.npy"))
        self.X_test = np.load(os.path.join(self.lstm_ready_dir, "X_test.npy"))
        self.y_test = np.load(os.path.join(self.lstm_ready_dir, "y_test.npy"))

        self.timestamps_train = np.load(
            os.path.join(self.lstm_ready_dir, "timestamps_train.npy"), allow_pickle=True
        )
        self.timestamps_test = np.load(
            os.path.join(self.lstm_ready_dir, "timestamps_test.npy"), allow_pickle=True
        )

        print("Data loaded successfully.")
        print(
            f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}"
        )
        print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")

    def build_model(self):
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        input_seq = Input(shape=input_shape, name="input_sequence")

        # -------------------------
        # Path A: Dilated CNN block
        # -------------------------
        x_a = Conv1D(
            filters=self.cnn_filters,
            kernel_size=3,
            padding="causal",
            dilation_rate=self.dilation_rate,
            activation="relu",
            kernel_regularizer=l2(self.l2_reg),
            name="dilated_conv_1",
        )(input_seq)

        x_a = Conv1D(
            filters=self.cnn_filters * 2,  # Increased filters for second layer
            kernel_size=3,
            padding="causal",
            dilation_rate=self.dilation_rate,
            activation="relu",
            kernel_regularizer=l2(self.l2_reg),
            name="dilated_conv_2",
        )(x_a)

        x_a = Dropout(self.dropout, name="dropout_a")(x_a)
        # Removed GlobalAveragePooling1D to retain temporal dimension
        # x_a remains with shape (None, 60, cnn_filters * 2)  # Add pooling

        # -------------------------
        # Path B: LSTM block
        # -------------------------
        x_b = LSTM(
            self.units,
            return_sequences=True,
            kernel_regularizer=l2(self.l2_reg),
            name="lstm_1",
        )(input_seq)
        x_b = Dropout(self.dropout, name="dropout_b1")(x_b)

        x_b = LSTM(
            int(self.units / 2),
            return_sequences=True,
            kernel_regularizer=l2(self.l2_reg),
            name="lstm_2",
        )(x_b)
        x_b = Dropout(self.dropout, name="dropout_b2")(x_b)

        # -------------------------
        # Merge Paths
        # -------------------------
        merged = Concatenate(name="merge_paths", axis=-1)(
            [x_a, x_b]
        )  # Both x_a and x_b retain temporal dimension

        # --------------
        # Multi-Head Attention
        # --------------
        att_output = MultiHeadAttention(
            num_heads=4,
            key_dim=self.cnn_filters + self.units // 2,
            name="multihead_attention",
        )(merged, merged)

        # ---------------
        # Residual Add
        # ---------------
        res_out = Add(name="residual_add")([att_output, merged])
        norm_out = LayerNormalization(name="layer_norm")(res_out)

        # ---------------
        # Dense Layers
        # ---------------
        gp_out = GlobalAveragePooling1D(name="global_avg_pool")(norm_out)
        gp_out = Dense(
            128, activation="relu", kernel_regularizer=l2(self.l2_reg), name="dense_1"
        )(gp_out)
        gp_out = Dropout(self.dropout, name="dropout_dense")(gp_out)
        output = Dense(
            units=1,
            activation="linear",
            kernel_regularizer=l2(self.l2_reg),
            name="output_dense",
        )(gp_out)

        # Define and compile model
        self.model = Model(
            inputs=input_seq, outputs=output, name="DualPath_LSTM_CNN_Attention"
        )
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            loss="mean_squared_error", optimizer=optimizer, metrics=["mae"]
        )

        print("Advanced model built successfully.")
        self.model.summary()

    def train_model(self):
        if self.model is None:
            raise ValueError(
                "Model not built. Call build_model() before train_model()."
            )

        # ------------------
        # Warm-Up Training
        # ------------------
        # Freeze all layers except the dense layers
        for layer in self.model.layers[:-2]:
            layer.trainable = False

        # Compile the model for warm-up
        self.model.compile(
            loss="mse", optimizer=Adam(learning_rate=0.01), metrics=["mae"]
        )

        # Train only the dense layers
        print("Training dense layers (warm-up)...")
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=3,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            verbose=1,
        )

        # Unfreeze all layers
        for layer in self.model.layers:
            layer.trainable = True

        # Re-compile the model for fine-tuning
        self.model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=["mae"],
        )

        # Callbacks
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
            verbose=1,
        )
        checkpoint_path = os.path.join(
            self.model_directory, f"{self.ticker}_best_advanced_model.keras"
        )
        model_checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1,
        )

        # Fine-tune the full model
        print("Fine-tuning the full model...")
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
        if self.model is None:
            raise ValueError("Model not built or loaded. Cannot evaluate.")

        loss, mae = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        y_pred = self.model.predict(self.X_test, verbose=0).flatten()

        if len(self.y_test) < 2:
            print("Not enough data points to compute Directional Accuracy.")
            directional_accuracy = np.nan
        else:
            direction_true = np.sign(self.y_test[1:] - self.y_test[:-1])
            direction_pred = np.sign(y_pred[1:] - self.y_test[:-1])

            direction_true = np.where(direction_true == 0, 0, direction_true)
            direction_pred = np.where(direction_pred == 0, 0, direction_pred)

            correct_directions = direction_true == direction_pred
            directional_accuracy = np.mean(correct_directions) * 100

        print(
            f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}, Directional Accuracy: {directional_accuracy:.2f}%"
        )
        return loss, mae, directional_accuracy

    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save.")
        self.model.save(self.model_path)
        print(f"Model saved to {self.model_path}")

    def run(self):
        self.load_data()
        self.build_model()
        history = self.train_model()
        self.evaluate_model()
        self.save_model()
        return history


if __name__ == "__main__":
    import dotenv

    dotenv.load_dotenv()
    ticker = os.getenv("TICKER", "AAPL")

    trainer = AdvancedModelTrainer(
        ticker=ticker,
        data_directory="app/luqman/data",
        model_directory="app/luqman/models",
        model_name_template="{ticker}_advanced_model.keras",
        units=32,
        cnn_filters=16,
        dilation_rate=2,
        dropout=0.1,
        learning_rate=0.005,
        epochs=3,
        batch_size=128,
        validation_split=0.15,
        patience=5,
        l2_reg=1e-5,
    )
    trainer.run()
