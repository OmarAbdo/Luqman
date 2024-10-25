# scripts/train_model.py
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from lstm_model import build_lstm_model
from feature_engineering import create_sequences


def train_model(x, y):
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, shuffle=False
    )
    model = build_lstm_model((x_train.shape[1], x_train.shape[2]))
    checkpoint = ModelCheckpoint(
        "models/best_model.h5", save_best_only=True, monitor="val_loss", mode="min"
    )
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint],
    )
    return model


if __name__ == "__main__":
    # Load sequences created earlier
    x, y = create_sequences(data)
    model = train_model(x, y)
