# scripts/lstm_model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Predicting the next 'Close' price
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


if __name__ == "__main__":
    input_shape = (60, 7)  # Example input shape: 60 time steps, 7 features
    model = build_lstm_model(input_shape)
    model.summary()
