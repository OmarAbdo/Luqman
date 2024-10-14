# scripts/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def add_technical_indicators(df):
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["RSI"] = compute_rsi(df["Close"], 14)  # You will define `compute_rsi()`
    # Add more indicators as needed...
    return df


def normalize_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    return scaled_data, scaler


def create_sequences(data, time_steps=60):
    x, y = [], []
    for i in range(len(data) - time_steps):
        x.append(data[i : (i + time_steps), :])
        y.append(data[i + time_steps, 3])  # Assuming column index 3 is 'Close'
    return np.array(x), np.array(y)


if __name__ == "__main__":
    df = pd.read_csv("data/aapl_hourly_data.csv")
    df = add_technical_indicators(df)
    data, scaler = normalize_data(
        df[["Open", "High", "Low", "Close", "Volume", "MA_10", "RSI"]]
    )
    x, y = create_sequences(data)
