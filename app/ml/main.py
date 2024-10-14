# main.py
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


def load_scaler():
    # Load scaler to transform input data (consider saving during training)
    pass


def make_prediction(model, recent_data):
    prediction = model.predict(np.expand_dims(recent_data, axis=0))
    return prediction


if __name__ == "__main__":
    # Load trained model and scaler
    model = load_model("models/best_model.h5")
    # Assuming recent_data has been loaded and pre-processed
    predicted_price = make_prediction(model, recent_data)
    print(f"Predicted next closing price: {predicted_price}")
