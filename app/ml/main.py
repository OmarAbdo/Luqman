import os
import subprocess
from dotenv import load_dotenv


class Luqman:
    def __init__(self, ticker):
        load_dotenv()  # Load environment variables from .env file
        self.ticker = ticker
        os.environ["TICKER"] = ticker  # Set the ticker environment variable

    def run_script(self, script_path):
        try:
            subprocess.run(["python", "-u", script_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing {script_path}: {e}")

    def fetch_data(self):
        print(f"Fetching data for {self.ticker}...")
        self.run_script("app/ml/scripts/features/stock_price/fmp.py")

    def add_technical_indicators(self):
        print("Adding technical indicators...")
        self.run_script(
            "app/ml/scripts/features/technical_indicators/technical_indicators.py"
        )
        print("Adding technical sentiment indicators...")
        self.run_script(
            "app/ml/scripts/features/technical_sentiment_indicators/technical_sentiment_indicators.py"
        )

    def standardize_data(self):
        print("Standardizing data...")
        self.run_script("app/ml/scripts/feature_engineering/data_standardizer.py")

    def prepare_lstm_data(self):
        print("Preparing LSTM sequences...")
        self.run_script("app/ml/scripts/neural_networks/lstm/lstm_data_prep.py")

    def train_lstm_model(self):
        print("Training LSTM model...")
        self.run_script("app/ml/scripts/neural_networks/lstm/lstm_model_trainer.py")

    def run_pipeline(self):
        self.fetch_data()
        self.add_technical_indicators()
        self.standardize_data()
        self.prepare_lstm_data()
        self.train_lstm_model()


if __name__ == "__main__":
    # Load the ticker from the environment file or set it directly here
    load_dotenv()  # Load environment variables from .env file
    ticker = os.getenv("TICKER", "AAPL")  # Default to AAPL if not set

    # Create an instance of Luqman and run the full pipeline
    luqman = Luqman(ticker)
    luqman.run_pipeline()
