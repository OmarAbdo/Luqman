# app/ml/main.py

"""
main.py

Orchestrates the data fetching, merging, feature engineering, data transformation,
and AI modeling pipeline in an object-oriented manner.

Phases:
-------
1. Data Preparation
2. Data Transformation
3. AI Modeling

Usage:
------
1. Ensure all necessary classes are implemented and importable.
2. Configure the required environment variables in the .env file.
3. Run the script:
       python app/ml/main.py
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Importing custom classes from their respective modules
from scripts.components.stock_price.alpha_vantage import AlphaVantageIntradayFetcher
from scripts.components.technical_indicators.technical_indicators import (
    TechnicalIndicators,
)
from scripts.components.technical_sentiment_indicators.technical_sentiment_indicators import (
    TechnicalSentimentIndicators,
)
from scripts.utils.merger import CSVDataMerger
from scripts.pipeline.data_pipeline import DataPipeline
from scripts.neural_networks.lstm.lstm_model_trainer import LSTMModelTrainer
from scripts.neural_networks.lstm.lstm_model_agent import ModelAgent


class Main:
    """
    Manages the execution of the data processing and modeling pipeline in an OOP style.
    """

    def __init__(self, ticker: str, config: dict):
        """
        Initializes the Main with the specified ticker and configuration.

        Args:
            ticker (str): Stock ticker symbol, e.g., "AAPL".
            config (dict): Configuration dictionary containing paths and parameters.
        """
        load_dotenv()  # Load environment variables from .env file
        self.ticker = ticker
        os.environ["TICKER"] = ticker  # Set the ticker environment variable

        # Define base directories
        self.base_dir = (
            Path(config.get("base_dir", "app/ml/data")) / self.ticker / "stock"
        )
        self.chunks_dir = self.base_dir / "chunks"
        self.models_dir = Path(config.get("models_dir", "app/ml/models")) / self.ticker

        # Ensure necessary directories exist
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components with parameters from config
        self.fetcher = AlphaVantageIntradayFetcher(
            ticker=self.ticker,
            interval=config.get("interval", "5min"),
            adjusted=config.get("adjusted", True),
            extended_hours=config.get("extended_hours", True),
            start_yearmonth=config.get("start_yearmonth"),
            end_yearmonth=config.get("end_yearmonth"),
            output_path=str(self.base_dir),
            api_key=os.getenv("ALPHA_VANTAGE_API"),
        )

        self.merger = CSVDataMerger(
            ticker=self.ticker,
            interval=config.get("interval", "5min"),
            base_dir=config.get("base_dir", "app/ml/data"),
        )

        self.tech_indicators = TechnicalIndicators(ticker=self.ticker)
        self.tech_sentiment_indicators = TechnicalSentimentIndicators(
            ticker=self.ticker
        )
        self.data_pipeline = DataPipeline(
            ticker=self.ticker,
            sequence_length=config.get("sequence_length"),  # default is 60
            test_size=config.get("test_size"),  # 80% train, 20% test
            sample_rate=config.get("sample_rate"),
        )

        self.model_trainer = LSTMModelTrainer(
            ticker=self.ticker,
            data_directory=config.get("base_dir"),
            model_directory=config.get("models_dir"),
            units=config.get("lstm_units"),
            dropout=config.get("dropout"),
            learning_rate=config.get("learning_rate"),
            epochs=config.get("epochs"),
            batch_size=config.get("batch_size"),
            validation_split=config.get("validation_split"),
            patience=config.get("patience"),
        )

        self.model_agent = ModelAgent(
            ticker=self.ticker,
            sequence_length=config.get("sequence_length"),
            test_size=config.get("test_size"),
            sample_rate=config.get("sample_rate"),
        )

    def run_phase1(self):
        """
        Executes Phase 1: Data Preparation
        """
        print("----- Phase 1: Data Preparation -----")
        print("1. Fetching Data...")
        self.fetcher.fetch_all_months()

        print("2. Merging Data...")
        self.merger.merge_chunks()

        print("3. Adding Technical Indicators...")
        self.tech_indicators().run()

        print("4. Adding Technical Sentiment Indicators...")
        self.tech_sentiment_indicators().run()

        print("Phase 1 Completed.\n")

    def run_phase2(self):
        """
        Executes Phase 2: Data Transformation
        """
        print("----- Phase 2: Data Transformation -----")
        print("1. Running Data Pipeline to Prepare LSTM Data...")
        self.data_pipeline.run()

        print("Phase 2 Completed.\n")

    def run_phase3(self):
        """
        Executes Phase 3: AI Modeling
        """
        print("----- Phase 3: AI Modeling -----")
        print("1. Training LSTM Model...")
        self.model_trainer.run()

        print("2. Running Model Agent...")
        self.model_agent.run()

        print("Phase 3 Completed.\n")

    def run_full_pipeline(self):
        """
        Runs the entire pipeline: Phase 1, Phase 2, and Phase 3.
        """
        self.run_phase1()
        self.run_phase2()
        self.run_phase3()
        print("----- Full Pipeline Execution Completed -----")


if __name__ == "__main__":
    # Load the ticker from the environment variable or set it directly here
    load_dotenv()  # Ensure this is called to load environment variables
    ticker = os.getenv("TICKER", "AAPL")  # Default to AAPL if not set

    # Configuration dictionary for pipeline parameters
    config = {
        "base_dir": "app/ml/data",  # Base directory where data is stored
        "models_dir": "app/ml/models",  # Directory to save trained models
        "interval": "5min",  # Data interval (e.g., '1min', '5min', etc.)
        "adjusted": True,  # Adjust for splits/dividends
        "extended_hours": True,  # Include extended hours data
        "start_yearmonth": "2014-07",  # Start month (e.g., '2014-07')
        "end_yearmonth": "2016-07",  # End month (e.g., '2016-07')
        "sequence_length": 90,  # Sequence length for LSTM
        "test_size": 0.2,  # Test size for train-test split
        "sample_rate": 1.0,  # Sampling rate (1.0 means use full dataset)
        "lstm_units": 256,  # Number of LSTM units
        "dropout": 0.3,  # Dropout rate
        "learning_rate": 0.001,  # Learning rate for optimizer
        "epochs": 20,  # Number of training epochs
        "batch_size": 64,  # Batch size for training
        "validation_split": 0.1,  # Validation split for training
        "patience": 3,  # Early stopping patience
    }

    # Create an instance of Main and run the full pipeline
    pipeline_manager = Main(ticker=ticker, config=config)
    pipeline_manager.run_phase1()
    # pipeline_manager.run_phase2()
    # pipeline_manager.run_phase3()
    # pipeline_manager.run_full_pipeline()
