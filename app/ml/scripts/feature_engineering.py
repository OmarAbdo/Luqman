# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

# Add the root directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from features.technical_indicators.technical_indicators import TechnicalIndicators
from features.sentiment_score.sentiment_score import SentimentScore
from features.macro_economic_analysis.macro_economic_analysis import MacroeconomicDataFetcher
from features.fundamental_analysis.fundamental_analysis import FundamentalAnalysis
from features.technical_sentiment_indicators.technical_sentiment_indicators import (
    TechnicalSentimentIndicators,
)

# [TODO] Test the merge of each feature class into the main dataset separately. this is completely unreliable 

class FeatureEngineer:
    """
    A class to handle feature engineering for stock data, including adding technical indicators,
    normalizing data, preparing sequences for AI models, and handling outliers.

    The `FeatureEngineer` class is responsible for:
    - Integrating various data features from multiple analysis classes (technical, sentiment, macroeconomic, etc.).
    - Handling outliers and missing values in the dataset.
    - Normalizing data for model readiness.
    - Creating sequences for time-series models like LSTMs.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def integrate_all_features(self):
        """
        Integrate all feature classes (technical indicators, sentiment analysis, etc.).
        """
        # Technical Indicators
        technical_indicators = TechnicalIndicators(self.data)
        self.data = technical_indicators.add_indicators()

        # Sentiment Analysis
        sentiment_analyzer = SentimentScore("AOOE7AD9CPPFTQHH")  # Replace with actual API key
        sentiment_data = sentiment_analyzer.get_sentiment_score("AAPL")
        sentiment_df = pd.DataFrame([sentiment_data]).set_index("timestamp")
        self.data = self.data.join(
            sentiment_df, how="left", rsuffix="_sentiment"
        )  # Join sentiment data

        # Macro-Economic Analysis
        macro_analyzer = MacroeconomicDataFetcher()
        all_data = []
        country_code = "US"
        start_year = "2010"
        end_year = "2023"
        for indicator, description in MacroeconomicDataFetcher.INDICATORS.items():
            print(f"Fetching data for {description} ({indicator})...")
            data = macro_analyzer.fetch_world_bank_data(
                country_code, indicator, start_year, end_year
            )
            if not data.empty:
                all_data.append(data)
        macro_raw_data = (
            pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        )
        macro_data = macro_analyzer.prepare_features(
            macro_analyzer.calculate_derived_metrics(macro_raw_data)
        )
        self.data = self.data.join(
            macro_data, how="left", rsuffix="_macro"
        )  # Join macro-economic data

        # Fundamental Analysis
        fundamental_analyzer = FundamentalAnalysis()
        fundamental_data = fundamental_analyzer.perform_analysis()
        self.data = self.data.join(
            fundamental_data, how="left", rsuffix="_fundamental"
        )  # Join fundamental data

        # Technical Sentiment Indicators
        technical_sentiment = TechnicalSentimentIndicators(self.data)
        sentiment_indicators_data = technical_sentiment.execute_analysis()
        self.data = self.data.join(
            sentiment_indicators_data, how="left", rsuffix="_technical_sentiment"
        )  # Join technical sentiment data

        return self

    def handle_outliers(self, z_threshold=3):
        """
        Handle outliers by capping values beyond a certain Z-score threshold.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            z_scores = (self.data[col] - mean) / std
            self.data[col] = np.where(
                z_scores > z_threshold,
                mean + z_threshold * std,
                np.where(
                    z_scores < -z_threshold, mean - z_threshold * std, self.data[col]
                ),
            )
        return self

    def handle_missing_values(self):
        """
        Handle missing values by forward-filling.
        """
        self.data.fillna(method="ffill", inplace=True)
        self.data.fillna(method="bfill", inplace=True)
        return self

    def normalize_data(self):
        """
        Normalize the data using Min-Max scaling.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        non_numeric_data = self.data.drop(columns=numeric_cols)

        scaled_numeric_data = pd.DataFrame(
            self.scaler.fit_transform(self.data[numeric_cols]),
            columns=numeric_cols,
            index=self.data.index,
        )

        self.data = pd.concat([scaled_numeric_data, non_numeric_data], axis=1)
        return self

    def create_sequences(self, sequence_length=60):
        """
        Create sequences of data for LSTM training and save them to CSV files.
        """
        sequences = []
        targets = []
        data_array = self.data.values

        for i in range(len(data_array) - sequence_length):
            sequences.append(data_array[i : i + sequence_length])
            targets.append(
                data_array[i + sequence_length, 3]
            )  # Assuming 'Close' is the 4th column

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Save sequences and targets to CSV files in the training data folder
        os.makedirs("app/ml/training_data", exist_ok=True)
        sequences_flattened = [
            seq.flatten() for seq in sequences
        ]  # Flatten each sequence for saving in CSV format
        sequences_df = pd.DataFrame(sequences_flattened)
        sequences_df.to_csv("app/ml/training_data/lstm_sequences.csv", index=False)

        targets_df = pd.DataFrame(targets, columns=["Target"])
        targets_df.to_csv("app/ml/training_data/lstm_targets.csv", index=False)

        return sequences, targets


if __name__ == "__main__":
    # Example usage
    data = pd.read_csv(
        "app/ml/data/AAPL/stock/AAPL_1d_5y.csv", index_col="Date", parse_dates=True
    )
    feature_engineer = FeatureEngineer(data)
    feature_engineer.integrate_all_features().handle_outliers().handle_missing_values().normalize_data()
    sequences, targets = feature_engineer.create_sequences()
    print(f"Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
