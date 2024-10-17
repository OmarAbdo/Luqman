# scripts/feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Placeholder imports for feature classes
from app.ml.scripts.features.technical_indicators import TechnicalIndicators
from app.ml.scripts.features.sentiment_analyzer import SentimentAnalyzer
from app.ml.scripts.features.macro_economic_analyzer import MacroEconomicAnalyzer
from app.ml.scripts.features.fundamental_analysis.fundamental_analyzer import FundamentalAnalyzer
from app.ml.scripts.features.technical_sentiment_indicators import (
    TechnicalSentimentIndicators,
)


class FeatureEngineer:
    """
    A class to handle feature engineering for stock data, including adding technical indicators,
    normalizing data, preparing sequences for LSTM models, and handling outliers.

    Documentation:

    The feature engineering process began with the goal of enriching our dataset for LSTM model training by adding
    relevant indicators and features that would help capture various aspects of market behavior. Initially, we focused
    on adding technical indicators such as Moving Averages, RSI, Bollinger Bands, and MACD. These indicators provide
    insights into trends, momentum, and volatility, which are crucial for short-term price action.

    As we progressed, we realized that there are multiple types of features that could be beneficial for our model,
    each providing unique perspectives on the market and company-specific behaviors. We identified the following
    types of features:

    1. Technical Indicators: Traditional price-based indicators that help understand price trends, volatility, and
       momentum.
    2. Sentiment Analysis Feature: Derived from social media, news articles, and other public sources to gauge
       market sentiment about a stock or sector.
    3. Macro-Economic Analysis Feature: Broader economic indicators such as interest rates, inflation, and GDP,
       providing context on the overall economic environment.
    4. Fundamental Analysis Feature: Focuses on the financial health of the company, including metrics like P/E
       ratio, Debt-to-Equity, and other financial ratios.
    5. Technical Sentiment Indicators: Metrics like the Volatility Index (VIX) and Put-Call Ratio, which reflect
       overall market sentiment and expectations of future volatility.

    To ensure modularity, scalability, and clean integration, we decided to create separate classes for each type of
    feature. Each feature class will be responsible for fetching, analyzing, and normalizing its respective data, and
    the output of each will be combined into the main dataset. This approach ensures that each feature type can be
    maintained independently and reused as needed.

    The `FeatureEngineer` class will serve as the central integration point, calling on each feature class and
    adding their output to the main dataset. This allows us to add more features in the future or modify existing
    ones without affecting the overall system architecture.

    Moving forward, we also plan to implement manual feature scaling and attention mechanisms to dynamically
    emphasize specific features during model training. This will help the LSTM model better understand which
    features are most important in different market conditions.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def integrate_all_features(self):
        """
        Integrate all feature classes (technical indicators, sentiment analysis, etc.)
        """
        # Technical Indicators
        technical_indicators = TechnicalIndicators(self.data)
        self.data = technical_indicators.add_indicators()

        # Sentiment Analysis
        sentiment_analyzer = SentimentAnalyzer()
        sentiment_data = sentiment_analyzer.get_sentiment_data()
        self.data = self.data.join(sentiment_data, how="left")  # Join sentiment data

        # Macro-Economic Analysis
        macro_analyzer = MacroEconomicAnalyzer()
        macro_data = macro_analyzer.get_macro_data()
        self.data = self.data.join(macro_data, how="left")  # Join macro-economic data

        # Fundamental Analysis
        fundamental_analyzer = FundamentalAnalyzer()
        fundamental_data = fundamental_analyzer.get_fundamental_data()
        self.data = self.data.join(
            fundamental_data, how="left"
        )  # Join fundamental data

        # Technical Sentiment Indicators
        technical_sentiment = TechnicalSentimentIndicators()
        sentiment_indicators_data = technical_sentiment.get_sentiment_indicators()
        self.data = self.data.join(
            sentiment_indicators_data, how="left"
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
        self.data = pd.DataFrame(
            self.scaler.fit_transform(self.data),
            columns=self.data.columns,
            index=self.data.index,
        )
        return self

    def create_sequences(self, sequence_length=60):
        """
        Create sequences of data for LSTM training.
        """
        sequences = []
        targets = []
        data_array = self.data.values

        for i in range(len(data_array) - sequence_length):
            sequences.append(data_array[i : i + sequence_length])
            targets.append(
                data_array[i + sequence_length, 3]
            )  # Assuming 'Close' is the 4th column

        return np.array(sequences), np.array(targets)


if __name__ == "__main__":
    # Example usage
    data = pd.read_csv(
        "app/ml/data/AAPL/AAPL_1d_5y.csv", index_col="Date", parse_dates=True
    )
    feature_engineer = FeatureEngineer(data)
    feature_engineer.integrate_all_features().handle_outliers().handle_missing_values().normalize_data()
    sequences, targets = feature_engineer.create_sequences()
    print(f"Sequences shape: {sequences.shape}, Targets shape: {targets.shape}")
