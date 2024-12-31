# scripts/tests/test_technical_indicators.py
import unittest
import pandas as pd
from app.ml.scripts.components.technical_indicators.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """
    Unit tests for the TechnicalIndicators class.
    """

    @classmethod
    def setUpClass(cls):
        # Sample data to test
        data = {
            "Date": pd.date_range(start="2023-01-01", periods=50, freq="D"),
            "Open": [100 + i for i in range(50)],
            "High": [105 + i for i in range(50)],
            "Low": [95 + i for i in range(50)],
            "Close": [100 + i for i in range(50)],
            "Volume": [1000 + 10 * i for i in range(50)],
        }
        cls.df = pd.DataFrame(data).set_index("Date")

    def test_add_indicators(self):
        """
        Test the addition of technical indicators to the dataframe.
        """
        technical_indicators = TechnicalIndicators(self.df)
        df_with_indicators = technical_indicators.add_indicators()

        # Check if the expected columns are present
        expected_columns = [
            "MA_10",
            "MA_50",
            "RSI",
            "BB_MA",
            "BB_std",
            "BB_upper",
            "BB_lower",
            "EMA_12",
            "EMA_26",
            "MACD",
            "Signal_Line",
        ]
        for col in expected_columns:
            self.assertIn(
                col, df_with_indicators.columns, f"{col} should be in the dataframe."
            )

        # Check if values are not NaN (after initial rows where calculations are not possible)
        non_nan_df = df_with_indicators.dropna()
        self.assertFalse(
            non_nan_df.isnull().values.any(),
            "Dataframe contains NaN values after indicator calculation.",
        )


if __name__ == "__main__":
    unittest.main()
