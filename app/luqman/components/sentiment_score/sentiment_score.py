import requests
import os
import csv
from datetime import datetime
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


class SentimentScore:
    """
    A class to gather sentiment analysis data using Alpha Vantage's sentiment analysis service.
    """

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_sentiment_score(self, company_ticker):
        """
        Get sentiment score for a specific company.

        :param company_ticker: The name of the company to get sentiment for.
        :return: A dictionary with sentiment score details.
        """
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.api_key,
            "tickers": company_ticker,
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract relevant sentiment score data
            sentiment_score = data.get("feed", [])
            if not sentiment_score:
                print("[WARNING] No sentiment data found for the given company.")
                return {}

            # Process and extract average sentiment score from all available articles
            scores = [
                article["overall_sentiment_score"]
                for article in sentiment_score
                if "overall_sentiment_score" in article
            ]
            if scores:
                avg_sentiment_score = sum(scores) / len(scores)
                result = {
                    "company_ticker": company_ticker,
                    "average_sentiment_score": avg_sentiment_score,
                    "timestamp": datetime.now().isoformat(),
                }
                return result
            else:
                print("[WARNING] Sentiment score data not available in the response.")
                return {}

        except requests.RequestException as e:
            print(f"[ERROR] Failed to get sentiment data from Alpha Vantage+: {e}")
            return {}

    def save_to_csv(self, data):
        """
        Save sentiment score data to a CSV file.

        :param data: The sentiment score data to save.
        """
        ticker = data["company_ticker"]
        dir_path = os.path.join("app/luqman/data", ticker, "sentiment_score")
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, "sentiment_score.csv")

        with open(file_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["company_ticker", "average_sentiment_score", "timestamp"])
            writer.writerow(
                [
                    data["company_ticker"],
                    data["average_sentiment_score"],
                    data["timestamp"],
                ]
            )

    def get_and_save_sentiment(self, company_ticker):
        """
        Streamline the process of getting and saving sentiment score for a company.

        :param company_ticker: The name of the company to get sentiment for.
        """
        sentiment_data = self.get_sentiment_score(company_ticker)
        if sentiment_data:
            self.save_to_csv(sentiment_data)
            print(f"Sentiment data for {company_ticker}: {sentiment_data}")


# Example usage
if __name__ == "__main__":
    alpha_vantage_api_key = "AOOE7AD9CPPFTQHH"
    sentiment_fetcher = SentimentScore(alpha_vantage_api_key)

    ticker = os.getenv("TICKER")
    sentiment_fetcher.get_and_save_sentiment(ticker)
