import requests
from datetime import datetime


class AlphaVantageSentiment:
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
                return {
                    "company_ticker": company_ticker,
                    "average_sentiment_score": avg_sentiment_score,
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                print("[WARNING] Sentiment score data not available in the response.")
                return {}

        except requests.RequestException as e:
            print(f"[ERROR] Failed to get sentiment data from Alpha Vantage: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    alpha_vantage_api_key = "AOOE7AD9CPPFTQHH"
    sentiment_fetcher = AlphaVantageSentiment(alpha_vantage_api_key)

    ticker = "AAPL"
    sentiment_score_data = sentiment_fetcher.get_sentiment_score(ticker)
    if sentiment_score_data:
        print(f"Sentiment data for {ticker}: {sentiment_score_data}")
