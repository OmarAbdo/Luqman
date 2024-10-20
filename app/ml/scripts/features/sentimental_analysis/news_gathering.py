# news_gathering.py

import requests
from googleapiclient.discovery import build
import tweepy
import os
import pandas as pd
from datetime import datetime
import yfinance as yf


class NewsGathering:
    """
    A class to gather relevant news content for financial sentiment analysis.
    This includes articles, tweets, and press releases that are impactful for the given stock ticker.
    """

    def __init__(
        self,
        google_api_key,
        google_cse_id,
        twitter_api_key,
        twitter_api_secret_key,
        twitter_access_token,
        twitter_access_token_secret,
    ):
        """
        Initialize the news gathering component with the required API keys.

        :param google_api_key: API key for Google Custom Search API.
        :param google_cse_id: Custom Search Engine ID for Google Custom Search API.
        :param twitter_api_key: API key for Twitter API.
        :param twitter_api_secret_key: API secret key for Twitter API.
        :param twitter_access_token: Access token for Twitter API.
        :param twitter_access_token_secret: Access token secret for Twitter API.
        """
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.twitter_api_key = twitter_api_key
        self.twitter_api_secret_key = twitter_api_secret_key
        self.twitter_access_token = twitter_access_token
        self.twitter_access_token_secret = twitter_access_token_secret

        # Initialize Twitter API
        auth = tweepy.OAuthHandler(self.twitter_api_key, self.twitter_api_secret_key)
        auth.set_access_token(
            self.twitter_access_token, self.twitter_access_token_secret
        )
        self.twitter_api = tweepy.API(auth)

    def get_company_name(self, ticker):
        """
        Get the company name from the ticker symbol using Yahoo Finance.

        :param ticker: The stock ticker symbol.
        :return: The name of the company.
        """
        try:
            company_info = yf.Ticker(ticker).info
            return company_info.get("longName", ticker)
        except Exception as e:
            print(f"Failed to fetch company name for {ticker}: {e}")
            return ticker

    def fetch_google_news(self, ticker):
        """
        Fetch news articles related to a specific stock ticker using Google Custom Search API.

        :param ticker: The stock ticker symbol to search for.
        :return: A list of news article content.
        """
        company_name = self.get_company_name(ticker)
        query = f"{ticker} {company_name} OR stock OR market OR company OR financial OR news"
        service = build("customsearch", "v1", developerKey=self.google_api_key)
        res = service.cse().list(q=query, cx=self.google_cse_id, num=10).execute()
        fetched_content = []

        if "items" in res:
            for item in res["items"]:
                try:
                    response = requests.get(item["link"])
                    if response.status_code == 200:
                        fetched_content.append(response.text)
                except requests.RequestException as e:
                    print(f"Failed to fetch data from {item['link']}: {e}")

        return fetched_content

    def fetch_twitter_data(self, ticker):
        """
        Fetch recent tweets related to a specific stock ticker using Twitter API.

        :param ticker: The stock ticker symbol to search for.
        :return: A list of tweet content.
        """
        company_name = self.get_company_name(ticker)
        query = (
            f"{ticker} OR {company_name} OR {ticker} stock OR NASDAQ OR market -fruit"
        )
        tweets = []
        try:
            for tweet in tweepy.Cursor(
                self.twitter_api.search_tweets,
                q=query,
                lang="en",
                result_type="recent",
                count=100,
            ).items(50):
                tweets.append(tweet.text)
        except tweepy.TweepError as e:
            print(f"Failed to fetch tweets: {e}")

        return tweets

    def save_news_data(self, ticker, news_content, tweets_content):
        """
        Save the gathered news data to a CSV file.

        :param ticker: The stock ticker symbol for which news data is being saved.
        :param news_content: A list of news articles.
        :param tweets_content: A list of tweets.
        """
        news_data = [
            {"content": content, "type": "news", "timestamp": datetime.now()}
            for content in news_content
        ]

        tweet_data = [
            {"content": tweet, "type": "tweet", "timestamp": datetime.now()}
            for tweet in tweets_content
        ]

        all_data = news_data + tweet_data
        news_df = pd.DataFrame(all_data)

        path = f"app/ml/data/{ticker}/news_gathering/news_data.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        news_df.to_csv(path, index=False)
        print(f"News data saved to {path}")


# Example usage
if __name__ == "__main__":
    google_api_key = "your_google_api_key_here"
    google_cse_id = "your_google_cse_id_here"
    twitter_api_key = "your_twitter_api_key_here"
    twitter_api_secret_key = "your_twitter_api_secret_key_here"
    twitter_access_token = "your_twitter_access_token_here"
    twitter_access_token_secret = "your_twitter_access_token_secret_here"

    news_gathering = NewsGathering(
        google_api_key,
        google_cse_id,
        twitter_api_key,
        twitter_api_secret_key,
        twitter_access_token,
        twitter_access_token_secret,
    )

    # Fetch and save news data for a specific ticker
    ticker = "AAPL"
    news_content = news_gathering.fetch_google_news(ticker)
    tweets_content = news_gathering.fetch_twitter_data(ticker)
    news_gathering.save_news_data(ticker, news_content, tweets_content)
