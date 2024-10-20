# news_gathering.py

import requests
from googleapiclient.discovery import build
import os
import pandas as pd
from datetime import datetime
import yfinance as yf
import praw  # Reddit API wrapper


class NewsGathering:
    """
    A class to gather relevant news content for financial sentiment analysis.
    This includes articles, Reddit posts, and press releases that are impactful for the given stock ticker.
    """

    def __init__(
        self,
        google_api_key,
        google_cse_id,
        reddit_client_id,
        reddit_client_secret,
        reddit_user_agent,
    ):
        """
        Initialize the news gathering component with the required API keys.

        :param google_api_key: API key for Google Custom Search API.
        :param google_cse_id: Custom Search Engine ID for Google Custom Search API.
        :param reddit_client_id: Client ID for Reddit API.
        :param reddit_client_secret: Client Secret for Reddit API.
        :param reddit_user_agent: User agent for Reddit API.
        """
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id

        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
        )

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

    def fetch_reddit_data(self, ticker):
        """
        Fetch recent Reddit posts related to a specific stock ticker using Reddit API.

        :param ticker: The stock ticker symbol to search for.
        :return: A list of Reddit post content.
        """
        company_name = self.get_company_name(ticker)
        query = f"{ticker} OR {company_name}"
        posts = []

        try:
            for submission in self.reddit.subreddit("all").search(query, limit=20):
                posts.append(submission.title + " " + submission.selftext)
        except Exception as e:
            print(f"Failed to fetch Reddit posts: {e}")

        return posts

    def save_news_data(self, ticker, news_content, reddit_content):
        """
        Save the gathered news data to a CSV file.

        :param ticker: The stock ticker symbol for which news data is being saved.
        :param news_content: A list of news articles.
        :param reddit_content: A list of Reddit posts.
        """
        news_data = [
            {"content": content, "type": "news", "timestamp": datetime.now()}
            for content in news_content
        ]

        reddit_data = [
            {"content": post, "type": "reddit", "timestamp": datetime.now()}
            for post in reddit_content
        ]

        all_data = news_data + reddit_data
        news_df = pd.DataFrame(all_data)

        path = f"app/ml/data/{ticker}/news_gathering/news_data.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        news_df.to_csv(path, index=False)
        print(f"News data saved to {path}")


# Example usage
if __name__ == "__main__":
    google_api_key = "your_google_api_key_here"
    google_cse_id = "your_google_cse_id_here"
    reddit_client_id = "your_reddit_client_id_here"
    reddit_client_secret = "your_reddit_client_secret_here"
    reddit_user_agent = "your_reddit_user_agent_here"

    news_gathering = NewsGathering(
        google_api_key,
        google_cse_id,
        reddit_client_id,
        reddit_client_secret,
        reddit_user_agent,
    )

    # Fetch and save news data for a specific ticker
    ticker = "AAPL"
    news_content = news_gathering.fetch_google_news(ticker)
    reddit_content = news_gathering.fetch_reddit_data(ticker)
    news_gathering.save_news_data(ticker, news_content, reddit_content)
