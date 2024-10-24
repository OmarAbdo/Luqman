# news_gathering.py

import requests
from datetime import datetime
import os
import json
import yfinance as yf
from news_sources.google import GoogleNews
from news_sources.reddit import RedditPosts


class NewsGathering:
    """
    A class to streamline the news gathering process for financial sentiment analysis.
    It manages different sources of news content (e.g., Google articles, Reddit posts).
    """

    def __init__(
        self,
        google_api_key,
        google_cse_id,
        reddit_client_id,
        reddit_client_secret,
        reddit_user_agent,
    ):
        self.google_news = GoogleNews(google_api_key, google_cse_id)
        self.reddit_posts = RedditPosts(
            reddit_client_id, reddit_client_secret, reddit_user_agent
        )

    def get_company_name(self, ticker):
        """
        Get the company name based on the ticker symbol.

        :param ticker: The stock ticker symbol.
        :return: The company name.
        """
        company = yf.Ticker(ticker)
        return company.info["shortName"]

    def gather_news_data(self, ticker):
        """
        Gather news data from Google and Reddit for the given ticker.

        :param ticker: The stock ticker symbol.
        :return: A tuple of Google news content and Reddit post content.
        """
        company_name = self.get_company_name(ticker)
        news_links = self.google_news.get_news_links(company_name, ticker)
        news_content = self.google_news.fetch_news(news_links)
        reddit_content = self.reddit_posts.fetch_posts(ticker)
        return news_content, reddit_content

    def save_news_data(self, ticker, news_content, reddit_content):
        """
        Save the gathered news data to separate JSON files.

        :param ticker: The stock ticker symbol for which news data is being saved.
        :param news_content: A list of news articles.
        :param reddit_content: A list of Reddit posts.
        """
        news_path = f"app/ml/data/{ticker}/news_gathering/google_news.json"
        reddit_path = f"app/ml/data/{ticker}/news_gathering/reddit_posts.json"
        os.makedirs(os.path.dirname(news_path), exist_ok=True)

        # Save Google News data
        with open(news_path, "w", encoding="utf-8") as news_file:
            json.dump(news_content, news_file, indent=4)
            print(f"[INFO] News data saved to {news_path}")

        # Save Reddit data
        with open(reddit_path, "w", encoding="utf-8") as reddit_file:
            json.dump(reddit_content, reddit_file, indent=4)
            print(f"[INFO] Reddit data saved to {reddit_path}")


# Example usage
if __name__ == "__main__":
    google_api_key = "AIzaSyAP87TlgbKQWk10xTXke6Kn6kHRyfuIB_I"
    google_cse_id = "2353131eb16b54bcd"  # Search engine ID

    reddit_client_id = "M1DFQZCTGE8tZYgAbsMT1A"
    reddit_client_secret = "HuIvXTtFxCQslw-JodBqEVu-xyA3ig"
    reddit_user_agent = "Luqman"


    news_gathering = NewsGathering(
        google_api_key,
        google_cse_id,
        reddit_client_id,
        reddit_client_secret,
        reddit_user_agent,
    )

    # Fetch and save news data for a specific ticker
    ticker = "AAPL"
    print(f"[INFO] Fetching news data for {ticker}")
    news_content, reddit_content = news_gathering.gather_news_data(ticker)
    print(f"[INFO] Saving news data for {ticker}")
    news_gathering.save_news_data(ticker, news_content, reddit_content)
