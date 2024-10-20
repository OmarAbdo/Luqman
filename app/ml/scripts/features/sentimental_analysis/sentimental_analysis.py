# sentiment_analysis.py

import openai
import os
import pandas as pd
import requests
from datetime import datetime


class SentimentAnalysis:
    """
    A class to perform sentiment analysis on financial news, press releases, and social media posts.
    This class uses GPT-4 for natural language processing to determine sentiment scores.
    """

    def __init__(self, api_key):
        """
        Initialize the sentiment analysis component with the required API key for GPT-4.

        :param api_key: API key for OpenAI's GPT-4 API.
        """
        openai.api_key = "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"

    def fetch_data(self, sources):
        """
        Fetch data from specified sources. The sources include news, press releases, and social media posts.

        :param sources: A list of data source URLs.
        :return: A list of strings representing the fetched content.
        """
        fetched_content = []
        for url in sources:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    fetched_content.append(response.text)
            except requests.RequestException as e:
                print(f"Failed to fetch data from {url}: {e}")
        return fetched_content

    def analyze_sentiment(self, content_list):
        """
        Use GPT-4 to perform sentiment analysis on the provided content.

        :param content_list: A list of strings where each string is the content to be analyzed.
        :return: A DataFrame with sentiment scores and other relevant information.
        """
        sentiment_data = []

        for content in content_list:
            response = openai.Completion.create(
                model="gpt-4",
                prompt=f"Analyze the sentiment of the following financial text: {content}\n\nProvide a sentiment score between -1 (very negative) to +1 (very positive), and a brief summary of the sentiment.",
                max_tokens=150,
            )
            sentiment_score = None
            sentiment_summary = None

            if response and "choices" in response and len(response.choices) > 0:
                text_response = response.choices[0].text.strip()
                try:
                    sentiment_score = float(
                        text_response.split("Sentiment score:")[1].split()[0]
                    )
                    sentiment_summary = text_response.split("Summary:")[1].strip()
                except (IndexError, ValueError):
                    sentiment_summary = text_response
                    sentiment_score = 0.0  # default to neutral if parsing fails

            sentiment_data.append(
                {
                    "content": content,
                    "sentiment_score": sentiment_score,
                    "sentiment_summary": sentiment_summary,
                    "timestamp": datetime.now(),
                }
            )

        return pd.DataFrame(sentiment_data)

    def save_sentiment_data(self, ticker, sentiment_df):
        """
        Save the sentiment data to a CSV file.

        :param ticker: The stock ticker symbol for which sentiment data is being saved.
        :param sentiment_df: DataFrame containing sentiment data.
        """
        path = f"app/ml/data/{ticker}/sentimental_analysis/sentiment_data.csv"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        sentiment_df.to_csv(path, index=False)
        print(f"Sentiment data saved to {path}")


# Example usage
if __name__ == "__main__":
    api_key = "your_openai_api_key_here"
    sentiment_analysis = SentimentAnalysis(api_key)

    # Example data sources (Replace with actual URLs)
    data_sources = [
        "https://example.com/financial-news-article-1",
        "https://example.com/financial-news-article-2",
    ]

    # Fetch, analyze and save sentiment data
    content_list = sentiment_analysis.fetch_data(data_sources)
    sentiment_df = sentiment_analysis.analyze_sentiment(content_list)
    ticker = "AAPL"
    sentiment_analysis.save_sentiment_data(ticker, sentiment_df)
