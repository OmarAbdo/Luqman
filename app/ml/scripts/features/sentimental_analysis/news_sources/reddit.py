from datetime import datetime
import praw 

class RedditPosts:
    """
    A class to gather Reddit posts related to a stock ticker.
    """

    def __init__(self, reddit_client_id, reddit_client_secret, reddit_user_agent):
        # Initialize Reddit API
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
        )

    def fetch_posts(self, ticker):
        """
        Fetch recent Reddit posts related to a specific stock ticker using Reddit API.

        :param ticker: The stock ticker symbol to search for.
        :return: A list of Reddit post content.
        """
        posts = []
        query = f"{ticker}"

        try:
            for submission in self.reddit.subreddit("all").search(query, limit=20):
                if submission.selftext.strip():
                    content = submission.selftext
                else:
                    content = "No detailed content provided."
                posts.append(
                    {
                        "title": submission.title,
                        "content": content,
                        "score": submission.score,
                        "timestamp": datetime.fromtimestamp(
                            submission.created_utc
                        ).isoformat(),
                    }
                )
                print(f"[INFO] Fetched Reddit post: {submission.title}")
        except Exception as e:
            print(f"[ERROR] Failed to fetch Reddit posts: {e}")

        return posts
