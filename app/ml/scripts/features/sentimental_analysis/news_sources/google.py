import requests
from datetime import datetime
from googleapiclient.discovery import build
from bs4 import BeautifulSoup


class GoogleNews:
    """
    A class to gather Google news articles using Google Custom Search API.
    """

    def __init__(self, google_api_key, google_cse_id):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.google_service = build("customsearch", "v1", developerKey=google_api_key)

    def get_news_links(self, company_name):
        """
        Use Google Custom Search API to retrieve news links.

        :param company_name: The name of the company.
        :return: A list of URLs for news articles.
        """
        try:
            result = (
                self.google_service.cse()
                .list(q=f"{company_name} news", cx=self.google_cse_id, num=10)
                .execute()
            )
            urls = [item["link"] for item in result.get("items", [])]
            print(f"[DEBUG] Retrieved {len(urls)} links from Google Custom Search.")
            return urls
        except Exception as e:
            print(f"[ERROR] Failed to get news links from Google Custom Search: {e}")
            return []

    def fetch_news(self, news_links):
        """
        Fetch news articles from the provided list of URLs.

        :param news_links: A list of URLs for news articles.
        :return: A list of news article content.
        """
        fetched_content = []

        for url in news_links:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200 and "text/html" in response.headers.get(
                    "Content-Type", ""
                ):
                    soup = BeautifulSoup(response.text, "html.parser")
                    article_text = " ".join(p.get_text() for p in soup.find_all("p"))
                    fetched_content.append(
                        {
                            "title": soup.title.string if soup.title else "No Title",
                            "link": url,
                            "content": article_text,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )
                    print(f"[INFO] Fetched content from: {url}")
                else:
                    print(f"[WARNING] Skipping non-news link: {url}")
            except requests.RequestException as e:
                print(f"[ERROR] Failed to fetch data from {url}: {e}")

        return fetched_content[:10]
