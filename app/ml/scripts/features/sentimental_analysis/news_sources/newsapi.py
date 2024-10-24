import requests
from datetime import datetime
from bs4 import BeautifulSoup


class NewsAPI:
    """
    A class to gather news articles using the NewsAPI.
    """

    def __init__(self, news_api_key="445a36e057ee4663bc15f403d76fd0b5"):
        self.news_api_key = news_api_key

    def get_news_links(self, company_name):
        """
        Get a list of relevant news links for the company using the NewsAPI.

        :param company_name: The name of the company.
        :return: A list of URLs for news articles.
        """
        url = (
            f"https://newsapi.org/v2/everything?q={company_name}&sortBy=publishedAt"
            f"&apiKey={self.news_api_key}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])
        except requests.RequestException as e:
            print(f"[ERROR] Failed to get news links from NewsAPI: {e}")
            return []

        valid_links = []
        for article in articles:
            link = article.get("url")
            if not link:
                continue
            valid_links.append(link)
            print(f"[DEBUG] Added valid link: {link}")

            if len(valid_links) >= 15:
                break

        return valid_links[:15]

    def fetch_news(self, urls):
        """
        Fetch news articles from the given URLs.

        :param urls: A list of URLs for news articles.
        :return: A list of news article content.
        """
        fetched_content = []

        for url in urls:
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

        return fetched_content
