import requests
from datetime import datetime
from bs4 import BeautifulSoup


class GoogleAPI:
    """
    A class to gather news articles using Google Custom Search API.
    """

    def __init__(self, google_api_key, google_cse_id):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id

    def get_news_links(self, company_name):
        """
        Get a list of relevant news links for the company using Google Custom Search API.

        :param company_name: The name of the company.
        :return: A list of URLs for news articles.
        """
        url = (
            f"https://www.googleapis.com/customsearch/v1?q={company_name}+news&cx={self.google_cse_id}"
            f"&num=15&key={self.google_api_key}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            items = response.json().get("items", [])
        except requests.RequestException as e:
            print(f"[ERROR] Failed to get news links from Google Custom Search: {e}")
            return []

        valid_links = []
        for item in items:
            link = item.get("link")
            if not link or self.is_disqualified_link(link):
                continue
            valid_links.append(link)
            print(f"[DEBUG] Added valid link: {link}")

            if len(valid_links) >= 15:
                break

        return valid_links[:15]

    def is_disqualified_link(self, link):
        """
        Check if a link should be disqualified based on certain criteria.

        :param link: The URL to check.
        :return: True if the link should be disqualified, False otherwise.
        """
        # Disqualify links that are homepages for broader topics rather than specific news articles
        if not any(char.isdigit() for char in link.split("/")[-1]) and not any(
            keyword in link for keyword in [".html", "article", "story", "news"]
        ):
            print(f"[WARNING] Skipping broad topic link: {link}")
            return True
        return False

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