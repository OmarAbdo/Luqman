import requests
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import yfinance as yf
import time
import random
from fake_useragent import UserAgent


class ScrapperNews:
    """
    A class to gather Google news articles using Selenium with DuckDuckGo search.
    """

    def __init__(self):
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.prioritized_sources = [
            "nytimes.com",
            "bbc.com",
            "theguardian.com",
            "npr.org",
            "aljazeera.com",
            "cnn.com",
            "reuters.com",
            "huffpost.com",
            "buzzfeednews.com",
            "vice.com",
            "washingtonpost.com",
            "wsj.com",
            "apnews.com",
            "usatoday.com",
            "theatlantic.com",
            "time.com",
            "politico.com",
            "vox.com",
            "thetimes.co.uk",
            "lemonde.fr",
            "spiegel.de",
            "faz.net",
            "elpais.com",
            "repubblica.it",
            "rainews.it",
            "telegraaf.nl",
            "corriere.it",
            "sueddeutsche.de",
            "aftenposten.no",
            "hs.fi",
            "dn.se",
        ]

    def get_news_links(self, company_name, ticker, prioritize_sources=True):
        """
        Get a list of relevant news links for the company using Selenium to perform a DuckDuckGo search.

        :param company_name: The name of the company.
        :param ticker: The stock ticker symbol.
        :param prioritize_sources: Whether to prioritize specific news sources.
        :return: A list of URLs for news articles.
        """
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={UserAgent().random}")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        query = f"{company_name} news"
        search_url = f"https://www.google.com/search?q={query}&tbm=nws"
        # search_url = f"https://www.bing.com/news/search?q={query}"
        # search_url = f"https://duckduckgo.com/?q={query}&t=h_&iar=news&ia=news"
        # search_url = f"https://news.search.yahoo.com/search?p={query}"
        # search_url = f"https://yandex.com/news/search?text={query}"

        driver.get(search_url)
        time.sleep(
            random.uniform(3, 5)
        )  # Allow time for the page to load with random delay

        valid_links = []
        while len(valid_links) < 15:
            try:
                links = driver.find_elements(By.XPATH, "//a[@href]")
                for link_element in links:
                    link = link_element.get_attribute("href")
                    # Disqualify links based on specific criteria
                    if not self.is_valid_news_link(link, ticker, prioritize_sources):
                        continue
                    if link not in valid_links:
                        valid_links.append(link)
                        print(f"[DEBUG] Added valid link: {link}")

                    if len(valid_links) >= 15:
                        break

                # Click the 'Next' button to go to the next page of results (DuckDuckGo may not have this feature)
                next_button = driver.find_elements(By.CLASS_NAME, "result--more__btn")
                if next_button:
                    next_button[0].click()
                    time.sleep(
                        random.uniform(3, 5)
                    )  # Allow time for the next page to load
                else:
                    print("[INFO] No more pages available in search results.")
                    break

            except Exception as e:
                print(f"[ERROR] Failed to get news links using Selenium: {e}")
                break

        driver.quit()
        return valid_links[:15]

    def is_valid_news_link(self, link, ticker, prioritize_sources):
        """
        Check if a link is valid based on certain criteria.

        :param link: The URL to check.
        :param ticker: The stock ticker symbol.
        :param prioritize_sources: Whether to prioritize specific news sources.
        :return: True if the link is valid, False otherwise.
        """
        # Disqualify links containing the company ticker
        if ticker and ticker in link:
            print(f"[WARNING] Skipping link containing ticker: {link}")
            return False
        # Disqualify non-prioritized sources if prioritize_sources is True
        if prioritize_sources and not any(
            source in link for source in self.prioritized_sources
        ):
            print(f"[WARNING] Skipping non-prioritized source: {link}")
            return False
        # Disqualify links that are homepages for broader topics rather than specific news articles
        if not any(char.isdigit() for char in link.split("/")[-1]) and not any(
            keyword in link for keyword in [".html", "article", "story", "news"]
        ):
            print(f"[WARNING] Skipping broad topic link: {link}")
            return False
        return True

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
