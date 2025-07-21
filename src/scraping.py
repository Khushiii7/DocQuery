import requests
from bs4 import BeautifulSoup
import os
import time

BASE_URL = "https://huggingface.co"
START_PATH = "/docs/transformers/"
SAVE_DIR = "C:/Users/khush/Documents/Project/DocQuery/data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

visited = set()

def is_valid_doc_link(href):
    return (
        href and
        href.startswith("/docs/transformers/") and
        not href.startswith("/docs/transformers/main/en/") and  # avoid language switcher
        "#" not in href and
        "?" not in href
    )

def scrape_page(path):
    if path in visited:
        return
    visited.add(path)
    url = BASE_URL + path
    print(f"Scraping: {url}")
    try:
        resp = requests.get(url)
        if resp.status_code != 200:
            print(f"Failed to fetch {url}")
            return
        soup = BeautifulSoup(resp.content, "html.parser")
        # Save HTML
        filename = path.replace("/docs/transformers/", "").replace("/", "_") or "index"
        with open(os.path.join(SAVE_DIR, f"{filename}.html"), "w", encoding="utf-8") as f:
            f.write(resp.text)
        # Find and crawl new links
        for link in soup.find_all("a"):
            href = link.get("href")
            if is_valid_doc_link(href):
                scrape_page(href)
        time.sleep(0.5)  # be polite to the server
    except Exception as e:
        print(f"Error scraping {url}: {e}")

if __name__ == "__main__":
    scrape_page(START_PATH)