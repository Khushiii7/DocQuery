from bs4 import BeautifulSoup
import os

def clean_html_files():
    os.makedirs("C:/Users/khush/Documents/Project/DocQuery/data/processed", exist_ok=True)
    raw_dir = "C:/Users/khush/Documents/Project/DocQuery/data/raw"
    processed_dir = "C:/Users/khush/Documents/Project/DocQuery/data/processed"

    for filename in os.listdir(raw_dir):
        with open(os.path.join(raw_dir, filename), "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

            main_tag = soup.find('main') 
            text = main_tag.get_text(separator="\n") if main_tag else ""

            with open(os.path.join(processed_dir, filename.replace(".html", ".txt")), "w", encoding="utf-8") as out:
                out.write(text)

if __name__ == "__main__":
    clean_html_files()
