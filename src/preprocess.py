# src/preprocess.py

import json
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_news(input_path="data/raw_news.json", output_path="data/cleaned_news.csv"):
    with open(input_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    rows = []
    for article in articles:
        combined_text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}"
        clean = clean_text(combined_text)
        rows.append({
            "source": article.get("source", {}).get("name", ""),
            "publishedAt": article.get("publishedAt", ""),
            "raw_text": combined_text,
            "clean_text": clean,
            "url": article.get("url", "")
        })

    df = pd.DataFrame(rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Cleaned {len(df)} articles and saved to {output_path}")

if __name__ == "__main__":
    preprocess_news()
