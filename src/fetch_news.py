# src/fetch_news.py
import requests
import json
from datetime import datetime, timedelta
import os

API_KEY = ''  # Replace with your NewsAPI key

def fetch_crime_news(query="crime", from_days_ago=7, page_size=100):
    try:
        url = 'https://newsapi.org/v2/everything'
        from_date = (datetime.now() - timedelta(days=from_days_ago)).strftime('%Y-%m-%d')

        params = {
            'q': query,
            'language': 'en',
            'from': from_date,
            'sortBy': 'publishedAt',
            'pageSize': page_size,
            'apiKey': API_KEY
        }

        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print("Failed to fetch news:", response.text)
            return []

        articles = response.json().get("articles", [])
        
        print(f"[+] Fetched {len(articles)} articles")
        
        # Ensure the 'data' folder exists relative to current directory
        os.makedirs("../data", exist_ok=True)

        # Save to file using relative path
        output_path = os.path.join("..", "data", "raw_news.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)

        return articles
        
    except Exception as e:
        print(f"Error in fetch_crime_news: {str(e)}")
        return []