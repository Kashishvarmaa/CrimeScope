# main.py
import os
import time
from datetime import datetime
from fetch_news import fetch_crime_news
from preprocess import preprocess_news
from ner_extractor import process_ner
from crime_classifier import classify_crime_types
from geocode_locations import geocode_locations
from trend_analysis import analyze_trends
from report_generator import generate_report_md, generate_report_pdf
from visualizer import show_visualization, plot_heatmap, show_crime_analysis
from keyword_cloud import generate_keyword_cloud
import pandas as pd 

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)


# Ensure required directories exist
os.makedirs("data/upload_analysis", exist_ok=True)
os.makedirs("reports/weekly", exist_ok=True)
os.makedirs("reports/uploaded", exist_ok=True)

# Crime classification keywords
CRIME_KEYWORDS = {
    'assault': ['assault', 'attack', 'battery', 'hit', 'beat'],
    'theft': ['theft', 'robbery', 'steal', 'stolen', 'larceny', 'burglary'],
    'murder': ['murder', 'homicide', 'kill', 'killed', 'slay'],
    'fraud': ['fraud', 'scam', 'cheat', 'swindle'],
    'sexual': ['rape', 'molest', 'abuse', 'harassment'],
    'drug': ['drug', 'narcotic', 'heroin', 'cocaine', 'opioid'],
    'cyber': ['cybercrime', 'hack', 'phishing', 'malware']
}

def get_data_path(filename):
    """Helper function to get consistent file paths"""
    return os.path.join(DATA_DIR, filename)

def run_pipeline(days_to_fetch=7, report_type="weekly"):
    """Run the complete crime analysis pipeline"""
    
    print("\n" + "="*50)
    print("Starting Crime Analysis Pipeline")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50 + "\n")
    
    try:
        # Step 1: Fetch crime news
        print("[1/8] Fetching crime news...")
        fetch_crime_news(from_days_ago=days_to_fetch)
        
        # Step 2: Preprocess news data
        print("\n[2/8] Preprocessing news data...")
        preprocess_news(
            input_path=get_data_path("raw_news.json"),
            output_path=get_data_path("cleaned_news.csv")
        )
        
        # Step 3: Extract named entities
        print("\n[3/8] Extracting named entities...")
        process_ner(
            input_path=get_data_path("cleaned_news.csv"),
            output_path=get_data_path("ner_output.csv")
        )
        

        # Step 4: Classify crime types
        print("\n[4/8] Classifying crime types...")

        df = pd.read_csv("data/news.csv")  # Step 1: Read CSV
        df = classify_crime_types(df, CRIME_KEYWORDS)  # Step 2: Classify and store the result

        
        # Step 5: Geocode locations
        print("\n[5/8] Geocoding locations...")
        geocode_locations(
            input_file=get_data_path("ner_output.csv"),
            output_file=get_data_path("geo_news.csv")
        )
        
        # Step 6: Generate keyword cloud
        print("\n[6/8] Generating keyword cloud...")
        generate_keyword_cloud(get_data_path("ner_output.csv"))
        
        # Step 7: Generate reports
        print("\n[7/8] Generating analysis reports...")
        generate_report_md(get_data_path("geo_news.csv"), report_type)
        generate_report_pdf(get_data_path("geo_news.csv"), report_type)
        
        # Step 8: Visualize data
        print("\n[8/8] Visualizing data...")
        analyze_trends()
        show_visualization(get_data_path("ner_output.csv"))
        plot_heatmap()

        
        print("\n" + "="*50)
        print("Pipeline completed successfully!")
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the complete pipeline
    run_pipeline(days_to_fetch=7, report_type="weekly")