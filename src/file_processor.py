# file_processor.py
import pandas as pd
import json
import matplotlib as plt
from preprocess import clean_text
from ner_extractor import extract_entities,process_ner
from crime_classifier import classify_crime_types
from geocode_locations import geocode_locations
from trend_analysis import analyze_trends
from report_generator import generate_report_md, generate_report_pdf
from keyword_cloud import generate_keyword_cloud
import PyPDF2
from io import StringIO
import os 

def extract_text_from_file(uploaded_file):
    """Extract text from either PDF or TXT file"""
    if uploaded_file.type == "text/plain":
        return str(uploaded_file.read(), "utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    return ""

def process_uploaded_file(uploaded_file, crime_keywords):
    """Complete processing pipeline for uploaded files"""
    # Step 1: Extract text
    raw_text = extract_text_from_file(uploaded_file)
    
    # Step 2: Preprocess
    clean = clean_text(raw_text)
    
    # Step 3: Create dataframe
    df = pd.DataFrame({
        "source": ["uploaded_file"],
        "publishedAt": [pd.Timestamp.now().strftime('%Y-%m-%d')],
        "raw_text": [raw_text],
        "clean_text": [clean],
        "url": [""]
    })
    
    # Step 4: NER Extraction
    all_locations = []
    all_crimes = []
    for text in df["clean_text"]:
        locations, crimes = extract_entities(text)
        all_locations.append(", ".join(locations))
        all_crimes.append(", ".join(crimes))
    df["locations"] = all_locations
    df["crime_types"] = all_crimes
    
    # Step 5: Crime Classification
    df = classify_crime_types(df, crime_keywords)
    
    # Step 6: Geocoding
    if not df.empty and 'locations' in df.columns:
        df = geocode_locations(df)
    
    return df

# In file_processor.py
def generate_analysis_artifacts(df, output_dir="data/upload_analysis"):
    """Generate all analysis outputs"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate keyword cloud
    wordcloud_fig = generate_keyword_cloud(df)
    wordcloud_path = f"{output_dir}/keyword_cloud.png"
    wordcloud_fig.savefig(wordcloud_path)
    
    # Generate reports
    csv_path = f"{output_dir}/analysis_results.csv"
    df.to_csv(csv_path, index=False)
    
    pdf_report_path = generate_report_pdf(df, f"{output_dir}/upload_report.pdf")
    md_report_path = generate_report_md(df, f"{output_dir}/upload_report.md")
    
    # Generate trend analysis (only if we have date data)
    trend_path = None
    if 'publishedAt' in df.columns:
        try:
            trend_fig = analyze_trends(df, return_fig=True)
            trend_path = f"{output_dir}/trend_analysis.png"
            trend_fig.savefig(trend_path)
            plt.close(trend_fig)
        except Exception as e:
            print(f"Could not generate trend analysis: {str(e)}")
    
    return {
        "wordcloud": wordcloud_path,
        "csv": csv_path,
        "pdf_report": pdf_report_path,
        "md_report": md_report_path,
        "trend_analysis": trend_path  # This might be None
    }