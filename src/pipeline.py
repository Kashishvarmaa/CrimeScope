# main.py
import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import nltk
from nltk.corpus import stopwords
import re
import string
from datetime import datetime, timedelta
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from wordcloud import WordCloud
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
import streamlit as st
from tqdm import tqdm
import PyPDF2
import logging
import pickle
from io import StringIO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK data
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "weekly"), exist_ok=True)
os.makedirs(os.path.join(REPORTS_DIR, "uploaded"), exist_ok=True)

# News API Key
API_KEY = '' # Replace with your NewsAPI key

# Crime keywords for classification
CRIME_KEYWORDS = {
    'assault': ['assault', 'attack', 'battery', 'hit', 'beat'],
    'theft': ['theft', 'robbery', 'steal', 'stolen', 'larceny', 'burglary'],
    'murder': ['murder', 'homicide', 'kill', 'killed', 'slay'],
    'fraud': ['fraud', 'scam', 'cheat', 'swindle'],
    'sexual': ['rape', 'molest', 'abuse', 'harassment'],
    'drug': ['drug', 'narcotic', 'heroin', 'cocaine', 'opioid'],
    'cyber': ['cybercrime', 'hack', 'phishing', 'malware']
}

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except Exception as e:
    logger.error(f"Failed to load spaCy model: {e}")
    raise

# Geocoding cache
GEOCACHE_FILE = os.path.join(DATA_DIR, "geocache.pkl")
geocache = {}
if os.path.exists(GEOCACHE_FILE):
    with open(GEOCACHE_FILE, 'rb') as f:
        geocache = pickle.load(f)

# Helper Functions
def get_data_path(filename):
    """Get consistent file paths"""
    return os.path.join(DATA_DIR, filename)

def fetch_crime_news(query="crime", from_days_ago=7, page_size=100):
    """Fetch crime news from NewsAPI"""
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
        response.raise_for_status()
        articles = response.json().get("articles", [])
        logger.info(f"Fetched {len(articles)} articles")
        
        output_path = get_data_path("raw_news.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        return articles
    except Exception as e:
        logger.error(f"Error fetching news: {e}")
        return []

def clean_text(text):
    """Clean text by removing URLs, punctuation, numbers, and stopwords"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = [word for word in text.split() if word not in stop_words]
    return " ".join(tokens)

def preprocess_news(input_path="raw_news.json", output_path="cleaned_news.csv"):
    """Preprocess news articles"""
    try:
        with open(get_data_path(input_path), "r", encoding="utf-8") as f:
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
        df.to_csv(get_data_path(output_path), index=False)
        logger.info(f"Cleaned {len(df)} articles and saved to {output_path}")
        return df
    except Exception as e:
        logger.error(f"Error preprocessing news: {e}")
        return pd.DataFrame()

def extract_entities(text):
    """Extract locations and crime types using spaCy and keywords"""
    doc = nlp(text)
    locations = list(set([ent.text for ent in doc.ents if ent.label_ == "GPE"]))
    crime_types = list(set([word for word in text.lower().split() if word in sum(CRIME_KEYWORDS.values(), [])]))
    return locations, crime_types

def process_ner(df):
    """Apply NER to DataFrame"""
    try:
        all_locations = []
        all_crimes = []
        for text in df["clean_text"]:
            locations, crimes = extract_entities(text)
            all_locations.append(", ".join(locations))
            all_crimes.append(", ".join(crimes))
        df["locations"] = all_locations
        df["crime_types"] = all_crimes
        return df
    except Exception as e:
        logger.error(f"Error in NER processing: {e}")
        return df

def classify_crime_types(df, crime_keywords):
    """Classify crimes based on keywords"""
    try:
        def classify(text):
            if pd.isna(text):
                return ""
            text = text.lower()
            labels = []
            for crime, keywords in crime_keywords.items():
                if any(keyword in text for keyword in keywords):
                    labels.append(crime)
            return ", ".join(labels) if labels else "other"
        df["crime_type"] = df["clean_text"].apply(classify)
        return df
    except Exception as e:
        logger.error(f"Error classifying crimes: {e}")
        return df
    



 # Helper function to clean location names
def clean_location_name(loc):
    """Remove special characters and normalize location names"""
    if not loc or pd.isna(loc):
        return None
    # Remove special characters and extra whitespace
    loc = re.sub(r'[^\w\s,]', '', loc).strip()
    # Replace multiple spaces with single space
    loc = re.sub(r'\s+', ' ', loc)
    return loc if loc else None

def geocode_locations(df):
    """Geocode locations with caching and improved error handling"""
    global geocache
    if 'locations' not in df.columns or df['locations'].isna().all():
        logger.warning("No locations to geocode")
        return df
    
    geolocator = Nominatim(user_agent="crime-analysis", timeout=5)  # Increased timeout
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=3)
    latitudes = []
    longitudes = []
    failed_locations = []
    
    for loc in tqdm(df['locations'], desc="Geocoding"):
        cleaned_loc = clean_location_name(loc)
        if not cleaned_loc:
            latitudes.append(None)
            longitudes.append(None)
            continue
        
        # Check cache
        if cleaned_loc in geocache:
            lat, lon = geocache[cleaned_loc]
        else:
            try:
                location = geocode(cleaned_loc)
                lat = location.latitude if location else None
                lon = location.longitude if location else None
                geocache[cleaned_loc] = (lat, lon)
                if not location:
                    failed_locations.append(cleaned_loc)
                    logger.warning(f"Failed to geocode: {cleaned_loc}")
            except Exception as e:
                lat, lon = None, None
                geocache[cleaned_loc] = (lat, lon)
                failed_locations.append(cleaned_loc)
                logger.error(f"Geocoding error for {cleaned_loc}: {e}")
        
        latitudes.append(lat)
        longitudes.append(lon)
    
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    
    # Save geocache
    with open(GEOCACHE_FILE, 'wb') as f:
        pickle.dump(geocache, f)
    
    if failed_locations:
        logger.info(f"Failed to geocode {len(failed_locations)} locations: {', '.join(set(failed_locations))}")
    
    return df

def generate_keyword_cloud(df):
    """Generate word cloud from text data"""
    try:
        text = " ".join(df['clean_text'].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(text)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        return fig
    except Exception as e:
        logger.error(f"Error generating keyword cloud: {e}")
        return plt.figure()

def generate_report_md(df, output_path, report_type="analysis"):
    """Generate markdown report"""
    try:
        crime_counts = df['crime_type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        location_counts = df['locations'].value_counts().reset_index() if 'locations' in df.columns else pd.DataFrame()
        location_counts.columns = ['Locations', 'Count'] if not location_counts.empty else []
        
        crime_plot_path = get_data_path("crime_type_distribution.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
        plt.savefig(crime_plot_path)
        plt.close()
        
        location_plot_path = None
        if not location_counts.empty:
            location_plot_path = get_data_path("top_crime_locations.png")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
            plt.savefig(location_plot_path)
            plt.close()
        
        report_content = f"""
# Crime Analysis Report ({report_type.capitalize()})

## Crime Type Distribution
![Crime Type Distribution]({crime_plot_path})
"""
        if location_plot_path:
            report_content += f"""
## Top Crime Locations
![Top Crime Locations]({location_plot_path})
"""
        report_content += f"""
## Summary Statistics
- **Total Cases Analyzed**: {len(df)}
- **Crime Categories Identified**: {len(crime_counts)}
- **Unique Locations Found**: {df['locations'].nunique() if 'locations' in df.columns else 0}
"""
        if 'publishedAt' in df.columns:
            report_content += f"""
## Temporal Analysis
- **Date Range**: {df['publishedAt'].min()} to {df['publishedAt'].max()}
"""
        report_content += f"""
**Report Generated on**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as file:
            file.write(report_content)
        logger.info(f"Markdown report saved as {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating markdown report: {e}")
        return None

def generate_report_pdf(df, output_path, report_type="analysis"):
    """Generate PDF report"""
    try:
        crime_counts = df['crime_type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        location_counts = df['locations'].value_counts().reset_index() if 'locations' in df.columns else pd.DataFrame()
        location_counts.columns = ['Locations', 'Count'] if not location_counts.empty else []
        
        crime_plot_path = get_data_path("crime_type_distribution.png")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
        plt.savefig(crime_plot_path)
        plt.close()
        
        location_plot_path = None
        if not location_counts.empty:
            location_plot_path = get_data_path("top_crime_locations.png")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
            plt.savefig(location_plot_path)
            plt.close()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        c = canvas.Canvas(output_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 750, f"Crime Analysis Report ({report_type.capitalize()})")
        c.setFont("Helvetica", 12)
        c.drawString(72, 730, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
        
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 700, "Crime Type Distribution")
        c.drawImage(crime_plot_path, 72, 450, width=450, height=250)
        
        if location_plot_path:
            c.showPage()
            c.setFont("Helvetica-Bold", 14)
            c.drawString(72, 750, "Top Crime Locations")
            c.drawImage(location_plot_path, 72, 500, width=450, height=250)
        
        c.showPage()
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 750, "Summary Statistics")
        c.setFont("Helvetica", 12)
        stats = [
            f"Total Cases Analyzed: {len(df)}",
            f"Crime Categories Identified: {len(crime_counts)}",
            f"Unique Locations Found: {df['locations'].nunique() if 'locations' in df.columns else 0}"
        ]
        if 'publishedAt' in df.columns:
            stats.append(f"Date Range: {df['publishedAt'].min()} to {df['publishedAt'].max()}")
        y_position = 700
        for stat in stats:
            c.drawString(72, y_position, stat)
            y_position -= 20
        c.save()
        logger.info(f"PDF report saved as {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
        return None

def analyze_trends(df, period='ME', return_fig=False):
    """Analyze crime trends by period (monthly 'ME' or weekly 'W')"""
    try:
        if 'publishedAt' not in df.columns or 'crime_type' not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "Insufficient data for trend analysis", ha='center', va='center')
            ax.axis('off')
            return fig if return_fig else plt.show()
        
        df['publishedAt'] = pd.to_datetime(df['publishedAt'])
        df.set_index('publishedAt', inplace=True)
        crime_trends = df.groupby('crime_type').resample(period).size().unstack(level=0)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        for crime in crime_trends.columns:
            crime_trends[crime].plot(ax=ax, label=crime, marker='o')
        ax.set_title(f"Crime Trends by Type ({'Monthly' if period == 'ME' else 'Weekly'})")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        ax.legend(title="Crime Type")
        plt.tight_layout()
        return fig if return_fig else plt.show()
    except Exception as e:
        logger.error(f"Error analyzing trends: {e}")
        return plt.figure()

def extract_text_from_file(uploaded_file):
    """Extract text from PDF or TXT file"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        return ""
    except Exception as e:
        logger.error(f"Error extracting text: {e}")
        return ""

def process_uploaded_file(uploaded_file, crime_keywords):
    """Process uploaded file through pipeline"""
    try:
        raw_text = extract_text_from_file(uploaded_file)
        clean = clean_text(raw_text)
        df = pd.DataFrame({
            "source": ["uploaded_file"],
            "publishedAt": [pd.Timestamp.now().strftime('%Y-%m-%d')],
            "raw_text": [raw_text],
            "clean_text": [clean],
            "url": [""]
        })
        df = process_ner(df)
        df = classify_crime_types(df, crime_keywords)
        if 'locations' in df.columns and not df['locations'].isna().all():
            df = geocode_locations(df)
        return df
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        return pd.DataFrame()

def generate_analysis_artifacts(df, output_dir=os.path.join(DATA_DIR, "upload_analysis")):
    """Generate analysis outputs for uploaded file"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        wordcloud_fig = generate_keyword_cloud(df)
        wordcloud_path = os.path.join(output_dir, "keyword_cloud.png")
        wordcloud_fig.savefig(wordcloud_path)
        plt.close(wordcloud_fig)
        
        csv_path = os.path.join(output_dir, "analysis_results.csv")
        df.to_csv(csv_path, index=False)
        
        pdf_report_path = generate_report_pdf(df, os.path.join(output_dir, "upload_report.pdf"), "file_analysis")
        md_report_path = generate_report_md(df, os.path.join(output_dir, "upload_report.md"), "file_analysis")
        
        trend_path = None
        if 'publishedAt' in df.columns:
            try:
                trend_fig = analyze_trends(df, return_fig=True)
                trend_path = os.path.join(output_dir, "trend_analysis.png")
                trend_fig.savefig(trend_path)
                plt.close(trend_fig)
            except Exception as e:
                logger.warning(f"Could not generate trend analysis: {e}")
        
        return {
            "wordcloud": wordcloud_path,
            "csv": csv_path,
            "pdf_report": pdf_report_path,
            "md_report": md_report_path,
            "trend_analysis": trend_path
        }
    except Exception as e:
        logger.error(f"Error generating artifacts: {e}")
        return {}

def plot_heatmap(df):
    """Generate heatmap with NaN handling"""
    try:
        df = df.dropna(subset=['latitude', 'longitude'])
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        if len(df) == 0:
            st.warning("No valid location data available for heatmap")
            return None
        
        mean_lat = df['latitude'].mean()
        mean_lon = df['longitude'].mean()
        m = folium.Map(location=[mean_lat, mean_lon], zoom_start=12)
        heat_data = df[['latitude', 'longitude']].values.tolist()
        HeatMap(heat_data).add_to(m)
        return m
    except Exception as e:
        st.error(f"Could not generate heatmap: {e}")
        return None

def run_pipeline(days_to_fetch=7, report_type="weekly"):
    """Run the complete crime analysis pipeline"""
    logger.info("Starting Crime Analysis Pipeline")
    try:
        # Fetch news
        fetch_crime_news(from_days_ago=days_to_fetch)
        
        # Preprocess
        df = preprocess_news(input_path="raw_news.json", output_path="cleaned_news.csv")
        if df.empty:
            raise ValueError("No data after preprocessing")
        
        # NER
        df = process_ner(df)
        df.to_csv(get_data_path("ner_output.csv"), index=False)
        
        # Classify crimes
        df = classify_crime_types(df, CRIME_KEYWORDS)
        
        # Geocode
        df = geocode_locations(df)
        df.to_csv(get_data_path("geo_news.csv"), index=False)
        
        # Generate keyword cloud
        wordcloud_fig = generate_keyword_cloud(df)
        wordcloud_fig.savefig(get_data_path("keyword_cloud.png"))
        plt.close(wordcloud_fig)
        
        # Generate reports
        pdf_path = os.path.join(REPORTS_DIR, report_type, f"crime_report_{report_type}.pdf")
        md_path = os.path.join(REPORTS_DIR, report_type, f"crime_report_{report_type}.md")
        generate_report_pdf(df, pdf_path, report_type)
        generate_report_md(df, md_path, report_type)
        
        # Analyze trends (monthly and weekly)
        monthly_fig = analyze_trends(df, period='ME', return_fig=True)
        monthly_fig.savefig(get_data_path("trend_analysis_monthly.png"))
        plt.close(monthly_fig)
        
        weekly_fig = analyze_trends(df, period='W', return_fig=True)
        weekly_fig.savefig(get_data_path("trend_analysis_weekly.png"))
        plt.close(weekly_fig)
        
        logger.info("Pipeline completed successfully")
        return df
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

# Streamlit Interface
def show_visualization():
    """Visualization dashboard"""
    st.title("🕵️‍♀️ Crime Pattern Visualization")
    st.markdown("Interactive visualizations of crime patterns from news reports.")
    
    try:
        df = pd.read_csv(get_data_path("geo_news.csv"))
        
        if st.checkbox("Show raw data"):
            st.dataframe(df.head(10))
        
        df = df.dropna(subset=['crime_type', 'locations'])
        
        # Filters
        st.sidebar.header("Filters")
        crime_types = st.sidebar.multiselect("Select Crime Types", options=df['crime_type'].unique(), default=df['crime_type'].unique())
        date_range = st.sidebar.date_input("Select Date Range", 
                                          [pd.to_datetime(df['publishedAt']).min().date(), 
                                           pd.to_datetime(df['publishedAt']).max().date()])
        
        filtered_df = df[
            (df['crime_type'].isin(crime_types)) &
            (pd.to_datetime(df['publishedAt']).dt.date >= date_range[0]) &
            (pd.to_datetime(df['publishedAt']).dt.date <= date_range[1])
        ]
        
        # Crime Type Distribution
        st.header("🔍 Crime Type Distribution")
        crime_counts = filtered_df['crime_type'].value_counts().reset_index()
        crime_counts.columns = ['Crime Type', 'Count']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Crime Type', data=crime_counts, ax=ax)
        st.pyplot(fig)
        
        # Top Crime Locations
        st.header("📍 Top Crime Locations")
        location_counts = filtered_df['locations'].value_counts().reset_index()
        location_counts.columns = ['Locations', 'Count']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Locations', data=location_counts.head(10), ax=ax)
        st.pyplot(fig)
        
        # Keyword Cloud
        st.header("☁️ Crime Keyword Cloud")
        wordcloud_fig = generate_keyword_cloud(filtered_df)
        st.pyplot(wordcloud_fig)
        
        # Heatmap
        st.header("🔥 Crime Heatmap")
        heatmap = plot_heatmap(filtered_df)
        if heatmap:
            folium_static(heatmap)
        
        # Trend Analysis
        st.header("📈 Crime Trends")
        period = st.selectbox("Select Trend Period", ["Monthly", "Weekly"])
        period_code = 'ME' if period == "Monthly" else 'W'
        trend_fig = analyze_trends(filtered_df, period=period_code, return_fig=True)
        st.pyplot(trend_fig)
        
        # Download Reports
        st.header("📊 Download Reports")
        if st.button("Generate Reports"):
            with st.spinner("Generating reports..."):
                report_paths = {
                    'pdf': get_data_path("weekly_crime_report.pdf"),
                    'markdown': get_data_path("weekly_crime_report.md")
                }
                generate_report_pdf(filtered_df, report_paths['pdf'], "weekly")
                generate_report_md(filtered_df, report_paths['markdown'], "weekly")
                
                with open(report_paths['pdf'], "rb") as f:
                    st.download_button(
                        "Download PDF Report",
                        f.read(),
                        "weekly_crime_report.pdf",
                        "application/pdf"
                    )
                with open(report_paths['markdown'], "rb") as f:
                    st.download_button(
                        "Download Markdown Report",
                        f.read(),
                        "weekly_crime_report.md",
                        "text/markdown"
                    )
    except Exception as e:
        st.error(f"Error in visualization: {e}")

def show_crime_analysis():
    """File upload and analysis section"""
    st.title("🔬 Crime File Analysis")
    st.markdown("Upload crime reports for detailed analysis.")
    
    uploaded_file = st.file_uploader("Choose a file (PDF or TXT)", type=['pdf', 'txt'])
    
    if uploaded_file is not None:
        with st.spinner("Analyzing file..."):
            try:
                df = process_uploaded_file(uploaded_file, CRIME_KEYWORDS)
                artifacts = generate_analysis_artifacts(df)
                
                st.success("Analysis complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Crimes Identified", len(df))
                with col2:
                    st.metric("Unique Crime Types", df['crime_type'].nunique())
                with col3:
                    st.metric("Locations Found", df['locations'].nunique())
                
                tab1, tab2, tab3, tab4 = st.tabs(["Crime Breakdown", "Locations", "Keyword Cloud", "Reports"])
                
                with tab1:
                    st.header("Crime Type Distribution")
                    crime_counts = df['crime_type'].str.split(', ').explode().value_counts()
                    fig, ax = plt.subplots(figsize=(10, 6))
                    crime_counts.plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title("Crime Type Distribution")
                    ax.set_xlabel("Crime Type")
                    ax.set_ylabel("Count")
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                
                with tab2:
                    st.header("Crime Locations")
                    has_geo_data = 'latitude' in df.columns and 'longitude' in df.columns
                    valid_geo_data = False
                    
                    if has_geo_data:
                        geo_df = df.dropna(subset=['latitude', 'longitude']).copy()
                        geo_df['latitude'] = pd.to_numeric(geo_df['latitude'], errors='coerce')
                        geo_df['longitude'] = pd.to_numeric(geo_df['longitude'], errors='coerce')
                        geo_df = geo_df.dropna(subset=['latitude', 'longitude'])
                        valid_geo_data = len(geo_df) > 0
                    
                    if valid_geo_data:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Map View")
                            first_valid = geo_df.iloc[0]
                            m = folium.Map(location=[first_valid['latitude'], first_valid['longitude']], zoom_start=10)
                            for _, row in geo_df.iterrows():
                                folium.Marker(
                                    [row['latitude'], row['longitude']],
                                    popup=f"{row.get('crime_type', 'Unknown')}: {row.get('locations', 'Unknown')}"
                                ).add_to(m)
                            folium_static(m)
                        with col2:
                            st.subheader("Heatmap View")
                            heatmap = plot_heatmap(geo_df)
                            if heatmap:
                                folium_static(heatmap)
                    else:
                        st.warning("No valid geographic coordinates found")
                    
                    st.subheader("Top Locations")
                    if 'locations' in df.columns:
                        location_counts = df['locations'].value_counts().reset_index()
                        fig, ax = plt.subplots()
                        sns.barplot(x='count', y='locations', data=location_counts.head(10).rename(columns={'index': 'locations'}), ax=ax)
                        st.pyplot(fig)
                
                with tab3:
                    st.header("Keyword Cloud")
                    st.image(artifacts['wordcloud'])
                    with open(artifacts['wordcloud'], "rb") as f:
                        st.download_button(
                            "Download Word Cloud",
                            f.read(),
                            "crime_keyword_cloud.png",
                            "image/png"
                        )
                
                with tab4:
                    st.header("Download Reports")
                    with open(artifacts['pdf_report'], "rb") as f:
                        st.download_button(
                            "Download PDF Report",
                            f.read(),
                            "crime_analysis_report.pdf",
                            "application/pdf"
                        )
                    with open(artifacts['md_report'], "rb") as f:
                        st.download_button(
                            "Download Markdown Report",
                            f.read(),
                            "crime_analysis_report.md",
                            "text/markdown"
                        )
                    st.download_button(
                        "Download Analysis Data (CSV)",
                        df.to_csv(index=False).encode('utf-8'),
                        "crime_analysis_data.csv",
                        "text/csv"
                    )
            except Exception as e:
                st.error(f"Analysis failed: {e}")

def show_chatbot():
    """Placeholder for chatbot interface"""
    st.title("💬 Crime Analysis Chatbot")
    st.markdown("Coming soon: Query crime data using natural language.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about crime patterns..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = f"I'm a placeholder response to: {prompt}"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main App Logic
if __name__ == "__main__":
    # Run pipeline to ensure data is available
    try:
        run_pipeline(days_to_fetch=7, report_type="weekly")
    except Exception as e:
        logger.error(f"Initial pipeline run failed: {e}")
        st.error(f"Failed to initialize data: {e}")
    
    # Streamlit app
    st.sidebar.title("Crime Analysis Portal")
    st.sidebar.markdown("---")
    app_mode = st.sidebar.radio("Navigation", ["Visualization", "Crime Analysis", "Chatbot"])
    
    st.sidebar.header("User Profile")
    user_name = st.sidebar.text_input("Name", "John Doe")
    user_role = st.sidebar.selectbox("Role", ["Analyst", "Officer", "Researcher"])
    st.sidebar.markdown(f"""
    **Current User:** {user_name}  
    **Role:** {user_role}  
    **Last Login:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
    """)
    
    if app_mode == "Visualization":
        show_visualization()
    elif app_mode == "Crime Analysis":
        show_crime_analysis()
    elif app_mode == "Chatbot":
        show_chatbot()