# trend_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_trends(df=None, return_fig=False):
    """Generate trend analysis with separate lines for each crime type"""
    if df is None:
        df = pd.read_csv("data/geo_news.csv")
    
    # Ensure we have the required columns
    if 'publishedAt' not in df.columns or 'crime_type' not in df.columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data for trend analysis", 
               ha='center', va='center')
        ax.axis('off')
        return fig if return_fig else plt.show()
    
    # Convert and set datetime index
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df.set_index('publishedAt', inplace=True)
    
    # Resample by crime type
    crime_trends = df.groupby('crime_type').resample('ME').size().unstack(level=0)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    for crime in crime_trends.columns:
        crime_trends[crime].plot(ax=ax, label=crime, marker='o')
    
    ax.set_title("Crime Trends by Type (Monthly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.legend(title="Crime Type")
    plt.tight_layout()
    
    return fig if return_fig else plt.show()