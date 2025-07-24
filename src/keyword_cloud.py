# keyword_cloud.py
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_keyword_cloud(text_data, output_path=None):
    """
    Generate a word cloud from text data
    Args:
        text_data: Either a DataFrame with 'cleaned_text' column or a string of concatenated text
        output_path: Optional path to save the image. If None, won't save to file.
    Returns:
        matplotlib figure object
    """
    # Handle both DataFrame and string input
    if isinstance(text_data, pd.DataFrame):
        crime_descriptions = " ".join(text_data['clean_text'].dropna().astype(str))
    else:
        crime_descriptions = str(text_data)

    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, 
                         background_color='white',
                         colormap='viridis',
                         max_words=100).generate(crime_descriptions)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Save if output path provided
    if output_path:
        plt.savefig(output_path)
        print(f"Word Cloud saved as {output_path}")
    
    return fig