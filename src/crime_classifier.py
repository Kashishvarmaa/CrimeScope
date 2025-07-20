# crime_classifier.py
import re
from collections import Counter

def classify_crime_types(df, crime_keywords):
    """Enhanced crime classification with multi-label support"""
    # Expanded crime keywords
    CRIME_KEYWORDS = {
        'theft': ['theft', 'robbery', 'stolen', 'loot', 'snatch', 'burglary'],
        'assault': ['assault', 'attack', 'battery', 'hit', 'beat', 'violence'],
        'fraud': ['fraud', 'scam', 'cheat', 'swindle', 'forgery', 'embezzlement'],
        'murder': ['murder', 'homicide', 'kill', 'slay', 'assassinate'],
        'sexual': ['rape', 'molest', 'abuse', 'harassment', 'assault'],
        'drug': ['drug', 'narcotic', 'heroin', 'cocaine', 'opioid', 'smuggle']
    }
    
    def detect_crimes(text):
        text = str(text).lower()
        crimes_found = []
        for crime, keywords in CRIME_KEYWORDS.items():
            if any(re.search(r'\b' + re.escape(keyword) + r'\b', text) for keyword in keywords):
                crimes_found.append(crime)
        return ', '.join(crimes_found) if crimes_found else 'unknown'
    
    df['crime_type'] = df['clean_text'].apply(detect_crimes)
    return df