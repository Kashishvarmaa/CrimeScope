# src/ner_extractor.py

import spacy
import pandas as pd
import os

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define common crime keywords (can expand later)
CRIME_KEYWORDS = [
    "murder", "theft", "robbery", "assault", "fraud", "kidnap", "rape", 
    "burglary", "arson", "violence", "scam", "cybercrime", "shooting"
]

def extract_entities(text):
    doc = nlp(text)
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
    crime_types = [word for word in text.lower().split() if word in CRIME_KEYWORDS]
    return list(set(locations)), list(set(crime_types))

def process_ner(input_path="data/cleaned_news.csv", output_path="data/ner_output.csv"):
    df = pd.read_csv(input_path)
    
    all_locations = []
    all_crimes = []

    for text in df["clean_text"]:
        locations, crimes = extract_entities(text)
        all_locations.append(", ".join(locations))
        all_crimes.append(", ".join(crimes))

    df["locations"] = all_locations
    df["crime_types"] = all_crimes

    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[+] Extracted NER info for {len(df)} articles -> saved to {output_path}")

if __name__ == "__main__":
    process_ner()
