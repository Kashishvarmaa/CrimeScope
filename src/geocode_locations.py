# import pandas as pd
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter
# from tqdm import tqdm
# import os

# def geocode_locations(input_file="data/ner_output.csv", output_file="data/geo_news.csv"):
#     # Load data
#     df = pd.read_csv(input_file)

#     # Initialize geocoder
#     geolocator = Nominatim(user_agent="crime-analysis")
#     geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

#     # Store geocoded data
#     latitudes = []
#     longitudes = []

#     tqdm.pandas(desc="Geocoding locations")

#     for loc in tqdm(df['locations']):
#         try:
#             location = geocode(loc)
#             if location:
#                 latitudes.append(location.latitude)
#                 longitudes.append(location.longitude)
#             else:
#                 latitudes.append(None)
#                 longitudes.append(None)
#         except:
#             latitudes.append(None)
#             longitudes.append(None)

#     df['latitude'] = latitudes
#     df['longitude'] = longitudes

#     os.makedirs("data", exist_ok=True)
#     df.to_csv(output_file, index=False)
#     print(f"[✓] Saved geocoded data to {output_file}")

# if __name__ == "__main__":
#     geocode_locations()


# geocode_locations.py
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

def geocode_locations(df):
    """Geocode locations in a DataFrame"""
    if not isinstance(df, pd.DataFrame):
        df = pd.read_csv(df)
    
    if 'locations' not in df.columns:
        return df
    
    geolocator = Nominatim(user_agent="crime-analysis")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    
    latitudes = []
    longitudes = []
    
    for loc in tqdm(df['locations']):
        try:
            location = geocode(loc)
            if location:
                latitudes.append(location.latitude)
                longitudes.append(location.longitude)
            else:
                latitudes.append(None)
                longitudes.append(None)
        except:
            latitudes.append(None)
            longitudes.append(None)
    
    df['latitude'] = latitudes
    df['longitude'] = longitudes
    
    return df