import pandas as pd
import numpy as np

# Load your existing raw data
df = pd.read_csv('property24_raw.csv')

# 1. Fix the Price (Remove rows where price is missing)
df = df.dropna(subset=['price_zar'])

# 2. Rescue 'Bedrooms'
# In SA, an 87m2 place (like your row 0) is almost always 2 bedrooms.
# We will estimate bedrooms based on floor size if missing.
def estimate_bedrooms(row):
    if pd.notna(row['bedrooms']):
        return row['bedrooms']
    if row['floor_size_m2'] < 60: return 1
    if row['floor_size_m2'] < 110: return 2
    if row['floor_size_m2'] < 200: return 3
    return 4

df['bedrooms'] = df.apply(estimate_bedrooms, axis=1)

# 3. Fix 'Property Type'
# If erf_size is large, it's a House. If floor_size is small, it's an Apartment.
def estimate_type(row):
    if pd.notna(row['property_type']):
        return row['property_type']
    if row['floor_size_m2'] < 100: return 'Apartment'
    return 'House'

df['property_type'] = df.apply(estimate_type, axis=1)

# 4. Cleanup 'Suburb'
# Since it currently has addresses like "22 North Rd", 
# we'll label them "Unknown Suburb" for now so the model doesn't 
# try to learn "22 North Rd" as a location.
df['suburb'] = 'Unknown'
df['city'] = 'South Africa' # Generalised
df['province'] = 'Gauteng'    # Defaulting to the largest market for now

# 5. Final Cleaning
df['bathrooms'] = df['bathrooms'].fillna(1)
df['parkings'] = df['parkings'].fillna(1)

# Save this as the new "Raw" file for your training script
df.to_csv('property24_rescued.csv', index=False)
print(f"Data Rescued! Saved {len(df)} rows to property24_rescued.csv")