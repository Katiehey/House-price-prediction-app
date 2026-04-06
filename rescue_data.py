import pandas as pd
import numpy as np
import re

# Load your existing raw data
df = pd.read_csv('property24_raw.csv')

# 1. CRITICAL: Remove Outliers and Garbage (Fixes the R1.5M Error)
# We drop anything above R30M (luxury farms/mansions) and below R150k (land/scams)
df = df.dropna(subset=['price_zar'])
df = df[(df['price_zar'] >= 150_000) & (df['price_zar'] <= 30_000_000)]

# 2. FIX CORRUPT DATA (The "900 Billion" Bedroom Fix)
# We convert to numeric and force any impossible numbers to NaN
for col in ['bedrooms', 'bathrooms', 'parkings', 'garages', 'floor_size_m2']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # If bedrooms > 15 or floor_size > 5000, it's likely a data error
        df.loc[df[col] > 5000, col] = np.nan 

# 3. RESCUE SUBURB FROM TITLE (The "Location" Fix)
def extract_location(row):
    title = str(row.get('title', ''))
    suburb_raw = str(row.get('suburb', ''))
    
    # Priority 1: If title contains "in [Suburb]", extract it
    if " in " in title:
        return title.split(" in ")[-1].strip()
    
    # Priority 2: If suburb doesn't look like an address (no numbers), keep it
    if suburb_raw != 'nan' and not any(char.isdigit() for char in suburb_raw):
        return suburb_raw
    
    return "Unknown"

df['suburb'] = df.apply(extract_location, axis=1)
df['province'] = 'Gauteng' # Keep as Gauteng if that was your scraper target

# 4. IMPROVED IMPUTATION
# We drop rows that have NO size data at all, as they are impossible to predict
df = df.dropna(subset=['floor_size_m2'])

def estimate_bedrooms(row):
    if pd.notna(row['bedrooms']) and row['bedrooms'] > 0:
        return row['bedrooms']
    if row['floor_size_m2'] < 65: return 1.0
    if row['floor_size_m2'] < 115: return 2.0
    if row['floor_size_m2'] < 250: return 3.0
    return 4.0

df['bedrooms'] = df.apply(estimate_bedrooms, axis=1)

# 5. PROPERTY TYPE ESTIMATION
def estimate_type(row):
    pt = str(row.get('property_type', ''))
    if pt != 'nan' and pt != 'None':
        return pt
    if row['floor_size_m2'] < 90: return 'Apartment'
    if row['bedrooms'] >= 3: return 'House'
    return 'Townhouse'

df['property_type'] = df.apply(estimate_type, axis=1)

# 6. FINAL POLISH (Safe Column Check)
columns_to_fill = {
    'bathrooms': 1.0,
    'parkings': 1.0,
    'garages': 0.0
}

for col, fill_value in columns_to_fill.items():
    if col in df.columns:
        df[col] = df[col].fillna(fill_value)
    else:
        # If the column is missing entirely, create it with the default value
        df[col] = fill_value

# Save the sanitized data
df.to_csv('property24_rescued.csv', index=False)

print("-" * 30)
print(f"✅ Data Rescued Successfully!")
print(f"Total Rows: {len(df)}")
print(f"Suburbs identified: {df['suburb'].nunique()}")
print(f"Sample Suburbs: {', '.join(df['suburb'].unique()[:5])}")
print("-" * 30)