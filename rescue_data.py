import pandas as pd
import numpy as np
import re
from streamlit_app import SUBURBS_BY_PROVINCE

# Load your existing raw data
df = pd.read_csv('property24_raw.csv')
print("REAL COLUMN NAMES IN CSV:", df.columns.tolist())

# 1. CRITICAL: Remove Outliers and Garbage
df = df.dropna(subset=['price_zar'])
df = df[(df['price_zar'] >= 150_000) & (df['price_zar'] <= 30_000_000)]

# 2. FIX CORRUPT DATA
for col in ['bedrooms', 'bathrooms', 'parkings', 'garages', 'floor_size_m2']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col] > 5000, col] = np.nan 

# 3. LOCATION LOGIC SETUP
MAJOR_SUBURBS = [
    "Sandton", "Bryanston", "Fourways", "Randburg", "Midrand", "Centurion", 
    "Rosebank", "Morningside", "Bedfordview", "Soweto", "Edenvale", "Boksburg",
    "Sea Point", "Camps Bay", "Constantia", "Rondebosch", "Claremont", "Stellenbosch",
    "Green Point", "Blouberg", "Somerset West", "George", "Knysna", "Paarl",
    "Umhlanga", "Ballito", "Durban North", "Berea", "Westville", "Hillcrest",
    "Gqeberha", "Summerstrand", "Jeffreys Bay", "Polokwane", "Nelspruit", 
    "Rustenburg", "Potchefstroom", "Bloemfontein", "Kimberley"
]

# 1. Create one big list of all suburbs from your app's dictionary
ALL_KNOWN_SUBURBS = []
for province_suburbs in SUBURBS_BY_PROVINCE.values():
    ALL_KNOWN_SUBURBS.extend(province_suburbs)

def extract_location_smart(text):
    if pd.isna(text) or str(text).strip() == "": 
        return "Unknown"
    
    text_str = str(text).strip()
    
    # 1. Strategy A: Check Official Suburbs first (KEEP THIS)
    for suburb in ALL_KNOWN_SUBURBS:
        if re.search(rf'\b{suburb}\b', text_str, re.IGNORECASE):
            return suburb

    # 2. Strategy B: Clean the "Last Word" logic (UPGRADED)
    words = text_str.split()
    if words:
        last_word = words[-1].replace(",", "").strip()
        
        # --- ADD THESE NEW FILTERS ---
        # A: Skip if it's just a number (Postal Code / Stand Number)
        if last_word.isdigit():
            return "Unknown"
        
        # B: Skip if it's too short or contains weird characters (like (C) or 14A)
        if len(last_word) < 3 or any(char in last_word for char in "()-"):
            return "Unknown"
            
        # C: Skip known road suffixes
        road_suffixes = ['Rd', 'Road', 'St', 'Street', 'Ave', 'Avenue', 'Dr', 'Drive']
        if last_word not in road_suffixes:
            return last_word.title()

    return "Unknown"

# --- IMPROVED STEP 4: MULTI-COLUMN LOCATION CLEANUP ---
def get_best_text_source(row):
    """
    Prioritize the cleanest location data first.
    """
    # 1. Try the actual 'suburb' column from the CSV first
    if pd.notna(row.get('suburb')) and str(row['suburb']).strip() != "":
        val = str(row['suburb']).strip()
        if val.lower() not in ['unknown', 'nan', 'none']:
            return val

    # 2. If suburb is empty, try the 'city' column
    if pd.notna(row.get('city')) and str(row['city']).strip() != "":
        val = str(row['city']).strip()
        if val.lower() not in ['unknown', 'nan', 'none']:
            return val
            
    # 3. Last resort: Extract from the 'title' (the address)
    if pd.notna(row.get('title')) and str(row['title']).strip() != "":
        return str(row['title'])
        
    return "Unknown"

# Apply the new multi-column source logic
df['source_text'] = df.apply(get_best_text_source, axis=1)
df['suburb'] = df['source_text'].apply(extract_location_smart)
df['suburb'] = df['suburb'].str.split(',').str[0].str.strip()

# Debugging prints
print("DEBUG: First 5 source texts used:", df['source_text'].head().tolist())
print("DEBUG: First 5 suburbs extracted:", df['suburb'].head().tolist())

# 5. SMART PROVINCE ASSIGNMENT
def guess_province(row):
    sub = str(row['suburb']).lower()
    
    # Check if the suburb/city belongs to a known province list
    for prov, suburbs in SUBURBS_BY_PROVINCE.items():
        if any(s.lower() in sub for s in suburbs):
            return prov
            
    # Hardcoded city fallbacks
    if "cape town" in sub or "stellenbosch" in sub: return "Western Cape"
    if "durban" in sub or "umhlanga" in sub: return "KwaZulu-Natal"
    if "joburg" in sub or "johannesburg" in sub or "pretoria" in sub: return "Gauteng"
    
    return 'Gauteng' # Default 

df['province'] = df.apply(guess_province, axis=1)

# 6. IMPROVED IMPUTATION
df = df.dropna(subset=['floor_size_m2'])

def estimate_bedrooms(row):
    if pd.notna(row['bedrooms']) and row['bedrooms'] > 0:
        return row['bedrooms']
    if row['floor_size_m2'] < 65: return 1.0
    if row['floor_size_m2'] < 115: return 2.0
    if row['floor_size_m2'] < 250: return 3.0
    return 4.0

df['bedrooms'] = df.apply(estimate_bedrooms, axis=1)

# 7. PROPERTY TYPE ESTIMATION
def estimate_type(row):
    pt = str(row.get('property_type', ''))
    if pt != 'nan' and pt != 'None' and pt != '':
        return pt
    if row['floor_size_m2'] < 90: return 'Apartment'
    if row['bedrooms'] >= 3: return 'House'
    return 'Townhouse'

df['property_type'] = df.apply(estimate_type, axis=1)

# 8. FINAL POLISH
columns_to_fill = {'bathrooms': 1.0, 'parkings': 1.0, 'garages': 0.0}
for col, fill_value in columns_to_fill.items():
    if col in df.columns:
        df[col] = df[col].fillna(fill_value)
    else:
        df[col] = fill_value

# Save the sanitized data
df = df[df['suburb'] != 'Unknown']
df.to_csv('property24_rescued.csv', index=False)

print("-" * 30)
print(f"✅ Data Rescued Successfully!")
print(f"Total Rows: {len(df)}")
print(f"Suburbs identified: {df['suburb'].nunique()}")
print(f"Sample Suburbs: {', '.join(df['suburb'].unique()[:5])}")
print("-" * 30)