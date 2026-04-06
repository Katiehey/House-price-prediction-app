"""
Fetch South African property listings from the Apify Property24 scraper actor
(ashersam01/property24-africa-scraper) and save the raw results to a CSV file.

Usage:
    python fetch_property24_data.py --token YOUR_APIFY_TOKEN [--max-items 2000]

The script will:
  1. Start a new actor run (or wait for an existing one).
  2. Poll until the run finishes.
  3. Flatten the result items into a tidy DataFrame.
  4. Save to property24_raw.csv for use by train_sa_model.py.
"""

import argparse
import json
import sys
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Apify helpers
# ---------------------------------------------------------------------------
ACTOR_ID = "ashersam01~property24-africa-scraper"
BASE_URL = "https://api.apify.com/v2"


def run_actor(token: str, max_items: int) -> str:
    """Start an actor run and return the run ID."""
    url = f"{BASE_URL}/acts/{ACTOR_ID}/runs?token={token}"
    
    payload = {
        "country": "south_africa",
        "maxItems": max_items,
        "startUrls": [{"url": "https://property24.com"}]
    }
    
    resp = requests.post(url, json=payload, timeout=30)
    
    if resp.status_code != 201 and resp.status_code != 200:
        print(f"Error Detail: {resp.text}")
        
    resp.raise_for_status()
    data = resp.json()
    run_id = data["data"]["id"]
    print(f"Actor run started: {run_id}")
    return run_id




def wait_for_run(token: str, run_id: str, poll_secs: int = 15) -> str:
    """Poll the run status until it finishes; return the default dataset ID."""
    url = f"{BASE_URL}/actor-runs/{run_id}?token={token}"
    while True:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        run = resp.json()["data"]
        status = run["status"]
        print(f"  Run status: {status}")
        if status in ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"):
            if status != "SUCCEEDED":
                raise RuntimeError(f"Actor run ended with status: {status}")
            return run["defaultDatasetId"]
        time.sleep(poll_secs)


def fetch_dataset(token: str, dataset_id: str) -> list[dict]:
    """Download all items from a dataset."""
    url = f"{BASE_URL}/datasets/{dataset_id}/items?token={token}&format=json&clean=true"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    items = resp.json()
    print(f"  Fetched {len(items)} items from dataset {dataset_id}")
    return items


# ---------------------------------------------------------------------------
# Data normalisation
# ---------------------------------------------------------------------------
# Map common raw field names from the scraper to our standard column names.
# Adjust these if the actor returns different key names.
FIELD_MAP = {
    # price
    "price": "price_zar",
    "listingPrice": "price_zar",

    # location
    "suburb": "suburb",
    "area": "suburb",
    "city": "city",
    "province": "province",

    # sizing
    "floorSize": "floor_size_m2",
    "floor_size": "floor_size_m2",
    "erfSize": "erf_size_m2",
    "erf_size": "erf_size_m2",
    "landSize": "erf_size_m2",

    # rooms
    "bedrooms": "bedrooms",
    "bathrooms": "bathrooms",
    "parkings": "parkings",
    "garages": "garages",
    "parking": "parkings",

    # type / features
    "propertyType": "property_type",
    "property_type": "property_type",
    "type": "property_type",
    "title": "title",
    "description": "description",
    "url": "url",
}

KEEP_COLUMNS = [
    "price_zar",
    "province",
    "city",
    "suburb",
    "property_type",
    "bedrooms",
    "bathrooms",
    "garages",
    "parkings",
    "floor_size_m2",
    "erf_size_m2",
    "title",
    "url",
]


def normalise_items(items: list[dict]) -> pd.DataFrame:
    """Flatten and rename raw scraper items into a clean DataFrame."""
    rows = []
    for item in items:
        row = {}
        for raw_key, std_key in FIELD_MAP.items():
            if raw_key in item:
                row[std_key] = item[raw_key]
        # Carry over anything not in the map for later inspection
        for k, v in item.items():
            if k not in row:
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)

    # Keep only our standard columns that exist in the data
    present = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[present]

        # Strip "R" and spaces before converting to numbers
    for num_col in ["price_zar", "floor_size_m2", "erf_size_m2", "bedrooms", "bathrooms", "garages", "parkings"]:
        if num_col in df.columns:
            # This regex removes everything except digits and dots
            df[num_col] = df[num_col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fetch Property24 listings via Apify")
    parser.add_argument("--token", required=True, help="Your Apify API token")
    parser.add_argument("--max-items", type=int, default=2000,
                        help="Maximum listings to fetch (default: 2000)")
    parser.add_argument("--dataset-id", default=None,
                        help="Skip actor run and fetch from an existing dataset ID")
    parser.add_argument("--output", default="property24_raw.csv",
                        help="Output CSV path (default: property24_raw.csv)")
    args = parser.parse_args()

    if args.dataset_id:
        dataset_id = args.dataset_id
        print(f"Using existing dataset: {dataset_id}")
    else:
        run_id = run_actor(args.token, args.max_items)
        dataset_id = wait_for_run(args.token, run_id)

    items = fetch_dataset(args.token, dataset_id)
    if not items:
        print("No items returned — check actor configuration.")
        sys.exit(1)

    df = normalise_items(items)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows to {args.output}")
    print(df.dtypes)
    print(df.head())


if __name__ == "__main__":
    main()
