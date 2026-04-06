"""
Train a South African house price prediction model from Property24 data.

Usage:
    python train_sa_model.py --input property24_raw.csv

Outputs:
    sa_house_price_pipeline.joblib  — trained sklearn pipeline
    sa_feature_list.json            — ordered feature names for the Streamlit app
    training_report.txt             — basic model diagnostics
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature definitions for the SA model
# ---------------------------------------------------------------------------
NUMERICAL_FEATURES = [
    "floor_size_m2",   # internal living area (m²)
    "erf_size_m2",     # plot / land size (m²)
    "bedrooms",
    "bathrooms",
    "garages",
    "parkings",
]

CATEGORICAL_FEATURES = [
    "province",
    "property_type",
    "suburb",          # high-cardinality but very predictive; OHE handles it fine
]

TARGET = "price_zar"

# ---------------------------------------------------------------------------
# Price band helper (used for reporting only)
# ---------------------------------------------------------------------------
PRICE_BANDS = [
    (0,        480_000,    "Affordable (< R480k)"),
    (480_000,  1_500_000,  "Middle (R480k – R1.5m)"),
    (1_500_000, 3_500_000, "Upper-Middle (R1.5m – R3.5m)"),
    (3_500_000, float("inf"), "Luxury (> R3.5m)"),
]


def assign_band(price: float) -> str:
    for lo, hi, label in PRICE_BANDS:
        if lo <= price < hi:
            return label
    return "Unknown"


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Raw rows: {len(df)}")

    # Drop rows without a target price
    df = df.dropna(subset=[TARGET])
    
    # Ensure Target is numeric
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce')
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET] > 0]

    # Sanity-check price range: drop extreme outliers (< R50k or > R30m)
    # Using R30M matches your rescue script logic
    df = df[(df[TARGET] >= 50_000) & (df[TARGET] <= 30_000_000)]

    # Normalise property_type labels - ADDED .astype(str) HERE
    if "property_type" in df.columns:
        df["property_type"] = (
            df["property_type"]
            .astype(str)  # Fixes the AttributeError
            .str.strip()
            .str.title()
            .replace({
                "House": "House",
                "Apartment": "Apartment",
                "Flat": "Apartment",
                "Townhouse": "Townhouse",
                "Cluster Home": "Cluster",
                "Cluster": "Cluster",
                "Vacant Land": "Vacant Land",
                "Farm": "Farm",
                "Commercial": "Commercial",
                "Nan": "Unknown" # Handle string-converted NaNs
            })
        )
    else:
        df["property_type"] = "Unknown"

    # Fill missing categoricals - ADDED .astype(str) HERE
    for col in ["province", "suburb"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown").str.strip()
        else:
            df[col] = "Unknown"

    # Ensure all numerical features exist and are numeric
    for col in NUMERICAL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"Clean rows: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, NUMERICAL_FEATURES),
        ("cat", cat_transformer, CATEGORICAL_FEATURES),
    ])

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame):
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples…")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    report_lines = [
        "=== SA House Price Model — Training Report ===",
        f"Training samples : {len(X_train):,}",
        f"Test samples     : {len(X_test):,}",
        "",
        f"R²   : {r2:.4f}",
        f"MAE  : R{mae:,.0f}",
        f"RMSE : R{rmse:,.0f}",
        "",
        "Price band distribution (test set):",
    ]
    bands = pd.Series(y_test).apply(assign_band).value_counts()
    for band, count in bands.items():
        report_lines.append(f"  {band}: {count}")

    report_lines += [
        "",
        "Numerical features: " + ", ".join(NUMERICAL_FEATURES),
        "Categorical features: " + ", ".join(CATEGORICAL_FEATURES),
    ]

    report = "\n".join(report_lines)
    print("\n" + report)

    with open("training_report.txt", "w") as f:
        f.write(report)
    print("\nSaved training_report.txt")

    return pipeline, mae, len(df)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SA house price model")
    parser.add_argument("--input", default="property24_raw.csv",
                        help="Path to cleaned CSV from fetch_property24_data.py")
    parser.add_argument("--model-out", default="sa_house_price_pipeline.joblib",
                        help="Output model path")
    parser.add_argument("--features-out", default="sa_feature_list.json",
                        help="Output feature list JSON path")
    args = parser.parse_args()

    df = load_and_clean(args.input)
    pipeline, mae_result, total_listings = train(df)

    joblib.dump(pipeline, args.model_out)
    print(f"Saved model to {args.model_out}")

    metadata = {
        "mae": float(mae_result),
        "listings": int(total_listings)
    }
    with open("model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved model_metadata.json")
    # --------------------------------

    feature_list = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    with open(args.features_out, "w") as f:
        json.dump(feature_list, f, indent=2)
    print(f"Saved feature list to {args.features_out}")


if __name__ == "__main__":
    main()
