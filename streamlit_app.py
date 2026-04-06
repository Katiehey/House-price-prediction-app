import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import os
from constants import SA_PROVINCES, SUBURBS_BY_PROVINCE, PROPERTY_TYPES, calculate_transfer_duty

# ---------------------------------------------------------------------------
# Paths — SA model takes priority; fall back to legacy Ames model with warning
# ---------------------------------------------------------------------------
SA_MODEL_PATH   = "sa_house_price_pipeline.joblib"
SA_FEATURES_PATH = "sa_feature_list.json"

LEGACY_MODEL_PATH    = "house_price_prediction_pipeline.joblib"
LEGACY_FEATURES_PATH = "feature_list.json"

SA_NUMERICAL_FEATURES   = ["floor_size_m2", "erf_size_m2", "bedrooms", "bathrooms", "garages", "parkings"]
SA_CATEGORICAL_FEATURES = ["province", "property_type", "suburb"]

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
using_sa_model = os.path.exists(SA_MODEL_PATH)

try:
    if using_sa_model:
        pipeline = joblib.load(SA_MODEL_PATH)
        with open(SA_FEATURES_PATH, "r") as f:
            raw_feature_names = json.load(f)
    else:
        pipeline = joblib.load(LEGACY_MODEL_PATH)
        with open(LEGACY_FEATURES_PATH, "r") as f:
            raw_feature_names = json.load(f)
except FileNotFoundError:
    st.error("No model file found. Run `train_sa_model.py` to generate `sa_house_price_pipeline.joblib`.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


# Load model metadata (MAE and Listing Count)
try:
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
        display_mae = metadata.get("mae", 890000)
        display_listings = metadata.get("listings", 1284)
except Exception:
    display_mae = 890000
    display_listings = 1284


# --- NEW: Load the actual rescued suburbs for smart filtering ---
try:
    if os.path.exists("property24_rescued.csv"):
        rescued_df = pd.read_csv("property24_rescued.csv")
        # New Filter: No numbers, at least 3 chars, not "Unknown"
        DYNAMIC_SUBURBS = {}
        for prov in SA_PROVINCES:
            raw_subs = rescued_df[rescued_df['province'] == prov]['suburb'].unique()
            valid_subs = [s for s in raw_subs if len(str(s)) > 2 and not any(c.isdigit() for c in str(s)) and s != "Unknown"]
            DYNAMIC_SUBURBS[prov] = valid_subs
    else:
        DYNAMIC_SUBURBS = {}
except Exception:
    DYNAMIC_SUBURBS = {}


# Updated Bond Fees for 2026
def calculate_bond_costs(bond_amount: float) -> dict:
    """Estimate 2026 bond registration and initiation fees."""
    initiation_fee = 6_037.50 # Fixed NCA cap for 2026
    # 2026 Deeds Office fee for bond registration
    registration_fee = 2_408 if bond_amount > 2_000_000 else 1_738 
    # Recommended attorney scale (~1.5% of value for lower bands)
    attorney_fee = 26_385 if bond_amount > 1_000_000 else 15_725 
    
    return {
        "Bank Initiation fee": initiation_fee,
        "Deeds Office Registration": registration_fee,
        "Bond Attorney fee (incl. VAT)": attorney_fee * 1.15,
    }

if __name__ == "__main__":


    # ---------------------------------------------------------------------------
    # Page config
    # ---------------------------------------------------------------------------
    st.set_page_config(
        page_title="SA Property Price Predictor",
        page_icon="🏠",
        layout="wide",
    )

    st.title("SA Property Price Predictor")
    st.caption("Powered by Property24 market data · Prices in ZAR · Areas in m²")

    if not using_sa_model:
        st.warning(
            "**Demo mode** — the SA-trained model has not been generated yet. "
            "The predictions shown use the legacy Ames (Iowa) model and are **not valid** for South African properties. "
            "Run `fetch_property24_data.py` then `train_sa_model.py` to generate the SA model.",
            icon="⚠️",
        )

    # ---------------------------------------------------------------------------
    # Main prediction form
    # ---------------------------------------------------------------------------
    st.header("Property Details")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Location")
        province = st.selectbox("Province", SA_PROVINCES)
        
        # PRIORITIZE: Use suburbs from the CSV, fall back to hardcoded list if CSV is missing
        if province in DYNAMIC_SUBURBS:
            suburb_options = sorted(DYNAMIC_SUBURBS[province])
            data_source_msg = "Showing suburbs from 2026 dataset"
        else:
            suburb_options = SUBURBS_BY_PROVINCE.get(province, ["Other"])
            data_source_msg = "Showing curated top markets"

        suburb = st.selectbox("Suburb", suburb_options + ["Other"])
        st.caption(f"📍 {data_source_msg}") # Let the user know why they see these names
        
        if suburb == "Other":
            suburb = st.text_input("Enter suburb name", value="", placeholder="e.g. Pofadder")
            st.info("💡 Since this location is not in our primary database, SA Property Predictor will use regional averages for the estimate.")

        st.subheader("Property Type")
        property_type = st.selectbox("Type", PROPERTY_TYPES)

    with col2:
        st.subheader("Size")
        floor_size_m2 = st.number_input("Floor size (m²)", min_value=0.0, value=120.0, step=5.0)
        erf_size_m2   = st.number_input("Erf / plot size (m²)", min_value=0.0, value=500.0, step=10.0,
                                        help="Leave 0 for apartments or sectional-title units")

        st.subheader("Rooms")
        bedrooms  = st.number_input("Bedrooms",  min_value=0, max_value=20, value=3, step=1)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2, step=1)
        garages   = st.number_input("Garages",   min_value=0, max_value=10, value=1, step=1)
        parkings  = st.number_input("Extra parking bays", min_value=0, max_value=10, value=0, step=1)

    # ---------------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------------
    st.subheader("Financial Preferences")
    deposit_pct = st.slider("Deposit (%)", min_value=0, max_value=50, value=10)

    if st.button("Predict Property Value", type="primary"):
        if floor_size_m2 <= 0 and erf_size_m2 <= 0:
            st.error("Please provide at least a Floor Size or Erf Size for a valid prediction.")
            st.stop()
        if using_sa_model:
            input_data = {
                "floor_size_m2": floor_size_m2,
                "erf_size_m2":   erf_size_m2,
                "bedrooms":      float(bedrooms),
                "bathrooms":     float(bathrooms),
                "garages":       float(garages),
                "parkings":      float(parkings),
                "province":      province,
                "property_type": property_type,
                "suburb":        suburb if suburb else "Unknown",
            }
            input_df = pd.DataFrame([input_data])
            input_df = input_df[raw_feature_names] 
            predicted_price = pipeline.predict(input_df)[0]
        else:
            # Legacy fallback logic...
            legacy_inputs = {feat: 0.0 for feat in raw_feature_names}
            # ... (keep your legacy mapping code here)
            input_df = pd.DataFrame([legacy_inputs])
            usd_price = pipeline.predict(input_df)[0]
            predicted_price = usd_price * 18.5

        # --- EVERYTHING BELOW IS NOW INSIDE THE BUTTON BLOCK ---
        st.success(f"Estimated Property Value: **R {predicted_price:,.0f}**")

        lower_bound = predicted_price * 0.85
        upper_bound = predicted_price * 1.15

        st.info(f"💡 **Market Estimate Range:** R {lower_bound:,.0f} — R {upper_bound:,.0f}")
        

        st.caption(f"Note: Current model has a Mean Absolute Error of **R {display_mae:,.0f}** based on **{display_listings:,}** local listings.")

        # Price band logic
        if predicted_price < 480_000:
            band = "Affordable (below R480k)"
        elif predicted_price < 1_500_000:
            band = "Middle market (R480k – R1.5m)"
        elif predicted_price < 3_500_000:
            band = "Upper-middle (R1.5m – R3.5m)"
        else:
            band = "Luxury (above R3.5m)"
        st.info(f"Market segment: {band}")

        # Cost breakdown calculations
        deposit = predicted_price * deposit_pct / 100
        bond_amount = predicted_price - deposit
        transfer_duty = calculate_transfer_duty(predicted_price)
        bond_costs = calculate_bond_costs(bond_amount)
        conveyancing = predicted_price * 0.015 

        # Create a more professional Cost Table
        cost_data = [
            {"Item": "Purchase Price", "Amount": predicted_price, "Type": "Market Value"},
            {"Item": "Deposit", "Amount": -deposit, "Type": "Out-of-pocket"},
            {"Item": "Transfer Duty (SARS)", "Amount": transfer_duty, "Type": "Fixed Tax"},
            {"Item": "Conveyancing Attorney", "Amount": conveyancing, "Type": "Estimate"},
            {"Item": "Bank Initiation Fee", "Amount": bond_costs["Bank Initiation fee"], "Type": "Fixed Fee"},
            {"Item": "Deeds Office Reg.", "Amount": bond_costs["Deeds Office Registration"], "Type": "Fixed Fee"},
            {"Item": "Bond Attorney Fee", "Amount": bond_costs["Bond Attorney fee (incl. VAT)"], "Type": "Estimate"}
        ]

        cost_df = pd.DataFrame(cost_data)
        # Format currency for display
        cost_df["Amount"] = cost_df["Amount"].apply(lambda x: f"R {x:,.0f}")
        st.table(cost_df)

        total_bond_fees = bond_costs["Bank Initiation fee"] + bond_costs["Deeds Office Registration"] + bond_costs["Bond Attorney fee (incl. VAT)"]
        total_upfront = transfer_duty + conveyancing + total_bond_fees + deposit
        st.metric("Total cash needed upfront", f"R {total_upfront:,.0f}")

    # ---------------------------------------------------------------------------
    # Standalone calculators
    # ---------------------------------------------------------------------------
    st.divider()
    st.header("Calculators")

    calc_tab1, calc_tab2 = st.tabs(["Bond Repayment", "Transfer Duty"])

    with calc_tab1:
        st.subheader("Monthly Bond Repayment")
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            loan_amount = st.number_input("Loan amount (R)", min_value=0.0, value=1_000_000.0, step=10_000.0)
        with bc2:
            # UPDATED: 2026 Prime Rate is ~10.25%
            interest_rate = st.number_input("Interest rate (%)", min_value=0.0, max_value=30.0, value=10.25, step=0.25,
                                            help="South African prime rate (April 2026) is 10.25%")
        with bc3:
            loan_term_years = st.number_input("Loan term (years)", min_value=1, max_value=30, value=20)

        if loan_amount > 0:
            n = loan_term_years * 12
            if interest_rate > 0:
                r = (interest_rate / 100) / 12
                monthly = loan_amount * r / (1 - (1 + r) ** -n)
            else:
            # 0% interest is just the loan divided by months
                monthly = loan_amount / n
        
            total_repaid = monthly * n
        
            st.metric("Monthly repayment", f"R {monthly:,.0f}")
            st.metric("Total repaid over term", f"R {total_repaid:,.0f}")
            st.metric("Total interest", f"R {total_repaid - loan_amount:,.0f}")

    with calc_tab2:
        st.subheader("Transfer Duty (SARS 2026/27 Rates)")
        td_price = st.number_input("Property purchase price (R)", min_value=0.0, value=1_500_000.0, step=50_000.0)
        duty = calculate_transfer_duty(td_price)
        if td_price <= 1_210_000:
            st.success(f"No transfer duty payable — properties below R1,210,000 are exempt.")
        else:
            st.metric("Transfer duty payable", f"R {duty:,.0f}")
            st.caption(f"That is {duty/td_price*100:.2f}% of the purchase price.")

    # ---------------------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------------------
    st.divider()
    st.caption(
        f"Data sourced from Property24 · Analyzed {display_listings:,} listings · "
        f"Model Accuracy (MAE): R {display_mae:,.0f} · "
        "Transfer duty rates per SARS 2026/27."
    )
