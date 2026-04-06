import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import os

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

# ---------------------------------------------------------------------------
# SA reference data
# ---------------------------------------------------------------------------
SA_PROVINCES = [
    "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
    "Limpopo", "Mpumalanga", "North West", "Free State", "Northern Cape",
]

# Curated suburb list per province (top property markets)
SUBURBS_BY_PROVINCE: dict[str, list[str]] = {
    "Gauteng": [
        "Sandton", "Rosebank", "Morningside", "Fourways", "Midrand",
        "Centurion", "Pretoria East", "Bryanston", "Randburg", "Edenvale",
        "Bedfordview", "Boksburg", "Soweto", "Alexandra", "Maboneng",
        "Northcliff", "Westcliff", "Houghton", "Parktown", "Melville",
    ],
    "Western Cape": [
        "Camps Bay", "Clifton", "Sea Point", "Green Point", "Waterfront",
        "Constantia", "Bishopscourt", "Claremont", "Rondebosch", "Newlands",
        "Stellenbosch", "Paarl", "Somerset West", "Strand", "George",
        "Knysna", "Plettenberg Bay", "Franschhoek", "Hermanus", "Mossel Bay",
    ],
    "KwaZulu-Natal": [
        "Umhlanga", "Ballito", "La Lucia", "Durban North", "Berea",
        "Glenwood", "Westville", "Hillcrest", "Pinetown", "Amanzimtoti",
        "Pietermaritzburg", "Howick", "Margate", "Port Shepstone", "Richards Bay",
    ],
    "Eastern Cape": [
        "Gqeberha (Port Elizabeth)", "Summerstrand", "Humewood", "Jeffreys Bay",
        "East London", "Vincent", "Beacon Bay", "Mdantsane", "Grahamstown",
    ],
    "Limpopo": [
        "Polokwane", "Tzaneen", "Phalaborwa", "Louis Trichardt", "Mokopane",
    ],
    "Mpumalanga": [
        "Mbombela (Nelspruit)", "White River", "Hazyview", "Witbank (eMalahleni)", "Secunda",
    ],
    "North West": [
        "Rustenburg", "Potchefstroom", "Klerksdorp", "Hartbeespoort", "Brits",
    ],
    "Free State": [
        "Bloemfontein", "Welkom", "Bethlehem", "Sasolburg", "Parys",
    ],
    "Northern Cape": [
        "Kimberley", "Upington", "Springbok", "De Aar", "Kuruman",
    ],
}

PROPERTY_TYPES = [
    "House",
    "Apartment",
    "Townhouse",
    "Cluster",
    "Vacant Land",
    "Farm",
    "Commercial",
]

# ---------------------------------------------------------------------------
# Bond & Transfer cost calculators (SA-specific)
# ---------------------------------------------------------------------------
TRANSFER_DUTY_BRACKETS = [
    (1_100_000, 0.00),
    (1_512_500, 0.03),
    (2_117_500, 0.06),
    (2_722_500, 0.08),
    (12_100_000, 0.11),
    (float("inf"), 0.13),
]
TRANSFER_DUTY_THRESHOLDS = [0, 1_100_000, 1_512_500, 2_117_500, 2_722_500, 12_100_000]


# ---------------------------------------------------------------------------
# Updated 2026/27 SARS Transfer Duty Brackets
# ---------------------------------------------------------------------------
def calculate_transfer_duty(price: float) -> float:
    """Calculate SARS transfer duty for 2026/27 tax year (Effective 1 April 2026)."""
    if price <= 1_210_000:
        return 0.0
    elif price <= 1_663_800:
        return (price - 1_210_000) * 0.03
    elif price <= 2_329_300:
        return 13_614 + (price - 1_663_800) * 0.06
    elif price <= 2_994_800:
        return 53_544 + (price - 2_329_300) * 0.08
    elif price <= 13_310_000:
        return 106_784 + (price - 2_994_800) * 0.11
    else:
        return 1_241_456 + (price - 13_310_000) * 0.13

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
    suburb_options = SUBURBS_BY_PROVINCE.get(province, ["Other"])
    suburb = st.selectbox("Suburb", suburb_options + ["Other"])
    if suburb == "Other":
        suburb = st.text_input("Enter suburb name", value="")

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
    
    try:
        df_for_count = pd.read_csv("property24_rescued.csv")
        listing_count = len(df_for_count)
    except:
        listing_count = 1284 # Fallback to your known rescued count

    st.caption(f"Note: Current model has a MAE of ~R890k based on {listing_count} local listings.")

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

    cost_items = {
        "Purchase price": predicted_price,
        "Deposit": -deposit,
        "Transfer duty (SARS)": transfer_duty,
        "Conveyancing attorney": conveyancing,
        **bond_costs,
    }

    cost_df = pd.DataFrame(
        [(k, f"R {v:,.0f}") for k, v in cost_items.items()],
        columns=["Item", "Amount"],
    )
    st.table(cost_df)

    total_upfront = transfer_duty + conveyancing + sum(bond_costs.values()) + deposit
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
    "Data sourced from Property24 via Apify scraper · "
    "Transfer duty rates per SARS 2026/27 · "
    "Bond costs are estimates only — consult a bond originator for exact figures."
)
