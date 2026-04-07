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
SA_MODEL_PATH    = "sa_house_price_pipeline.joblib"
SA_FEATURES_PATH = "sa_feature_list.json"

LEGACY_MODEL_PATH    = "house_price_prediction_pipeline.joblib"
LEGACY_FEATURES_PATH = "feature_list.json"

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
# Model metadata (MAE and listing count from training)
# ---------------------------------------------------------------------------
try:
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
        display_mae = metadata.get("mae", 890_000)
        # "listings" is the full dataset size; training set is ~80% of that
        total_listings = metadata.get("listings", 1_284)
        training_listings = int(total_listings * 0.8)
except Exception:
    display_mae = 890_000
    total_listings = 1_284
    training_listings = int(total_listings * 0.8)

# ---------------------------------------------------------------------------
# Load actual rescued suburbs per province for smart filtering
# ---------------------------------------------------------------------------
PROVINCE_LISTING_COUNTS: dict[str, int] = {}
try:
    if os.path.exists("property24_rescued.csv"):
        rescued_df = pd.read_csv("property24_rescued.csv")
        DYNAMIC_SUBURBS: dict[str, list[str]] = {}
        for prov in SA_PROVINCES:
            prov_df = rescued_df[rescued_df["province"] == prov]
            PROVINCE_LISTING_COUNTS[prov] = len(prov_df)
            raw_subs = prov_df["suburb"].unique()
            valid_subs = [
                s for s in raw_subs
                if len(str(s)) > 2
                and not any(c.isdigit() for c in str(s))
                and s != "Unknown"
            ]
            DYNAMIC_SUBURBS[prov] = sorted(valid_subs)
    else:
        DYNAMIC_SUBURBS = {}
except Exception:
    DYNAMIC_SUBURBS = {}

# ---------------------------------------------------------------------------
# Conveyancing fee — SA Law Society recommended tariff scale (+ 15% VAT)
# ---------------------------------------------------------------------------
def calculate_conveyancing(price: float) -> float:
    """SA Law Society recommended conveyancing tariff scale, VAT-inclusive."""
    _BRACKETS = [
        (100_000,    4_750,  0.0),
        (250_000,    4_750,  0.0150),
        (400_000,    7_000,  0.0130),
        (800_000,    8_950,  0.0120),
        (2_000_000,  13_750, 0.0100),
        (4_000_000,  25_750, 0.0075),
        (8_000_000,  40_750, 0.0050),
        (float("inf"), 60_750, 0.0025),
    ]
    prev_threshold = 0
    fee = 0.0
    for threshold, base, rate in _BRACKETS:
        if price <= prev_threshold:
            break
        chunk = min(price, threshold) - prev_threshold
        if prev_threshold == 0:
            fee = base
        else:
            fee += chunk * rate
        prev_threshold = threshold
        if price <= threshold:
            break
    return fee * 1.15  # VAT

# ---------------------------------------------------------------------------
# Bond registration costs (2026 Deeds Office scale)
# ---------------------------------------------------------------------------
def calculate_bond_costs(bond_amount: float) -> dict:
    initiation_fee = 6_037.50
    registration_fee = 2_408 if bond_amount > 2_000_000 else 1_738
    attorney_fee = 26_385 if bond_amount > 1_000_000 else 15_725
    return {
        "Bank initiation fee": initiation_fee,
        "Deeds Office registration": registration_fee,
        "Bond attorney fee (incl. VAT)": attorney_fee * 1.15,
    }

# ---------------------------------------------------------------------------
# Price band
# ---------------------------------------------------------------------------
def get_price_band(price: float) -> str:
    if price < 480_000:
        return "Affordable (below R480k)"
    elif price < 1_500_000:
        return "Middle market (R480k – R1.5m)"
    elif price < 3_500_000:
        return "Upper-middle (R1.5m – R3.5m)"
    return "Luxury (above R3.5m)"

# ===========================================================================
# PAGE LAYOUT
# ===========================================================================
st.set_page_config(
    page_title="SA Property Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — bond repayment calculator (always visible while filling the form)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Bond Repayment")
    st.caption("Quick calculator — update as you go")

    sb_loan = st.number_input(
        "Loan amount (R)", min_value=0.0, value=900_000.0, step=10_000.0, key="sb_loan"
    )
    sb_rate = st.number_input(
        "Interest rate (%)", min_value=0.0, max_value=30.0, value=10.25, step=0.25,
        help="SA prime rate (April 2026) is 10.25%", key="sb_rate"
    )
    sb_term = st.number_input(
        "Loan term (years)", min_value=1, max_value=30, value=20, key="sb_term"
    )

    if sb_loan > 0:
        n = sb_term * 12
        if sb_rate > 0:
            r = (sb_rate / 100) / 12
            monthly = sb_loan * r / (1 - (1 + r) ** -n)
        else:
            monthly = sb_loan / n
        total_repaid = monthly * n

        st.metric("Monthly repayment", f"R {monthly:,.0f}")
        st.metric("Total interest paid", f"R {total_repaid - sb_loan:,.0f}")

    st.divider()
    st.caption(
        f"Model trained on **{training_listings:,} listings**  \n"
        f"Mean Absolute Error: **R {display_mae:,.0f}**"
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("SA Property Price Predictor")
st.caption("Powered by Property24 market data · Prices in ZAR · Areas in m²")

if not using_sa_model:
    st.warning(
        "**Demo mode** — the SA-trained model has not been generated yet. "
        "Predictions use the legacy Ames (Iowa) model and are **not valid** for SA properties. "
        "Run `fetch_property24_data.py` then `train_sa_model.py` to generate the SA model.",
        icon="⚠️",
    )

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.header("Property Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Location")

    # Build province label with listing count when data is available
    if PROVINCE_LISTING_COUNTS:
        province_labels = [
            f"{p} ({PROVINCE_LISTING_COUNTS.get(p, 0)} listings)"
            for p in SA_PROVINCES
        ]
        province_label = st.selectbox("Province", province_labels)
        province = province_label.split(" (")[0]
    else:
        province = st.selectbox("Province", SA_PROVINCES)

    # Suburb list — prefer dynamic (from dataset) over hardcoded
    if province in DYNAMIC_SUBURBS and DYNAMIC_SUBURBS[province]:
        suburb_options = DYNAMIC_SUBURBS[province]
        data_source_msg = "Suburbs from 2026 dataset"
    else:
        suburb_options = SUBURBS_BY_PROVINCE.get(province, [])
        data_source_msg = "Curated top markets (no dataset coverage for this province)"

    suburb_choice = st.selectbox("Suburb", suburb_options + ["Other"])
    st.caption(f"📍 {data_source_msg}")

    custom_suburb_used = suburb_choice == "Other"
    if custom_suburb_used:
        suburb = st.text_input("Enter suburb name", value="", placeholder="e.g. Pofadder")
        st.info(
            "This suburb is not in our training data — the model will use provincial and "
            "property-type averages. The estimate may be less accurate."
        )
    else:
        suburb = suburb_choice

    st.subheader("Property Type")
    property_type = st.selectbox("Type", PROPERTY_TYPES)

with col2:
    st.subheader("Size")
    floor_size_m2 = st.number_input("Floor size (m²)", min_value=0.0, value=120.0, step=5.0)

    # Default erf size to 0 for unit types where a plot is irrelevant
    no_erf_types = {"Apartment", "Townhouse"}
    erf_default = 0.0 if property_type in no_erf_types else 500.0
    erf_size_m2 = st.number_input(
        "Erf / plot size (m²)",
        min_value=0.0,
        value=erf_default,
        step=10.0,
        help="Typically 0 for apartments and townhouses (sectional title).",
    )

    st.subheader("Rooms")
    bedrooms  = st.number_input("Bedrooms",          min_value=0, max_value=20, value=3, step=1)
    bathrooms = st.number_input("Bathrooms",          min_value=0, max_value=20, value=2, step=1)
    garages   = st.number_input("Garages",            min_value=0, max_value=10, value=1, step=1)
    parkings  = st.number_input("Extra parking bays", min_value=0, max_value=10, value=0, step=1)

# ---------------------------------------------------------------------------
# Financial preferences (outside columns so it spans full width)
# ---------------------------------------------------------------------------
st.subheader("Financial Preferences")
deposit_pct = st.slider("Deposit (%)", min_value=0, max_value=50, value=10)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
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
        input_df = pd.DataFrame([input_data])[raw_feature_names]
        predicted_price = pipeline.predict(input_df)[0]
    else:
        # SA model is required for meaningful predictions — legacy Ames model
        # cannot be meaningfully mapped to SA inputs.
        st.error(
            "The SA model (`sa_house_price_pipeline.joblib`) is required. "
            "Run `fetch_property24_data.py` then `train_sa_model.py` to generate it."
        )
        st.stop()

    # --- Result summary ---
    lower_bound = max(0, predicted_price - display_mae)
    upper_bound = predicted_price + display_mae

    st.success(f"### Estimated Property Value: **R {predicted_price:,.0f}**")
    st.info(
        f"Indicative range (±model MAE of R {display_mae:,.0f}):  "
        f"**R {lower_bound:,.0f} — R {upper_bound:,.0f}**"
    )

    if custom_suburb_used and suburb:
        st.warning(
            f"**Note:** '{suburb}' was not seen during training. The suburb was not factored "
            "into this estimate — only province, property type, and size were used.",
            icon="⚠️",
        )

    st.caption(f"Market segment: **{get_price_band(predicted_price)}**")

    # --- Cost breakdown ---
    st.subheader("Purchase Cost Breakdown")

    deposit       = predicted_price * deposit_pct / 100
    bond_amount   = predicted_price - deposit
    transfer_duty = calculate_transfer_duty(predicted_price)
    conveyancing  = calculate_conveyancing(predicted_price)
    bond_costs    = calculate_bond_costs(bond_amount)
    total_bond_fees = sum(bond_costs.values())
    total_upfront = transfer_duty + conveyancing + total_bond_fees + deposit

    # Two-section table: buyer costs then buyer contribution
    costs_rows = [
        ("Transfer Duty (SARS 2026/27)", transfer_duty,   "Government tax"),
        ("Conveyancing attorney (Law Society tariff)", conveyancing, "Estimate"),
        ("Bank initiation fee",          bond_costs["Bank initiation fee"],          "Fixed fee"),
        ("Deeds Office registration",    bond_costs["Deeds Office registration"],    "Fixed fee"),
        ("Bond attorney fee (incl. VAT)", bond_costs["Bond attorney fee (incl. VAT)"], "Estimate"),
    ]
    contribution_rows = [
        ("Deposit",  deposit,  f"{deposit_pct}% of purchase price"),
    ]

    def fmt(v: float) -> str:
        return f"R {v:,.0f}"

    st.markdown("**Transaction costs**")
    costs_df = pd.DataFrame(
        [(item, fmt(amt), note) for item, amt, note in costs_rows],
        columns=["Item", "Amount", "Note"],
    )
    st.table(costs_df)

    st.markdown("**Your cash contribution**")
    contrib_df = pd.DataFrame(
        [(item, fmt(amt), note) for item, amt, note in contribution_rows],
        columns=["Item", "Amount", "Note"],
    )
    st.table(contrib_df)

    st.metric("Total cash needed on transfer day", fmt(total_upfront))
    st.caption(
        "This is the cash required to register transfer. "
        "It excludes monthly bond repayments — see the Bond Repayment calculator in the sidebar."
    )

# ---------------------------------------------------------------------------
# Transfer Duty calculator tab (bond repayment moved to sidebar)
# ---------------------------------------------------------------------------
st.divider()
st.header("Transfer Duty Calculator")
st.caption("SARS 2026/27 rates")

td_price = st.number_input(
    "Property purchase price (R)", min_value=0.0, value=1_500_000.0, step=50_000.0, key="td_calc"
)
duty = calculate_transfer_duty(td_price)
if td_price <= 1_210_000:
    st.success("No transfer duty payable — properties below R1,210,000 are exempt.")
else:
    st.metric("Transfer duty payable", f"R {duty:,.0f}")
    st.caption(f"That is {duty / td_price * 100:.2f}% of the purchase price.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    f"Data sourced from Property24 · Model trained on {training_listings:,} listings · "
    f"Mean Absolute Error: R {display_mae:,.0f} · "
    "Transfer duty per SARS 2026/27 · Bond costs are estimates only."
)
