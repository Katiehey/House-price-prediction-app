
import streamlit as st
import joblib
import json
import pandas as pd
import numpy as np
import os

# Define the paths where your files are saved within Kaggle
MODEL_PATH = 'house_price_prediction_pipeline.joblib'
FEATURES_PATH = 'feature_list.json'

# --- Load the Model and Feature List ---
try:
    pipeline = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, 'r') as f:
        raw_feature_names = json.load(f)
except FileNotFoundError:
    st.error("Model or feature list files not found. Ensure they are in /kaggle/working/")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# --- Define Features (Use the correct capitalization from your training script!) ---
# These lists must match your training script exactly
numerical_features = ['lotarea', 'yearremodadd', 'masvnrarea', 'bsmtfinsf1', 'bsmtunfsf', 'totalbsmtsf', '1stflrsf', '2ndflrsf', 'grlivarea', 'fireplaces', 'garagecars', 'poolarea', 'yrsold']
categorical_features = ['neighborhood', 'overallqual', 'overallcond', 'bsmtexposure', 'kitchenqual', 'saletype']

# --- Streamlit UI ---
st.title("House Price Prediction App")

# Create a dictionary to store user inputs
user_inputs = {}

# Use columns for a cleaner layout
col1, col2 = st.columns(2)

# Generate inputs for numerical features
with col1:
    st.subheader("Numerical Inputs")
    for feature in numerical_features:
        user_inputs[feature] = st.number_input(f"Enter value for {feature}", value=0.0)

# Generate inputs for categorical features
with col2:
    st.subheader("Categorical Inputs")
    for feature in categorical_features:
        # Define options for dropdowns based on your training data unique values
        if feature == 'Neighborhood':
            options = ['Blmngtn', 'Blueste', 'BrDale', 'BrkSide', 'ClearCr', 'CollgCr', 'Crawfor', 'Edwards', 'Gilbert', 'IDotRR', 'MeadowV', 'Mitchel', 'NAmes', 'NoRidge', 'NPkVill', 'NridgHt', 'NWames', 'OldTown', 'SWISU', 'Sawyer', 'SawyerW', 'Somerst', 'StoneBr', 'Timber', 'Veenker']
        elif feature == 'OverallQual' or feature == 'OverallCond':
            options = [10,9,8,7,6,5,4,3,2,1]
        elif feature == 'BsmtExposure':
            options = ['Gd', 'Av', 'Mn', 'No', 'NA']
        elif feature == 'KitchenQual':
            options = ['Ex', 'Gd', 'TA', 'Fa', 'Po']
        elif feature == 'SaleType':
            options = ['WD', 'CWD', 'VWD', 'New', 'COD', 'Con', 'ConLW', 'ConLI', 'ConLD', 'Oth']
        else:
            options = ['N/A'] # Fallback
        
        user_inputs[feature] = st.selectbox(f"Select option for {feature}", options=options)

# The prediction button
if st.button("Predict SalePrice"):
    # Convert inputs dictionary to a DataFrame (single row)
    input_df = pd.DataFrame([user_inputs])

    # The pipeline handles all preprocessing
    prediction = pipeline.predict(input_df)
    
    st.success(f"The predicted house price is: ${prediction[0]:,.2f}")

