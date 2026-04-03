import os
import streamlit as st
import numpy as np
import pickle

# ---------------- LOAD MODEL ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

with open(model_path, "rb") as f:
    w, b, mean, std, y_mean, y_std = pickle.load(f)

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 House Price Predictor")
st.markdown("Estimate Indian house prices using a linear regression model trained on synthetic data.")
st.divider()
# ---------------- INPUTS ---------------- #
col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Area (sqft)", min_value=300, max_value=10000, value=1000, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)

with col2:
    age = st.number_input("Age of House (years)", min_value=0, max_value=100, value=5)
    location_score = st.slider("Location Score", 1, 10, 5, help="1 = poor locality, 10 = prime locality")
    garage = st.selectbox("Garage", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

st.divider()

# ---------------- PREDICTION ---------------- #
if st.button("🔍 Predict Price", use_container_width=True):
    try:
        features = np.array([area, bedrooms, bathrooms, age, location_score, garage], dtype=float)

        # Scale features using training stats
        features_scaled = (features - mean) / std

        # Predict in scaled space
        scaled_pred = np.dot(features_scaled, w) + b

        # Convert back to actual price
        predicted_price = scaled_pred * y_std + y_mean

        # Sanity clamp — no negative prices
        predicted_price = max(500_000, predicted_price)

        # ✅ Display result
        st.success(f"### 🏠 Estimated Price: ₹{predicted_price:,.0f}")

        # Breakdown
        st.markdown(f"""
        | Factor | Value |
        |---|---|
        | Area | {area} sqft |
        | Bedrooms | {bedrooms} |
        | Bathrooms | {bathrooms} |
        | House Age | {age} years |
        | Location Score | {location_score}/10 |
        | Garage | {'Yes' if garage else 'No'} |
        | **Predicted Price** | **₹{predicted_price:,.0f}** |
        """)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ---------------- MODEL DETAILS ---------------- #
with st.expander("🔍 Model Details"):
    feature_names = ["Area_sqft", "Bedrooms", "Bathrooms", "Age", "Location_Score", "Garage"]
    st.write("**Feature Weights (w):**")
    for name, weight in zip(feature_names, w):
        st.write(f"  {name}: `{weight:.4f}`")
    st.write(f"**Bias (b):** `{b:.4f}`")
    st.write(f"**Target Mean (y_mean):** ₹`{y_mean:,.0f}`")
    st.write(f"**Target Std (y_std):** ₹`{y_std:,.0f}`")