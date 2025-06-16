import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model, scaler, and feature list
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# App title
st.title("Microcredit Loan Eligibility Prediction")
st.markdown("Enter borrower details below to check loan eligibility.")

# Input Fields
loan_amount = st.number_input("Loan Amount (৳)", min_value=1000, max_value=500000, step=5000, value=20000)
income = st.number_input("Family Monthly Income (৳)", min_value=1000, max_value=500000, step=1000, value=10000)
savings = st.number_input("Total Family Savings (৳)", min_value=0, max_value=500000, step=1000, value=3000)
loan_duration = st.number_input("Loan Duration (Days)", min_value=7, max_value=720, step=7, value=120)

# Repayment Ratio slider
repayment_ratio = st.slider("Estimated Repayment Ratio (0.0–1.0)", min_value=0.0, max_value=1.0, step=0.05, value=0.75)

# Threshold for decision
threshold = st.slider("Approval Threshold", min_value=0.5, max_value=0.9, step=0.01, value=0.65)

# Fill in all other features as 0.0
extra_features = {name: 0.0 for name in feature_names if name not in [
    'Loan Amount', 'Family Income in Taka', 'Total Savings',
    'Debt_to_Income', 'Repayment_Ratio', 'Savings_to_Income', 'Loan_Duration_Days'
]}

# Feature Engineering
debt_to_income = loan_amount / (income + 1)
savings_to_income = savings / (income + 1)

# Construct the input dictionary
user_input = {
    "Loan Amount": loan_amount,
    "Family Income in Taka": income,
    "Total Savings": savings,
    "Debt_to_Income": debt_to_income,
    "Repayment_Ratio": repayment_ratio,
    "Savings_to_Income": savings_to_income,
    "Loan_Duration_Days": loan_duration
}
user_input.update(extra_features)

# Prepare input for model
input_df = pd.DataFrame([user_input], columns=feature_names)
input_scaled = scaler.transform(input_df)

# Prediction Logic
if st.button("Check Eligibility"):
    probability = model.predict_proba(input_scaled)[0][1]

    if probability >= threshold:
        st.success(f"✅ Eligible for loan (Model Confidence: {probability:.2f})")
        prediction = 1
    else:
        st.error(f"❌ High risk – not eligible (Model Confidence: {probability:.2f})")
        prediction = 0

    # Optional: Flag borderline cases
    if abs(probability - threshold) < 0.05:
        st.info("ℹ️ Note: This decision is close to the threshold. Consider manual review.")

    # Debug / Explanation Section
    st.markdown("---")
    st.subheader("Prediction Details")
    st.write("Model Output Probability:", round(probability, 4))
    st.write("Approval Threshold Used:", threshold)
    st.write("Scaled Input Features:")
    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names))
