import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model, scaler, and feature list
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# App title
st.title("Microcredit Loan Eligibility Prediction")
st.markdown("Provide borrower details to check loan eligibility.")

# Core user inputs
loan_amount = st.number_input("Loan Amount (৳)", min_value=1000, max_value=500000, step=5000, value=20000)
income = st.number_input("Family Monthly Income (৳)", min_value=1000, max_value=500000, step=1000, value=10000)
loan_duration = st.number_input("Loan Duration (Days)", min_value=7, max_value=720, step=7, value=120)

repayment_ratio = st.slider("Estimated Repayment Ratio (0.0–1.0)", min_value=0.0, max_value=1.0, step=0.05, value=0.75)
threshold = st.slider("Approval Threshold", min_value=0.3, max_value=0.9, step=0.01, value=0.60)

# Engineered feature
debt_to_income = loan_amount / (income + 1)

# Build input dictionary (without savings)
user_input = {
    "Loan Amount": loan_amount,
    "Family Income in Taka": income,
    "Debt_to_Income": debt_to_income,
    "Repayment_Ratio": repayment_ratio,
    "Loan_Duration_Days": loan_duration,
    "Total Savings": 0.0,  # neutralize if required by scaler
    "Savings_to_Income": 0.0  # neutralize bad behavior
}

# Fill any one-hot or missing columns with 0.0
for col in feature_names:
    if col not in user_input:
        user_input[col] = 0.0

# Align and scale
input_df = pd.DataFrame([user_input])[feature_names]
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Check Eligibility"):
    probability = model.predict_proba(input_scaled)[0][1]

    if probability >= threshold:
        st.success(f"✅ Eligible for loan (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ High risk – not eligible (Confidence: {probability:.2f})")

    if abs(probability - threshold) < 0.05:
        st.info("ℹ️ Close to threshold. Consider manual review.")

    # Output debugging info
    st.markdown("---")
    st.subheader("Prediction Details")
    st.write("Model Output Probability:", round(probability, 4))
    st.write("Approval Threshold Used:", threshold)

    st.subheader("Raw Input Features")
    st.dataframe(pd.DataFrame([user_input]))

    st.subheader("Scaled Features Sent to Model")
    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names))
