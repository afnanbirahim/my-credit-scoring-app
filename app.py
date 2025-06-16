import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load artifacts
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# UI
st.title("Microcredit Loan Eligibility Prediction")
st.markdown("Enter applicant information below.")

# Inputs
loan_amount = st.number_input("Loan Amount (৳)", 1000, 500000, step=5000, value=20000)
income = st.number_input("Family Monthly Income (৳)", 1000, 500000, step=1000, value=10000)
savings = st.number_input("Total Family Savings (৳)", 0, 1000000, step=1000, value=3000)
loan_duration = st.number_input("Loan Duration (Days)", 7, 720, step=7, value=120)
repayment_ratio = st.slider("Estimated Repayment Ratio", 0.0, 1.0, 0.75, 0.05)
threshold = st.slider("Approval Threshold", 0.3, 0.9, 0.6, 0.01)

# Engineered
debt_to_income = loan_amount / (income + 1)

# Build input
user_input = {
    "Loan Amount": loan_amount,
    "Family Income in Taka": income,
    "Total Savings": savings,
    "Debt_to_Income": debt_to_income,
    "Repayment_Ratio": repayment_ratio,
    "Loan_Duration_Days": loan_duration
}

# Remove problematic feature
if "Savings_to_Income" in feature_names:
    feature_names.remove("Savings_to_Income")

# Fill rest with 0
for col in feature_names:
    if col not in user_input:
        user_input[col] = 0.0

# Align
input_df = pd.DataFrame([user_input])[feature_names]
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Check Eligibility"):
    prob = model.predict_proba(input_scaled)[0][1]

    if prob >= threshold:
        st.success(f"✅ Eligible (Confidence: {prob:.2f})")
    else:
        st.error(f"❌ High risk (Confidence: {prob:.2f})")

    if abs(prob - threshold) < 0.05:
        st.info("ℹ️ Close to decision threshold. Manual review suggested.")

    st.markdown("---")
    st.subheader("Prediction Details")
    st.write("Confidence Score:", round(prob, 4))
    st.write("Threshold Used:", threshold)

    st.subheader("Input Sent to Model")
    st.dataframe(pd.DataFrame([user_input]))
