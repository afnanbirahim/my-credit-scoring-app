import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load saved model and preprocessing artifacts
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# App title and description
st.title("Microcredit Loan Eligibility Predictor")
st.markdown("This tool predicts loan eligibility based on borrower details using a stacked ML model.")

# User input fields
loan_amount = st.number_input("Loan Amount (৳)", min_value=1000, max_value=500000, step=5000, value=20000)
income = st.number_input("Family Monthly Income (৳)", min_value=1000, max_value=500000, step=1000, value=10000)
loan_duration = st.number_input("Loan Duration (Days)", min_value=7, max_value=720, step=7, value=120)
interest_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=100.0, step=0.5, value=24.0)
installments = st.number_input("Number of Installments", min_value=1, max_value=60, step=1, value=12)
installment_amount = st.number_input("Installment Amount (৳)", min_value=100, max_value=50000, step=100, value=3800)
repayment_ratio = st.slider("Estimated Repayment Ratio", min_value=0.0, max_value=1.0, step=0.05, value=0.75)
threshold = st.slider("Approval Threshold", min_value=0.3, max_value=0.9, step=0.01, value=0.60)

# Engineered feature
debt_to_income = loan_amount / (income + 1)

# Input dictionary using only expected features
user_input = {
    "Loan Amount": loan_amount,
    "Family Income in Taka": income,
    "Debt_to_Income": debt_to_income,
    "Repayment_Ratio": repayment_ratio,
    "Loan_Duration_Days": loan_duration,
    "Interest Rate": interest_rate,
    "Number of Installment": installments,
    "Installment Amount": installment_amount
}

# Fill missing expected features with 0.0
for col in feature_names:
    if col not in user_input:
        user_input[col] = 0.0

# Create aligned input DataFrame
input_df = pd.DataFrame([user_input])[feature_names]
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Check Eligibility"):
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = int(probability >= threshold)

    if prediction == 1:
        st.success(f"✅ Eligible for loan (Confidence: {probability:.2f})")
    else:
        st.error(f"❌ High risk – not eligible (Confidence: {probability:.2f})")

    if abs(probability - threshold) < 0.05:
        st.info("ℹ️ This prediction is close to the decision boundary. Manual review advised.")

    # Debug info
    st.markdown("---")
    st.subheader("Model Input (Raw)")
    st.dataframe(pd.DataFrame([user_input]))

    st.subheader("Input After Scaling")
    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names))

    st.subheader("Prediction Probability")
    st.write("Model Confidence (Probability of Approval):", round(probability, 4))
