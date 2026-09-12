import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Risk Predictor",
    page_icon="📉",
    layout="wide"
)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    features = joblib.load("models/model_features.joblib")
    return model, scaler, features

model, scaler, feature_cols = load_artifacts()

st.title("📉 Telco Customer Churn Risk Predictor")
st.markdown("Adjust subscriber parameters on the left to evaluate retention probability and risk factors.")

# Sidebar - Input parameters
st.sidebar.header("Customer Profile")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0)
total_charges = float(tenure * monthly_charges)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
payment_method = st.sidebar.selectbox(
    "Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

# Build raw input dictionary with exact numeric encodings used during training
input_dict = {
    "gender": 1 if gender == "Male" else 0,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": float(tenure),
    "PhoneService": 1,
    "PaperlessBilling": 1 if paperless == "Yes" else 0,
    "MonthlyCharges": float(monthly_charges),
    "TotalCharges": total_charges,
    "MultipleLines": "No",
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": tech_support,
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaymentMethod": payment_method
}

# Preprocess categorical inputs to match training dummy columns
input_df = pd.DataFrame([input_dict])
categorical_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False, dtype=int)

# Align columns with model training features (all initialized as 0.0 floats)
final_input = pd.DataFrame(0.0, index=[0], columns=feature_cols)
for col in input_encoded.columns:
    if col in final_input.columns:
        final_input[col] = input_encoded[col].astype(float)

# Scale numeric features
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
final_input[num_cols] = scaler.transform(final_input[num_cols])

# Run Inference
churn_prob = model.predict_proba(final_input)[0][1]

# Main Dashboard Display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Assessment Result")
    if churn_prob >= 0.5:
        st.error(f"⚠️ High Churn Risk: **{churn_prob:.1%} Probability**")
    else:
        st.success(f"✅ Healthy Account: **{(1 - churn_prob):.1%} Retention Probability**")

    st.progress(float(churn_prob))
    st.caption("Calculated via Cost-Balanced Random Forest Ensemble")

with col2:
    st.subheader("Key Risk Analysis")
    recommendations = []
    if contract == "Month-to-month":
        recommendations.append("🚩 **High Risk Contract**: Customer is on Month-to-month. Offer a promotional annual plan.")
    if internet_service == "Fiber optic" and tech_support == "No":
        recommendations.append("🚩 **Service Friction**: Fiber Optic subscriber without Technical Support. High attrition correlation.")
    if payment_method == "Electronic check":
        recommendations.append("🚩 **Payment Method Friction**: Electronic check payments correlate heavily with churn.")
    if tenure <= 6:
        recommendations.append("🚩 **Early Lifecycle**: Customer is in the vulnerable onboarding window (≤ 6 months).")

    if recommendations:
        for rec in recommendations:
            st.markdown(rec)
    else:
        st.markdown("✨ Account metrics indicate low operational friction.")