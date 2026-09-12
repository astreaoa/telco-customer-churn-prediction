import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Telco Customer Churn & Retention Tool",
    page_icon="📊",
    layout="centered"
)

# Load serialized artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    features = joblib.load("models/model_features.joblib")
    return model, scaler, features

model, scaler, feature_cols = load_artifacts()

# App Title & Overview
st.title("📊 Subscriber Churn Risk Evaluator")
st.write("Assess customer retention probability, analyze risk drivers, and determine retention action points.")

# -------------------------------------------------------------
# Explanatory Guide
# -------------------------------------------------------------
with st.expander("ℹ️ Field Explanations & How This Data Is Used"):
    st.markdown("""
    * **Currency & Monthly Bill:** The recurring amount billed per cycle. Higher bills without bundled add-ons historically correlate with elevated churn.
    * **Estimated Monthly Income:** Used to calculate the **Bill-to-Income Burden** on the subscriber's personal budget.
    * **Contract Duration:** The contractual commitment length. Month-to-month contracts lack lock-in and carry the highest churn velocity.
    * **Tenure:** Number of months the account has been active. The first 1–6 months represent the highest risk drop-off window.
    * **Payment Method:** Electronic checks require manual monthly payment effort, whereas automated bank drafts create friction-free continuity.
    * **Subscribed Services:** Tree ensembles evaluate service bundle density—customers with multiple security and support add-ons have higher switching costs.
    """)

st.divider()

# -------------------------------------------------------------
# Section 1: Profile & Billing
# -------------------------------------------------------------
st.subheader("1. Customer Profile & Financials")

col_curr, col_inc = st.columns(2)
with col_curr:
    currency = st.selectbox("Preferred Currency", ["USD ($)", "EUR (€)", "GBP (£)", "GHS (GH₵)"])
    sym = currency.split()[0]
with col_inc:
    est_income = st.number_input(f"Est. Monthly Income ({sym})", min_value=100.0, max_value=50000.0, value=2500.0, step=100.0)

col1, col2 = st.columns(2)
with col1:
    contract = st.selectbox("Contract Duration", ["Month-to-month", "One year", "Two year"])
    tenure = st.slider("Customer Tenure (Months Active)", min_value=1, max_value=72, value=12)

with col2:
    monthly_charges = st.number_input(f"Monthly Bill ({sym})", min_value=18.0, max_value=150.0, value=70.0, step=5.0)
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

# -------------------------------------------------------------
# Section 2: Subscribed Services Selection
# -------------------------------------------------------------
st.subheader("2. Subscribed Services & Add-ons")

serv_col1, serv_col2 = st.columns(2)

with serv_col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Primary Internet Connection", ["Fiber optic", "DSL", "No Internet"])

with serv_col2:
    tech_support = st.selectbox("Technical Support Add-on", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security Suite", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection Plan", ["No", "Yes", "No internet service"])

with st.expander("Streaming Add-ons"):
    stream_col1, stream_col2 = st.columns(2)
    with stream_col1:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    with stream_col2:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

# -------------------------------------------------------------
# Section 3: Feature Engineering & Inference
# -------------------------------------------------------------
total_charges = float(tenure * monthly_charges)

input_dict = {
    "gender": 1,
    "SeniorCitizen": 0,
    "Partner": 0,
    "Dependents": 0,
    "tenure": float(tenure),
    "PhoneService": 1 if phone_service == "Yes" else 0,
    "PaperlessBilling": 1,
    "MonthlyCharges": float(monthly_charges),
    "TotalCharges": total_charges,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service if internet_service != "No Internet" else "No",
    "OnlineSecurity": online_security,
    "OnlineBackup": "No",
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaymentMethod": payment_method
}

input_df = pd.DataFrame([input_dict])
categorical_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=False, dtype=int)

# Align input vector schema to trained model
final_input = pd.DataFrame(0.0, index=[0], columns=feature_cols)
for col in input_encoded.columns:
    if col in final_input.columns:
        final_input[col] = input_encoded[col].astype(float)

final_input[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
    final_input[["tenure", "MonthlyCharges", "TotalCharges"]]
)

churn_prob = model.predict_proba(final_input)[0][1]
burden_ratio = (monthly_charges / est_income) * 100

st.divider()

# -------------------------------------------------------------
# Section 4: Results & Risk Diagnostics
# -------------------------------------------------------------
st.subheader("3. Risk Diagnostics & Retention Action Plan")

res1, res2 = st.columns([1, 1])

with res1:
    if churn_prob >= 0.5:
        st.error("🚨 **High Churn Vulnerability**")
        st.metric(label="Predicted Churn Probability", value=f"{churn_prob:.1%}")
    else:
        st.success("✅ **Healthy Account Status**")
        st.metric(label="Retention Probability", value=f"{(1 - churn_prob):.1%}")

    st.progress(float(churn_prob))
    st.caption(f"Estimated Budget Impact: This bill represents **{burden_ratio:.1f}%** of monthly income.")

with res2:
    st.markdown("**Key Risk Factors Detected:**")
    risk_factors = []

    if contract == "Month-to-month":
        risk_factors.append("• **Contract:** Month-to-month accounts experience >40% attrition rate.")
    if tenure <= 6:
        risk_factors.append("• **Tenure:** First 6 months represent the most volatile onboarding phase.")
    if internet_service == "Fiber optic" and tech_support == "No":
        risk_factors.append("• **Service:** Fiber optic without technical support leads to unresolved service friction.")
    if payment_method == "Electronic check":
        risk_factors.append("• **Billing:** Manual payment methods add friction compared to automated deductions.")
    if burden_ratio > 8.0:
        risk_factors.append("• **Budget:** Service bill exceeds 8% of customer's estimated monthly income.")

    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.write("• Customer maintains a stable setup with high loyalty indicators.")
