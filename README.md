# 📉 Telco Customer Churn Prediction & Risk Analysis

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://telco-churn-astreaoa.streamlit.app)

An end-to-end Machine Learning classification pipeline analyzing subscription attrition drivers and evaluating predictive models to detect churn-prone accounts.

---

## 📌 Executive Summary
Acquiring new subscribers is significantly more expensive than retaining existing accounts. This project investigates a cohort of **7,043 customers** from a telecommunications provider to:
- Clean messy structural artifacts (whitespace strings in `TotalCharges`).
- Uncover behavioral friction points via Exploratory Data Analysis (EDA).
- Train cost-sensitive models (**Logistic Regression** vs. **Random Forest**) to optimize for **Recall** on churned customers under a 73.5% / 26.5% class imbalance.
- Interpret the primary features driving customer attrition.

---

## 🔍 Key Findings & Feature Drivers
From the tree ensemble's feature importance analysis:
* **Contract Commitment:** Month-to-month contracts exhibit an attrition rate of **~42.7%**, whereas 1-year (~11.3%) and 2-year (~2.8%) commitments provide massive retention moats.
* **Fiber Optic & Pricing:** High monthly charges paired with Fiber Optic service drove higher churn probabilities when unaccompanied by technical support add-ons.
* **Payment Method Risk:** Accounts paying via manual electronic checks demonstrated substantially higher churn than those utilizing automated bank deductions.
* **Onboarding Friction:** Churn density peaks sharply within the first **1–6 months** of customer tenure.

---

## 📊 Model Performance

| Model | ROC-AUC | Churn Recall | Notes |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **0.842** | **~78.3%** | Interpretable linear baseline with balanced class weights |
| **Random Forest Classifier** | **0.843** | **~79.1%** | Balanced tree ensemble capturing non-linear interactions |

---

## 📁 Repository Structure

```text
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Sourced from Kaggle BlastChar
├── models/
│   ├── churn_model.joblib                    # Serialized Random Forest classifier
│   ├── scaler.joblib                         # Pre-fitted StandardScaler
│   └── model_features.joblib                 # Feature column order alignment
├── notebooks/
│   └── 01_churn_eda_and_baseline.ipynb       # End-to-end data pipeline & modeling
├── app.py                                    # Interactive Streamlit dashboard
├── .gitignore
├── requirements.txt
└── README.md
