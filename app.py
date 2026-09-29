import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Telco Ltd. - Customer Churn Evaluator", layout="wide")

# Determine absolute path to model.pkl in the same folder as app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')


@st.cache_resource
def load_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


pipeline = load_pipeline()

st.title("📊 Telco Ltd. - Customer Retention Risk Predictor")
st.markdown(
    "Decision-support tool for non-technical account managers at **Telco Ltd.** to evaluate customer churn risk."
)

if pipeline is None:
    st.error(
        f"`model.pkl` file not found at `{MODEL_PATH}`! Please ensure `model.pkl` is committed to GitHub."
    )
    st.stop()

st.sidebar.header("Customer Profile Settings")
tenure = st.sidebar.slider("Tenure (Months)", 1, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
contract_type = st.sidebar.selectbox(
    "Contract Type", ["Month-to-month", "One year", "Two year"]
)
payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
)
paperless_billing = st.sidebar.radio(
    "Paperless Billing Active?", ["Yes", "No"], horizontal=True
)
tech_support = st.sidebar.radio(
    "Has Premium Tech Support?", ["Yes", "No"], horizontal=True
)

input_df = pd.DataFrame([
    {
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly_charges,
        "Contract_Type": contract_type,
        "Payment_Method": payment_method,
        "Paperless_Billing": paperless_billing,
        "Tech_Support": tech_support,
    }
])

st.subheader("Predictive Risk Assessment")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("#### Selected Customer Profile")
    st.dataframe(
        input_df.T.rename(columns={0: "Value"}), use_container_width=True
    )

with col2:
    churn_prob = pipeline.predict_proba(input_df)[0][1]
    st.metric(
        label="Predicted Attrition Risk", value=f"{churn_prob * 100:.1f}%"
    )

    if churn_prob >= 0.50:
        st.error(
            "🚨 HIGH CHURN RISK: High probability of account termination. Initiate immediate account management follow-up."
        )
    else:
        st.success(
            "✅ LOW CHURN RISK: Account status stable. Maintain standard retention protocols."
        )

st.divider()
st.subheader("Model Decision Drivers (Explainability)")
st.write(
    "1. **Contract Type**: Month-to-month contracts are the strongest driver of customer churn."
)
st.write(
    "2. **Tenure**: Customer longevity significantly buffers against attrition."
)
st.write(
    "3. **Monthly Charges**: Higher charges escalate churn sensitivity for uncommitted accounts."
)
