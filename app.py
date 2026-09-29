import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(
    page_title="Telco Ltd. - Customer Churn Evaluator", layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Telco_churn.csv")


@st.cache_resource
def load_and_train_pipeline():
    if not os.path.exists(DATA_PATH):
        return None

    # Load and clean Kaggle dataset
    df_raw = pd.read_csv(DATA_PATH)
    df_raw["TotalCharges"] = pd.to_numeric(
        df_raw["TotalCharges"], errors="coerce"
    )
    df_raw = df_raw.dropna()

    # Map to schema
    df = pd.DataFrame({
        "Tenure_Months": df_raw["tenure"],
        "Monthly_Charges": df_raw["MonthlyCharges"],
        "Contract_Type": df_raw["Contract"],
        "Payment_Method": df_raw["PaymentMethod"],
        "Paperless_Billing": df_raw["PaperlessBilling"],
        "Tech_Support": df_raw["TechSupport"].replace(
            {"No internet service": "No"}
        ),
        "Churn": df_raw["Churn"].map({"Yes": 1, "No": 0}),
    })

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    num_features = ["Tenure_Months", "Monthly_Charges"]
    cat_features = [
        "Contract_Type",
        "Payment_Method",
        "Paperless_Billing",
        "Tech_Support",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                cat_features,
            ),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(X, y)
    return pipeline


pipeline = load_and_train_pipeline()

st.title("📊 ABC Ltd. — Customer Retention Risk Predictor")
st.markdown(
    "Decision-support tool for non-technical account managers at **ABC Ltd.** to evaluate customer churn risk."
)

if pipeline is None:
    st.error(
        f"`Telco_churn.csv` not found in repo! Please upload `Telco_churn.csv` to the main GitHub folder."
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
