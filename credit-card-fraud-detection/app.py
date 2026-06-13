import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/fraud_model.pkl")

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.big-title{
    font-size:42px;
    font-weight:700;
    color:white;
}

.subtitle{
    color:#A0A0A0;
    font-size:18px;
}

.metric-card{
    background:#1E293B;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.result-box{
    padding:25px;
    border-radius:15px;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================

st.markdown(
    """
    <div class='big-title'>
    💳 Credit Card Fraud Detection Dashboard
    </div>
    <div class='subtitle'>
    AI Powered Transaction Risk Analysis System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# DASHBOARD METRICS
# =========================

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Model", "Random Forest")

with c2:
    st.metric("Status", "Active")

with c3:
    st.metric("Risk Engine", "Online")

with c4:
    st.metric("Version", "1.0")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================

left, right = st.columns([2,1])

with left:

    st.subheader("📝 Transaction Details")

    amt = st.number_input(
        "Transaction Amount ($)",
        min_value=0.0,
        value=100.0
    )

    category = st.selectbox(
        "Transaction Category",
        [
            "shopping_pos",
            "shopping_net",
            "grocery_pos",
            "food_dining",
            "gas_transport",
            "entertainment",
            "misc_pos"
        ]
    )

    merchant = st.text_input(
        "Merchant Name",
        "Amazon"
    )

with right:

    st.subheader("👤 Customer Information")

    gender = st.selectbox(
        "Gender",
        ["M", "F"]
    )

    state = st.text_input(
        "State",
        "NY"
    )

    city_pop = st.number_input(
        "City Population",
        min_value=100,
        value=50000
    )

    job = st.text_input(
        "Occupation",
        "Engineer"
    )

st.markdown("")

# =========================
# PREDICT BUTTON
# =========================

if st.button("🔍 Analyze Transaction", use_container_width=True):

    sample = pd.DataFrame({
        "trans_date_trans_time": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "merchant": [merchant],
        "category": [category],
        "gender": [gender],
        "state": [state],
        "city_pop": [city_pop],
        "job": [job],
        "amt": [amt]
    })

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.markdown("---")

    st.subheader("📊 Risk Assessment Report")

    risk_score = probability * 100

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Fraud Probability",
            f"{risk_score:.2f}%"
        )

    with col2:
        st.metric(
            "Transaction Amount",
            f"${amt:,.2f}"
        )

    with col3:
        st.metric(
            "Risk Level",
            "HIGH" if prediction == 1 else "LOW"
        )

    st.progress(float(probability))

    if prediction == 1:

        st.error(
            f"""
            🚨 FRAUD ALERT

            This transaction appears suspicious.

            Estimated Fraud Probability: {risk_score:.2f}%
            """
        )

    else:

        st.success(
            f"""
            ✅ TRANSACTION APPROVED

            Transaction appears legitimate.

            Fraud Probability: {risk_score:.2f}%
            """
        )

    st.markdown("---")

    st.subheader("📋 Transaction Summary")

    summary = pd.DataFrame({
        "Field":[
            "Merchant",
            "Category",
            "Amount",
            "State",
            "Occupation"
        ],
        "Value":[
            merchant,
            category,
            f"${amt}",
            state,
            job
        ]
    })

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Developed by Aryan Mali | CODSOFT Machine Learning Internship Project"
)