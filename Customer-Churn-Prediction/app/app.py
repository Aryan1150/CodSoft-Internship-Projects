# ==========================================
# CUSTOMER CHURN PREDICTION WEB APP
# Developed by Aryan Mali
# ==========================================

import streamlit as st
import pandas as pd
import pickle

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ==========================================
# LOAD MODEL & SCALER
# ==========================================

model = pickle.load(
    open("models/random_forest.pkl", "rb")
)

scaler = pickle.load(
    open("models/scaler.pkl", "rb")
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.big-title {
    text-align:center;
    font-size:48px;
    font-weight:bold;
    color:#1f4e79;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:gray;
}

.card {
    background-color:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.1);
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<p class="big-title">🏦 Bank Customer Churn Prediction</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether a customer is likely to leave the bank</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================
# KPI SECTION
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Model Accuracy",
        value="86%"
    )

with col2:
    st.metric(
        label="Customers",
        value="10,000"
    )

with col3:
    st.metric(
        label="Countries",
        value="3"
    )

st.markdown("---")

# ==========================================
# SIDEBAR INPUTS
# ==========================================

st.sidebar.header("📋 Customer Information")

credit_score = st.sidebar.slider(
    "Credit Score",
    300,
    900,
    650
)

age = st.sidebar.slider(
    "Age",
    18,
    100,
    35
)

tenure = st.sidebar.slider(
    "Tenure (Years)",
    0,
    20,
    5
)

balance = st.sidebar.number_input(
    "Account Balance",
    min_value=0.0,
    value=50000.0
)

num_products = st.sidebar.selectbox(
    "Number of Products",
    [1, 2, 3, 4]
)

has_card = st.sidebar.selectbox(
    "Has Credit Card?",
    ["Yes", "No"]
)

active_member = st.sidebar.selectbox(
    "Is Active Member?",
    ["Yes", "No"]
)

salary = st.sidebar.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

geography = st.sidebar.selectbox(
    "Country",
    ["France", "Germany", "Spain"]
)

# ==========================================
# CUSTOMER SUMMARY
# ==========================================

st.subheader("👤 Customer Profile")

summary_df = pd.DataFrame({
    "Feature": [
        "Credit Score",
        "Age",
        "Tenure",
        "Balance",
        "Products",
        "Salary",
        "Gender",
        "Country",
        "Active Member"
    ],
    "Value": [
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        salary,
        gender,
        geography,
        active_member
    ]
})

st.dataframe(
    summary_df,
    use_container_width=True
)

# ==========================================
# PREPARE DATA
# ==========================================

has_card = 1 if has_card == "Yes" else 0
active_member = 1 if active_member == "Yes" else 0

gender_male = 1 if gender == "Male" else 0

geo_germany = 1 if geography == "Germany" else 0
geo_spain = 1 if geography == "Spain" else 0

# ==========================================
# PREDICT BUTTON
# ==========================================

if st.button(
    "🔮 Predict Customer Churn",
    use_container_width=True
):

    input_data = pd.DataFrame([[
        credit_score,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        active_member,
        salary,
        gender_male,
        geo_germany,
        geo_spain
    ]],
    columns=[
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Gender_Male",
        "Geography_Germany",
        "Geography_Spain"
    ])

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    risk_score = probability * 100

    st.markdown("---")

    st.subheader("📈 Prediction Result")

    st.progress(int(risk_score))

    st.write(
        f"### Churn Risk Score: {risk_score:.2f}%"
    )

    if prediction == 1:

        st.error(
            f"""
            ⚠ HIGH CHURN RISK

            This customer is likely to leave the bank.

            Risk Probability: {risk_score:.2f}%
            """
        )

    else:

        st.success(
            f"""
            ✅ CUSTOMER LIKELY TO STAY

            This customer is expected to remain with the bank.

            Confidence Score: {100-risk_score:.2f}%
            """
        )

# ==========================================
# ABOUT SECTION
# ==========================================

st.markdown("---")

st.subheader("📊 Project Information")

st.write("""
This Machine Learning project predicts whether a customer is likely to leave a bank.

### Technologies Used

- Python
- Pandas
- Scikit-Learn
- Random Forest
- Streamlit

### Features

- Customer Churn Prediction
- Interactive Dashboard
- Probability-Based Risk Analysis
- Business Intelligence Insights

### Dataset

Bank Customer Churn Dataset (10,000 Records)

Countries Included:
- France
- Germany
- Spain
""")

st.markdown("---")

st.markdown(
    """
    <center>
    Developed by <b>Aryan Mali</b> 🚀
    </center>
    """,
    unsafe_allow_html=True
)