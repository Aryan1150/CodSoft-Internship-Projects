import streamlit as st
import joblib

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Spam SMS Detector",
    page_icon="📩",
    layout="centered"
)

# =========================
# CUSTOM CSS (PRO UI)
# =========================
st.markdown("""
    <style>
        .main {
            background-color: #0f172a;
        }

        h1 {
            color: #38bdf8;
            text-align: center;
            font-size: 40px;
        }

        .subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .card {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
            backdrop-filter: blur(10px);
        }

        .result-box {
            padding: 15px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }

        .spam {
            background-color: #7f1d1d;
            color: #fecaca;
        }

        .ham {
            background-color: #14532d;
            color: #bbf7d0;
        }

        .btn {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("models/spam_model.pkl")

# =========================
# HEADER
# =========================
st.markdown("<h1>📩 Spam SMS Detector</h1>", unsafe_allow_html=True)
st.markdown("<br><hr><center style='color:gray'>© 2026 Aryan Mali | AI/ML Project - Spam Detection System</center>", unsafe_allow_html=True)
# =========================
# INPUT UI
# =========================
st.markdown("<div class='card'>", unsafe_allow_html=True)

msg = st.text_area("✍️ Enter your SMS message here", height=150)

predict_btn = st.button("🔍 Predict")

st.markdown("</div>", unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if predict_btn:
    if msg.strip() == "":
        st.warning("⚠️ Please enter a message first")
    else:
        prediction = model.predict([msg])[0]

        if prediction == 1:
            st.markdown("""
                <div class='result-box spam'>
                    🚨 SPAM MESSAGE DETECTED
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='result-box ham'>
                    ✅ HAM MESSAGE (Safe)
                </div>
            """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
