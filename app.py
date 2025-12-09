# app.py

import streamlit as st
import pandas as pd
import pickle
import requests
import io
import os

# =====================
# 1. Konfigurasi
# =====================

st.set_page_config(
    page_title="Prediksi Customer Churn - Telco",
    page_icon="📡",
    layout="centered"
)

USE_GITHUB_MODEL = False  # ubah ke True kalau mau load dari raw GitHub

GITHUB_MODEL_URL = (
    "https://raw.githubusercontent.com/USERNAME/REPO/main/models/churn_pipeline.pkl"
)

LOCAL_MODEL_PATH = "models/churn_pipeline.pkl"

# =====================
# 2. Load model
# =====================

@st.cache_resource
def load_model():
    if USE_GITHUB_MODEL:
        resp = requests.get(GITHUB_MODEL_URL)
        resp.raise_for_status()
        return pickle.load(io.BytesIO(resp.content))
    else:
        with open(LOCAL_MODEL_PATH, "rb") as f:
            return pickle.load(f)

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =====================
# 3. UI
# =====================

st.title("📡 Prediksi Customer Churn – Perusahaan Telekomunikasi")
st.markdown(
    "Aplikasi ini menggunakan model **K-Nearest Neighbors (KNN)** dengan preprocessing "
    "data (outlier capping, encoding, dan normalisasi) untuk memprediksi risiko churn pelanggan."
)

st.subheader("Input Data Pelanggan")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (bulan)", min_value=1, max_value=60, value=24)
    usage = st.number_input("Usage Frequency", min_value=1, max_value=30, value=15)
    support = st.number_input("Support Calls", min_value=0, max_value=20, value=3)

with col2:
    delay = st.number_input("Payment Delay (hari)", min_value=0, max_value=30, value=5)
    spend = st.number_input("Total Spend", min_value=100, max_value=1000, value=600)
    last = st.number_input("Last Interaction (hari)", min_value=1, max_value=30, value=10)

gender = st.selectbox("Gender", ["Male", "Female"])
sub_type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])

# =====================
# 4. Bentuk dataframe input
# =====================

input_df = pd.DataFrame(
    {
        "Age": [age],
        "Tenure": [tenure],
        "Usage Frequency": [usage],
        "Support Calls": [support],
        "Payment Delay": [delay],
        "Total Spend": [spend],
        "Last Interaction": [last],
        "Gender": [gender],
        "Subscription Type": [sub_type],
        "Contract Length": [contract],
    }
)

st.write("### Preview Data Input")
st.dataframe(input_df)

# =====================
# 5. Prediksi
# =====================

if st.button("Prediksi Churn"):
    try:
        pred = pipeline.predict(input_df)[0]
        prob = pipeline.predict_proba(input_df)[0][1]

        st.write("---")
        st.subheader("Hasil Prediksi")

        if pred == 1:
            st.error(f"⚠️ Pelanggan **berisiko churn**.\n\nProbabilitas churn: **{prob:.2f}**")
            st.markdown(
                "- Pertimbangkan penawaran khusus atau program loyalitas.\n"
                "- Cek riwayat **Support Calls** dan **Payment Delay** untuk intervensi proaktif."
            )
        else:
            st.success(f"✅ Pelanggan **tidak berisiko churn**.\n\nProbabilitas churn: **{prob:.2f}**")
            st.markdown(
                "- Pelanggan ini cenderung loyal.\n"
                "- Bisa menjadi target program **upsell** atau **cross-sell**."
            )

    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")

st.write("---")
st.caption("Model: KNN + Preprocessing (IQR capping, One-Hot Encoding, Min-Max Scaling)")
