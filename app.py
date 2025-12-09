import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier

# ==========================
# 1. Load dataset dari GitHub
# ==========================

DATA_URL = "https://raw.githubusercontent.com/USERNAME/REPO/main/customer_churn.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

df = load_data()

# ==========================
# 2. Preprocessing + Training
# ==========================

numerical_cols = [
    "Age", "Tenure", "Usage Frequency", "Support Calls",
    "Payment Delay", "Total Spend", "Last Interaction"
]

categorical_cols = ["Gender", "Subscription Type", "Contract Length"]

target_col = "Churn"

# Outlier capping (IQR)
def cap_outliers(df, cols):
    df = df.copy()
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower, lower, df[col])
        df[col] = np.where(df[col] > upper, upper, df[col])
    return df

df[numerical_cols] = cap_outliers(df, numerical_cols)[numerical_cols]

X = df[numerical_cols + categorical_cols]
y = df[target_col]

# Preprocessing pipeline
numeric_transformer = Pipeline([("scaler", MinMaxScaler())])
categorical_transformer = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Model
model = KNeighborsClassifier(n_neighbors=5)

# Pipeline lengkap
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train model
pipeline.fit(X, y)

# ==========================
# 3. Streamlit UI
# ==========================

st.title("📡 Prediksi Customer Churn – Tanpa File Model (.pkl)")
st.write("Model dilatih **langsung di dalam aplikasi** setiap kali dijalankan.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure (bulan)", 1, 60, 24)
    usage = st.number_input
