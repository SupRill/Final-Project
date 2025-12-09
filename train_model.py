# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import pickle
import os

# =====================
# 1. Load dataset
# =====================

# Misal dataset lokal
df = pd.read_csv("customer_churn.csv")

# Drop baris dengan missing values (sesuai laporan)
df = df.dropna().copy()

# =====================
# 2. Definisikan fitur & target
# =====================

target_col = "Churn"

numerical_cols = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

categorical_cols = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]

X = df[numerical_cols + categorical_cols]
y = df[target_col]

# =====================
# 3. Fungsi outlier capping
# =====================

def cap_outliers_iqr(df, cols):
    df_capped = df.copy()
    for col in cols:
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_capped[col] = np.where(df_capped[col] < lower, lower, df_capped[col])
        df_capped[col] = np.where(df_capped[col] > upper, upper, df_capped[col])
    return df_capped

X[numerical_cols] = cap_outliers_iqr(X, numerical_cols)[numerical_cols]

# =====================
# 4. Train-test split
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================
# 5. Preprocessing: scaler + encoder (dalam ColumnTransformer)
# =====================

numeric_transformer = Pipeline(
    steps=[
        ("scaler", MinMaxScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# =====================
# 6. Model: KNN
# =====================

knn_model = KNeighborsClassifier(n_neighbors=5)  # sesuaikan dengan hasil tuningmu

# =====================
# 7. Pipeline lengkap (preprocess + model)
# =====================

churn_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", knn_model),
    ]
)

# =====================
# 8. Training
# =====================

churn_pipeline.fit(X_train, y_train)

# =====================
# 9. Simpan pipeline
# =====================

os.makedirs("models", exist_ok=True)
with open("models/churn_pipeline.pkl", "wb") as f:
    pickle.dump(churn_pipeline, f)

print("Pipeline churn tersimpan di models/churn_pipeline.pkl")
