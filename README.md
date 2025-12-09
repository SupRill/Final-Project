# Telco Customer Churn – Streamlit App

Aplikasi web untuk memprediksi risiko customer churn pada perusahaan telekomunikasi menggunakan:

- Data preprocessing (IQR outlier capping, encoding, normalisasi)
- Model K-Nearest Neighbors (KNN)
- Deployment dengan Streamlit Cloud
- Model pipeline disimpan di GitHub (`models/churn_pipeline.pkl`)

## Cara Menjalankan Lokal

```bash
pip install -r requirements.txt
python train_model.py  # melatih model & menyimpan pipeline
streamlit run app.py
