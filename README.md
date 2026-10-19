# FraudLens - Real-Time Credit Card Fraud Detection

Streamlit web app serving an XGBoost fraud classifier trained on the European cardholders dataset (284,807 transactions, 0.17% fraud).

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` - web app (single transaction, batch CSV, threshold slider)
- `model.json`, `scaler.json` - trained model and scaler (exported from the notebook)
- `sample_transactions.csv` - 100 demo transactions (50 fraud, 50 legit)
- `export_model.py` - converts `fraud_engine.joblib` from the notebook into the files above
