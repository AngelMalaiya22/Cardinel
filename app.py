import json
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="FraudLens", page_icon="🛡️", layout="wide")

# ---------- load model once ----------
@st.cache_resource
def load_engine():
    model = XGBClassifier()
    model.load_model("model.json")
    with open("scaler.json") as f:
        sc = json.load(f)
    return model, sc

model, sc = load_engine()
FEATURES = sc["features"]          # Time, V1..V28, Amount
CENTER = dict(zip(["Time", "Amount"], sc["center"]))
SCALE = dict(zip(["Time", "Amount"], sc["scale"]))

def predict(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].astype(float).copy()
    for c in ("Time", "Amount"):
        X[c] = (X[c] - CENTER[c]) / SCALE[c]
    return model.predict_proba(X)[:, 1]

@st.cache_data
def load_sample():
    return pd.read_csv("sample_transactions.csv")

# ---------- header ----------
st.title("🛡️ FraudLens – Real-Time Credit Card Fraud Detection")
st.caption("XGBoost model trained on the European cardholders dataset (284,807 transactions, 0.17% fraud).")

threshold = st.sidebar.slider("Fraud threshold", 0.05, 0.95, 0.50, 0.05,
    help="Lower = catch more fraud but more false alarms. Higher = fewer false alarms but more missed fraud.")
st.sidebar.markdown("---")
st.sidebar.write("**How it works**\n\nThe model outputs a fraud probability. "
                 "If it is ≥ the threshold, the transaction is blocked.")

tab1, tab2, tab3 = st.tabs(["🔍 Single transaction", "📂 Batch (CSV)", "ℹ️ About"])

# ---------- tab 1: single ----------
with tab1:
    sample = load_sample()
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("🎲 Load random FRAUD sample"):
        st.session_state["row"] = sample[sample.Class == 1].sample(1).iloc[0]
    if c2.button("🎲 Load random LEGIT sample"):
        st.session_state["row"] = sample[sample.Class == 0].sample(1).iloc[0]

    row = st.session_state.get("row")
    if row is None:
        row = sample.iloc[0]
        st.info("Showing a default sample. Use the buttons above, or edit any value below.")
    else:
        st.caption(f"Actual label of loaded sample: **{'FRAUD' if row['Class']==1 else 'LEGIT'}**")

    st.subheader("Transaction details")
    a, b = st.columns(2)
    amount = a.number_input("Amount", value=float(row["Amount"]), format="%.2f")
    time_s = b.number_input("Time (seconds since first transaction)", value=float(row["Time"]))
    with st.expander("PCA features V1–V28 (anonymised by the bank)"):
        cols = st.columns(4)
        v = {f"V{i}": cols[(i-1) % 4].number_input(f"V{i}", value=float(row[f"V{i}"]), format="%.4f", key=f"v{i}_{row.name}")
             for i in range(1, 29)}

    if st.button("Check transaction", type="primary"):
        txn = pd.DataFrame([{"Time": time_s, **v, "Amount": amount}])
        p = float(predict(txn)[0])
        st.progress(min(p, 1.0))
        if p >= threshold:
            st.error(f"🚨 FRAUD – BLOCK  (fraud probability {p:.2%})")
        else:
            st.success(f"✅ LEGIT – APPROVE  (fraud probability {p:.2%})")

# ---------- tab 2: batch ----------
with tab2:
    st.write("Upload a CSV with columns `Time, V1…V28, Amount` (a `Class` column is optional).")
    st.download_button("Download sample CSV", load_sample().to_csv(index=False), "sample_transactions.csv", "text/csv")
    up = st.file_uploader("Upload CSV", type="csv")
    if up:
        data = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["Fraud_Probability"] = predict(data)
            data["Decision"] = np.where(data["Fraud_Probability"] >= threshold, "FRAUD", "LEGIT")
            m1, m2, m3 = st.columns(3)
            m1.metric("Transactions", len(data))
            m2.metric("Flagged as fraud", int((data.Decision == "FRAUD").sum()))
            m3.metric("Fraud rate", f"{(data.Decision == 'FRAUD').mean():.2%}")
            if "Class" in data.columns:
                tp = int(((data.Decision == "FRAUD") & (data.Class == 1)).sum())
                st.write(f"Frauds actually caught: **{tp} / {int((data.Class == 1).sum())}**")
            show = ["Time", "Amount", "Fraud_Probability", "Decision"] + (["Class"] if "Class" in data.columns else [])
            st.dataframe(data[show].sort_values("Fraud_Probability", ascending=False), use_container_width=True)
            st.download_button("Download results", data.to_csv(index=False), "fraud_predictions.csv", "text/csv")

# ---------- tab 3: about ----------
with tab3:
    st.markdown("""
**Pipeline:** EDA → duplicate removal → stratified split → RobustScaler (Time, Amount) → SMOTE comparison →
5 models (LR, DT, RF, XGBoost, LightGBM) → evaluation with precision, recall, F1, PR-AUC.

**Why XGBoost without SMOTE?** It gave the best F1 (0.833) and PR-AUC (0.825) with 95.9% precision.
SMOTE raised recall but caused many more false alarms, so the threshold slider lets you trade between the two.

**Note:** `V1–V28` are PCA components, so they cannot be interpreted directly. This is a demo, not a bank-grade system.
""")
