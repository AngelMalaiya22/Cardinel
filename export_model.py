"""Run ONCE next to fraud_engine.joblib (created by the notebook).
Converts it into version-independent files the web app loads:
  model.json   (XGBoost native format)
  scaler.json  (RobustScaler centre/scale for Time & Amount)
  sample_transactions.csv (small demo file, 50 fraud + 50 legit)
"""
import json, joblib, pandas as pd

engine = joblib.load("fraud_engine.joblib")
model, scaler, features = engine["model"], engine["scaler"], engine["features"]

if not hasattr(model, "save_model"):
    raise SystemExit("Best model is not XGBoost. Re-run the notebook's best model as XGBoost, or tell me and I'll adapt the app.")

model.save_model("model.json")
with open("scaler.json", "w") as f:
    json.dump({"features": features,
               "center": scaler.center_.tolist(),
               "scale": scaler.scale_.tolist()}, f)

# small demo file (raw, unscaled values) - the full 150 MB dataset is NOT needed for deployment
raw = pd.read_csv("creditcard.csv").drop_duplicates()
demo = pd.concat([raw[raw.Class == 1].sample(50, random_state=1),
                  raw[raw.Class == 0].sample(50, random_state=1)]).sample(frac=1, random_state=1)
demo.to_csv("sample_transactions.csv", index=False)
print("Done: model.json, scaler.json, sample_transactions.csv")
