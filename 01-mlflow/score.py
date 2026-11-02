"""How good is the model currently @staging on the data in your working dir? Usage: python score.py [alias]"""
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
alias = sys.argv[1] if len(sys.argv) > 1 else "staging"
name = os.getenv("MODEL_NAME", "fraud-detector")

df = pd.read_csv("data/transactions.csv")
X, y = df.drop(columns="is_fraud").assign(online=lambda d: d.online.astype(int)), df.is_fraud
model = mlflow.pyfunc.load_model(f"models:/{name}@{alias}")
proba = model.predict(X)[:, 1]  # column 1 = P(fraud)
print(f"{name}@{alias} on {len(df)} rows: AUC {roc_auc_score(y, proba):.3f} | AP {average_precision_score(y, proba):.3f}")
