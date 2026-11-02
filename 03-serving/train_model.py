"""Train the fraud model on data/transactions.csv and write model.joblib.

    uv run train_model.py
"""
import hashlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = Path("data/transactions.csv")

df = pd.read_csv(DATA)
X, y = df.drop(columns="is_fraud"), df["is_fraud"]
X = X.assign(online=X.online.astype(int))
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

model = Pipeline([
    ("pre", ColumnTransformer([("merchant", OneHotEncoder(handle_unknown="ignore"), ["merchant"])],
                              remainder="passthrough")),
    ("clf", HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_depth=3, random_state=42)),
]).fit(X_tr, y_tr)

version = hashlib.md5(DATA.read_bytes()).hexdigest()[:8]  # the version names the data it came from
joblib.dump({"model": model, "version": version}, "model.joblib")
print(f"model.joblib written (version {version}), AUC on held-out data: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.3f}")
