"""Score a CSV of transactions. Usage: python predict.py in.csv out.csv"""
import logging
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
import pandas as pd

# the model lists its full training environment; the image only needs what is in pyproject.toml
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)

src, dst = sys.argv[1], sys.argv[2]

model = mlflow.pyfunc.load_model("model")
df = pd.read_csv(src)
X = df.drop(columns="is_fraud", errors="ignore").assign(online=lambda d: d.online.astype(int))
df["fraud_proba"] = model.predict(X)[:, 1]
df.to_csv(dst, index=False)

print(f"scored {len(df)} rows | mean P(fraud) {df.fraud_proba.mean():.3f} | max {df.fraud_proba.max():.3f}")
