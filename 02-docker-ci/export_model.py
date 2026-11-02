"""Download the model behind an alias into ./model. Usage: python export_model.py [alias]"""
import os
import shutil
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
alias = sys.argv[1] if len(sys.argv) > 1 else "staging"
name = os.getenv("MODEL_NAME", "fraud-detector")

shutil.rmtree("model", ignore_errors=True)
mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}@{alias}", dst_path="model")
print(f"{name}@{alias} -> model/")
