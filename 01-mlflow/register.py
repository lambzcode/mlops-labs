"""Put a run's model in the Registry and give it an alias. Usage: python register.py RUN_ID [alias]"""
import os
import sys

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
NAME = os.getenv("MODEL_NAME", "fraud-detector")

run_id, alias = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "staging")
version = mlflow.register_model(f"runs:/{run_id}/model", NAME)
# TODO 4.1: give the new model version the alias (MlflowClient has a method for it)
print(f"{NAME} v{version.version} is now @{alias}")
