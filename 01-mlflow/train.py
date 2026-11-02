"""Train the fraud model and log everything that matters to MLflow.
Usage: python train.py [key=value ...]   e.g. python train.py max_depth=6
"""
import os
import subprocess
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep the terminal clean

import mlflow
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = "data/transactions.csv"
TARGET = "is_fraud"


def load_params(overrides: list[str]) -> dict:
    params = yaml.safe_load(open("params.yaml"))
    for item in overrides:
        key, value = item.split("=")
        params[key] = type(params[key])(value)
    return params


def fit(params: dict, df: pd.DataFrame):
    """Pure function: same params + same data => same model. No MLflow in here."""
    X, y = df.drop(columns=TARGET), df[TARGET]
    X = X.assign(online=X.online.astype(int))
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=params["test_size"], stratify=y, random_state=params["seed"]
    )
    pre = ColumnTransformer(
        [("merchant", OneHotEncoder(handle_unknown="ignore"), ["merchant"])],
        remainder="passthrough",
    )
    clf = HistGradientBoostingClassifier(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_depth=params["max_depth"],
        random_state=params["seed"],
    )
    model = Pipeline([("pre", pre), ("clf", clf)]).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {"auc": roc_auc_score(y_te, proba), "avg_precision": average_precision_score(y_te, proba)}
    return model, metrics, X_te


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def data_md5() -> str:
    """The data's fingerprint, read from the small .dvc file that Git tracks."""
    return yaml.safe_load(open(f"{DATA}.dvc"))["outs"][0]["md5"]


if __name__ == "__main__":
    params = load_params(sys.argv[1:])
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("EXPERIMENT", "fraud"))

    df = pd.read_csv(DATA)
    with mlflow.start_run() as run:
        model, metrics, X_te = fit(params, df)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "git_commit": git("rev-parse", "HEAD"),
                "git_dirty": str(bool(git("status", "--porcelain"))),
                "data_md5": data_md5(),
                "data_rows": str(len(df)),
            }
        )
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"],  # we trust our own model
            pyfunc_predict_fn="predict_proba",  # serve fraud probabilities, not 0/1 labels
            signature=infer_signature(X_te, model.predict_proba(X_te)),
            input_example=X_te.head(3),
        )
        print(f"AUC {metrics['auc']:.4f} | AP {metrics['avg_precision']:.4f}")
        print(f"run_id = {run.info.run_id}")
