"""The 1e-6 test. Usage: python repro_check.py RUN_ID
Retrains from the CURRENT checkout and compares with what MLflow remembers."""
import os
import sys

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # keep the terminal clean

import mlflow
import pandas as pd

from train import data_md5, fit, git, load_params

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
run = mlflow.get_run(sys.argv[1])
tags, logged = run.data.tags, run.data.metrics

# params come from the run, not from params.yaml: we want what was REALLY used
params = {k: type(v)(run.data.params[k]) for k, v in load_params([]).items()}

checks = {
    "code  (git commit)": git("rev-parse", "HEAD") == tags["git_commit"],
    "data  (dvc md5)": ____,  # TODO 5.4: True when the current data md5 equals the one tagged on the run
}
_, metrics, _ = fit(params, pd.read_csv("data/transactions.csv"))
for name in logged:
    delta = abs(metrics[name] - logged[name])
    checks[f"{name:<14} delta={delta:.1e}"] = delta < 1e-6

for label, ok in checks.items():
    print("PASS" if ok else "FAIL", label)
print("\nREPRODUCED" if all(checks.values()) else "\nNOT REPRODUCIBLE")
sys.exit(not all(checks.values()))
