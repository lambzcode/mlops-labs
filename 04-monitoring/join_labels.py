"""Layer 4: join the logged predictions with the labels that arrived later.

    uv run join_labels.py            # AUC per cohort of 1000 predictions
Exit code 1 when the last cohort is below the gate (0.75 in the lab, 0.85 in the session 2 CI): the same
check, run by a scheduler, is how an outcome alert works.
"""
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score

GATE, COHORT = 0.75, 1000   # the lab model scores about 0.77 on normal traffic

preds = pd.read_json("predictions.jsonl", lines=True)
labels = pd.read_csv("labels.csv").drop_duplicates("request_id")
df = preds.merge(labels, on="request_id").sort_values("ts").reset_index(drop=True)
df["cohort"] = df.index // COHORT
rows = []
for c, g in df.groupby("cohort"):
    if len(g) == COHORT and g.is_fraud.nunique() == 2:
        rows.append((c, len(g), int(g.is_fraud.sum()), roc_auc_score(g.is_fraud, g.fraud_proba)))
if not rows:
    sys.exit("not enough labelled predictions yet: send more traffic")
print("cohort  rows  frauds   AUC")
for c, n, f, auc in rows:
    print(f"{c:>6} {n:>5} {f:>7}  {auc:.3f}")
last = rows[-1][3]
print(f"\nlast cohort AUC {last:.3f}: " + ("ok" if last >= GATE else f"BELOW THE GATE ({GATE})"))
sys.exit(0 if last >= GATE else 1)
