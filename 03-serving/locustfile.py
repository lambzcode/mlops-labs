"""Load test. Usage, one user class at a time:

    uv run locust -f locustfile.py SingleUser --headless -u 50 -r 10 -t 30s --host http://localhost:8000 --only-summary
"""
import random

import pandas as pd
from locust import HttpUser, between, task

# real transactions from data/, so the load looks like production traffic
ROWS = (
    pd.read_csv("data/transactions.csv")
    .drop(columns="is_fraud")
    .sample(2000, random_state=0)
    .to_dict("records")
)


class SingleUser(HttpUser):
    """One shopper = one request about every second, so N users is roughly N requests/s."""

    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        self.client.post("/v1/predict", json=random.choice(ROWS))


class BatchUser(HttpUser):
    """A back-office job that sends 20 transactions at once."""

    wait_time = between(0.5, 1.5)

    @task
    def predict_batch(self):
        self.client.post("/v1/predict_batch", json={"transactions": random.sample(ROWS, 20)})
