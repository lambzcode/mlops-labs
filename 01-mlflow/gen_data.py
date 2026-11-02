"""Generate a month of fake card transactions. Usage: python gen_data.py jan|feb"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MERCHANTS = ["grocery", "restaurant", "gas", "travel", "electronics", "crypto"]

# month -> (seed, rows, how fraudsters behave that month)
MONTHS = {
    "jan": dict(seed=1, rows=20_000, style="heist"),  # big, far, at night
    "feb": dict(seed=2, rows=30_000, style="testing"),  # small, bursty, online
}


def generate(month: str) -> pd.DataFrame:
    cfg = MONTHS[month]
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["rows"]
    df = pd.DataFrame(
        {
            "amount": rng.lognormal(3.3, 1.0, n).round(2),
            "hour": rng.integers(0, 24, n),
            "distance_km": rng.exponential(8, n).round(1),
            "merchant": rng.choice(MERCHANTS, n, p=[0.35, 0.25, 0.15, 0.08, 0.12, 0.05]),
            "online": rng.random(n) < 0.4,
            "tx_last_24h": rng.poisson(2, n),
        }
    )
    night = df.hour.between(0, 5)
    risky = df.merchant.isin(["electronics", "crypto"])
    if cfg["style"] == "heist":
        z = -5.6 + 0.012 * df.amount + 1.2 * night + 1.5 * risky + 0.03 * df.distance_km
    else:  # card testing: tiny amounts, bursts of transactions, online
        z = -6.2 + 0.9 * df.online + 0.7 * df.tx_last_24h - 0.01 * df.amount + 0.5 * night
    df["is_fraud"] = rng.random(n) < 1 / (1 + np.exp(-z))
    return df


if __name__ == "__main__":
    month = sys.argv[1]
    df = generate(month)
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/transactions.csv", index=False)
    print(f"{month}: {len(df)} transactions, {df.is_fraud.mean():.2%} fraud")
