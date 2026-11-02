"""Replay transactions against the API. One command per incident.

    uv run traffic.py normal               # steady traffic, everything green
    uv run traffic.py burst                # huge batches in parallel
    uv run traffic.py amount3              # every amount times 3
    uv run traffic.py crypto               # crypto goes from 5 to 17 percent
    uv run traffic.py concept              # same inputs, fraudsters change tactics

Options: --rate 20 (requests per second), --minutes 3, --url http://localhost:8000
Every single-transaction run appends request_id,is_fraud to labels.csv.
"""
import argparse
import asyncio
import csv
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

MODES = ["normal", "burst", "amount3", "crypto", "concept"]
LIVE_FROM = 10_000          # rows before this one are the training window


def load(mode: str) -> pd.DataFrame:
    df = pd.read_csv("data/transactions.csv").iloc[LIVE_FROM:].reset_index(drop=True)
    if mode == "amount3":
        df["amount"] *= 3
    if mode == "crypto":
        w = df.merchant.map(lambda m: 4.0 if m == "crypto" else 1.0)     # about 17 percent crypto
        df = df.sample(len(df), weights=w, replace=True, random_state=0).reset_index(drop=True)
    if mode == "concept":
        # same inputs, new fraud rule: the February behaviour of session 1 (small, online, bursty)
        rng = np.random.default_rng(7)
        night = df.hour.between(0, 5)
        z = -6.2 + 0.9 * df.online + 0.7 * df.tx_last_24h - 0.01 * df.amount + 0.5 * night
        df["is_fraud"] = rng.random(len(df)) < 1 / (1 + np.exp(-z))
    return df


async def one(client: httpx.AsyncClient, row: dict, out: csv.writer, stats: dict) -> None:
    label = row.pop("is_fraud")
    try:
        r = await client.post("/predict", json=row)
        stats["ok" if r.status_code == 200 else "err"] += 1
        if r.status_code == 200:
            out.writerow([r.headers["X-Request-Id"], int(label)])
    except httpx.HTTPError:
        stats["err"] += 1


async def batches(client: httpx.AsyncClient, df: pd.DataFrame, stats: dict, end: float) -> None:
    rows = df.drop(columns="is_fraud").head(1000).to_dict("records")
    async def worker():
        while time.time() < end:
            try:
                r = await client.post("/predict_batch", json={"transactions": rows}, timeout=60)
                stats["ok" if r.status_code == 200 else "err"] += 1
            except httpx.HTTPError:
                stats["err"] += 1
    await asyncio.gather(*[worker() for _ in range(4)])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=MODES)
    ap.add_argument("--rate", type=float, default=10)
    ap.add_argument("--minutes", type=float, default=3)
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args()
    df = load(a.mode)
    end = time.time() + a.minutes * 60
    stats = {"ok": 0, "err": 0}
    new = not Path("labels.csv").exists()
    with open("labels.csv", "a", newline="") as f:
        out = csv.writer(f)
        if new:
            out.writerow(["request_id", "is_fraud"])
        async with httpx.AsyncClient(base_url=a.url, timeout=10) as client:
            print(f"{a.mode}: {a.minutes:g} min, Ctrl+C to stop")
            if a.mode == "burst":
                await batches(client, df, stats, end)
            else:
                tasks, i, t = [], 0, time.time()
                while time.time() < end:
                    tasks.append(asyncio.create_task(one(client, df.iloc[i % len(df)].to_dict(), out, stats)))
                    i += 1
                    t += 1 / a.rate
                    await asyncio.sleep(max(0, t - time.time()))
                    if i % int(a.rate * 10) == 0:
                        print(f"  sent {i}, ok {stats['ok']}, errors {stats['err']}")
                await asyncio.gather(*tasks)
    print(f"done: ok {stats['ok']}, errors {stats['err']}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("stopped")
