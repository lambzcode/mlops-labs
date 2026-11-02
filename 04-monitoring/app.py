"""Session 3 API, to be instrumented with the four monitoring layers. Run with ONE worker:
    uv run uvicorn app:app
Fill in the four TODOs. The metrics and the JSON log are already declared;
until a TODO is filled, its metric stays flat at zero.
"""
import asyncio
import logging
import os
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import structlog
from fastapi import FastAPI, Request, Response
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, Field

log = logging.getLogger("uvicorn.error")

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
BUDGET_MS = 200
THRESHOLD = 0.5
WINDOW = 300          # the last 300 amounts are compared with the training data
PSI_EVERY = 20        # recompute the PSI every 20 predictions

# Layer 1: infrastructure
LATENCY = Histogram("fraud_request_seconds", "Request latency", ["endpoint"],
                    buckets=[0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6])
PREDICTIONS = Counter("fraud_predictions_total", "Predictions served", ["model_version"])
# Layer 2: input
INPUT_PSI = Gauge("fraud_input_psi", "PSI of the last window against training", ["feature"])
# Layer 3: prediction
SCORES = Histogram("fraud_score", "Fraud probability", buckets=[i / 10 for i in range(1, 10)] + [0.95])
FLAGGED = Counter("fraud_flagged_total", "Predictions above the threshold", ["model_version"])

state: dict = {}
recent_amounts: deque = deque(maxlen=WINDOW)
seen = 0

# Layer 4: outcome. One JSON line per prediction, labels are joined offline.
ALLOWED = {"request_id", "ts"}   # allow-list. TODO 4: add what join_labels.py needs; never the amount, card or name
structlog.configure(
    processors=[structlog.processors.add_log_level, structlog.processors.JSONRenderer()],
    logger_factory=structlog.WriteLoggerFactory(file=open("predictions.jsonl", "a", buffering=1)),
)
plog = structlog.get_logger()


def redact(event: dict) -> dict:
    """Drop every field that is not on the allow-list: card, name and amount never reach the log."""
    return {k: v for k, v in event.items() if k in ALLOWED}


def psi(reference_edges: np.ndarray, reference_share: np.ndarray, sample: np.ndarray) -> float:
    counts = np.histogram(np.clip(sample, reference_edges[0], reference_edges[-1]), reference_edges)[0]
    live = np.clip(counts / counts.sum(), 1e-4, None)
    ref = np.clip(reference_share, 1e-4, None)
    return float(np.sum((live - ref) * np.log(live / ref)))


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(MODEL_PATH)                                    # as in session 3
    state.update(model=bundle["model"], version=bundle["version"])
    train = pd.read_csv("data/transactions.csv").amount.head(10000)     # the training window
    state["edges"] = np.quantile(train, np.linspace(0, 1, 11))          # deciles: 10 bins
    state["share"] = np.full(10, 0.1)
    INPUT_PSI.labels("amount").set(0)
    log.info("loaded %s (version %s)", MODEL_PATH, state["version"])
    yield
    state.clear()


app = FastAPI(title="Fraud scoring", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


class Transaction(BaseModel):
    amount: float = Field(gt=0)
    hour: int = Field(ge=0, le=23)
    distance_km: float = Field(ge=0)
    merchant: Literal["grocery", "restaurant", "gas", "travel", "electronics", "crypto"]
    online: bool
    tx_last_24h: int = Field(ge=0)


class Prediction(BaseModel):
    fraud_proba: float
    model_version: str
    model_ms: float


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    fraud_proba: list[float]
    model_version: str
    model_ms: float


def score(txs: list[Transaction]) -> list[float]:
    X = pd.DataFrame([t.model_dump() for t in txs]).assign(online=lambda d: d.online.astype(int))
    return state["model"].predict_proba(X)[:, 1].tolist()


async def timed_score(txs: list[Transaction]) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    proba = await asyncio.to_thread(score, txs)
    return proba, (time.perf_counter() - t0) * 1000


def observe(txs: list[Transaction], proba: list[float]) -> None:
    """Layers 2 and 3: called once per scored transaction."""
    global seen
    v = state["version"]
    for tx, p in zip(txs, proba):
        PREDICTIONS.labels(v).inc()
        SCORES.observe(p)
        if False:                    # TODO 3: flag the transaction when its score reaches THRESHOLD
            FLAGGED.labels(v).inc()
        recent_amounts.append(tx.amount)
        seen += 1
    if seen % PSI_EVERY < len(txs) and len(recent_amounts) == WINDOW:
        INPUT_PSI.labels("amount").set(0)   # TODO 2: replace 0 by psi(edges, share, the recent amounts as a numpy array)


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    seconds = time.perf_counter() - t0
    if request.url.path.startswith("/predict"):
        LATENCY.labels(request.url.path).observe(0)   # TODO 1: replace 0 by the request duration, in seconds
    response.headers["X-Process-Time-Ms"] = f"{seconds * 1000:.1f}"
    if seconds * 1000 > BUDGET_MS:
        log.warning("SLOW %s %s %d %.0f ms", request.method, request.url.path,
                    response.status_code, seconds * 1000)
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "model_version": state["version"]}


@app.post("/predict")
async def predict(tx: Transaction, response: Response) -> Prediction:
    proba, ms = await timed_score([tx])
    request_id = str(uuid.uuid4())
    response.headers["X-Request-Id"] = request_id
    observe([tx], proba)
    plog.info("prediction", **redact({
        "request_id": request_id, "model_version": state["version"],
        "fraud_proba": round(proba[0], 4), "merchant": tx.merchant, "ts": time.time(),
        "amount": tx.amount,             # dropped by redact(): not on the allow-list
    }))
    return Prediction(fraud_proba=proba[0], model_version=state["version"], model_ms=ms)


@app.post("/predict_batch")
async def predict_batch(batch: Batch) -> BatchPrediction:
    proba, ms = await timed_score(batch.transactions)
    observe(batch.transactions, proba)
    return BatchPrediction(fraud_proba=proba, model_version=state["version"], model_ms=ms)
