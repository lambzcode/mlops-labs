"""Serve the fraud model as a REST API.

    uv run uvicorn app:app --port 8000
"""
import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal  # noqa: F401  (used by TODO 1)

import joblib
import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
BUDGET_MS = 200
log = logging.getLogger("uvicorn.error")
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once at startup: load the model, not on every request.
    If the file is missing the service still starts: /health says so and /predict answers 503."""
    try:
        bundle = joblib.load(MODEL_PATH)
        state.update(model=bundle["model"], version=bundle["version"])
        log.info("loaded %s (version %s)", MODEL_PATH, state["version"])
    except FileNotFoundError:
        log.error("no model at %s: running without a model", MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Fraud scoring API",
    description="Scores a card payment. Called by the payment team on every transaction.",
    version="1.0",
    lifespan=lifespan,
)
# Needed when a web page served from another origin calls this API from the browser.
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["*"], allow_headers=["*"])


class Transaction(BaseModel):
    """The contract: what a caller must send."""

    model_config = ConfigDict(extra="forbid")  # a misspelled field is an error, not silently ignored

    # TODO 1: make this contract strict and documented. For every field use Field(...) with:
    #   - a constraint: amount > 0, hour from 0 to 23, distance_km and tx_last_24h >= 0
    #   - a description and an example, which /docs will display
    # and replace `str` by a Literal of the six merchants:
    #   grocery, restaurant, gas, travel, electronics, crypto
    amount: float
    hour: int
    distance_km: float
    merchant: str
    online: bool
    tx_last_24h: int


class Prediction(BaseModel):
    fraud_proba: float = Field(description="Probability that the payment is fraud, between 0 and 1")
    model_version: str = Field(description="Which model answered: needed when a customer disputes a decision")
    model_ms: float = Field(description="Time spent scoring, in milliseconds")


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=1000)


class BatchPrediction(BaseModel):
    fraud_proba: list[float]
    model_version: str
    model_ms: float


def score(txs: list[Transaction]) -> list[float]:
    """Preprocessing lives HERE, so callers never have to know that the model wants online as 0/1."""
    X = pd.DataFrame([t.model_dump() for t in txs]).assign(online=lambda d: d.online.astype(int))
    return state["model"].predict_proba(X)[:, 1].tolist()


async def timed_score(txs: list[Transaction]) -> tuple[list[float], float]:
    t0 = time.perf_counter()
    proba = await asyncio.to_thread(score, txs)  # keep the event loop free while the CPU works
    return proba, (time.perf_counter() - t0) * 1000


@app.middleware("http")
async def tracing(request: Request, call_next):
    """Give every request an id and a timing header: what you grep for at 3 am."""
    request.state.request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:8])
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-Process-Time-Ms"] = f"{ms:.1f}"
    if ms > BUDGET_MS:
        log.warning("SLOW %s %s %d %.0f ms", request.method, request.url.path, response.status_code, ms)
    return response


@app.exception_handler(Exception)
async def unexpected_error(request: Request, exc: Exception):
    """Never leak a stack trace to the caller; log it and return a small, stable error body."""
    rid = getattr(request.state, "request_id", "-")
    log.exception("unhandled error, request %s", rid)
    return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": rid})


@app.get("/health", tags=["ops"])
async def health():
    """Is the service ready to score? A load balancer reads this."""
    if "model" not in state:
        return JSONResponse(status_code=503, content={"status": "degraded", "detail": "model not loaded"})
    return {"status": "ok", "model_version": state["version"]}


v1 = APIRouter(prefix="/v1", tags=["scoring"])  # the version is part of the URL: v2 can change the contract


@v1.post("/predict", responses={503: {"description": "No model loaded"}})
async def predict(tx: Transaction) -> Prediction:
    """Score one transaction."""
    # TODO 2: when no model is loaded, answer 503 with a clear message instead of crashing with a 500.
    # Copy the pattern used in predict_batch below.
    proba, ms = await timed_score([tx])
    return Prediction(fraud_proba=proba[0], model_version=state["version"], model_ms=ms)


@v1.post("/predict_batch", responses={503: {"description": "No model loaded"}})
async def predict_batch(batch: Batch) -> BatchPrediction:
    """Score up to 1000 transactions in one call."""
    if "model" not in state:
        raise HTTPException(status_code=503, detail="model not loaded")
    proba, ms = await timed_score(batch.transactions)
    return BatchPrediction(fraud_proba=proba, model_version=state["version"], model_ms=ms)


app.include_router(v1)
