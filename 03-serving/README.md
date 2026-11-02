# Session 3 lab: what makes a good API

The payment team will call your fraud model on every card payment. A model in a notebook is not
usable by them: they need a **contract** (what to send, what comes back), **documentation** they
can read without asking you, **clear errors**, and a service that **fails safely**.

In this lab you serve the model with FastAPI and check each of these, one at a time.

| Word | Meaning |
|---|---|
| Contract | the exact fields a caller must send and what they get back |
| 422 | "your request is well-formed but invalid": the caller's fault |
| 500 | "something broke on our side" |
| 503 | "the service is up but cannot answer right now" (for example, no model) |
| Health check | a route a load balancer calls to know if the service is ready |

You edit three places, each marked `TODO`: the `Transaction` class and `/v1/predict` in `app.py`, and one test in `test_app.py`.

## Setup

```bash
uv sync
uv run train_model.py                      # writes model.joblib, the fraud model
uv run uvicorn app:app --port 8000 --reload
```

Use two terminals: one for the API, one for `curl` and tests.

## Step 1: serve it and read the docs

```bash
curl localhost:8000/health
```

Open http://localhost:8000/docs. FastAPI built this page from your code: routes, fields,
types, and a "Try it out" button. Then open http://localhost:8000/openapi.json: this is the
machine-readable contract that other teams can generate a client from.

Read `app.py` and notice:

- the model is loaded **once**, in `lifespan`, not on every request
- routes live under `/v1`: a future `/v2` can change the contract without breaking callers
- every answer carries `model_version`; every response carries `X-Request-ID` and `X-Process-Time-Ms` headers

## Step 2: call it from the terminal

Write the `curl` command yourself: a POST on `/v1/predict` with a `content-type` header and this JSON body.

```json
{"amount": 42.5, "hour": 14, "distance_km": 3.2, "merchant": "grocery", "online": false, "tx_last_24h": 1}
```

Then send a suspicious one: 900 at 3 am, 80 km away, `crypto`, online, 6 transactions in the last 24 hours.
Which probability do you get?

## Step 3: the contract (TODO 1)

Send nonsense:

```bash
curl -i -X POST localhost:8000/v1/predict -H 'content-type: application/json' \
  -d '{"amount": -5, "hour": 99, "distance_km": 3.2, "merchant": "casino", "online": false, "tx_last_24h": 1}'
```

The service happily answers `200` with a score. A confident wrong answer is worse than an error,
because nobody notices it. Run the tests to see the same problem:

```bash
uv run pytest -q          # 3 failed
```

**TODO 1:** in `app.py`, make `Transaction` strict and documented. For each field use `Field(...)` with a
constraint (`amount` above 0, `hour` from 0 to 23, the two counts at least 0), a `description` and an
`examples` value, and replace `merchant: str` by a `Literal` of the six merchants.

Save (the service reloads), then:

```bash
uv run pytest -q          # only the TODO 3 test still fails
```

Send the nonsense again: you now get `422` and a body that names every wrong field.
Reload `/docs`: each field now has a description and an example.
Notice that `extra="forbid"` (already in the class) also rejects a field with a typo.

## Step 4: fail safely (TODO 2)

What if the model file is missing? Stop the API and start it with no model:

```bash
MODEL_PATH=missing.joblib uv run uvicorn app:app --port 8000
curl -i localhost:8000/health
```

`/health` answers `503` and `degraded`: a load balancer would stop sending traffic. Now call `/v1/predict`:
it crashes with a `500`, because `state["model"]` does not exist.

**TODO 2:** in `/v1/predict`, answer `503` with `model not loaded` when there is no model. Copy the pattern
from `/v1/predict_batch` just below. Check with the same command. Then restart with the real model.

## Step 5: test it (TODO 3)

Open `test_app.py`. The fixture starts the app with `TestClient`, which runs the startup too.

**TODO 3:** write the last test: `hour=99` must give `422`.

```bash
uv run pytest -q          # 6 passed
```

## Step 6: load test (optional)

Locust simulates shoppers, about one request per second each. One user class at a time:

```bash
uv run uvicorn app:app --port 8000 --no-access-log
uv run locust -f locustfile.py SingleUser --headless -u 50 -r 10 -t 30s --host http://localhost:8000 --only-summary
```

Read the percentile table: `50%` is the median, `95%` is P95, and `Failures` must be 0.
Try 50, 150 and 250 users. Expect P95 to stay low, then climb fast once requests queue up.

## Questions to discuss

1. Why is `422` a different status from `500`? Who reads each one first?
2. `/health` returns `503` when the model is missing. What would a load balancer do with that?
3. Why is `/v1` in the URL? What would you do to change a field name without breaking the payment team?
4. The tests load the real model from `model.joblib`. What does that mean for a CI pipeline?
5. `extra="forbid"` rejects unknown fields. When might you prefer to ignore them instead?
