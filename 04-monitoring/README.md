# Session 4 lab: four-layer monitoring

Your fraud API answers quickly and never crashes. Is it still right? You will instrument it on four
layers, watch it on a dashboard, then break its input and see which alert fires first.

| Layer | Question | Signal |
|---|---|---|
| 1 Infrastructure | is the service up and fast? | P95 latency, request rate |
| 2 Input | do the requests still look like the training data? | PSI of `amount` |
| 3 Prediction | do the scores still look the same? | flagged share, score distribution |
| 4 Outcome | are the predictions still right? | AUC per cohort, once labels arrive |

## Setup

You need Docker running. The model is loaded from a file, as in session 3.

```bash
uv sync
uv run train_model.py               # writes model.joblib from data/transactions.csv
docker compose up -d                # Prometheus, Alertmanager, Grafana
uv run uvicorn app:app              # the API, with ONE worker, on port 8000
```

If port 8000 is already taken, stop what uses it: Prometheus looks for the API there.

| Tool | Address |
|---|---|
| API metrics | http://localhost:8000/metrics/ |
| Prometheus | http://localhost:9090 |
| Alertmanager | http://localhost:9093 |
| Grafana (dashboard "Fraud API: four layers") | http://localhost:3000 |

The stack is provided: look at `prometheus/prometheus.yml`, `prometheus/alerts.yml` and the dashboard
before you start. Your job is the code in `app.py`.

## Step 1: layer 1, infrastructure

TODO 1 in `app.py`: the `timing` middleware must record the request duration in the `LATENCY` histogram.

```bash
uv run traffic.py normal            # steady traffic; leave it running in another terminal
```

Open Grafana: the latency and request rate panels move. In Prometheus, run
`histogram_quantile(0.95, sum by (le) (rate(fraud_request_seconds_bucket[1m])))`.

## Step 2: layer 2, input

TODO 2: set the `fraud_input_psi` gauge from the last 300 amounts. The `psi` function is provided; it compares
the live window with the 10 deciles of the training data.

With normal traffic the PSI stays near 0.

## Step 3: layer 3, prediction

TODO 3: count a prediction as flagged when its score reaches `THRESHOLD`.

In Grafana, note the flagged share and the median score under normal traffic.

## Step 4: layer 4, outcome and privacy

Every prediction is logged as one JSON line in `predictions.jsonl`. The log must never contain the amount,
a card number or a name, so `redact` keeps only the fields on an allow-list.

TODO 4: complete `ALLOWED` with what `join_labels.py` needs, and nothing sensitive.

```bash
head -2 predictions.jsonl           # no amount in there
uv run join_labels.py               # AUC per cohort of 1000 predictions
```

`traffic.py` writes the true label of each request to `labels.csv`, as if it arrived weeks later.
The join on `request_id` is what makes the outcome layer possible.

## Step 5: the alert

In `prometheus/alerts.yml`, TODO 5: set the threshold of the `InputDrift` rule to the PSI level at which
you act (session 5 rule of thumb). Then reload:

```bash
docker compose restart prometheus
```

## Step 6: incident 1, the inputs change

```bash
uv run traffic.py amount3           # every amount multiplied by 3
```

Predict first: which panels move, which alert fires, in which order? Then watch:

- Grafana: the PSI panel crosses its line, the flagged share rises
- Prometheus, Alerts tab: `InputDrift` goes from pending to firing
- Alertmanager: the alert arrives, grouped by name

## Step 7: incident 2, the world changes

Restart normal traffic for a few minutes, then:

```bash
uv run traffic.py concept           # same inputs, fraudsters change tactics
uv run join_labels.py
```

The inputs look the same, so the PSI stays quiet. Which layer sees the damage, and how late?

## Step 8: incident 3, the service is slow

```bash
uv run traffic.py burst             # huge batches in parallel
```

Watch the P95 panel and the request rate. Does the P95 cross the 0.2 s budget on your machine? If not, how would you find the load where it does?

## Questions to discuss

1. In step 6, why did layers 2 and 3 move while layer 1 stayed green?
2. In step 7, which layer saw it, and what does that say about delayed labels?
3. Why does the log keep an allow-list instead of removing a list of forbidden fields?
4. Why is `for: 1m` fine in this lab and wrong in production?
5. Why should `feature` be a label of the PSI gauge, and `request_id` never be one?

## Clean up

```bash
docker compose down
```
