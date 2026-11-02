# Session 1: Reproducibility with MLflow and DVC

A model in production raises questions months later. An auditor asks which data it was
trained on. A bug report asks which code. A rollback asks which version came before.

Project: a card fraud classifier. Goal: from a commit hash and a run id, a teammate
rebuilds the model with metrics equal to within 1e-6.

**How this lab works.** Some commands and some lines of code are left out on purpose.
Look for `TODO`: complete the blank, run it, and check the result matches what is shown.

## 0. Setup

```bash
cd 01-mlflow
uv sync
alias py="uv run python"; alias dvc="uv run dvc"
export EXPERIMENT=fraud-<yourname>
```

Start MLflow in a second terminal and leave it running. Runs go to `mlflow.db`, models to
`mlartifacts/` (both ignored by Git):

```bash
cd 01-mlflow
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

Open http://localhost:5000. An empty UI means it works.

<details><summary>Port 5000 is busy, or shared server</summary>

```bash
# macOS often uses :5000 for AirPlay Receiver. Pick another port, in both terminals:
uv run mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
export MLFLOW_TRACKING_URI=http://localhost:5001

# shared server
export MLFLOW_TRACKING_URI=http://<host>:5000
```
</details>



## 1. Version the data

```bash
git init && dvc init
# TODO 1.1: register $HOME/dvc-storage as the default DVC remote, named "storage"
#           (a folder stands in for S3). Hint: dvc remote add --help
py gen_data.py jan                             # 20,000 transactions
# TODO 1.2: put data/transactions.csv under DVC control, then upload it to the remote
git add -A && git commit -m "january data"
# TODO 1.3: tag this commit v1-jan
```

Check with `dvc status -c` (it should say the cache and remote are in sync) and `git tag`.

```bash
cat data/transactions.csv.dvc
```

```yaml
outs:
- md5: 9f2c...
  path: transactions.csv
```

Git tracks the hash, not the CSV. Commit hash gives data hash gives the exact bytes.


## 2. Track a run

```bash
py train.py
```

```
AUC 0.7723 | AP 0.3019
run_id = 04a7...
```

In the UI, open the run and read its tags: `git_commit`, `git_dirty`, `data_md5`.
These answer "which code, which data" for this model.

`fit()` in `train.py` contains no MLflow calls. Logging happens in the `__main__` block only.
The tree was committed before training, so `git_dirty` is `False`. Keep that order.

**TODO 2.1:** `train.py` does not log everything yet. Before running it, complete the
four `TODO` lines in its `__main__` block: log the params, log the metrics, and set the
`git_dirty` and `data_md5` tags. Look at the helpers `git()` and `data_md5()` above it.
Without them the run has no way to answer "which code, which data".


## 3. Compare experiments

Hyperparameters live in `params.yaml`. CLI overrides are logged too.

```bash
for d in 2 4 6 8; do py train.py max_depth=$d; done
# TODO 3.1: with max_depth=6, run two more trainings with learning_rate 0.03 and 0.3
```

UI: select all runs, click Compare, open the parallel coordinates plot.

Question: which run has the best AP, and why do we rank on AP and not AUC when about 2%
of transactions are fraud?


## 4. Model registry

```bash
export RUN=<run_id of your best run>
# TODO 4.1: register that run's model. First complete register.py (one TODO), then
#           run it with $RUN. Expected output: "fraud-detector v1 is now @staging"
py score.py                       # loads models:/fraud-detector@staging
```

UI: Models tab, `fraud-detector`. Consumers load by name and alias, never by file path.

MLflow stages are deprecated. Aliases (`@staging`, `@production`) replace them: a movable
label on a version.


## 5. Reproduce from a fresh clone

Same checkout:

```bash
py repro_check.py $RUN
```

```
PASS code  (git commit)
PASS data  (dvc md5)
PASS auc            delta=0.0e+00
PASS avg_precision  delta=0.0e+00

REPRODUCED
```

Now as a teammate who has nothing but the repo and the run id:

```bash
git status                           # must be clean: a clone only sees commits
git clone . ../teammate && cd ../teammate
# TODO 5.1: go to the commit stored in the run's git_commit tag
# TODO 5.2: install the exact libraries from uv.lock
# TODO 5.3: download the data version this commit points to, from the DVC remote
py repro_check.py $RUN
```

`repro_check.py` has one TODO too: it never checks the data fingerprint. Complete it
before you run it (compare with the `code` check right above it).

Same result on a fresh clone is the deliverable of this session.

If `dvc pull` reports `missing-files`, you forgot `dvc push`.


## 6. New data arrives

Back in your main repo. February data has a different fraud pattern: small amounts, bursts
of transactions, mostly online.

```bash
cd ../01-mlflow
py gen_data.py feb
# TODO 6.1: tell DVC the file changed
py score.py
```

```
fraud-detector@staging on 30000 rows: AUC 0.515
```

The January model is at chance level on February data. Retrain, commit first:

```bash
# TODO 6.2: upload the new data to the remote, commit the new .dvc pointer, tag v2-feb
py train.py max_depth=6
export RUN2=<new run_id>
py register.py $RUN2              # v2 takes the @staging alias
py score.py                       # same command, now scores v2
```

Go back to the January state, code and data:

```bash
# TODO 6.3: switch to tag v1-jan AND make data/transactions.csv match it
py score.py                       # sees 20,000 rows again
# TODO 6.4: come back to February the same way
```

Question: `git checkout` alone is not enough here. Why, and which command completes it?

This performance drop is data drift. Session 5 detects it automatically.
