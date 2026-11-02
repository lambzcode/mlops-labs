# Session 2: Docker and a first CI pipeline

## Why

Session 1 pinned code, data and libraries. The remaining gap is the machine: operating
system, system libraries, Python itself. "Works on my laptop" is the failure we remove.


```
your repo --push--> GitHub Actions --build + test--> image --push--> GHCR (registry)
```

## Setup

You need Docker Desktop running (`docker info` succeeds), a GitHub account, and your
session 1 repo with a model registered under the `staging` alias (MLflow running).

Copy the starter files from the course folder `02-docker/` to the root of your repo:

```bash
cp <course>/02-docker/{Dockerfile,.dockerignore,pyproject.toml,uv.lock,predict.py,export_model.py} .
```

## Docker basics

Run your first container, then go inside one:

```bash
docker run --rm python:3.12-slim python -c "import sys; print(sys.version)"

docker run --rm -it python:3.12-slim bash
cat /etc/os-release        # a different OS from your laptop
ls /Users /home            # none of your files are here
exit
```

A container is disposable. This time give it a name and do not use `--rm`, then leave a
file inside:

```bash
docker run -it --name scratch python:3.12-slim bash
echo "hello" > /tmp/note.txt
exit
```

Prediction: if you start another container from the same image, will `/tmp/note.txt`
be there? Write your answer down, then check:

```bash
# TODO 1.1: start a new container from python:3.12-slim (removed when you exit)
#           and look for /tmp/note.txt
# TODO 1.2: list all containers, including the stopped ones. Which one holds your file?
# TODO 1.3: go back into the container named scratch and look for the file again.
#           Hint: docker start --help (you need two flags to get a shell back)
# TODO 1.4: delete the scratch container
```

Each `docker run` creates a new container from the image. Your file lived in the first
one only. Anything you want to keep must come from the image or from a volume (step 2).

## Package and run the model

Take the model out of the registry, by alias, into a folder. Add a small sample for tests
and keep generated files out of Git:

```bash
py export_model.py                                 # models:/fraud-detector@staging -> model/
head -101 data/transactions.csv > sample.csv       # header + 100 rows
echo "out.csv" >> .gitignore
```

`model/` is a self-contained package: the pipeline with its preprocessing, the signature,
and the libraries it was trained with. Now read the Dockerfile. Every line is one layer:

```dockerfile
FROM python:3.12-slim                                    # base: OS + Python
COPY --from=ghcr.io/astral-sh/uv:0.12.22 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./                           # 1. dependencies first
RUN uv sync --locked --no-dev --no-install-project       # exactly what uv.lock says

COPY model/ model/                                       # 2. then the model
COPY predict.py .                                        # 3. then the code

ENTRYPOINT ["python", "predict.py"]                      # what runs at startup
```

Build, then run. The container sees your files only through a volume (`-v`):

```bash
docker build -t fraud-scorer:0.1 .
docker run --rm -v "$PWD:/data" fraud-scorer:0.1 /data/sample.csv /data/out.csv
head -3 out.csv                # the new fraud_proba column
```

```
scored 100 rows | mean P(fraud) 0.018 | max 0.297
```

`uv sync --locked` installs exactly what `uv.lock` records, like in session 1, and fails if
the lock is out of date. This small project lists only what scoring needs: training needs
DVC, serving does not.

## Two habits

**Layer cache.** Change the code and rebuild:

```bash
echo "# tweak" >> predict.py
docker build -t fraud-scorer:0.2 .
```

The `RUN uv pip install` step shows `CACHED`: Docker reuses every layer above the first
change, which is why dependencies are copied before code. Now change the dependencies
instead (`echo "# tweak" >> pyproject.toml`) and watch the install rerun.

**Secrets.** Never put a secret in an image. Anyone who can pull it can read it:

```bash
docker build -q -t leaky -f - . <<'EOF'
FROM fraud-scorer:0.1
ENV API_KEY=sk-demo-123
EOF
docker inspect leaky --format '{{.Config.Env}}'     # the key is right there
docker rmi leaky
```

Pass secrets at runtime instead: `docker run -e API_KEY=...` or `--env-file .env`.
`.dockerignore` keeps `.env` out of the build context.

## Push to GitHub

Create a repository (private is fine) and push. `model/` and `sample.csv` are committed;
`.env`, `.venv` and `out.csv` are not.

```bash
git add -A && git commit -m "docker packaging"
gh repo create fraud-mlops --private --source . --push
```

Without the GitHub CLI: create an empty repo on github.com, then
`git remote add origin git@github.com:<you>/fraud-mlops.git`, `git branch -M main`,
`git push -u origin main`.

## The workflow

GitHub runs any file in `.github/workflows/`. A workflow is a list of shell commands on
a fresh Ubuntu machine: your terminal on someone else's computer.

```
push to main --> checkout --> build --> smoke test --> push to GHCR
                                            |
                                  red here: nothing is published
```

Create `.github/workflows/docker.yml` and fill in the blanks. The permissions and the
last step are given: the token every run gets for free logs in to GHCR and pushes the
image tagged with the first 7 characters of the commit.

```yaml
name: docker

on:
  push:
    branches: [main]

permissions:
  contents: read
  packages: write

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v7

      - name: Build
        run: docker build -t fraud-scorer:ci .

      - name: Smoke test
        run: |
          docker run --rm -v "$PWD:/data" fraud-scorer:ci /data/sample.csv /data/out.csv
          head -3 out.csv

      - name: Push to GHCR
        env:
          TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          echo "$TOKEN" | docker login ghcr.io -u "$GITHUB_ACTOR" --password-stdin
          IMAGE=ghcr.io/$(echo "$GITHUB_REPOSITORY" | tr A-Z a-z):${GITHUB_SHA:0:7}
          docker tag fraud-scorer:ci "$IMAGE"
          docker push "$IMAGE"
```

Hints: the branch is the one you pushed. The build context is `.`. The two arguments are
the same as in step 2, with the sample file as input and `out.csv` as output. Registry
names must be lowercase, hence the `tr`.

```bash
git add .github && git commit -m "ci: build, test, publish" && git push
```

On GitHub, open the Actions tab, click the run, then the step names to expand their logs.
Green check: your image builds and scores a CSV on a machine you have never touched.
Then open the Packages section in your repo's right sidebar: the image is there, tagged
with the commit hash. Image, commit and (through session 1) data and model are tied together.

## Break it

CI is useful when it fails loudly.

1. In the `Dockerfile`, change `COPY model/ model/` to `COPY modle/ modle/`. Commit, push.
   The run turns red: open the failing step and find the line that explains why. Revert, push.
2. Make `predict.py` crash (for example read `sys.argv[3]`). The build passes but the
   smoke test fails. This is why we test the container, not only build it.
