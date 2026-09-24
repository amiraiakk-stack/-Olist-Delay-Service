# Olist Delay Inference Service

Predicts whether an Olist order will arrive late. Pipeline logic lives in `src/` and is
served through a FastAPI inference API; training and serving import the same
feature/preprocessing code, so what's served matches what training measured.
`notebooks/notebook_4.ipynb` is the only notebook left, kept for EDA context.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for every folder, config value, and tool.

## Layout

- `app/`: FastAPI schemas, routes, CLI.
- `src/`: extraction, labeling, splitting, features, preprocessing, training, validation,
  inference, logging, metrics, Great Expectations suite.
- `scripts/train_pipeline.py`: full training run, used by the `train` stage in `dvc.yaml`.
  Needs `DB_USER`, `DB_PASS`, `DB_NAME` in the environment.
- `config/config.yaml`: the one runtime config file.
- `saved_artifacts/`: model artifacts loaded at startup. Gitignored (see below).
- `tests/`: `pytest -q` runs everything.
- `data/`, `models/`: reserved, currently empty.

## Retraining

```bash
python -m scripts.train_pipeline --artifacts-dir saved_artifacts
# or: dvc repro train
```

Needs a real Olist Postgres database. If `MLFLOW_TRACKING_URI` is set, also logs to MLflow
and registers the model under `olist-delay-classifier` with a `champion` alias; set
`model.source: mlflow` in `config/config.yaml` to serve that version instead of local files.

Intermediate datasets are DVC-tracked (pointer files in git, bytes in DVC's cache). No
remote configured yet - add one with `dvc remote add -d storage <url>`, then `dvc push`.

## Fresh clone

`saved_artifacts/` is gitignored, so a clone has no model until you either train for real
(above) or build a synthetic fixture:

```bash
make fixture-artifacts   # or: python -m scripts.build_fixture_artifacts
```

CI does this automatically. Never point production traffic at fixture artifacts.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest -q
uvicorn app.main:app --reload
```

API at `http://localhost:8000`, docs at `/docs`. CLI: `python -m app.cli request.example.json`.

## API routes

- `GET /health`, `GET /model`, `GET /metrics`
- `POST /predict`: one order in, prediction/label/probability/model version out.
- `POST /predict/batch`: bounded batch inference.

## Containers

```bash
cp .env.example .env
docker compose up --build
```

Works on a clean machine, no database needed - a one-shot fixture service seeds the model
volume before `api` starts. Also brings up Postgres (MLflow's backend), MLflow, Prometheus.
For a real deployment, run `train_pipeline.py` against the live database and mount its
output over the `model_artifacts` volume instead.

## Quality gates

```bash
pre-commit run --all-files
pytest -q
ruff check app src tests
```

CI (`.github/workflows/ci.yml`) lints, formats, and tests on every push/PR; only pushes the
Docker image to `ghcr.io` on `main`.

## Input validation

Every request is checked twice against the `data:` contract in `config/config.yaml`: a
pandas guard (`src/validation.py`) for missing/out-of-range/unknown values, then a Great
Expectations suite (`src/expectations.py`) that also catches type issues, like a numeric
field arriving as a string. Any failure returns HTTP 422 - no silent defaults.

## Monitoring

`GET /metrics` exposes Prometheus counters (`src/monitoring.py`): request counts, latency,
errors, and prediction distribution. `docker compose up` runs Prometheus against
`monitoring/alerts.yml` (service down, high error rate, high latency, prediction-class
skew as a drift proxy) - no Alertmanager route wired up yet, so alerts show in Prometheus's
UI but don't page anyone. Accepted predictions are logged to `logs/predictions.jsonl` with
`order_id` and `predicted_at` for later evaluation against real outcomes.
