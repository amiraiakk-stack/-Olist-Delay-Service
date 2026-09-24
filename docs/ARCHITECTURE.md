# Architecture Reference

Every folder, every `config/config.yaml` value, and every tool in this repo, and why it's
there. Complements the task-oriented README.

## Folders

| Path | Contents | Why |
|---|---|---|
| `app/` | `main.py` (FastAPI app/routes), `schemas.py` (Pydantic models), `cli.py` (offline CLI) | The only layer that knows about HTTP/CLI. `src/` stays framework-agnostic. |
| `src/` | extraction, labeling, splitting, features, preprocessing, training, validation, expectations, inference, registry, monitoring, logging_utils, fixtures | One copy of every pipeline step. Training and serving both import from here, so there's no separate "prod version" of the logic that can drift. |
| `scripts/` | `train_pipeline.py` (real, DB-backed), `build_fixture_artifacts.py` (synthetic, for CI/local dev) | Two entry points because they answer different questions: real training needs a live DB; a placeholder model for tests doesn't and must never be mistaken for one. |
| `config/` | `config.yaml` | Every tunable in one place instead of scattered constants. |
| `tests/` | one file per concern: extraction, training, leakage, expectations, inference, registry, API | Runs together with `pytest -q`. |
| `notebooks/` | `notebook_4.ipynb` only | Pure EDA with no pipeline step to become - kept as context for feature choices. The other five were deleted once their logic moved to `src/` (notebook 1 also hardcoded a DB password, which didn't help). |
| `monitoring/` | `prometheus.yml`, `alerts.yml` | Observability as code, versioned with the service it watches. |
| `data/`, `models/` | empty, `.gitkeep` only | Reserved for raw data / promoted models if the project outgrows the current setup. |
| `saved_artifacts/` | gitignored except `.gitkeep`, populated at runtime | What `app/main.py` loads when `model.source: local`. Not committed - a trained model is a build output. |
| `eda_plots/` | gitignored | `plt.savefig()` output from running `notebook_4.ipynb` - a build output, not a tracked asset; the notebook's own rendered outputs already show the plots on GitHub. |
| `logs/` | gitignored, created at runtime | `inference.log` + `predictions.jsonl`. |
| `mlruns/` | gitignored except README | Local MLflow tracking data. |
| `.dvc/` | `config` (no remote yet), `cache/` (gitignored) | See "Data versioning" below. |
| `.github/workflows/ci.yml` | GitHub Actions | See "CI/CD" below. |

## `config/config.yaml`

```yaml
service:
  model_version: "1.0.0"     # returned by /model, stamped on every logged prediction
  log_file: logs/inference.log
  prediction_log_file: logs/predictions.jsonl
  max_batch_size: 1000        # cap enforced by InferenceService.predict_batch

model:
  source: local                # or "mlflow" to load the registered model instead
  artifacts_dir: saved_artifacts
  model_file: final_best_model.joblib
  # ...num_imputer/cat_imputer/encoder/scaler files: one joblib dump per fitted object
  mlflow_registered_model_name: olist-delay-classifier
  mlflow_model_alias: champion  # flipping this promotes a model without a redeploy

data:
  required_columns: [...]      # pandas guard rejects a request missing any of these
  numeric_ranges: {col: [min, max]}
  column_types: {col: dtype}   # used only by the GE suite - catches type coercion bugs
  categorical_values: {col: [...]}
  max_missing_ratio: 0.5

monitoring:
  latency_warning_ms: 1000     # mirrored in monitoring/alerts.yml, keep in sync
  error_rate_warning: 0.05
```

## Tools

| Tool | Role |
|---|---|
| FastAPI + uvicorn | HTTP layer, free OpenAPI docs from the Pydantic schemas |
| Pydantic | first validation layer - type/shape errors rejected before pandas runs |
| pandas / numpy | tabular manipulation throughout `src/` |
| scikit-learn | imputers, encoder, scaler, baselines, metrics |
| LightGBM / XGBoost | candidate classifiers, handle the ~1:12 class imbalance via `scale_pos_weight` |
| joblib | serializing the model and fitted transformers |
| SQLAlchemy + psycopg2 | reading source tables from Postgres, credentials from env only |
| DVC | versions the large intermediate datasets and defines the `train` stage |
| Great Expectations | second validation layer, catches what the pandas guard can't |
| MLflow | experiment tracking + model registry (`src/registry.py`) |
| prometheus-client | in-process metrics on `/metrics` |
| Prometheus | scrapes the service, evaluates `monitoring/alerts.yml` |
| pytest + httpx | test runner + FastAPI `TestClient` |
| ruff | lint + format, one tool for both, enforced in CI |
| pre-commit | runs ruff before each commit |
| Docker / docker-compose | packages the API with its dependencies for a one-command local stack |
| GitHub Actions | CI/CD |

## CI/CD

Two jobs, `test` then `image` (`needs: test`):

1. `test`: installs deps, runs `ruff check`/`ruff format --check`, builds a fixture model,
   runs `pytest -q`.
2. `image`: always builds the Docker image; only pushes to `ghcr.io/<owner>/<repo>` on a
   push to `main`.

## Data versioning and experiment tracking

1. `scripts/train_pipeline.py` (or `build_fixture_artifacts.py` for CI/local dev) produces
   the datasets and fitted artifacts.
2. `dvc add`/`dvc commit` records their hashes in git; the bytes live in `.dvc/cache/` and,
   once a remote is configured, in that remote after `dvc push`.
3. If `MLFLOW_TRACKING_URI` is set, the run also logs to MLflow and registers the model with
   a `champion` alias.
4. At serve time, `app/main.py` loads artifacts from local disk or the MLflow registry
   depending on `model.source` - both paths produce the same `ArtifactBundle`.
