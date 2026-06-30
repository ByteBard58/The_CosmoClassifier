# CosmoClassifier Design

## Purpose

CosmoClassifier is a machine-learning web application for classifying Sloan Digital Sky Survey DR18 celestial objects as `GALAXY`, `STAR`, or `QSO`. The application exposes an interactive browser interface for single-object and CSV batch prediction, backed by a serialized scikit-learn-compatible pipeline.

The design goal is to keep model inference simple and reproducible:

- Accept a small, documented set of astronomical input features.
- Apply the same feature engineering used during training.
- Preserve model feature ordering exactly.
- Return a class label and confidence distribution with minimal latency.
- Fall back to model training only when required artifacts are missing.

## High-Level Architecture

```text
Browser UI
  |
  | GET /
  | POST /predict
  | POST /predict/file
  v
FastAPI application: app/app.py
  |
  | startup lifespan
  v
Serialized model artifacts
  - models/estimator.pkl
  - models/column_names.pkl
  |
  | missing artifacts
  v
Training pipeline: models/fit.py
  |
  v
Dataset: Datasets/SDSS_DR18.csv
```

The app is organized into four main areas:

- `app/`: FastAPI service, Pydantic validation schema, static assets, and HTML UI.
- `models/`: training code and serialized inference artifacts.
- `Datasets/`: local SDSS DR18 dataset used for training or artifact regeneration.
- `notebooks/` and `reports/`: research and exploratory analysis material.

## Runtime Components

### FastAPI Service

The API is defined in `app/app.py`. It creates a `FastAPI` instance with a lifespan hook that loads the trained estimator and expected column names at startup.

Key routes:

- `GET /`: serves `app/templates/index.html`.
- `GET /health`: returns application metadata and operational status.
- `POST /predict`: accepts a JSON payload for one celestial object and returns one prediction.
- `POST /predict/file`: accepts a CSV upload and returns predictions for all rows.

Static assets are served from `app/static` under `/static`.

### Frontend

The frontend is a static single-page experience made from:

- `app/templates/index.html`
- `app/static/style.css`
- `app/static/script.js`

The UI has three main tabs:

- Single prediction form.
- Batch CSV prediction form.
- Model information view.

Client-side JavaScript handles tab navigation, form submission, CSV upload controls, result rendering, probability bars, toast messages, and a batch doughnut chart.

The frontend performs basic usability checks, but the backend remains the source of truth for validation.

### Model Artifacts

Inference depends on two joblib files:

- `models/estimator.pkl`: the trained imbalanced-learn/scikit-learn pipeline.
- `models/column_names.pkl`: the feature order used during training, including the final `class` target column.

On application startup, `load_or_create_models()` checks for both files. If either is absent, it calls `models.fit.main()` to regenerate them from the dataset. This makes local setup easier, but it means a missing artifact can trigger a potentially long training job during service startup.

## Data Flow

### Single Prediction

1. The browser collects these fields:
   - `ra`
   - `dec`
   - `redshift`
   - `psfMag_r`
   - `u`
   - `g`
   - `r`
   - `i`
   - `z`
2. The frontend sends JSON to `POST /predict`.
3. FastAPI validates the request using `UserInput` from `app/schema/validation.py`.
4. `preprocess_data()` computes color features:
   - `u_g_color = u - g`
   - `g_r_color = g - r`
   - `r_i_color = r - i`
   - `i_z_color = i - z`
5. Raw magnitude columns `u`, `g`, `r`, `i`, and `z` are removed from the inference payload.
6. The service builds a feature array in the exact order stored in `column_names.pkl`, skipping `class`.
7. The pipeline runs `predict()` and `predict_proba()`.
8. Numeric labels are mapped to public class names:
   - `0 -> GALAXY`
   - `1 -> STAR`
   - `2 -> QSO`
9. The API returns:

```json
{
  "message": "prediction successful",
  "prediction": "GALAXY",
  "probabilities": {
    "GALAXY": 0.981,
    "STAR": 0.012,
    "QSO": 0.007
  }
}
```

### Batch Prediction

1. The browser uploads a `.csv` file to `POST /predict/file`.
2. The backend verifies the file extension.
3. The CSV is parsed with pandas.
4. The backend requires the exact column list and order:

```text
ra, dec, redshift, psfMag_r, u, g, r, i, z
```

5. All values are converted to floats.
6. The same color features are generated vectorially.
7. Raw magnitude columns are dropped.
8. Columns are reordered to match the model feature order.
9. Predictions and probabilities are returned as arrays.

## Training Design

Training is implemented in `models/fit.py`.

### Dataset Preparation

`data_collection()` reads `Datasets/SDSS_DR18.csv` by default. The path can be overridden with the `PATH_DS` environment variable.

The training data preparation performs these steps:

1. Drop identifier and metadata fields that are not intended for prediction.
2. Map target labels:
   - `GALAXY -> 0`
   - `STAR -> 1`
   - `QSO -> 2`
3. Keep the reduced feature set:
   - `ra`
   - `dec`
   - `redshift`
   - `u`
   - `g`
   - `r`
   - `i`
   - `z`
   - `psfMag_r`
   - `class`
4. Engineer the four color contrast features.
5. Drop the raw `u`, `g`, `r`, `i`, and `z` columns.
6. Move `class` to the final column.
7. Return `x`, `y`, and `column_names`.

### Model Selection

`model()` splits data into train and test sets with stratification. It then uses an imbalanced-learn `Pipeline` with:

- `SimpleImputer(strategy="median")`
- `StandardScaler`
- `SMOTE`
- optional dimensionality reduction
- classifier

`RandomizedSearchCV` evaluates candidate configurations across random forest, logistic regression, and XGBoost variants. The best estimator is refit and saved as the final pipeline.

The checked-in README describes the selected production model as logistic regression with L1 penalty, `saga` solver, and `C = 10`, but the code is capable of selecting other candidates when retrained.

## Validation

Single-object requests use Pydantic constraints:

- `ra`: 0 to 360
- `dec`: -90 to 90
- `redshift`: -2 to 10
- `psfMag_r`: -30 to 30
- `u`, `g`, `r`, `i`, `z`: -30 to 30
- infinite and NaN values are rejected

Validation errors are flattened into a frontend-friendly response:

```json
{
  "message": "validation failed",
  "error": "Invalid ra: Input should be greater than or equal to 0"
}
```

Batch requests currently validate file type, exact column order, and numeric compatibility. They do not apply the same per-field numeric ranges used by the single prediction schema.

## Deployment

The Dockerfile builds from `python:3.11-slim`, installs `requirements.txt`, copies the repository, exposes port `8000`, and starts the app with Gunicorn using Uvicorn workers:

```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 app.app:app
```

For local development, the app can be started with:

```bash
uvicorn app.app:app --reload
```

## Design Constraints and Tradeoffs

- The service is intentionally state-light. The model and column names are loaded once into `app.state`.
- Feature order is controlled by `column_names.pkl`, reducing the risk of training/inference mismatch.
- Startup self-healing improves local usability, but production deployments should normally include prebuilt artifacts to avoid long startup times.
- CSV batch validation is strict about column order. This keeps implementation simple and predictable, but it can be less forgiving for users.
- The frontend is static and directly coupled to the current API shape. This is simple to deploy, though larger UI changes may benefit from stronger shared contracts or generated API types.
- Both Flask and FastAPI are present in dependencies, but the active application is FastAPI.

## Future Improvements

- Add automated tests for validation, feature engineering, single prediction, and batch prediction.
- Share feature engineering logic between training and inference through a common module.
- Apply Pydantic-equivalent range validation to batch CSV rows.
- Add a maximum accepted row count for batch prediction to protect memory and response time.
- Store model metadata, training date, metrics, feature order, and dataset version alongside the serialized estimator.
- Avoid retraining during production startup; fail fast when required artifacts are missing.
- Align README runtime and framework wording with the current FastAPI implementation.
- Consider returning class probabilities for batch predictions as objects keyed by class name, matching the single prediction response.
