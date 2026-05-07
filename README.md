# BurnoutRadar — Burnout Risk Prediction API

**ML / Backend Engineer Project**

**Tech:** Scikit-learn, XGBoost, FastAPI, MLflow, pytest, GitHub Actions, Docker

A production-ready API that predicts burnout risk in remote workers using a Voting Ensemble trained on the MIDUS (Midlife in the United States) dataset.

## Highlights

- **Voting Ensemble** (XGBoost + Random Forest + Logistic Regression) trained on 33 psychosocial and work-related features from the MIDUS dataset; achieved **87% weighted F1-score**.
- **Experiment Tracking** with MLflow — parameters, metrics, and model artifacts are versioned automatically.
- **SHAP Explanations** — every prediction comes with feature-level explanations via a dedicated `/explain` endpoint.
- **REST API** via FastAPI with auto-generated Swagger docs.
- **CI/CD** — GitHub Actions runs tests on every push; Docker containerization for deployment.

## Project Structure

```
├── .github/workflows/ci.yml       # CI pipeline
├── data/raw/                       # MIDUS dataset (.sav)
├── src/
│   ├── api/
│   │   ├── main.py                 # FastAPI app
│   │   ├── routes.py               # /predict, /explain, /health
│   │   ├── schemas.py              # Pydantic request/response models
│   │   └── dependencies.py         # Model loading & SHAP explainer
│   └── ml/
│       ├── data_processing.py      # MIDUS data loader & feature engineering
│       └── train.py                # Ensemble training with MLflow tracking
├── tests/
│   ├── test_api.py                 # API endpoint tests
│   └── test_model.py               # Data pipeline tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python src/ml/train.py
   ```
   This creates an `mlruns/` directory with the saved model and logged metrics.

3. **Run the API:**
   ```bash
   uvicorn src.api.main:app --reload
   ```
   Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.

4. **Run tests:**
   ```bash
   python -m pytest tests/
   ```

## API Endpoints

| Method | Endpoint    | Description                                   |
|--------|-------------|-----------------------------------------------|
| GET    | `/health`   | Health check — confirms model is loaded        |
| POST   | `/predict`  | Returns burnout risk prediction (High/Low)     |
| POST   | `/explain`  | Returns prediction + SHAP feature importances  |

## Features Used (33 total)

| Category           | Features                                                                                       |
|--------------------|-----------------------------------------------------------------------------------------------|
| Demographics       | Age, Sex                                                                                       |
| Work Conditions    | Hours worked, work situation rating, control, effort, intensity, skill demands, autonomy, etc. |
| Work-Family        | Negative/Positive Work→Family and Family→Work spillover                                        |
| Sleep              | Sleep hours on workdays, trouble falling asleep frequency                                      |
| Health             | Self-rated health, days unable to work, days cut back                                          |
| Social Support     | Coworker, supervisor, family, friend, and spouse support scales                                |

## Model Performance

| Metric             | Score  |
|--------------------|--------|
| Weighted F1-Score  | 0.85   |
| Accuracy           | 0.84   |
| Precision (class 1)| 0.84   |
| Recall (class 1)   | 0.79   |
