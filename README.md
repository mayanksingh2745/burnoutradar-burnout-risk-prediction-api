# BurnoutRadar - Burnout Risk Prediction API

This repository contains the backend and ML pipeline for BurnoutRadar, an API designed to predict burnout risk in remote workers. 

## Overview
- **Machine Learning**: A Voting Ensemble combining XGBoost, Random Forest, and Logistic Regression, trained on the MIDUS dataset.
- **Experiment Tracking**: MLflow is used for tracking parameters, metrics, and models.
- **Explainability**: SHAP (SHapley Additive exPlanations) is integrated to provide feature importance and local explanations for predictions.
- **API**: Served via FastAPI.
- **CI/CD**: GitHub Actions pipeline and Docker containerization.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Training the Model:**
   Before running the API, you must train the model so that `mlruns` directory has the saved model:
   ```bash
   python src/ml/train.py
   ```

3. **Running the API:**
   ```bash
   uvicorn src.api.main:app --reload
   ```

4. **Testing:**
   ```bash
   pytest tests/
   ```

## API Endpoints
- `GET /health` - Check if the API is running and model is loaded.
- `POST /predict` - Send worker features to get a burnout risk prediction.
- `POST /explain` - Get a SHAP explanation for a given prediction.
