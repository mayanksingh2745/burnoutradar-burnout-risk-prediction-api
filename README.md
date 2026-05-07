# BurnoutRadar: Burnout Risk Prediction API

This is a well-structured AI Engineering project for predicting burnout risk based on employee data. 

## Project Structure

- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and model prototyping.
- `models/`: Saved model artifacts (e.g., .pkl, .joblib).
- `src/`: Core source code.
  - `api/`: FastAPI application code.
  - `ml/`: Machine learning training, preprocessing, and inference logic.
  - `utils/`: Helper functions and logging.
  - `config.py`: Configuration and environment variables.
- `tests/`: Unit and integration tests.

## Setup Instructions

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`

## Running the API

```bash
uvicorn src.api.main:app --reload
```
The API will be available at http://127.0.0.1:8000
