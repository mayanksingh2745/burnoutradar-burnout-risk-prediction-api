import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path=None):
    """
    Mock function to load and preprocess the MIDUS dataset.
    In a real scenario, this would load from a CSV or database and perform actual preprocessing.
    """
    # Generating mock data for the sake of the project structure
    # Let's assume we have 5 features: age, hours_worked, stress_level, sleep_hours, support_level
    np.random.seed(42)
    n_samples = 1000
    
    X = pd.DataFrame({
        'age': np.random.randint(22, 65, n_samples),
        'hours_worked': np.random.normal(45, 10, n_samples),
        'stress_level': np.random.randint(1, 10, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'support_level': np.random.randint(1, 5, n_samples)
    })
    
    # Target variable: 1 for High Burnout Risk, 0 for Low
    # Creating a synthetic relationship
    risk_score = (X['hours_worked'] * 0.1 + X['stress_level'] * 0.5 - X['sleep_hours'] * 0.8 - X['support_level'] * 0.4)
    y = (risk_score > risk_score.median()).astype(int)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
