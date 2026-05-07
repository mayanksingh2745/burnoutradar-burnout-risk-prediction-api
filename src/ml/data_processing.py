import pandas as pd
import numpy as np
import os
import pyreadstat
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path=None):
    """
    Load and preprocess the MIDUS dataset (.sav) for burnout prediction.
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
            'data', 'raw', 'MR2_P1_SURVEY_N2154_20251003.sav'
        )
    
    # Check if file exists, else fallback to mock for safety
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place the .sav file there.")

    print(f"Loading data from {data_path}...")
    df, meta = pyreadstat.read_sav(data_path)
    
    # Feature Mapping
    # age -> RB1PRAGE (Respondent's calculated age)
    # hours_worked -> RB1PB12 (Hours work for pay at main job)
    # stress_level -> RB1SF27 (Other ongoing stress at work)
    # sleep_hours -> RB1SA53A (Hours of sleep on workdays)
    # support_level -> RB1SF41D (Supervisor help/support) or RB1SF41A (Coworker help/support)
    
    features = {
        'age': 'RB1PRAGE',
        'hours_worked': 'RB1PB12',
        'stress_level': 'RB1SF27',  # 1-5 scale typically
        'sleep_hours': 'RB1SA53A',
        'support_level': 'RB1SF41A' # Coworker support
    }
    
    # Target variables for composite burnout risk
    # RB1SF27: Ongoing stress at work
    # RB1SF38B: Job stress makes irritable at home
    target_cols = ['RB1SF27', 'RB1SF38B']
    
    # Ensure columns exist
    cols_to_keep = list(features.values()) + ['RB1SF38B']
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")
        
    df_subset = df[cols_to_keep].copy()
    
    # Handle missing values - MIDUS often uses 9, 98, 99 for missing/refused
    # We will replace them with NaN and impute with median
    # We'll assume values > 90 are missing codes for age/hours, and 8,9 for scales
    df_subset.replace([98, 99], np.nan, inplace=True)
    
    # Convert scale 8, 9 to NaN for likert scales
    for col in ['RB1SF27', 'RB1SA53A', 'RB1SF41A', 'RB1SF38B']:
        df_subset.loc[df_subset[col] >= 8, col] = np.nan
        
    # Impute missing values with median
    df_subset.fillna(df_subset.median(), inplace=True)
    
    # Create target: Burnout Risk
    # High stress at work + stress makes irritable at home
    composite_stress = df_subset['RB1SF27'] + df_subset['RB1SF38B']
    # 1 if composite stress is above median, else 0
    y = (composite_stress > composite_stress.median()).astype(int)
    
    # Rename features
    X = df_subset[list(features.values())].rename(columns={v: k for k, v in features.items()})
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data successfully processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
