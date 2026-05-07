import pandas as pd
import numpy as np
import os
import pyreadstat
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(data_path=None):
    """
    Load and preprocess the MIDUS dataset (.sav) for burnout prediction.
    Uses a rich set of work-related, health, and psychosocial features.
    Constructs a composite burnout risk target from multiple stress/exhaustion indicators.
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
            'data', 'raw', 'MR2_P1_SURVEY_N2154_20251003.sav'
        )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please place the .sav file there.")

    print(f"Loading data from {data_path}...")
    df, meta = pyreadstat.read_sav(data_path)
    
    # ── Feature Mapping ──────────────────────────────────────────────────
    # We use a richer feature set that captures multiple burnout dimensions:
    #   Demographics, Work conditions, Sleep, Social support, Health, Coping
    
    features = {
        # Demographics
        'age': 'RB1PRAGE',                     # Respondent's calculated age
        # Work conditions
        'hours_worked': 'RB1PB12',             # Hours work for pay at main job
        'work_situation_rating': 'RB1SF1',     # Rate current work situation
        'work_control': 'RB1SF5',              # Rate amount control over work situation
        'work_intensive': 'RB1SF39A',          # Work intensively at job
        'work_demands_combine': 'RB1SF39J',    # Work demands hard to combine
        'coworker_support': 'RB1SF41A',        # Coworker help/support
        'supervisor_support': 'RB1SF41D',      # Supervisor help/support
        # Sleep
        'sleep_hours': 'RB1SA53A',             # Hours of sleep on workdays
        'trouble_falling_asleep': 'RB1SA57A',  # Trouble fall asleep (frequency)
        # Health
        'self_health': 'RB1PA3',               # Compare overall health to others your age
        'days_unable_work': 'RB1PA4',          # Days unable to work because of health (30 days)
        'days_cutback_work': 'RB1PA5',         # Days cut back work because of health (30 days)
    }
    
    # ── Target Variables ─────────────────────────────────────────────────
    # Multiple indicators of burnout / psychological distress:
    target_indicators = {
        'stress_at_work': 'RB1SF27',           # Other ongoing stress at work (12 months)
        'job_stress_irritable': 'RB1SF38B',    # Job stress makes irritable at home
        'home_stress_irritable': 'RB1SF38L',   # Home stress makes irritable at job
        'sleep_prevent': 'RB1SF38K',           # Home chores prevent sleep to do job
    }
    
    # Ensure all columns exist
    all_cols = list(features.values()) + list(target_indicators.values())
    missing_cols = [c for c in all_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns in dataset: {missing_cols}")
        
    df_subset = df[all_cols].copy()
    
    # ── Handle Missing Values ────────────────────────────────────────────
    # MIDUS uses special codes: 7=Don't Know, 8=Refused, 9=Not Applicable,
    # 97/98/99 for larger scales. Replace with NaN.
    
    # For Likert scales (1-5 or 1-10), treat >= 7 as missing
    likert_cols = [
        'RB1SF27', 'RB1SF38B', 'RB1SF38L', 'RB1SF38K',
        'RB1SF1', 'RB1SF5', 'RB1SF39A', 'RB1SF39J',
        'RB1SF41A', 'RB1SF41D', 'RB1SA57A', 'RB1PA3'
    ]
    for col in likert_cols:
        df_subset.loc[df_subset[col] >= 7, col] = np.nan
    
    # For count/hours columns, treat 98/99 as missing
    count_cols = ['RB1PB12', 'RB1SA53A', 'RB1PA4', 'RB1PA5', 'RB1PRAGE']
    for col in count_cols:
        df_subset.loc[df_subset[col] >= 97, col] = np.nan
    
    # Drop rows where we don't have enough target info (need at least 2 target indicators)
    target_cols_list = list(target_indicators.values())
    df_subset = df_subset.dropna(subset=target_cols_list, thresh=2)
    
    # Impute remaining missing values with median
    df_subset = df_subset.fillna(df_subset.median())
    
    # ── Construct Target Variable ────────────────────────────────────────
    # Create a composite burnout score from 4 indicators, then use
    # the 60th percentile as the threshold for "High Risk" to get a 
    # more balanced split (~40% positive class).
    
    composite = (
        df_subset['RB1SF27'] +      # work stress
        df_subset['RB1SF38B'] +      # job stress -> irritable at home
        df_subset['RB1SF38L'] +      # home stress -> irritable at job
        df_subset['RB1SF38K']        # sleep issues from home chores
    )
    
    threshold = composite.quantile(0.60)
    y = (composite > threshold).astype(int)
    
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # ── Prepare Features ─────────────────────────────────────────────────
    X = df_subset[list(features.values())].rename(
        columns={v: k for k, v in features.items()}
    )
    
    # Split into train and test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data successfully processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
