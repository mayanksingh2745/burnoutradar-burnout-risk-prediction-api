import pandas as pd
import numpy as np
import os
import pyreadstat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path=None):
    """
    Load and preprocess the MIDUS dataset (.sav) for burnout prediction.
    Uses a comprehensive set of work, health, psychosocial, and coping features.
    Constructs burnout risk from multiple validated burnout indicators.
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
    features = {
        # Demographics
        'age': 'RB1PRAGE',
        'sex': 'RB1PRSEX',
        # Work conditions
        'hours_worked': 'RB1PB12',
        'work_situation_rating': 'RB1SF1',
        'work_control': 'RB1SF5',
        'work_effort': 'RB1SF6',
        'work_intensive': 'RB1SF39A',
        'work_skill_level': 'RB1SF39C',
        'work_choice_how': 'RB1SF39E',
        'work_choice_what': 'RB1SF39F',
        'work_decisions': 'RB1SF39G',
        'work_demands_combine': 'RB1SF39J',
        'work_absorbed': 'RB1SF39K',
        'coworker_support': 'RB1SF41A',
        'coworker_listen': 'RB1SF41B',
        'supervisor_support': 'RB1SF41D',
        'supervisor_listen': 'RB1SF41E',
        'pride_work': 'RB1SF43B',
        'others_respect_work': 'RB1SF43C',
        # Work-Family Spillover (constructed scales)
        'neg_work_to_family': 'RB1SNEGWF',
        'neg_family_to_work': 'RB1SNEGFW',
        'pos_work_to_family': 'RB1SPOSWF',
        'pos_family_to_work': 'RB1SPOSFW',
        # Sleep
        'sleep_hours': 'RB1SA53A',
        'trouble_falling_asleep': 'RB1SA57A',
        # Health
        'self_health': 'RB1PA3',
        'days_unable_work': 'RB1PA4',
        'days_cutback_work': 'RB1PA5',
        # Social Support (constructed scales)
        'coworker_support_scale': 'RB1SJCCS',
        'supervisor_support_scale': 'RB1SJCSS',
        'family_support': 'RB1SKINPO',
        'friend_support': 'RB1SFDSPO',
        'spouse_support': 'RB1SSPEMP',
    }
    
    # ── Target Variables ─────────────────────────────────────────────────
    target_indicators = {
        'stress_at_work': 'RB1SF27',
        'job_stress_irritable': 'RB1SF38B',
        'home_stress_irritable': 'RB1SF38L',
        'sleep_prevent': 'RB1SF38K',
        'job_harder_home': 'RB1SF38N',
    }
    
    # Verify all columns exist
    all_cols = list(features.values()) + list(target_indicators.values())
    available_cols = [c for c in all_cols if c in df.columns]
    missing_cols = [c for c in all_cols if c not in df.columns]
    
    if missing_cols:
        print(f"Warning: Missing columns (will be excluded): {missing_cols}")
        # Remove missing features
        features = {k: v for k, v in features.items() if v in df.columns}
        target_indicators = {k: v for k, v in target_indicators.items() if v in df.columns}
    
    all_cols = list(features.values()) + list(target_indicators.values())
    df_subset = df[all_cols].copy()
    
    # ── Handle Missing Values ────────────────────────────────────────────
    # Likert scales (1-5 typically): treat >= 7 as missing
    likert_cols = [
        'RB1SF27', 'RB1SF38B', 'RB1SF38L', 'RB1SF38K', 'RB1SF38N',
        'RB1SF1', 'RB1SF5', 'RB1SF6',
        'RB1SF39A', 'RB1SF39C', 'RB1SF39E', 'RB1SF39F', 'RB1SF39G',
        'RB1SF39J', 'RB1SF39K',
        'RB1SF41A', 'RB1SF41B', 'RB1SF41D', 'RB1SF41E',
        'RB1SF43B', 'RB1SF43C',
        'RB1SA57A', 'RB1PA3'
    ]
    for col in likert_cols:
        if col in df_subset.columns:
            df_subset.loc[df_subset[col] >= 7, col] = np.nan
    
    # Count/hours columns: treat >= 97 as missing
    count_cols = ['RB1PB12', 'RB1SA53A', 'RB1PA4', 'RB1PA5', 'RB1PRAGE']
    for col in count_cols:
        if col in df_subset.columns:
            df_subset.loc[df_subset[col] >= 97, col] = np.nan
    
    # Constructed scales: replace extreme missing codes
    scale_cols = [
        'RB1SNEGWF', 'RB1SNEGFW', 'RB1SPOSWF', 'RB1SPOSFW',
        'RB1SJCCS', 'RB1SJCSS', 'RB1SKINPO', 'RB1SFDSPO', 'RB1SSPEMP'
    ]
    for col in scale_cols:
        if col in df_subset.columns:
            df_subset.loc[df_subset[col] >= 98, col] = np.nan
    
    # Drop rows with missing target indicators (need at least 3 of 5)
    target_cols_list = [v for v in target_indicators.values() if v in df_subset.columns]
    df_subset = df_subset.dropna(subset=target_cols_list, thresh=3)
    
    # Impute remaining NaNs with median
    df_subset = df_subset.fillna(df_subset.median())
    
    # ── Construct Target Variable ────────────────────────────────────────
    # Composite burnout score from all available target indicators
    composite = sum(df_subset[col] for col in target_cols_list)
    
    # Use median as threshold for balanced 50/50 split
    threshold = composite.median()
    y = (composite > threshold).astype(int)
    
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # ── Prepare Features ─────────────────────────────────────────────────
    feature_cols = [v for v in features.values() if v in df_subset.columns]
    feature_names = [k for k, v in features.items() if v in df_subset.columns]
    X = df_subset[feature_cols].copy()
    X.columns = feature_names
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data successfully processed. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test
