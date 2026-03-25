import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Base feature columns (original data)
BASE_FEATURE_COLUMNS = ["overrep", "fourzerostart", "eventwins", "playerpopulation"]

# Temporal feature columns (engineered from time-series)
TEMPORAL_FEATURE_COLUMNS = [
    "winrate_change_last_period",
    "winrate_std_dev_3periods",
    "consistency_score",
    "avg_historical_winrate",
    "peak_winrate",
    "appearance_months",
    "popularity_momentum",
    "popularity_trend_strength",
    "trend_direction_encoded",
]

TARGET_COLUMN = "winrate"


def split_features_and_target(df, include_temporal=False):
    """
    Splits the DataFrame into features (X) and target (y).
    """
    feature_cols = BASE_FEATURE_COLUMNS.copy()
    
    if include_temporal:
        # Only include numeric temporal features (skip categorical ones that need encoding)
        temporal_numeric = [
            "winrate_change_last_period",
            "winrate_std_dev_3periods",
            "consistency_score",
            "avg_historical_winrate",
            "peak_winrate",
            "appearance_months",
            "popularity_momentum",
            "popularity_trend_strength",
            "trend_direction_encoded",
        ]
        # Only add if present in dataframe
        for col in temporal_numeric:
            if col in df.columns:
                feature_cols.append(col)
    
    X = df[feature_cols]
    y = df[TARGET_COLUMN]
    return X, y


def scale_features(X):
    """
    Scales the feature columns using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def chronological_train_test_split(df, test_fraction=0.3, temporal_col="timestamp"):
    """
    Splits data chronologically to prevent time-series data leakage.
    Uses the most recent data as test set, earlier data as training set.
    
    Important: For time-series data, random train/test split causes data leakage.
    This function enforces temporal ordering: train on past, test on future.
    
    Args:
        df: DataFrame with temporal data
        test_fraction: Fraction of time periods reserved for testing [0.0, 1.0]
        temporal_col: Column name containing timestamps (default: "timestamp")
    
    Returns:
        Tuple of (train_df, test_df)
    
    Example:
        train, test = chronological_train_test_split(df_historical, test_fraction=0.3)
        X_train, y_train = split_features_and_target(train, include_temporal=True)
        X_test, y_test = split_features_and_target(test, include_temporal=True)
    """
    if temporal_col not in df.columns:
        raise ValueError(f"Column '{temporal_col}' not found in DataFrame")
    
    df = df.sort_values(temporal_col).reset_index(drop=True)
    
    # Get unique time periods (not factions)
    unique_periods = df[temporal_col].unique()
    num_periods = len(unique_periods)
    
    # Calculate split point
    split_idx = int(num_periods * (1 - test_fraction))
    cutoff_date = unique_periods[split_idx]
    
    # Split based on date threshold
    train_df = df[df[temporal_col] < cutoff_date].copy()
    test_df = df[df[temporal_col] >= cutoff_date].copy()
    
    return train_df, test_df


def get_faction_statistics(df, group_by="faction"):
    """
    Computes per-faction statistics useful for understanding meta shifts.
    
    Args:
        df: DataFrame with temporal data
        group_by: Grouping column (default: "faction")
    
    Returns:
        DataFrame with aggregated statistics per faction:
        - mean_winrate, std_winrate: Average and volatility
        - max_winrate, min_winrate: Peak and trough performance
        - win_momentum: Trend (recent mean - historical mean)
        - seasons_appeared: Number of time periods in data
    """
    stats = df.groupby(group_by).agg({
        "winrate": ["mean", "std", "max", "min"],
        "season": "count",
        "playerpopulation": "mean",
    }).round(2)
    
    stats.columns = ["mean_winrate", "std_winrate", "max_winrate", "min_winrate", 
                     "seasons_appeared", "avg_playerpopulation"]
    
    # Calculate momentum (recent - historical)
    stats["win_momentum"] = (
        df.groupby(group_by)["winrate"].apply(
            lambda x: x.iloc[-1] - x.iloc[:-1].mean() if len(x) > 1 else 0
        ).round(2)
    )
    
    return stats
