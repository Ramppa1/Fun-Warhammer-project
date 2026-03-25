"""
Temporal feature engineering for Warhammer meta analysis.
Generates momentum, volatility, and historical trend features from time-series data.
"""

import pandas as pd
import numpy as np


def engineer_temporal_features(historical_df):
    """
    Engineers temporal features for each faction across time periods.
    
    Args:
        historical_df: DataFrame from load_historical_data() with columns:
                      season, timestamp, faction, winrate, overrep, fourzerostart, eventwins, playerpopulation
    
    Returns:
        DataFrame with additional temporal feature columns:
        - winrate_change_last_period: Change in winrate from previous season (+/-)
        - trend_direction: Categorical (up/stable/down) based on recent direction
        - winrate_std_dev_3periods: Rolling std deviation (volatility) over last 3 periods
        - consistency_score: Inverse of volatility (0-1 scale, higher = more stable)
        - avg_historical_winrate: Mean winrate across all previous periods
        - peak_winrate: Best winrate achieved in history
        - appearance_months: How many distinct time periods faction appeared in
        - popularity_momentum: Change in player population
        - popularity_trend_strength: Interaction feature (popularity_momentum * winrate_trend)
    """
    df = historical_df.copy()
    
    # Sort by faction and timestamp to ensure chronological order
    df = df.sort_values(["faction", "timestamp"]).reset_index(drop=True)
    
    # Initialize feature columns with NaN
    df["winrate_change_last_period"] = np.nan
    df["trend_direction"] = "stable"
    df["winrate_std_dev_3periods"] = np.nan
    df["consistency_score"] = np.nan
    df["avg_historical_winrate"] = np.nan
    df["peak_winrate"] = np.nan
    df["appearance_months"] = np.nan
    df["popularity_momentum"] = np.nan
    df["popularity_trend_strength"] = np.nan
    
    # Group by faction and compute temporal features
    for faction in df["faction"].unique():
        faction_data = df[df["faction"] == faction].copy()
        faction_indices = df[df["faction"] == faction].index
        
        # 1. Winrate change from previous period
        winrate_change = faction_data["winrate"].diff()
        df.loc[faction_indices, "winrate_change_last_period"] = winrate_change.values
        
        # 2. Trend direction (up/stable/down based on last 2 periods)
        trend_direction = []
        for i, wr_change in enumerate(winrate_change):
            if pd.isna(wr_change):
                trend_direction.append("stable")  # First period
            elif wr_change > 2:
                trend_direction.append("up")
            elif wr_change < -2:
                trend_direction.append("down")
            else:
                trend_direction.append("stable")
        df.loc[faction_indices, "trend_direction"] = trend_direction
        
        # 3. Volatility: Rolling std over last 3 periods
        rollstd = faction_data["winrate"].rolling(window=3, min_periods=1).std()
        df.loc[faction_indices, "winrate_std_dev_3periods"] = rollstd.values
        
        # 4. Consistency score (inverse of volatility, scaled 0-1)
        # Max possible std for 0-100 range is ~28.9 (at values 0, 50, 100)
        consistency = 1 - (rollstd / 30).clip(0, 1)  # Clip at 30 for extreme volatility
        df.loc[faction_indices, "consistency_score"] = consistency.values
        
        # 5. Average historical winrate up to current period (cumulative mean, excluding current)
        avg_hist_wr = faction_data["winrate"].expanding(min_periods=1).mean().shift(1)
        df.loc[faction_indices, "avg_historical_winrate"] = avg_hist_wr.values
        
        # 6. Peak winrate achieved so far
        peak_wr = faction_data["winrate"].expanding(min_periods=1).max()
        df.loc[faction_indices, "peak_winrate"] = peak_wr.values
        
        # 7. Appearance count (how many seasons has this faction existed?)
        appearance_count = faction_data["season"].rank(method="first")
        df.loc[faction_indices, "appearance_months"] = appearance_count.values
        
        # 8. Popularity momentum (change in player population)
        pop_momentum = faction_data["playerpopulation"].diff()
        df.loc[faction_indices, "popularity_momentum"] = pop_momentum.values
    
    # 9. Interaction feature: popularity_momentum * winrate_trend_strength
    df["popularity_trend_strength"] = (
        df["popularity_momentum"] * df["winrate_change_last_period"]
    )
    
    # Fill NaN winrate_change for first period with 0
    df["winrate_change_last_period"].fillna(0, inplace=True)
    
    # Fill NaN avg_historical_winrate with current winrate for first period (no history)
    first_period_mask = df["avg_historical_winrate"].isna()
    df.loc[first_period_mask, "avg_historical_winrate"] = df.loc[first_period_mask, "winrate"]
    
    # Fill NaN popularity_momentum with 0 for first period
    df["popularity_momentum"].fillna(0, inplace=True)
    df["popularity_trend_strength"].fillna(0, inplace=True)
    
    return df


def get_temporal_feature_columns():
    """
    Returns list of engineered temporal feature column names.
    Useful for model selection and preprocessing.
    """
    return [
        "winrate_change_last_period",
        "trend_direction",  # Categorical, needs encoding
        "winrate_std_dev_3periods",
        "consistency_score",
        "avg_historical_winrate",
        "peak_winrate",
        "appearance_months",
        "popularity_momentum",
        "popularity_trend_strength",
    ]


def get_numeric_temporal_features():
    """
    Returns list of numeric (non-categorical) temporal features.
    Use this for scaling and model input.
    """
    return [
        "winrate_change_last_period",
        "winrate_std_dev_3periods",
        "consistency_score",
        "avg_historical_winrate",
        "peak_winrate",
        "appearance_months",
        "popularity_momentum",
        "popularity_trend_strength",
    ]


def encode_trend_direction(df):
    """
    Encodes the categorical 'trend_direction' column to numeric.
    
    Args:
        df: DataFrame with 'trend_direction' column
    
    Returns:
        DataFrame with additional 'trend_direction_encoded' column
        (down=-1, stable=0, up=1)
    """
    trend_mapping = {"down": -1, "stable": 0, "up": 1}
    df["trend_direction_encoded"] = df["trend_direction"].map(trend_mapping)
    return df


def get_features_for_modeling(historical_df, include_temporal=True):
    """
    Prepares a dataframe with original and (optionally) engineered features for modeling.
    
    Args:
        historical_df: Raw historical DataFrame
        include_temporal: If True, adds engineered temporal features
    
    Returns:
        DataFrame with features ready for preprocessing and modeling
    """
    df = historical_df.copy()
    
    if include_temporal:
        df = engineer_temporal_features(df)
        df = encode_trend_direction(df)
    
    return df
