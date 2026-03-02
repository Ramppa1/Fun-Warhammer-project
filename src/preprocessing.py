import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["overrep", "fourzerostart", "eventwins", "playerpopulation"]
TARGET_COLUMN = "winrate"

def split_features_and_target(df):
    """
    Splits the DataFrame into features (X) and target (y).
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return X, y

def scale_features(X):
    """
    Scales the feature columns using StandardScaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler