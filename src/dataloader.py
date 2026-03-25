import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "meta_overview.csv"
HISTORICAL_DATA_PATH = Path(__file__).parent.parent / "data" / "historical_meta.csv"

def load_data():
    """
    Loads the current snapshot data from the CSV file and returns it as a pandas DataFrame.
    """
    column_names = ["faction", "winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation"]
    df = pd.read_csv(DATA_PATH, quoting=3, skiprows=1, names=column_names)
    df = _standardize_column_names(df)
    _validate_data(df)
    return df

def load_latest_snapshot():
    """
    Alias for load_data() - loads the current meta snapshot.
    """
    return load_data()

def load_historical_data():
    """
    Loads historical meta data spanning multiple seasons/time periods.
    Returns a DataFrame with columns: season, timestamp, faction, winrate, overrep, fourzerostart, eventwins, playerpopulation.
    """
    df = pd.read_csv(HISTORICAL_DATA_PATH)
    df.columns = df.columns.str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    numeric_columns = {"winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation", "season"}
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    _validate_historical_data(df)
    return df

def merge_historical_with_current(current_snapshot, season_number=None, timestamp=None):
    """
    Merges the current snapshot with historical data into a unified time-series DataFrame.
    
    Args:
        current_snapshot: DataFrame with current meta data (from load_data())
        season_number: Season number for the current snapshot (auto-detected if None)
        timestamp: Timestamp for current snapshot (defaults to today)
    
    Returns:
        Merged historical + current DataFrame with all time periods.
    """
    if timestamp is None:
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")
    
    if season_number is None:
        # Auto-detect season number as max from historical + 1
        hist = load_historical_data()
        season_number = int(hist["season"].max()) + 1
    
    current_copy = current_snapshot.copy()
    current_copy["season"] = season_number
    current_copy["timestamp"] = timestamp
    
    hist = load_historical_data()
    merged = pd.concat([hist, current_copy], ignore_index=True, sort=False)
    merged = merged.sort_values(["timestamp", "faction"]).reset_index(drop=True)
    
    return merged

def _standardize_column_names(df):
    """
    Standardizes the column names to lowercase and replaces spaces with underscores.
    """
    df = df.map(lambda x: x.strip().strip('"') if isinstance(x, str) else x)
    
    numeric_columns = {"winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation"}
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _validate_data(df):
    """
    Validates the data to ensure it meets the expected format and constraints.
    This is a placeholder function and should be implemented based on specific validation requirements.
    """
    excepted_columns = {"faction", "winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation"}

    if not excepted_columns.issubset(df.columns):
        missing_cols = set(excepted_columns) - set(df.columns)
        raise ValueError(f"Missing columns in the data: {missing_cols}")
    
    if df.isnull().values.any():
        raise ValueError("Data contains null values. Please clean the data before proceeding.")
    
    if len(df) < 10:
        raise ValueError("Data contains fewer than 10 rows. Please ensure there is enough data for analysis.")
    
    if (df["winrate"] < 0).any() or (df["winrate"] > 100).any():
        raise ValueError("Winrate values must be between 0 and 100.")

def _validate_historical_data(df):
    """
    Validates historical time-series data.
    """
    required_columns = {"season", "timestamp", "faction", "winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation"}
    
    if not required_columns.issubset(df.columns):
        missing_cols = set(required_columns) - set(df.columns)
        raise ValueError(f"Missing columns in historical data: {missing_cols}")
    
    if df.isnull().values.any():
        raise ValueError("Historical data contains null values.")
    
    if len(df) < 10:
        raise ValueError("Insufficient historical data (need >=10 rows).")
    
    if (df["winrate"] < 0).any() or (df["winrate"] > 100).any():
        raise ValueError("Historical winrate values must be between 0 and 100.")
    
    # Check temporal continuity (at least 2 distinct time periods)
    unique_timestamps = df["timestamp"].nunique()
    if unique_timestamps < 2:
        raise ValueError(f"Need at least 2 distinct time periods. Found: {unique_timestamps}")