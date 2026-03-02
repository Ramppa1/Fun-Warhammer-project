import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/meta_overview.csv")

def load_data():
    """
    Loads the data from the CSV file and returns it as a pandas DataFrame.
    """
    column_names = ["faction", "winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation"]
    df = pd.read_csv(DATA_PATH, quoting=3, skiprows=1, names=column_names)
    _validate_data(df) #Placeholder for data validation function
    return df

def _standardize_column_names(df):
    """
    Standardizes the column names to lowercase and replaces spaces with underscores.
    """
    df.columns = (
        df.columns.str.strip()
        .str.replace('"', '')
        .str.replace(" ", "_")
        .str.replace("-", "")
        .str.lower()
    )

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