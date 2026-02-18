import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/meta_overview.csv")

def load_data():
    """
    Loads the data from the CSV file and returns it as a pandas DataFrame.
    """

    df = pd.read_csv(DATA_PATH)
    df = _standardize_column_names(df)
    _validate_data(df) #Placeholder for data validation function
    return df

def _standardize_column_names(df):
    """
    Standardizes the column names to lowercase and replaces spaces with underscores.
    """
    df.columns = (
        df.columns.str.strip().str.replace("", "_").str.replace("-", "").str.lower()
    )

    numeric_columns = ["winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df