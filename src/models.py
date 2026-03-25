import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import randint, uniform
# ============================================================================
# REGRESSION MODELS (Winrate Prediction)
# ============================================================================

def train_linear_regression(X_train, y_train):
    """
    Trains a Linear Regression model on the provided training data.
    
    Args:
        X_train: Features array or DataFrame
        y_train: Target values
    
    Returns:
        Trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a Random Forest Regressor on the provided training data.
    
    Args:
        X_train: Features array or DataFrame
        y_train: Target values
        n_estimators: Number of trees in the forest
        random_state: Seed for reproducibility
    
    Returns:
        Trained RandomForestRegressor model
    """
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest_tuned(X_train, y_train, n_iter=20, random_state=42, cv=5):
    """
    Trains a Random Forest Regressor with hyperparameter tuning via RandomizedSearchCV.
    Faster than GridSearchCV; good for initial exploration.
    
    Args:
        X_train: Features array or DataFrame
        y_train: Target values
        n_iter: Number of parameter combinations to sample
        random_state: Seed for reproducibility
        cv: Number of cross-validation folds
    
    Returns:
        Best trained RandomForestRegressor model from search
    """
    param_dist = {
        "n_estimators": randint(50, 300),
        "max_depth": [None, 5, 10, 15, 20],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2"],
    }
    
    rf = RandomForestRegressor(random_state=random_state)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=n_iter, cv=cv, 
        scoring="r2", n_jobs=-1, random_state=random_state
    )
    search.fit(X_train, y_train)
    
    return search.best_estimator_


# ============================================================================
# CLASSIFICATION MODELS (Faction Dominance Prediction)
# ============================================================================

def create_dominance_target(df, top_n=3):
    """
    Creates a binary classification target: "Is faction in top-N dominant?"
    
    Args:
        df: DataFrame with 'faction', 'winrate', and 'timestamp' columns
        top_n: Number of top factions to consider as "dominant"
    
    Returns:
        DataFrame with additional 'is_dominant' column (1=top N, 0=not)
    """
    df = df.copy()
    df["is_dominant"] = 0
    
    # For each time period, label top-N factions as dominant
    for period in df["timestamp"].unique():
        period_mask = df["timestamp"] == period
        top_factions = df[period_mask].nlargest(top_n, "winrate")["faction"].values
        df.loc[period_mask & df["faction"].isin(top_factions), "is_dominant"] = 1
    
    return df


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Trains a Logistic Regression classifier for dominance prediction.
    Simple, interpretable baseline.
    
    Args:
        X_train: Features array or DataFrame
        y_train: Binary target (1=dominant, 0=not)
        random_state: Seed for reproducibility
    
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest_classifier(X_train, y_train, n_estimators=100, random_state=42):
    """
    Trains a Random Forest Classifier for dominance prediction.
    More flexible than logistic regression.
    
    Args:
        X_train: Features array or DataFrame
        y_train: Binary target (1=dominant, 0=not)
        n_estimators: Number of trees
        random_state: Seed for reproducibility
    
    Returns:
        Trained RandomForestClassifier model
    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest_classifier_tuned(X_train, y_train, n_iter=20, random_state=42, cv=5):
    """
    Trains a Random Forest Classifier with hyperparameter tuning.
    
    Args:
        X_train: Features array or DataFrame
        y_train: Binary target (1=dominant, 0=not)
        n_iter: Number of parameter combinations to sample
        random_state: Seed for reproducibility
        cv: Number of cross-validation folds
    
    Returns:
        Best trained RandomForestClassifier model from search
    """
    param_dist = {
        "n_estimators": randint(50, 300),
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": ["sqrt", "log2"],
    }
    
    rf = RandomForestClassifier(random_state=random_state)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=n_iter, cv=cv, 
        scoring="roc_auc", n_jobs=-1, random_state=random_state
    )
    search.fit(X_train, y_train)
    
    return search.best_estimator_


# ============================================================================
# CLUSTERING MODELS (Faction Archetypes)
# ============================================================================

def discover_faction_archetypes(df, n_clusters=4, include_temporal=False):
    """
    Clusters factions into archetypes using their meta characteristics.
    Works on the latest time period or aggregated statistics.
    
    Args:
        df: Historical DataFrame with faction data
        n_clusters: Number of archetypes to identify (e.g., 3-5)
        include_temporal: If True, uses temporal features; else uses base features
    
    Returns:
        Dict with keys:
        - 'model': Trained KMeans model
        - 'cluster_labels': Array of cluster assignments per faction (latest period)
        - 'cluster_centers': Cluster centers for interpretation
        - 'factions': List of faction names (latest period, sorted by cluster)
    """
    # Get latest time period data
    latest_period = df["timestamp"].max()
    latest_df = df[df["timestamp"] == latest_period].copy()
    
    # Select features for clustering
    base_features = ["winrate", "overrep", "fourzerostart", "eventwins", "playerpopulation"]
    temporal_features = [
        "winrate_change_last_period", "consistency_score", 
        "avg_historical_winrate", "peak_winrate"
    ]
    
    if include_temporal:
        feature_cols = base_features + [col for col in temporal_features if col in latest_df.columns]
    else:
        feature_cols = base_features
    
    X = latest_df[feature_cols].values
    
    # Scale features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Prepare output
    cluster_assignments = pd.DataFrame({
        "faction": latest_df["faction"].values,
        "cluster": labels,
        "winrate": latest_df["winrate"].values,
        "playerpopulation": latest_df["playerpopulation"].values,
    }).sort_values("cluster")
    
    return {
        "model": kmeans,
        "scaler": scaler,
        "cluster_labels": labels,
        "cluster_centers": kmeans.cluster_centers_,
        "assignments": cluster_assignments,
        "feature_cols": feature_cols,
    }


def get_archetype_names(cluster_df, base_df):
    """
    Interprets faction clusters and suggests archetype names based on characteristics.
    
    Args:
        cluster_df: Output from discover_faction_archetypes()['assignments']
        base_df: DataFrame with features for interpretation
    
    Returns:
        Dict mapping cluster_id -> archetype_name (suggested)
    """
    archetypes = {}
    
    for cluster_id in cluster_df["cluster"].unique():
        cluster_factions = cluster_df[cluster_df["cluster"] == cluster_id]
        avg_winrate = cluster_factions["winrate"].mean()
        avg_pop = cluster_factions["playerpopulation"].mean()
        
        # Heuristic naming based on winrate and popularity
        if avg_winrate > 52 and avg_pop > 4:
            name = "Dominant Meta"
        elif avg_winrate > 52 and avg_pop <= 4:
            name = "Hidden Gems"
        elif avg_winrate <= 52 and avg_pop > 4:
            name = "Balanced Favorites"
        else:
            name = "Niche Picks"
        
        archetypes[cluster_id] = name
    
    return archetypes


# ============================================================================
# TIME-SERIES FORECASTING
# ============================================================================

def prepare_faction_timeseries(df, faction):
    """
    Extracts time-series data for a single faction.
    
    Args:
        df: Historical DataFrame sorted by timestamp
        faction: Faction name
    
    Returns:
        Dict with:
        - 'timestamps': Array of timestamps
        - 'winrates': Array of winrates over time
        - 'series': Pandas Series (winrate indexed by timestamp)
    """
    faction_data = df[df["faction"] == faction].sort_values("timestamp")
    
    return {
        "timestamps": faction_data["timestamp"].values,
        "winrates": faction_data["winrate"].values,
        "series": pd.Series(
            faction_data["winrate"].values, 
            index=faction_data["timestamp"].values
        ),
        "faction": faction,
    }


def simple_exponential_smoothing_forecast(series, alpha=0.3, periods=1):
    """
    Simple exponential smoothing forecast (baseline time-series method).
    Good for meta data with gradual trends.
    
    Args:
        series: Pandas Series or array of historical values
        alpha: Smoothing factor (0-1; higher = more weight to recent data)
        periods: Number of periods ahead to forecast
    
    Returns:
        Array of forecasted values (length = periods)
    """
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = np.array(series)
    
    # Exponential smoothing
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])
    
    # Forecast next periods (constant from last smoothed value)
    forecasts = np.repeat(smoothed[-1], periods)
    
    return forecasts


try:
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    
    def train_arima_model(series, order=(1, 1, 1)):
        """
        Trains an ARIMA model for time-series forecasting.
        Works well with 4-20 observations per faction.
        
        Args:
            series: Pandas Series of historical values
            order: (p, d, q) tuple for ARIMA hyperparameters
        
        Returns:
            Trained ARIMA model
        """
        model = ARIMA(series, order=order)
        result = model.fit()
        return result
    
    def forecast_with_arima(arima_result, periods=1):
        """
        Forecasts future values using trained ARIMA model.
        
        Args:
            arima_result: Result from train_arima_model()
            periods: Number of periods to forecast
        
        Returns:
            Array of forecasted values
        """
        forecast = arima_result.get_forecast(steps=periods)
        return forecast.predicted_mean.values
    
    ARIMA_AVAILABLE = True
    
except ImportError:
    ARIMA_AVAILABLE = False
    def train_arima_model(*args, **kwargs):
        raise ImportError("statsmodels not installed. Use pip install statsmodels")
    def forecast_with_arima(*args, **kwargs):
        raise ImportError("statsmodels not installed. Use pip install statsmodels")