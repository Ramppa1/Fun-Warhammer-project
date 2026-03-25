# Warhammer 40K Meta Intelligence System

Advanced temporal analysis of Warhammer 40K competitive faction balance using machine learning and network analysis.

## Overview

This project analyzes tournament meta data across multiple seasons to:
- **Predict faction dominance** (will faction be top-3 next season?)
- **Identify faction archetypes** (glass cannons, balanced, meta favorites, niche picks)
- **Forecast winrate trends** (how will faction perform over time?)
- **Analyze meta stability** (is the meta healthy and diverse?)
- **Visualize faction dominance networks** (who counters whom, hierarchies)

## Project Structure

```
data/
├── meta_overview.csv          # Current meta snapshot
└── historical_meta.csv         # Historical time-series (4+ seasons)

src/
├── dataloader.py              # Load snapshot & historical data
├── preprocessing.py           # Train/test split, feature scaling, chronological splitting
├── features.py                # Temporal feature engineering (momentum, volatility, trends)
├── models.py                  # Regression, classification, clustering, forecasting
├── evaluation.py              # Model metrics (R², MAE, classification scores)
├── network.py                 # Faction dominance graph analysis
└── analysis.ipynb             # Aggregated insights & balance reports

notebooks/
├── 1_preprocess.ipynb         # Data loading, exploration, validation
├── 2_modeling.ipynb           # Train/evaluate regression & classification
└── 3_network_analysis.ipynb   # Network graphs, community detection
```

## Features Implemented

### Phase 1: Data Foundation ✅
- Historical data CSV with 3 seasons (4-14 factions per season)
- Data loader for snapshot + historical time-series merging
- Temporal feature engineering:
  - `winrate_change_last_period`: Momentum indicator
  - `consistency_score`: Volatility-based stability metric
  - `avg_historical_winrate`: Cumulative moving average
  - `peak_winrate`: Historical best performance
  - Plus 4 more (popularity trends, appearance count, etc.)
- Chronological train/test split (prevents time-series data leakage)

### Phase 2: Intelligence Layer ✅
**Regression Models:**
- Linear Regression (baseline)
- Random Forest Regressor (base)
- **Random Forest with Hyperparameter Tuning** (production option)

**Classification Models** (predicts top-3 dominance):
- Logistic Regression (interpretable baseline)
- Random Forest Classifier (flexible)
- **Random Forest Classifier with Tuning** (production option)

**Clustering** (identifies faction archetypes):
- K-Means clustering on latest season data
- Automatic archetype naming (Dominant Meta, Hidden Gems, etc.)

**Time-Series Forecasting:**
- Simple Exponential Smoothing (baseline)
- **ARIMA support** (requires statsmodels; graceful fallback)

### Phase 3: Network Analysis (Ready to implement)
- Centrality metrics (betweenness, eigenvector, closeness)
- Temporal graph evolution
- Community detection (Louvain algorithm)

## Installation

```bash
# Clone repository
cd "Warhammer project/Fun-Warhammer-project"

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Loading Data

```python
from src.dataloader import load_data, load_historical_data, merge_historical_with_current

# Load current snapshot
current = load_data()

# Load historical time-series
historical = load_historical_data()

# Merge both
combined = merge_historical_with_current(current, season_number=4)
```

### Temporal Feature Engineering

```python
from src.features import engineer_temporal_features, encode_trend_direction

# Add temporal features
df_with_features = engineer_temporal_features(historical)
df_with_features = encode_trend_direction(df_with_features)
```

### Preprocessing (Chronological Split)

```python
from src.preprocessing import chronological_train_test_split, split_features_and_target

# Split by time (train on past, test on future)
train, test = chronological_train_test_split(df_with_features, test_fraction=0.3)

# Get features & target
X_train, y_train = split_features_and_target(train, include_temporal=True)
X_test, y_test = split_features_and_target(test, include_temporal=True)
```

### Model Training

**Regression:**
```python
from src.models import train_random_forest_tuned
model = train_random_forest_tuned(X_train, y_train)
predictions = model.predict(X_test)
```

**Classification (Dominance Prediction):**
```python
from src.models import create_dominance_target, train_random_forest_classifier_tuned

# Create target
df_classified = create_dominance_target(historical, top_n=3)
train2, test2 = chronological_train_test_split(df_classified)

X_train2, y_train2 = split_features_and_target(train2, include_temporal=True)
X_test2, y_test2 = split_features_and_target(test2, include_temporal=True)

model = train_random_forest_classifier_tuned(X_train2, y_train2)
proba = model.predict_proba(X_test2)[:, 1]  # Probability of dominance
```

**Clustering (Archetypes):**
```python
from src.models import discover_faction_archetypes, get_archetype_names

result = discover_faction_archetypes(historical, n_clusters=4, include_temporal=True)
archetypes = get_archetype_names(result["assignments"], historical)
print(result["assignments"])  # DataFrame with cluster assignments
```

**Time-Series Forecasting:**
```python
from src.models import prepare_faction_timeseries, train_arima_model, forecast_with_arima

aeldari_ts = prepare_faction_timeseries(historical, "Aeldari")
arima_model = train_arima_model(aeldari_ts["series"], order=(1, 1, 1))
forecast = forecast_with_arima(arima_model, periods=1)
print(f"Forecasted Aeldari winrate: {forecast[0]:.1f}%")
```

## Model Evaluation

```python
from src.evaluation import r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, roc_auc_score

# Regression
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

# Classification
acc = accuracy_score(y_test2, model.predict(X_test2))
auc = roc_auc_score(y_test2, proba)
```

## Data Format

### Current Snapshot (meta_overview.csv)
```
Faction,WinRate,OverRep,FourZeroStart,EventWins,PlayerPopulation
Aeldari,55,1.58,12,166,5
...
```

### Historical Data (historical_meta.csv)
```
Season,Timestamp,Faction,WinRate,OverRep,FourZeroStart,EventWins,PlayerPopulation
1,2025-09-01,Aeldari,48,1.12,8,132,3
2,2025-11-15,Aeldari,52,1.35,10,148,4
...
```

## Next Steps (Phase 3 & 4)

### Phase 3: Network Analysis
- [ ] Add centrality metrics to `src/network.py`
- [ ] Implement temporal graph evolution
- [ ] Community detection (faction cliques)
- [ ] Meta stability index (entropy metric)

### Phase 4: Production Readiness
- [ ] Create `src/config.py` for hyperparameters
- [ ] Add unit tests for dataloader, preprocessing
- [ ] Build Streamlit dashboard (`src/dashboard.py`)
- [ ] API endpoint for live predictions (`src/api.py`)
- [ ] Implement auto-retraining scheduler

## Key Design Decisions

1. **Chronological Train/Test Split:** Time-series data must be split temporally (train on past, test on future) to avoid data leakage from random splitting.

2. **Temporal Features:** Engineered features (momentum, volatility, history) capture meta trends that raw features miss.

3. **Optional Dependencies:** ARIMA forecasting is optional via graceful fallback to exponential smoothing if statsmodels unavailable.

4. **Latest Period Clustering:** Faction archetypes are identified from the most recent season only (most relevant).

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, networkx
- Optional: statsmodels (for ARIMA), xgboost (for advanced classification)

See `requirements.txt` for full version specifications.

## Contributing

Structure for adding features:
1. Add rawdata to `data/` (CSV format)
2. Add processing functions to `src/preprocessing.py` or `src/features.py`
3. Add models to `src/models.py`
4. Add evaluation metrics to `src/evaluation.py`
5. Test in notebooks (1_preprocess → 2_modeling → 3_network_analysis)

## References

- Win Rate: Faction victory percentage across tournaments
- Over-Representation: Meta prevalence (1.0 = expected, >1 = overplayed)
- Four-Zero Starts: Tournament consistency metric (perfect records at start)
- Event Wins: Total tournament victories by faction
- Player Population: Active player base (thousands)
