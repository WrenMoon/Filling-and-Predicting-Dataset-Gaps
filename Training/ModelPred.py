import os
import pickle

import numpy as np
import pandas as pd
from numpy import float32
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow import keras

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_PATH = 'Data/Chlor_A data.csv'
FILLED_PATH = 'Data/Filled_Chlorophyll_Data.csv'
MODELS_DIR = 'Training/models'

MODEL_3_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_model.keras')
SCALER_3_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_scaler.pkl')

MODEL_9_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_model.keras')
SCALER_9_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_scaler.pkl')

# Station sets (same as training)
EST_COLUMNS_3 = ['Est2', 'Est5', 'Est8']
EST_COLUMNS_9 = ['Est1', 'Est2', 'Est3', 'Est4', 'Est5', 'Est6', 'Est7', 'Est8', 'Est9']
TARGET_COL = 'Est5'

# ------------------------------------------------------------------
# Helper: prepare features for a given model (NEW VERSION)
# ------------------------------------------------------------------
def prepare_features_for_model(df, est_columns, min_required_neighbors):

    target = TARGET_COL
    time_features = ['week', 'month', 'sin_doy', 'cos_doy']

    required_cols = est_columns + time_features
    temp = df[required_cols].copy()

    valid_indices = []
    feature_rows = []
    spatial_means = []

    # Neighbor columns (no target)
    neighbor_cols = [c for c in est_columns if c != target]

    for idx, row in temp.iterrows():
        neighbors = row[neighbor_cols]
        available = neighbors.dropna()

        # Not enough neighbors -> skip
        if len(available) < min_required_neighbors:
            continue

        # Compute spatial mean from available neighbors
        spatial_mean = available.mean()

        # Fill missing neighbors with spatial mean
        neighbors_filled = neighbors.fillna(spatial_mean)

        # Compute deviations from spatial mean
        deviations = neighbors_filled - spatial_mean

        # Build feature vector: time + spatial_mean + deviations
        # Order must match training: ['week', 'month', 'sin_doy', 'cos_doy', 'spatial_mean'] + deviations
        feature_vals = list(row[time_features].values) + [spatial_mean] + list(deviations.values)

        if np.isnan(feature_vals).any():
            continue

        valid_indices.append(idx)
        feature_rows.append(np.array(feature_vals, dtype=float32))
        spatial_means.append(spatial_mean)

    if not feature_rows:
        # Return empty arrays with correct dimensions
        n_features = len(time_features) + 1 + len(neighbor_cols)  # +1 for spatial_mean
        return np.empty((0, n_features), dtype=float32), np.empty(0, dtype=float32), pd.Index([])

    X = np.vstack(feature_rows)
    spatial_means = np.array(spatial_means, dtype=float32)
    indices = pd.Index(valid_indices)
    return X, spatial_means, indices

# ------------------------------------------------------------------
# Load data and add time features (must match training scripts)
# ------------------------------------------------------------------
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data = data.sort_index()

# Time features
data['week'] = data.index.isocalendar().week.astype('int32')
data['month'] = data.index.month.astype('int32')
data['dayofyear'] = data.index.dayofyear.astype('int32')
data['sin_doy'] = np.sin(2 * np.pi * data['dayofyear'] / 365.25)
data['cos_doy'] = np.cos(2 * np.pi * data['dayofyear'] / 365.25)

# ------------------------------------------------------------------
# Load models and scalers
# ------------------------------------------------------------------
print("Loading models and scalers...")
model_3 = keras.models.load_model(MODEL_3_PATH)
with open(SCALER_3_PATH, 'rb') as f:
    scaler_3 = pickle.load(f)

model_9 = keras.models.load_model(MODEL_9_PATH)
with open(SCALER_9_PATH, 'rb') as f:
    scaler_9 = pickle.load(f)

# ------------------------------------------------------------------
# 3‑point model predictions
# ------------------------------------------------------------------
# Training logic: at least 1 neighbor (Est2 or Est8) present
X3, spatial_means_3, idx3 = prepare_features_for_model(
    data, EST_COLUMNS_3, min_required_neighbors=1
)
print(f"3‑point model: making predictions for {len(idx3)} dates.")

pred3_full = np.full(shape=(len(data),), fill_value=np.nan, dtype=float32)

if len(idx3) > 0:
    X3_scaled = scaler_3.transform(X3)
    # Model predicts CORRECTION to spatial mean
    corrections_3 = model_3.predict(X3_scaled, verbose=0).flatten().astype(float32)
    
    # Reconstruct: Est5 = spatial_mean + correction
    y3_pred = spatial_means_3 + corrections_3

    # Map predictions back to full index
    idx_pos = data.index.get_indexer(idx3)
    pred3_full[idx_pos] = y3_pred

# ------------------------------------------------------------------
# 9‑point model predictions
# ------------------------------------------------------------------
# Training logic: at least half of 8 neighbors (4) present
X9, spatial_means_9, idx9 = prepare_features_for_model(
    data,
    EST_COLUMNS_9,
    min_required_neighbors=4
)
print(f"9‑point model: making predictions for {len(idx9)} dates.")

pred9_full = np.full(shape=(len(data),), fill_value=np.nan, dtype=float32)

if len(idx9) > 0:
    X9_scaled = scaler_9.transform(X9)
    # Model predicts CORRECTION to spatial mean
    corrections_9 = model_9.predict(X9_scaled, verbose=0).flatten().astype(float32)
    
    # Reconstruct: Est5 = spatial_mean + correction
    y9_pred = spatial_means_9 + corrections_9

    idx_pos = data.index.get_indexer(idx9)
    pred9_full[idx_pos] = y9_pred

# ------------------------------------------------------------------
# Compute MAPE (for info; excludes NaNs and real gaps)
# ------------------------------------------------------------------
y_true = data[TARGET_COL].values

mask3 = ~np.isnan(y_true) & ~np.isnan(pred3_full)
if mask3.sum() > 0:
    mape3 = mean_absolute_percentage_error(y_true[mask3], pred3_full[mask3])
    print(f"3‑point model MAPE (on observed days): {mape3 * 100:.2f}%")
else:
    print("3‑point model: no valid data for MAPE calculation.")

mask9 = ~np.isnan(y_true) & ~np.isnan(pred9_full)
if mask9.sum() > 0:
    mape9 = mean_absolute_percentage_error(y_true[mask9], pred9_full[mask9])
    print(f"9‑point model MAPE (on observed days): {mape9 * 100:.2f}%")
else:
    print("9‑point model: no valid data for MAPE calculation.")

# ------------------------------------------------------------------
# Write predictions into Filled_Chlorophyll_Data.csv
# ------------------------------------------------------------------
pred3_series = pd.Series(pred3_full, index=data.index, name='3 Point Prediction')
pred9_series = pd.Series(pred9_full, index=data.index, name='9 Point Prediction')

if os.path.exists(FILLED_PATH) and os.path.getsize(FILLED_PATH) > 0:
    filled_df = pd.read_csv(FILLED_PATH, index_col=0, parse_dates=True)
    # Align to main data index
    filled_df = filled_df.reindex(data.index)
else:
    filled_df = pd.DataFrame(index=data.index)

filled_df['3 Point Prediction'] = pred3_series
filled_df['9 Point Prediction'] = pred9_series

filled_df.to_csv(FILLED_PATH)

print(f"\nSaved predictions to: {FILLED_PATH}")
print("="*60)