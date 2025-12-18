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
MODELS_DIR = 'models'

MODEL_3_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_model.keras')
SCALER_3_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_scaler.pkl')

MODEL_9_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_model.keras')
SCALER_9_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_scaler.pkl')

# Station sets
EST_COLUMNS_3 = ['Est2', 'Est5', 'Est8']
EST_COLUMNS_9 = ['Est1', 'Est2', 'Est3', 'Est4', 'Est5', 'Est6', 'Est7', 'Est8', 'Est9']

# ------------------------------------------------------------------
# Helper: prepare features for a given model
# ------------------------------------------------------------------
def prepare_features_for_model(df, est_columns):
    """
    df: DataFrame with all Est columns present and 'week' column.
    est_columns: list of station columns including 'Est5'.

    Returns:
        X (np.array), indices (Index) where features are valid.
    Strategy:
        - For each row, fill missing non-target stations with mean of
          available stations that day.
        - Drop rows where all non-target stations are NaN.
    """
    target = 'Est5'
    temp = df[est_columns + ['week']].copy()

    valid_indices = []
    feature_rows = []

    for idx, row in temp.iterrows():
        # Separate target & neighbors
        neighbors = row[est_columns].drop(labels=[target])
        available_neighbors = neighbors.dropna()

        # If no neighbors at all, skip this row
        if available_neighbors.empty:
            continue

        # Fill missing neighbors with mean of available neighbors
        mean_val = available_neighbors.mean()
        neighbors_filled = neighbors.fillna(mean_val)

        # Rebuild row with filled neighbors + week
        row_filled = pd.concat([
            pd.Series({col: neighbors_filled[col] for col in neighbors_filled.index}),
            pd.Series({'week': row['week']})
        ])

        # Features: week + all Est columns except Est5
        feature_cols = ['week'] + [col for col in est_columns if col != target]
        feature_row = row_filled[feature_cols]

        # If any feature is still NaN, skip row
        if feature_row.isna().any():
            continue

        valid_indices.append(idx)
        feature_rows.append(feature_row.values.astype(float32))

    if not feature_rows:
        return np.empty((0, len(est_columns)), dtype=float32), pd.Index([])

    X = np.vstack(feature_rows)
    indices = pd.Index(valid_indices)
    return X, indices

# ------------------------------------------------------------------
# Load data and models
# ------------------------------------------------------------------
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)

# Add week-of-year feature for all dates
data['week'] = data.index.isocalendar().week.astype('int32')

# Load 3‑point model & scaler
model_3 = keras.models.load_model(MODEL_3_PATH)
with open(SCALER_3_PATH, 'rb') as f:
    scaler_3 = pickle.load(f)

# Load 9‑point model & scaler
model_9 = keras.models.load_model(MODEL_9_PATH)
with open(SCALER_9_PATH, 'rb') as f:
    scaler_9 = pickle.load(f)

# ------------------------------------------------------------------
# Predictions with 3‑point model
# ------------------------------------------------------------------
X3, idx3 = prepare_features_for_model(data, EST_COLUMNS_3)
print(f"3‑point model: making predictions for {len(idx3)} dates.")

pred3_full = np.full(shape=(len(data),), fill_value=np.nan, dtype=float32)

if len(idx3) > 0:
    X3_scaled = scaler_3.transform(X3)
    y3_pred = model_3.predict(X3_scaled).flatten().astype(float32)
    # Map predictions back to full index
    idx_pos = data.index.get_indexer(idx3)
    pred3_full[idx_pos] = y3_pred

# ------------------------------------------------------------------
# Predictions with 9‑point model
# ------------------------------------------------------------------
X9, idx9 = prepare_features_for_model(data, EST_COLUMNS_9)
print(f"9‑point model: making predictions for {len(idx9)} dates.")

pred9_full = np.full(shape=(len(data),), fill_value=np.nan, dtype=float32)

if len(idx9) > 0:
    X9_scaled = scaler_9.transform(X9)
    y9_pred = model_9.predict(X9_scaled).flatten().astype(float32)
    idx_pos = data.index.get_indexer(idx9)
    pred9_full[idx_pos] = y9_pred

# ------------------------------------------------------------------
# Compute MAPE (optional, for information)
# ------------------------------------------------------------------
y_true = data['Est5'].values

mask3 = ~np.isnan(y_true) & ~np.isnan(pred3_full)
if mask3.sum() > 0:
    mape3 = mean_absolute_percentage_error(y_true[mask3], pred3_full[mask3])
    print(f"3‑point model MAPE: {mape3 * 100:.2f}%")
else:
    print("3‑point model: no valid data for MAPE calculation.")

mask9 = ~np.isnan(y_true) & ~np.isnan(pred9_full)
if mask9.sum() > 0:
    mape9 = mean_absolute_percentage_error(y_true[mask9], pred9_full[mask9])
    print(f"9‑point model MAPE: {mape9 * 100:.2f}%")
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

print(f"Saved predictions to: {FILLED_PATH}")