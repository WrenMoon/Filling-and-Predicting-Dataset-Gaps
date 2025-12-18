import os
import pickle

import numpy as np
import pandas as pd
from numpy import float32
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_PATH = 'Data/Chlor_A data.csv'
DATES_PATH = 'Data/Dates.csv'
MODELS_DIR = 'Training/models'
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_model.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_scaler.pkl')

# 9‑point model uses all nine stations (Est1–Est9) with Est5 as the main target
EST_COLUMNS_9 = ['Est1', 'Est2', 'Est3', 'Est4', 'Est5', 'Est6', 'Est7', 'Est8', 'Est9']

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
dates = pd.read_csv(DATES_PATH)

start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']

# Restrict to training window
train_df = data.loc[start_date:end_date].copy()

# Add week-of-year feature
train_df['week'] = train_df.index.isocalendar().week.astype('int32')

# ------------------------------------------------------------------
# Helper to fill missing inputs (except target) for training
# ------------------------------------------------------------------
def fill_missing(row, target='Est5', est_columns=None):
    """
    If target is NaN -> discard row.
    Otherwise, require at least half of the surrounding stations to be present.
    Fill missing station values with mean of available stations.
    """
    if est_columns is None:
        est_columns = EST_COLUMNS_9

    if pd.isna(row[target]):
        return None  # drop from training

    surrounding = row[est_columns].drop(labels=[target])
    available_count = surrounding.notna().sum()

    # Require at least half of non-target stations to be present
    if available_count < len(est_columns) / 2:
        return None

    # Replace missing values among surrounding stations by their mean
    mean_val = surrounding.dropna().mean()
    for col in surrounding.index:
        if pd.isna(row[col]):
            row[col] = mean_val

    return row

# Apply fill_missing row-wise only to relevant columns + week
subset_cols = EST_COLUMNS_9 + ['week']
filled_rows = train_df[subset_cols].apply(
    lambda r: fill_missing(r, target='Est5', est_columns=EST_COLUMNS_9),
    axis=1
)

# Drop rows where fill_missing returned None
filled_train_df = filled_rows.dropna()

# ------------------------------------------------------------------
# Build training dataset
# ------------------------------------------------------------------
feature_cols = ['week'] + [col for col in EST_COLUMNS_9 if col != 'Est5']
X_train = filled_train_df[feature_cols].astype(float32).values
y_train = filled_train_df['Est5'].astype(float32).values

print(f"Training 9‑point model on {X_train.shape[0]} samples, {X_train.shape[1]} features.")

# ------------------------------------------------------------------
# Scale features
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

# ------------------------------------------------------------------
# Define model
# ------------------------------------------------------------------
input_dim = X_train_scaled.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(256, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu', kernel_regularizer='l2'),
    layers.Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_percentage_error']
)

# ------------------------------------------------------------------
# Train model
# ------------------------------------------------------------------
model.fit(
    X_train_scaled,
    y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    verbose=2
)

# ------------------------------------------------------------------
# Save model and scaler
# ------------------------------------------------------------------
model.save(MODEL_PATH)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"Saved 9‑point model to: {MODEL_PATH}")
print(f"Saved 9‑point scaler to: {SCALER_PATH}")