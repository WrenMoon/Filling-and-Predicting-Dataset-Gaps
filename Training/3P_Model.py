import os
import pickle
import random

import numpy as np
import pandas as pd
from numpy import float32
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
SEED = 42

DATA_PATH = 'Data/Chlor_A data.csv'
DATES_PATH = 'Data/Dates.csv'
MODELS_DIR = 'Training/models'
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_model.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_scaler.pkl')

# 3‑point model uses three stations: Est2, Est5 (target), Est8
EST_COLUMNS_3 = ['Est2', 'Est5', 'Est8']

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data = data.sort_index()
dates = pd.read_csv(DATES_PATH)

start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']

# Restrict to training window
train_df = data.loc[start_date:end_date].copy()

print(f"Date range: {start_date} to {end_date}")
print(f"Rows in train_df before preprocessing: {len(train_df)}")

# ------------------------------------------------------------------
# Temporal features
# ------------------------------------------------------------------
train_df['week'] = train_df.index.isocalendar().week.astype('int32')
train_df['dayofyear'] = train_df.index.dayofyear.astype('int32')
train_df['sin_doy'] = np.sin(2 * np.pi * train_df['dayofyear'] / 365.25)
train_df['cos_doy'] = np.cos(2 * np.pi * train_df['dayofyear'] / 365.25)

# Month as additional seasonal indicator
train_df['month'] = train_df.index.month.astype('int32')

# ------------------------------------------------------------------
# Build training dataset with robust missing handling
# ------------------------------------------------------------------
cols_needed = EST_COLUMNS_3 + ['week', 'month', 'sin_doy', 'cos_doy']
train_df = train_df[cols_needed]

# 1) Require target (Est5) to be present
stations = train_df[EST_COLUMNS_3].copy()
mask_target_present = stations['Est5'].notna()
stations = stations[mask_target_present]

# 2) Require at least one neighbor (Est2 or Est8) available
neighbors = stations.drop(columns=['Est5'])
available_neighbors = neighbors.notna().sum(axis=1)
stations = stations[available_neighbors >= 1]

# 3) Impute missing neighbor(s) with row-wise mean of available neighbor(s)
neighbors = stations.drop(columns=['Est5'])
row_means = neighbors.mean(axis=1, skipna=True)

for col in neighbors.columns:
    missing_mask = stations[col].isna()
    stations.loc[missing_mask, col] = row_means[missing_mask]

# 4) Join back time features
filled_train_df = stations.join(
    train_df[['week', 'month', 'sin_doy', 'cos_doy']], how='left'
)

print(f"Rows in filled_train_df after missing handling: {len(filled_train_df)}")

# ------------------------------------------------------------------
# Features and target
# ------------------------------------------------------------------
feature_cols = ['week', 'month', 'sin_doy', 'cos_doy'] + [col for col in EST_COLUMNS_3 if col != 'Est5']

X = filled_train_df[feature_cols].astype(float32).values
y = filled_train_df['Est5'].astype(float32).values

print(f"Training 3‑point model on {X.shape[0]} samples, {X.shape[1]} features.")

# ------------------------------------------------------------------
# Scale features
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# ------------------------------------------------------------------
# Time‑aware train/validation split (no shuffling)
# ------------------------------------------------------------------
n_samples = X_scaled.shape[0]
if n_samples < 20:
    raise ValueError("Not enough samples to train reliably. Check date range.")

split_idx = int(n_samples * 0.8)
X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

# ------------------------------------------------------------------
# Define optimized model (smaller for ~2400 samples)
# ------------------------------------------------------------------
input_dim = X_train.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(48, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.15),
    layers.Dense(24, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dropout(0.15),
    layers.Dense(12, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dense(1)
], name='3point_gapfill')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.Huber(delta=0.5),  # tuned for skewed chlorophyll data
    metrics=[
        keras.metrics.MeanAbsoluteError(name='mae'),
        keras.metrics.MeanAbsolutePercentageError(name='mape'),
        keras.metrics.RootMeanSquaredError(name='rmse'),
    ]
)

model.summary()

# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=40,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
]

# ------------------------------------------------------------------
# Train model
# ------------------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=32,
    validation_data=(X_val, y_val),
    shuffle=False,
    callbacks=callbacks,
    verbose=2
)

# ------------------------------------------------------------------
# Evaluate on validation set
# ------------------------------------------------------------------
y_val_pred = model.predict(X_val, verbose=0).flatten()

val_mae = mean_absolute_error(y_val, y_val_pred)
val_rmse = root_mean_squared_error(y_val, y_val_pred)
val_mape = mean_absolute_percentage_error(y_val, y_val_pred)

print("\n" + "="*60)
print("VALIDATION PERFORMANCE (3‑point model)")
print("="*60)
print(f"  MAE  = {val_mae:.4f}")
print(f"  RMSE = {val_rmse:.4f}")
print(f"  MAPE = {val_mape * 100:.2f}%")
print("="*60)

# ------------------------------------------------------------------
# Save model and scaler
# ------------------------------------------------------------------
model.save(MODEL_PATH)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nSaved 3‑point model to:   {MODEL_PATH}")
print(f"Saved 3‑point scaler to: {SCALER_PATH}")