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

MODEL_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_model.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'gapfill_9point_scaler.pkl')

# 9‑point model uses all nine stations (Est1–Est9) with Est5 as the main target
EST_COLUMNS_9 = ['Est1', 'Est2', 'Est3', 'Est4', 'Est5', 'Est6', 'Est7', 'Est8', 'Est9']

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# ------------------------------------------------------------------
# Load data, restricted to training windown
# ------------------------------------------------------------------
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data = data.sort_index()
dates = pd.read_csv(DATES_PATH)

start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']

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
train_df['month'] = train_df.index.month.astype('int32')

# ------------------------------------------------------------------
# handling gaps in 9‑station inputs
# ------------------------------------------------------------------
stations = train_df[EST_COLUMNS_9].copy()

mask_target_present = stations['Est5'].notna()
stations = stations[mask_target_present]

neighbors = stations.drop(columns=['Est5'])
available_counts = neighbors.notna().sum(axis=1)

min_required = int(np.ceil((len(EST_COLUMNS_9) - 1) / 2.0))  # half of 8 neighbors
enough_neighbors = available_counts >= min_required
stations = stations[enough_neighbors]

neighbors = stations.drop(columns=['Est5'])
row_means = neighbors.mean(axis=1, skipna=True)

for col in neighbors.columns:
    missing_mask = stations[col].isna()
    stations.loc[missing_mask, col] = row_means[missing_mask]

# Join back time features
filled_train_df = stations.join(
    train_df[['week', 'month', 'sin_doy', 'cos_doy']], how='left'
)

print(f"Rows in filled_train_df after missing handling: {len(filled_train_df)}")

# Spatial mean calculations
neighbor_cols = [c for c in EST_COLUMNS_9 if c != 'Est5']
neighbors_df = filled_train_df[neighbor_cols]

filled_train_df['spatial_mean'] = neighbors_df.mean(axis=1, skipna=True)

for col in neighbor_cols:
    filled_train_df[f'{col}_dev'] = neighbors_df[col] - filled_train_df['spatial_mean']

# ------------------------------------------------------------------
# Features and target
# ------------------------------------------------------------------
# Features: time + spatial_mean + deviations
feature_cols = (
    ['week', 'month', 'sin_doy', 'cos_doy', 'spatial_mean'] +
    [f'{col}_dev' for col in neighbor_cols]
)

X = filled_train_df[feature_cols].astype(float32).values

# Target: predict the CORRECTION to spatial mean (Suggestion 2.a)
y = (filled_train_df['Est5'] - filled_train_df['spatial_mean']).astype(float32).values

print(f"Training 9‑point model on {X.shape[0]} samples, {X.shape[1]} features.")
print(f"Target: correction to spatial mean (mean={y.mean():.4f}, std={y.std():.4f})")

# ------------------------------------------------------------------
# Model Architecture 
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

n_samples = X_scaled.shape[0]
if n_samples < 20:
    raise ValueError("Not enough samples to train reliably. Check date range.")

split_idx = int(n_samples * 0.8)
X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# keep spatial_mean for validation reconstruction
spatial_mean_val = filled_train_df['spatial_mean'].iloc[split_idx:].values
est5_true_val = filled_train_df['Est5'].iloc[split_idx:].values

print(f"Train samples: {X_train.shape[0]}, Val samples: {X_val.shape[0]}")

input_dim = X_train.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
    layers.Dense(1)
], name='9point_gapfill_v2')

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='mae',
    metrics=[
        keras.metrics.MeanAbsoluteError(name='mae'),
        keras.metrics.MeanAbsolutePercentageError(name='mape'),
        keras.metrics.RootMeanSquaredError(name='rmse'),
    ]
)

model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
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

# Evaluate on validation set
y_val_correction_pred = model.predict(X_val, verbose=0).flatten()

# Reconstruct Est5 = spatial_mean + correction
y_val_pred = spatial_mean_val + y_val_correction_pred

val_mae = mean_absolute_error(est5_true_val, y_val_pred)
val_rmse = root_mean_squared_error(est5_true_val, y_val_pred)
val_mape = mean_absolute_percentage_error(est5_true_val, y_val_pred)

# Also compute baseline (spatial mean alone)
baseline_mae = mean_absolute_error(est5_true_val, spatial_mean_val)
baseline_rmse = root_mean_squared_error(est5_true_val, spatial_mean_val)
baseline_mape = mean_absolute_percentage_error(est5_true_val, spatial_mean_val)

print("\n" + "="*60)
print("VALIDATION PERFORMANCE (9‑point model)")
print("="*60)
print("Baseline (Spatial Mean only):")
print(f"  MAE  = {baseline_mae:.4f}")
print(f"  RMSE = {baseline_rmse:.4f}")
print(f"  MAPE = {baseline_mape * 100:.2f}%")
print("-"*60)
print("Neural Model (Spatial Mean + Correction):")
print(f"  MAE  = {val_mae:.4f}")
print(f"  RMSE = {val_rmse:.4f}")
print(f"  MAPE = {val_mape * 100:.2f}%")
print("-"*60)
print(f"Improvement: RMSE {baseline_rmse - val_rmse:+.4f}, MAPE {(baseline_mape - val_mape)*100:+.2f}%")
print("="*60)

# ------------------------------------------------------------------
# Save model and scaler
# ------------------------------------------------------------------
model.save(MODEL_PATH)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\nSaved 9‑point model to:   {MODEL_PATH}")
print(f"Saved 9‑point scaler to: {SCALER_PATH}")