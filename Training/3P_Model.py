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

MODEL_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_model.keras')
SCALER_PATH = os.path.join(MODELS_DIR, 'gapfill_3point_scaler.pkl')

# 3‑point model uses three stations: Est2, Est5 (target), Est8
EST_COLUMNS_3 = ['Est2', 'Est5', 'Est8']

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
dates = pd.read_csv(DATES_PATH)

start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']

# Restrict to training window
train_df = data.loc[start_date:end_date].copy()

print(f"Date range: {start_date} to {end_date}")
print(f"Rows in train_df before dropna: {len(train_df)}")

# Add week-of-year feature
train_df['week'] = train_df.index.isocalendar().week.astype('int32')

# ------------------------------------------------------------------
# Build training dataset
# ------------------------------------------------------------------
# For 3-point model we simply drop rows with any missing values
train_df = train_df[EST_COLUMNS_3 + ['week']].dropna()

print(f"Rows in train_df after dropna: {len(train_df)}")

# Features: week + all Est columns except Est5 (target)
feature_cols = ['week'] + [col for col in EST_COLUMNS_3 if col != 'Est5']
X_train = train_df[feature_cols].astype(float32).values
y_train = train_df['Est5'].astype(float32).values

print(f"Training 3‑point model on {X_train.shape[0]} samples, {X_train.shape[1]} features.")

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

print(f"Saved 3‑point model to: {MODEL_PATH}")
print(f"Saved 3‑point scaler to: {SCALER_PATH}")