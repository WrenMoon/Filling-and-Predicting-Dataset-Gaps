import pandas as pd
import numpy as np
import pickle
import os
import warnings

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# USER SETTINGS
# ============================================================================
# Time window (same as your main analysis script)
START_DATE = '2015-08-23'
END_DATE   = '2016-06-28'

# Gap settings (index within this window, not calendar date)
GAP_START_IDX = 90   # you can change this
GAP_LENGTH    = 30    # you can change this
GAP_END_IDX   = GAP_START_IDX + GAP_LENGTH  # exclusive

# File paths
DATA_PATH = 'Data/Chlor_A Data.csv'
MODEL_3P_PATH = 'Training/models/gapfill_3point_model.keras'
SCALER_3P_PATH = 'Training/models/gapfill_3point_scaler.pkl'
MODEL_9P_PATH = 'Training/models/gapfill_9point_model.keras'
SCALER_9P_PATH = 'Training/models/gapfill_9point_scaler.pkl'

OUTPUT_CSV = 'GapFill_Est5_Comparison.csv'

print("="*80)
print("Simple Gap-Fill Comparison Script")
print("Outputs a CSV with Original Est5 (no gap) + 3-Point, 9-Point, Linear-filled series")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1] Loading data...")
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
data = data.sort_index()

# Extract time window for Est5
target_data = data.loc[START_DATE:END_DATE, 'Est5'].copy()
original_series = target_data.dropna()
original_dates = original_series.index
original_values = original_series.values.copy()  # This is the TRUE original (no gap)

print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Total days (after dropping NaNs in Est5): {len(original_values)}")

# Basic checks for gap position
if GAP_END_IDX > len(original_values):
    raise ValueError(
        f"GAP_END_IDX={GAP_END_IDX} exceeds data length={len(original_values)}. "
        "Adjust GAP_START_IDX and GAP_LENGTH."
    )

print(f"Gap indices: {GAP_START_IDX} to {GAP_END_IDX-1} "
      f"({GAP_LENGTH} points, within the cleaned Est5 series)")

# ============================================================================
# 2. INTRODUCE ARTIFICIAL GAP
# ============================================================================
print("\n[2] Introducing artificial gap in Est5...")
gapped_values = original_values.copy()
gapped_values[GAP_START_IDX:GAP_END_IDX] = np.nan

# This will be our reference for the output index
full_index = original_dates

# ============================================================================
# 3. GAP-FILLING METHODS
# ============================================================================
filled_linear = None
filled_3p = None
filled_9p = None

print("\n[3] Applying gap-filling methods...")

# -------------------------------
# 3.1 Linear Interpolation
# -------------------------------
print("  - Linear Interpolation...")
series_with_gap = pd.Series(gapped_values, index=full_index)
filled_linear = series_with_gap.interpolate(method='linear').values
print("    ✓ Linear interpolation done")

# -------------------------------
# 3.2 3-Point ANN Model
# -------------------------------
print("  - 3-Point ANN Model...")
try:
    if not os.path.exists(MODEL_3P_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_3P_PATH}")
    if not os.path.exists(SCALER_3P_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_3P_PATH}")

    model_3p = keras.models.load_model(MODEL_3P_PATH)
    with open(SCALER_3P_PATH, 'rb') as f:
        scaler_3p: StandardScaler = pickle.load(f)

    # Need Est2, Est5, Est8 and time features in same window
    full_data_3p = data.loc[START_DATE:END_DATE, ['Est2', 'Est5', 'Est8']].copy()
    # Drop rows where Est5 was originally NaN so index aligns with original_series
    full_data_3p = full_data_3p.loc[original_dates]

    full_data_3p['week'] = full_data_3p.index.isocalendar().week
    full_data_3p['dayofyear'] = full_data_3p.index.dayofyear
    full_data_3p['sin_doy'] = np.sin(2 * np.pi * full_data_3p['dayofyear'] / 365.25)
    full_data_3p['cos_doy'] = np.cos(2 * np.pi * full_data_3p['dayofyear'] / 365.25)
    full_data_3p['month'] = full_data_3p.index.month

    filled_3p = gapped_values.copy()
    for i in range(GAP_START_IDX, GAP_END_IDX):
        row = full_data_3p.iloc[i]

        # Spatial mean from Est2 and Est8
        spatial_mean = np.nanmean([row['Est2'], row['Est8']])

        est2_dev = row['Est2'] - spatial_mean
        est8_dev = row['Est8'] - spatial_mean

        features = np.array([[
            row['week'],
            row['month'],
            row['sin_doy'],
            row['cos_doy'],
            spatial_mean,
            est2_dev,
            est8_dev
        ]], dtype=np.float32)

        features_scaled = scaler_3p.transform(features)
        correction = model_3p.predict(features_scaled, verbose=0)[0, 0]
        filled_3p[i] = spatial_mean + correction

    print("    ✓ 3-Point ANN filling done")

except Exception as e:
    print(f"    ✗ Error in 3-Point model: {e}")
    print("    → Falling back to linear interpolation for 3-point series.")
    filled_3p = filled_linear.copy()

# -------------------------------
# 3.3 9-Point ANN Model
# -------------------------------
print("  - 9-Point ANN Model...")
try:
    if not os.path.exists(MODEL_9P_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_9P_PATH}")
    if not os.path.exists(SCALER_9P_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_9P_PATH}")

    model_9p = keras.models.load_model(MODEL_9P_PATH)
    with open(SCALER_9P_PATH, 'rb') as f:
        scaler_9p: StandardScaler = pickle.load(f)

    all_stations = ['Est1', 'Est2', 'Est3', 'Est4',
                    'Est5', 'Est6', 'Est7', 'Est8', 'Est9']
    full_data_9p = data.loc[START_DATE:END_DATE, all_stations].copy()
    # Align to same index as original_series
    full_data_9p = full_data_9p.loc[original_dates]

    full_data_9p['week'] = full_data_9p.index.isocalendar().week
    full_data_9p['dayofyear'] = full_data_9p.index.dayofyear
    full_data_9p['sin_doy'] = np.sin(2 * np.pi * full_data_9p['dayofyear'] / 365.25)
    full_data_9p['cos_doy'] = np.cos(2 * np.pi * full_data_9p['dayofyear'] / 365.25)
    full_data_9p['month'] = full_data_9p.index.month

    neighbor_stations = [s for s in all_stations if s != 'Est5']
    filled_9p = gapped_values.copy()

    for i in range(GAP_START_IDX, GAP_END_IDX):
        row = full_data_9p.iloc[i]

        neighbor_values = [row[s] for s in neighbor_stations]
        spatial_mean = np.nanmean(neighbor_values)

        feature_list = [
            row['week'],
            row['month'],
            row['sin_doy'],
            row['cos_doy'],
            spatial_mean
        ]
        for station in neighbor_stations:
            dev = row[station] - spatial_mean
            feature_list.append(dev)

        features = np.array([feature_list], dtype=np.float32)
        features_scaled = scaler_9p.transform(features)
        correction = model_9p.predict(features_scaled, verbose=0)[0, 0]
        filled_9p[i] = spatial_mean + correction

    print("    ✓ 9-Point ANN filling done")

except Exception as e:
    print(f"    ✗ Error in 9-Point model: {e}")
    print("    → Falling back to linear interpolation for 9-point series.")
    filled_9p = filled_linear.copy()

# ============================================================================
# 4. BUILD OUTPUT DATAFRAME & SAVE
# ============================================================================
print("\n[4] Building output DataFrame and saving CSV...")

# Original_Est5 = the TRUE original data (no gap introduced)
out_df = pd.DataFrame({
    'Original_Est5': original_values,  # <-- NO GAP, the true original
    'Linear_Filled': filled_linear,
    'ThreePoint_Filled': filled_3p,
    'NinePoint_Filled': filled_9p
}, index=full_index)

out_df.to_csv(OUTPUT_CSV)

print(f"  ✓ Saved: {OUTPUT_CSV}")
print("\nPreview of the first few rows:")
print(out_df.head())

print("\nPreview of gap region (rows around the gap):")
preview_start = max(0, GAP_START_IDX - 2)
preview_end = min(len(out_df), GAP_END_IDX + 2)
print(out_df.iloc[preview_start:preview_end])

print("\n" + "="*80)
print("GAP-FILL COMPARISON CSV CREATED")
print("="*80)
print(f"\nThe CSV contains:")
print(f"  - Original_Est5: True original data (no gap)")
print(f"  - Linear_Filled: Gap filled with linear interpolation")
print(f"  - ThreePoint_Filled: Gap filled with 3-point ANN model")
print(f"  - NinePoint_Filled: Gap filled with 9-point ANN model")
print(f"\nGap location: indices {GAP_START_IDX} to {GAP_END_IDX-1}")