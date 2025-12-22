import os
import pandas as pd
import numpy as np
import random
from datetime import timedelta
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MAX_RUNS = 10000
GAP_DURATIONS = [timedelta(days=i) for i in range(7, 22)]

DATES_PATH = 'Data/Dates.csv'
DATA_PATH = 'Data/Chlor_A data.csv'
FILLED_PATH = 'Data/Filled_Chlorophyll_Data.csv'
ACCURACY_PATH = 'Data/accuracy.csv'

# ------------------------------------------------------------------
# Load initial configuration
# ------------------------------------------------------------------
dates_df = pd.read_csv(DATES_PATH)
start_date = pd.to_datetime(dates_df.loc[0, 'start_date'])
end_date = pd.to_datetime(dates_df.loc[0, 'end_date'])

# Load data once (reused across all iterations)
original_data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
filled_df = pd.read_csv(FILLED_PATH, index_col=0, parse_dates=True)

# Identify station columns for spatial averaging
EST_COLUMNS = [c for c in original_data.columns if c.startswith('Est')]
NEIGHBOR_COLS = [c for c in EST_COLUMNS if c != 'Est5']

# Calculate maximum offset
max_gap_start_offset = (end_date - start_date).days - max(GAP_DURATIONS).days

# Clear accuracy.csv
with open(ACCURACY_PATH, 'w') as f:
    f.write('')

print(f"Starting fast accuracy calculation for {MAX_RUNS} gaps...")
print(f"Gap durations: {[g.days for g in GAP_DURATIONS]} days")
print("-" * 60)

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
successful_runs = 0
total_attempts = 0
accumulated_results = []

while successful_runs < MAX_RUNS:
    total_attempts += 1
    
    # Randomly choose a gap duration and start
    gap = random.choice(GAP_DURATIONS)
    gap_start_offset = random.randint(1, max_gap_start_offset - gap.days)
    gap_start = start_date + timedelta(days=gap_start_offset)
    gap_end = gap_start + gap
    
    # Adjust gap boundaries (matching your original logic)
    gap_start_adjusted = gap_start - timedelta(days=1)
    gap_end_adjusted = gap_end + timedelta(days=1)
    
    # ------------------------------------------------------------------
    # Extract values for this date range
    # ------------------------------------------------------------------
    try:
        values = original_data['Est5'].loc[start_date:end_date].copy()
    except Exception as e:
        print(f"[Attempt {total_attempts}] Failed to extract values: {e}")
        continue
    
    # Create artificial gap
    mask = (values.index >= gap_start_adjusted) & (values.index <= gap_end_adjusted)
    gap_index = values.index[mask]
    
    if len(gap_index) == 0:
        continue  # No data in gap, skip
    
    values_with_gap = values.copy()
    values_with_gap[mask] = np.nan
    
    # ------------------------------------------------------------------
    # Apply traditional + spatial interpolation methods
    # ------------------------------------------------------------------
    methods = {}
    
    # 1) Temporal Linear Interpolation (time series of Est5 only)
    methods['Linear Interpolation'] = values_with_gap.interpolate(method='linear')
    
    # 2) Polynomial Interpolation (degree 2)
    try:
        valid_indices = values_with_gap.dropna().index
        valid_values = values_with_gap.dropna().values
        if len(valid_values) >= 3:
            poly_interp = interp1d(
                valid_indices.astype(np.int64) // 10**9,
                valid_values,
                kind='quadratic',
                fill_value='extrapolate'
            )
            methods['Polynomial'] = pd.Series(
                poly_interp(values_with_gap.index.astype(np.int64) // 10**9),
                index=values_with_gap.index
            )
        else:
            methods['Polynomial'] = values_with_gap.copy()
    except Exception:
        methods['Polynomial'] = values_with_gap.copy()
    
    # 3) Cubic Interpolation
    try:
        methods['Cubic'] = values_with_gap.interpolate(method='cubic')
    except Exception:
        methods['Cubic'] = values_with_gap.copy()
    
    # 4) Mean Imputation (temporal mean of Est5)
    mean_value = values_with_gap.mean()
    methods['Mean Imputation'] = values_with_gap.fillna(mean_value)
    
    # 5) Spatial Linear Interpolation (spatial mean of surrounding stations)
    #
    # For each date in the gap, take the mean of all available neighboring
    # stations (Est1–Est9 except Est5) for that date, and use that as the
    # estimate of Est5. Outside the gap we keep the original Est5 values.
    try:
        # Neighbor values for all dates in the gap
        neighbor_vals = original_data.loc[gap_index, NEIGHBOR_COLS]
        spatial_mean = neighbor_vals.mean(axis=1, skipna=True)  # row-wise mean
        
        spatial_interp = values_with_gap.copy()
        spatial_interp.loc[gap_index] = spatial_mean
        
        methods['Spatial Linear Interpolation'] = spatial_interp
    except Exception:
        # If something goes wrong, fall back to just leaving NaNs in the gap
        methods['Spatial Linear Interpolation'] = values_with_gap.copy()
    
    # 6) Add neural network predictions
    if '3 Point Prediction' in filled_df.columns:
        methods['3 Point Prediction'] = filled_df['3 Point Prediction']
    if '9 Point Prediction' in filled_df.columns:
        methods['9 Point Prediction'] = filled_df['9 Point Prediction']
    
    # ------------------------------------------------------------------
    # Calculate metrics for all methods
    # ------------------------------------------------------------------
    results = {
        'gap_start': gap_start_adjusted,
        'gap_length': (gap_end_adjusted - gap_start_adjusted).days
    }
    
    methods_with_valid_data = 0
    
    for method_name, method_values in methods.items():
        # Align true and predicted values on the gap index
        try:
            y_true = values.loc[gap_index]
            y_pred = method_values.loc[gap_index]
        except Exception:
            results[f'{method_name} (MAPE)'] = None
            results[f'{method_name} (RMSE)'] = None
            continue
        
        # Combine and drop NaN/inf rows
        pair = pd.concat(
            [y_true.rename('true'), y_pred.rename('pred')],
            axis=1
        )
        pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(pair) == 0:
            results[f'{method_name} (MAPE)'] = None
            results[f'{method_name} (RMSE)'] = None
            continue
        
        y_t = pair['true'].to_numpy(dtype=float)
        y_p = pair['pred'].to_numpy(dtype=float)
        
        try:
            mape = mean_absolute_percentage_error(y_t, y_p)
            rmse = root_mean_squared_error(y_t, y_p)
            
            if not np.isfinite(mape) or not np.isfinite(rmse):
                results[f'{method_name} (MAPE)'] = None
                results[f'{method_name} (RMSE)'] = None
                continue
            
            results[f'{method_name} (MAPE)'] = mape
            results[f'{method_name} (RMSE)'] = rmse
            methods_with_valid_data += 1
            
        except Exception:
            results[f'{method_name} (MAPE)'] = None
            results[f'{method_name} (RMSE)'] = None
            continue
    
    # ------------------------------------------------------------------
    # Only count as successful if at least one method produced valid data
    # ------------------------------------------------------------------
    if methods_with_valid_data == 0:
        continue
    
    # Store results
    accumulated_results.append(results)
    successful_runs += 1
    
    # Determine best method for logging
    valid_rmse = {
        k.replace(' (RMSE)', ''): v
        for k, v in results.items()
        if k.endswith('(RMSE)') and v is not None
    }
    
    if valid_rmse:
        best_method = min(valid_rmse, key=valid_rmse.get)
        best_rmse = valid_rmse[best_method]
        print(
            f"[{successful_runs}/{MAX_RUNS}] ✓ Gap {gap_start_adjusted.date()}–{gap_end_adjusted.date()}: "
            f"Best = {best_method} (RMSE: {best_rmse:.4f}), "
            f"Valid methods: {methods_with_valid_data}/{len(methods)}"
        )
    else:
        print(
            f"[{successful_runs}/{MAX_RUNS}] ✓ Gap {gap_start_adjusted.date()}–{gap_end_adjusted.date()}: "
            f"Recorded (no valid RMSE values)"
        )
    
    # Save to CSV every 100 runs (reduces I/O overhead)
    if successful_runs % 100 == 0:
        accuracy_df = pd.DataFrame(accumulated_results)
        accuracy_df.set_index('gap_start', inplace=True)
        
        if os.path.exists(ACCURACY_PATH) and os.path.getsize(ACCURACY_PATH) > 0:
            existing_df = pd.read_csv(ACCURACY_PATH, index_col=0, parse_dates=True)
            updated_df = pd.concat([existing_df, accuracy_df])
        else:
            updated_df = accuracy_df
        
        updated_df.to_csv(ACCURACY_PATH)
        accumulated_results = []  # Clear buffer

# ------------------------------------------------------------------
# Save any remaining results
# ------------------------------------------------------------------
if accumulated_results:
    accuracy_df = pd.DataFrame(accumulated_results)
    accuracy_df.set_index('gap_start', inplace=True)
    
    if os.path.exists(ACCURACY_PATH) and os.path.getsize(ACCURACY_PATH) > 0:
        existing_df = pd.read_csv(ACCURACY_PATH, index_col=0, parse_dates=True)
        updated_df = pd.concat([existing_df, accuracy_df])
    else:
        updated_df = accuracy_df
    
    updated_df.to_csv(ACCURACY_PATH)

print("-" * 60)
print(f"Completed! {successful_runs} runs out of {total_attempts} attempts.")
print(f"Success rate: {100 * successful_runs / total_attempts:.1f}%")
print(f"\nResults saved to: {ACCURACY_PATH}")
print("\nTo generate summary plots, run: python Accuracy_Plotting.py")