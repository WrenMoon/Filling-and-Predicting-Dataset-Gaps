import os
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error

# Load dates
dates = pd.read_csv('Data/Dates.csv')
start_date = pd.to_datetime(dates.loc[0, 'start_date'])
end_date = pd.to_datetime(dates.loc[0, 'end_date'])
gap_start = pd.to_datetime(dates.loc[0, 'gap_start']) - timedelta(days=1)
gap_end = pd.to_datetime(dates.loc[0, 'gap_end']) + timedelta(days=1)

# Load original data
original_data = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
values = original_data['Est5'].loc[start_date:end_date].copy()

# Create artificial gap
mask = (values.index >= gap_start) & (values.index <= gap_end)
gap_index = values.index[mask]
values_with_gap = values.copy()
values_with_gap[mask] = np.nan

# Load neural network predictions
filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)

# Initialize results dictionary
results = {
    'gap_start': gap_start,
    'gap_length': (gap_end - gap_start).days
}

# ------------------------------------------------------------------
# Apply traditional interpolation methods
# ------------------------------------------------------------------
methods = {}

# Linear Interpolation
methods['Linear Interpolation'] = values_with_gap.interpolate(method='linear')

# Cubic Spline Interpolation
methods['Cubic Spline'] = values_with_gap.interpolate(method='cubic')

# Polynomial Interpolation (degree 2)
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
except:
    methods['Polynomial'] = values_with_gap.copy()

# Cubic Interpolation
methods['Cubic'] = values_with_gap.interpolate(method='cubic')

# Mean Imputation
mean_value = values_with_gap.mean()
methods['Mean Imputation'] = values_with_gap.fillna(mean_value)

# Add neural network predictions
if '3 Point Prediction' in filled_df.columns:
    methods['3 Point Prediction'] = filled_df['3 Point Prediction']
if '9 Point Prediction' in filled_df.columns:
    methods['9 Point Prediction'] = filled_df['9 Point Prediction']

# ------------------------------------------------------------------
# Calculate metrics for all methods
# ------------------------------------------------------------------
invalid_run = False
methods_with_valid_data = 0

for method_name, method_values in methods.items():
    # Align true and predicted values on the gap index
    y_true = values.loc[gap_index]
    y_pred = method_values.loc[gap_index]
    
    # Combine and drop NaN/inf rows
    pair = pd.concat(
        [y_true.rename('true'), y_pred.rename('pred')],
        axis=1
    )
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(pair) == 0:
        # No valid data for this method, but don't fail the entire run
        results[f'{method_name} (MAPE)'] = None
        results[f'{method_name} (RMSE)'] = None
        continue
    
    y_t = pair['true'].to_numpy(dtype=float)
    y_p = pair['pred'].to_numpy(dtype=float)
    
    try:
        mape = mean_absolute_percentage_error(y_t, y_p)
        rmse = root_mean_squared_error(y_t, y_p)
        
        if not np.isfinite(mape) or not np.isfinite(rmse):
            # Non-finite metrics for this method, but don't fail the entire run
            results[f'{method_name} (MAPE)'] = None
            results[f'{method_name} (RMSE)'] = None
            continue
        
        results[f'{method_name} (MAPE)'] = mape
        results[f'{method_name} (RMSE)'] = rmse
        methods_with_valid_data += 1
        
    except Exception as e:
        # Error for this method, but don't fail the entire run
        results[f'{method_name} (MAPE)'] = None
        results[f'{method_name} (RMSE)'] = None
        continue

# ------------------------------------------------------------------
# Only skip if NO methods produced valid data
# ------------------------------------------------------------------
if methods_with_valid_data == 0:
    print("Skipping gap: no methods produced valid metrics.")
    exit(1)

# ------------------------------------------------------------------
# Save results to accuracy.csv (NO FILTERING)
# ------------------------------------------------------------------
accuracy_df = pd.DataFrame([results])
accuracy_df.set_index('gap_start', inplace=True)

csv_file_path = 'Data/accuracy.csv'
if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    existing_df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    updated_df = pd.concat([existing_df, accuracy_df])
else:
    updated_df = accuracy_df

updated_df.to_csv(csv_file_path)

# Determine which method was best (for informational purposes only)
valid_rmse = {k.replace(' (RMSE)', ''): v for k, v in results.items() 
              if k.endswith('(RMSE)') and v is not None}

if valid_rmse:
    best_method = min(valid_rmse, key=valid_rmse.get)
    best_rmse = valid_rmse[best_method]
    print(f"✓ Gap {gap_start.date()}–{gap_end.date()}: Best = {best_method} (RMSE: {best_rmse:.4f}), Valid methods: {methods_with_valid_data}/{len(methods)}")
else:
    print(f"✓ Gap {gap_start.date()}–{gap_end.date()}: Recorded (no valid RMSE values)")

exit(0)