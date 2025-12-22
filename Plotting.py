import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, root_mean_squared_error
from datetime import timedelta
from scipy.interpolate import interp1d
from matplotlib import rcParams

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load dates
dates = pd.read_csv('Data/Dates.csv')

# Load filled values from CSV (neural network predictions)
filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)

start_date = pd.to_datetime(dates.loc[0, 'start_date'])
end_date = pd.to_datetime(dates.loc[0, 'end_date'])
gap_start = pd.to_datetime(dates.loc[0, 'gap_start']) - timedelta(days=1)
gap_end = pd.to_datetime(dates.loc[0, 'gap_end']) + timedelta(days=1)

plot_start = max(gap_start - timedelta(days=50), start_date)
plot_end = min(gap_end + timedelta(days=50), end_date)

# Load original data for MAPE calculation
original_data = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
values = original_data['Est5'].loc[start_date:end_date].copy()

# Create artificial gap
mask = (values.index >= gap_start) & (values.index <= gap_end)
gap_index = values.index[mask]
values_with_gap = values.copy()
values_with_gap[mask] = np.nan

# ------------------------------------------------------------------
# Apply traditional interpolation methods
# ------------------------------------------------------------------
methods = {}

# Linear Interpolation
methods['Linear Interpolation'] = values_with_gap.interpolate(method='linear')


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

# Add neural network predictions from filled_df
if '3 Point Prediction' in filled_df.columns:
    methods['3 Point Prediction'] = filled_df['3 Point Prediction']
if '9 Point Prediction' in filled_df.columns:
    methods['9 Point Prediction'] = filled_df['9 Point Prediction']

# ------------------------------------------------------------------
# Calculate metrics for all methods
# ------------------------------------------------------------------
mape_results = {}
rmse_results = {}
invalid_run = False

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
        print(f"Skipping gap: no valid data for method '{method_name}' in gap window.")
        invalid_run = True
        break
    
    y_t = pair['true'].to_numpy(dtype=float)
    y_p = pair['pred'].to_numpy(dtype=float)
    
    try:
        mape = mean_absolute_percentage_error(y_t, y_p)
        rmse = root_mean_squared_error(y_t, y_p)
        
        if not np.isfinite(mape) or not np.isfinite(rmse):
            print(f"Skipping gap: non-finite metric for method '{method_name}'.")
            invalid_run = True
            break
        
        mape_results[method_name] = mape
        rmse_results[method_name] = rmse
    except Exception as e:
        print(f"Error calculating metrics for '{method_name}': {e}")
        invalid_run = True
        break

# If anything was invalid (NaNs, no overlap, etc.), scrap this gap
if invalid_run or not mape_results or not rmse_results:
    raise SystemExit(1)

# ----------------------------------------------------------------------
# Filter: keep only gaps where a neural net is the best model
# ----------------------------------------------------------------------
candidate_methods = ['Linear Interpolation', '3 Point Prediction', '9 Point Prediction']

# Keep only candidates that actually have an RMSE computed
valid_candidates = {
    m: rmse_results[m]
    for m in candidate_methods
    if m in rmse_results and rmse_results[m] is not None
}

# We need Linear + at least one NN to compare
if 'Linear Interpolation' not in valid_candidates or len(valid_candidates) < 2:
    print("Skipping gap: not enough valid methods (need Linear + at least one NN).")
    raise SystemExit(2)

best_method = min(valid_candidates, key=valid_candidates.get)

if best_method not in ['3 Point Prediction', '9 Point Prediction']:
    print(f"Skipping gap: best method is '{best_method}', not a neural network.")
    raise SystemExit(3)

# ------------------------------------------------------------------
# Prepare to plot (only selected methods for visual clarity)
# ------------------------------------------------------------------
plot_values = values.loc[plot_start:plot_end]
plot_methods = {}

# Only plot selected methods for clarity
methods_to_plot = ['Linear Interpolation', 'Cubic', '3 Point Prediction', '9 Point Prediction']

for method_name in methods_to_plot:
    if method_name in methods:
        plot_methods[method_name] = methods[method_name].loc[gap_start:gap_end]

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

plt.figure(figsize=(8, 3))
plt.plot(plot_values.index, plot_values, label='Original Data', color='blue', linewidth=2)

for method_name, method_values in plot_methods.items():
    rmse_str = f'{rmse_results.get(method_name, "N/A"):.4f}' if rmse_results.get(method_name) is not None else "N/A"
    
    # Custom labels for neural networks
    if method_name == '3 Point Prediction':
        label = f'3 Input Model (RMSE: {rmse_str})'
    elif method_name == '9 Point Prediction':
        label = f'9 Input Model (RMSE: {rmse_str})'
        plt.plot(
            method_values.index,
            method_values,
            label=label,
            color='black',
            linestyle='--',
            linewidth=4
        )
        continue
    else:
        label = f'{method_name} (RMSE: {rmse_str})'
    
    plt.plot(
        method_values.index,
        method_values,
        label=label,
        linestyle='--'
    )

plt.ylabel('Chlorophyll Concentration (mg/m\u00b3)\n', fontsize=7)
plt.legend(loc='best', prop={'size': 6})
plt.xticks(rotation='horizontal', fontsize=6)
plt.ylim(0, 1.75)
plt.grid()

plot_filename = f'plots/Chlorophyll_Gap_Filling_{gap_start.strftime("%Y%m%d")}_{gap_end.strftime("%Y%m%d")}.png'
plt.savefig(plot_filename)
plot_filename = f'plots/Chlorophyll_Gap_Filling_{gap_start.strftime("%Y%m%d")}_{gap_end.strftime("%Y%m%d")}.eps'
plt.savefig(plot_filename)
plt.close()