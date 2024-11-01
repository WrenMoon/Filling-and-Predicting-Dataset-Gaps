import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from datetime import timedelta
from matplotlib import rcParams

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load dates
dates = pd.read_csv('Data/Dates.csv')

# Load filled values from CSV
filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)

start_date = pd.to_datetime(dates.loc[0, 'start_date'])
end_date = pd.to_datetime(dates.loc[0, 'end_date'])
gap_start = pd.to_datetime(dates.loc[0, 'gap_start']) - timedelta(days=1)
gap_end = pd.to_datetime(dates.loc[0, 'gap_end']) + timedelta(days=1)

plot_start = max(gap_start - timedelta(days=50), start_date)
plot_end = min(gap_end + timedelta(days=50), end_date)

# Load original data for MAPE calculation
original_data = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
values = original_data['Est5'].loc[start_date:end_date]

# Define the gap mask
mask = (values.index >= gap_start) & (values.index <= gap_end)

# Initialize dictionaries to hold MAPE and RMSE results
mape_results = {}
rmse_results = {}

# Calculate MAPE and RMSE for each method regardless of filter
for method in filled_df.columns:
    # Ensure y_true and y_pred are aligned by reindexing
    y_true = values[mask].reindex(filled_df.index)
    y_pred = filled_df[method][mask].reindex(values.index)
    
    # Drop NaNs in both series to avoid mismatches
    y_true, y_pred = y_true.dropna(), y_pred.dropna()

    if len(y_true) > 0 and len(y_pred) > 0:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mape_results[method] = mape
        rmse_results[method] = rmse
    else:
        mape_results[method] = None
        rmse_results[method] = None

# Check if the neural network has the lowest MAPE only if filtering is enabled

# Create a DataFrame to store accuracy results
accuracy_df = pd.DataFrame(index=[gap_start])

# Add MAPE and RMSE for 3 Point Prediction and 9 Point Prediction
accuracy_df['3 Point Prediction (MAPE)'] = mape_results.get('3 Point Prediction')
accuracy_df['3 Point Prediction (RMSE)'] = rmse_results.get('3 Point Prediction')
accuracy_df['9 Point Prediction (MAPE)'] = mape_results.get('9 Point Prediction')
accuracy_df['9 Point Prediction (RMSE)'] = rmse_results.get('9 Point Prediction')
accuracy_df['Linear Interpolation (RMSE)'] = rmse_results.get('Linear Interpolation')
accuracy_df['Linear Interpolation (MAPE)'] = mape_results.get('Linear Interpolation')

# Calculate gap length
gap_length = (gap_end - gap_start).days

# Add gap_length to the DataFrame
accuracy_df['gap_length'] = gap_length

# Save the results to a CSV file, appending if the file already exists
csv_file_path = 'Data/accuracy.csv'
if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
    # Load the existing data
    existing_df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    # Append new data
    updated_df = pd.concat([existing_df, accuracy_df])
else:
    # Create a new file if it doesn't exist or is empty
    updated_df = accuracy_df

# Save the updated DataFrame to CSV
updated_df.to_csv(csv_file_path)

# Prepare to plot

plot_values = values.loc[plot_start:plot_end]
plot_filled_df = filled_df.loc[gap_start:gap_end]

plt.figure(figsize=(8, 3))
plt.plot(plot_values.index, plot_values, label='Original Data', color='blue', linewidth=2)

for method in plot_filled_df.columns:
    if method != '9 Point Prediction' and method != '3 Point Prediction' and method != 'Mean Imputation' and method != 'Cubic' and method != 'Polynomial':
        rmse_str = f'{rmse_results.get(method, "N/A"):.4f}' if rmse_results.get(method) is not None else "N/A"
        plt.plot(
            plot_filled_df.index,
            plot_filled_df[method],
            label=f'{method} (RMSE: {rmse_str})',
            linestyle='--'
        )
    elif method == '3 Point Prediction':
        rmse_str = f'{rmse_results.get(method, "N/A"):.4f}' if rmse_results.get(method) is not None else "N/A"
        plt.plot(
            plot_filled_df.index,
            plot_filled_df[method],
            label=f'3 Input Model (RMSE: {rmse_str})',
            linestyle='--'
        )

    elif method == '9 Point Prediction':
        rmse_str = f'{rmse_results.get(method, "N/A"):.4f}' if rmse_results.get(method) is not None else "N/A"
        plt.plot(
            plot_filled_df.index,
            plot_filled_df[method],
            label=f'9 Input Model (RMSE: {rmse_str})',
            color='black',
            linestyle='--',
            linewidth=4
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
