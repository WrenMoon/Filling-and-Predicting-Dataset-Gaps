import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from datetime import timedelta

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load dates
dates = pd.read_csv('Data/Dates.csv')
apply_filter = dates.loc[0, 'filter'] == 'true'

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

# Initialize MAPE dictionary
mape_results = {}
plot_saved = False

# Calculate MAPE for each method regardless of filter
for method in filled_df.columns:
    # Ensure y_true and y_pred are aligned by reindexing
    y_true = values[mask].reindex(filled_df.index)
    y_pred = filled_df[method][mask].reindex(values.index)
    
    # Drop NaNs in both series to avoid mismatches
    y_true, y_pred = y_true.dropna(), y_pred.dropna()

    if len(y_true) > 0 and len(y_pred) > 0:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mape_results[method] = mape
    else:
        mape_results[method] = None

# Check if the neural network has the lowest MAPE only if filtering is enabled
if apply_filter:
    nn_mape = mape_results.get('Neural Network Prediction', float('inf'))
    if nn_mape == min(mape for mape in mape_results.values() if mape is not None):
        plot_saved = True
else:
    # Allow plot saving without the neural network MAPE check
    plot_saved = True

# Plot if filter is off or neural network MAPE is lowest
if plot_saved:
    plot_values = values.loc[plot_start:plot_end]
    plot_filled_df = filled_df.loc[gap_start:gap_end]

    plt.figure(figsize=(15, 10))
    plt.plot(plot_values.index, plot_values, label='Original Data', color='blue', linewidth=2)

    for method in plot_filled_df.columns:
        plt.plot(
            plot_filled_df.index,
            plot_filled_df[method],
            label=f'{method} (MAPE: {mape_results.get(method, "N/A") * 100:.2f}%)' if mape_results.get(method) is not None else f'{method} (MAPE: N/A)',
            linestyle='--'
        )

    plt.title('Chlorophyll Data Gap Filling Methods (Focused on Gap Region)')
    plt.xlabel('Date')
    plt.ylabel('Chlorophyll Concentration')
    plt.legend(loc='best')
    plt.xticks(rotation='vertical')
    plt.ylim(0, 1)
    plt.grid()

    plot_filename = f'plots/Chlorophyll_Gap_Filling_{gap_start.strftime("%Y%m%d")}_{gap_end.strftime("%Y%m%d")}.png'
    plt.savefig(plot_filename)
    plt.close()

    # Mark plot as successfully saved if filtering
    with open('Data/plot_mape_status.txt', 'w') as file:
        file.write("success" if plot_saved else "failure")
