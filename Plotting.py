import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
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
MainEst = dates.loc[0, 'primary_data_point']

plot_start = max(gap_start - timedelta(days=50), start_date)
plot_end = min(gap_end + timedelta(days=50), end_date)

# Load original data for MAPE calculation
original_data = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
values = original_data[MainEst].loc[start_date:end_date]

# Define the gap mask
mask = (values.index >= gap_start) & (values.index <= gap_end)

# Initialize dictionaries to hold MAPE and RMSE results
mape_results = {}
rmse_results = {}
plot_saved = False

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
if apply_filter:
    nn_mape = mape_results.get('Neural Network Prediction', float('inf'))
    if nn_mape == min(mape for mape in mape_results.values() if mape is not None):
        plot_saved = True
else:
    # Allow plot saving without the neural network MAPE check
    plot_saved = True

# Create a DataFrame to store accuracy results
accuracy_df = pd.DataFrame(index=[gap_start])

# Add MAPE and RMSE for 3 Point Prediction and 9 Point Prediction
accuracy_df['3 Point Prediction (MAPE)'] = mape_results.get('3 Point Prediction')
accuracy_df['3 Point Prediction (RMSE)'] = rmse_results.get('3 Point Prediction')
accuracy_df['9 Point Prediction (MAPE)'] = mape_results.get('9 Point Prediction')
accuracy_df['9 Point Prediction (RMSE)'] = rmse_results.get('9 Point Prediction')

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
if plot_saved:
    plot_values = values.loc[plot_start:plot_end]
    plot_filled_df = filled_df.loc[gap_start:gap_end]

    plt.figure(figsize=(15, 10))
    plt.plot(plot_values.index, plot_values, label='Original Data', color='blue', linewidth=2)

    for method in plot_filled_df.columns:
        mape_str = f'{mape_results.get(method, "N/A") * 100:.2f}%' if mape_results.get(method) is not None else "N/A"
        rmse_str = f'{rmse_results.get(method, "N/A"):.4f}' if rmse_results.get(method) is not None else "N/A"
        plt.plot(
            plot_filled_df.index,
            plot_filled_df[method],
            label=f'{method} (MAPE: {mape_str}, RMSE: {rmse_str})',
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

# Additional code to handle plotting accuracy if the accuracy.csv is empty
if updated_df.empty:
    print("Accuracy DataFrame is empty. No accuracy plots can be generated.")
else:
    # Create scatter plots for MAPE and RMSE
    plt.figure(figsize=(14, 12))

    # Plot for MAPE
    plt.subplot(2, 1, 1)
    plt.scatter(updated_df['gap_length'], updated_df['3 Point Prediction (MAPE)'], color='blue', label='3 Point Prediction MAPE', alpha=0.6)
    plt.scatter(updated_df['gap_length'], updated_df['9 Point Prediction (MAPE)'], color='orange', label='9 Point Prediction MAPE', alpha=0.6)
    plt.plot(updated_df['gap_length'], updated_df['3 Point Prediction (MAPE)'], color='blue', alpha=0.3)
    plt.plot(updated_df['gap_length'], updated_df['9 Point Prediction (MAPE)'], color='orange', alpha=0.3)

    plt.title('MAPE vs Gap Length')
    plt.xlabel('Gap Length (days)')
    plt.ylabel('MAPE')
    plt.legend()
    plt.grid()

    # Plot for RMSE
    plt.subplot(2, 1, 2)
    plt.scatter(updated_df['gap_length'], updated_df['3 Point Prediction (RMSE)'], color='blue', label='3 Point Prediction RMSE', alpha=0.6)
    plt.scatter(updated_df['gap_length'], updated_df['9 Point Prediction (RMSE)'], color='orange', label='9 Point Prediction RMSE', alpha=0.6)
    plt.plot(updated_df['gap_length'], updated_df['3 Point Prediction (RMSE)'], color='blue', alpha=0.3)
    plt.plot(updated_df['gap_length'], updated_df['9 Point Prediction (RMSE)'], color='orange', alpha=0.3)

    plt.title('RMSE vs Gap Length')
    plt.xlabel('Gap Length (days)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()

    # Save the accuracy plots
    plt.tight_layout()
    plt.savefig('plots/Accuracy_Plots.png')
    plt.close()
