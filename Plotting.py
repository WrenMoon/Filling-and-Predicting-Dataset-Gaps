from datetime import timedelta
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load filled values from CSV
filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)
dates = pd.read_csv('Data/Dates.csv')

start_date = pd.to_datetime(dates.loc[0, 'start_date'])
end_date = pd.to_datetime(dates.loc[0, 'end_date']) 
gap_start = (pd.to_datetime(dates.loc[0, 'gap_start']) - timedelta(days=1))
gap_end = (pd.to_datetime(dates.loc[0, 'gap_end']) + timedelta(days=1))

plot_start = max(gap_start - timedelta(days=50), start_date)
plot_end = min(gap_end + timedelta(days=50), end_date)

# Load original data for MAPE calculation
original_data = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
values = original_data['Est2'].loc[start_date:end_date]

# Define the gap mask
mask = (values.index >= gap_start) & (values.index <= gap_end)

# Initialize a dictionary to hold MAPE values
mape_results = {}

# Calculate MAPE for each filling method
for method in filled_df.columns:
    # Filter original and filled values using the mask
    y_true = values[mask]
    y_pred = filled_df[method][mask]
    
    # Ensure there are valid entries before calculating MAPE
    if len(y_true) > 0 and len(y_pred) > 0:
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mape_results[method] = mape
    else:
        mape_results[method] = None

# Check if neural network has the lowest MAPE
neural_net_mape = mape_results.get('Neural Network Prediction')
if neural_net_mape is not None and all(neural_net_mape <= m for m in mape_results.values() if m is not None):
    
    # Select the data for plotting only in the gap time
    plot_values = values.loc[plot_start:plot_end]  # Original data for full range
    plot_filled_df = filled_df.loc[gap_start:gap_end]  # Filled data only in the gap time

    # Plotting the original data
    plt.figure(figsize=(15, 10))
    plt.plot(plot_values.index, plot_values, label='Original Data', color='blue', linewidth=2)

    # Plotting the filled data only within the gap
    for method in plot_filled_df.columns:
        plt.plot(plot_filled_df.index, plot_filled_df[method], 
                 label=f'{method} (MAPE: {mape_results[method] * 100:.2f}%)' if mape_results[method] is not None else f'{method} (MAPE: N/A)', linestyle='--')

    # Add title and labels
    plt.title('Chlorophyll Data Gap Filling Methods (Focused on Gap Region)')
    plt.xlabel('Date')
    plt.ylabel('Chlorophyll Concentration')
    plt.legend(loc='best')
    plt.xticks(rotation='vertical')
    plt.ylim(0, 1)
    plt.grid()
    
    # Save the plot with formatted gap dates
    plot_filename = f'plots/Chlorophyll_Gap_Filling_{gap_start.strftime("%Y%m%d")}_{gap_end.strftime("%Y%m%d")}.png'
    plt.savefig(plot_filename)
    plt.close()  # Close the plot to free up memory
    
    print("Plot saved")  # Indicate a successful save in the output for the master script to track
else:
    print("Plot not saved as the neural network did not achieve the lowest MAPE.")
