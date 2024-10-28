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
gap_start = pd.to_datetime(dates.loc[0, 'gap_start'])
gap_end = pd.to_datetime(dates.loc[0, 'gap_end'])

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

# Calculate the gap length in days
gap_length = (gap_end - gap_start).days

# Prepare data for logging into a CSV
log_data = []
for method, mape in mape_results.items():
    log_data.append({
        'gap_start': gap_start,
        'gap_end': gap_end,
        'gap_length': gap_length,
        'method': method,
        'mape': mape if mape is not None else np.nan  # Use NaN for missing values
    })

# Create a DataFrame from log_data
log_df = pd.DataFrame(log_data)

# Append log_df to the accuracy.csv file
accuracy_file = 'Data/accuracy.csv'
if os.path.exists(accuracy_file):
    log_df.to_csv(accuracy_file, mode='a', header=False, index=False)
else:
    log_df.to_csv(accuracy_file, mode='w', header=True, index=False)

# Plotting all methods on the same graph
plt.figure(figsize=(15, 10))
plt.plot(values.index, values, label='Original Data', color='blue', linewidth=2)

for method in filled_df.columns:
    plt.plot(filled_df.index, filled_df[method], 
             label=f'{method} (MAPE: {mape_results[method] * 100:.2f}%)' if mape_results[method] is not None else f'{method} (MAPE: N/A)', linestyle='--')

# Add title and labels
plt.title('Chlorophyll Data Gap Filling Methods')
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
