import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load filled values from CSV
filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)
dates = pd.read_csv('Data/Dates.csv')

start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']
gap_start = dates.loc[0, 'gap_start']
gap_end = dates.loc[0, 'gap_end']

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
plt.show()
