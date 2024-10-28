import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

# Load filled values from CSV
filled_df = pd.read_csv('Filled_Chlorophyll_Data_with_Predictions.csv', index_col=0, parse_dates=True)

# Load original data for MAPE calculation
original_data = pd.read_csv('Chlor_A data.csv', index_col=0, parse_dates=True)
values = original_data['Est2'].loc['2013-08-14':'2014-01-01']

# Define the gap mask
mask = (values.index >= '2013-11-01') & (values.index <= '2013-12-30')

# Initialize a dictionary to hold MAPE values
mape_results = {}

# Calculate MAPE for each filling method
for method in filled_df.columns:
    mape = mean_absolute_percentage_error(values[mask], filled_df[method][mask])
    mape_results[method] = mape

# Plotting all methods on the same graph
plt.figure(figsize=(15, 10))
plt.plot(values.index, values, label='Original Data', color='blue', linewidth=2)

for method in filled_df.columns:
    plt.plot(filled_df.index, filled_df[method], 
             label=f'{method} (MAPE: {mape_results[method] * 100:.2f}%)', linestyle='--')

# Add title and labels
plt.title('Chlorophyll Data Gap Filling Methods')
plt.xlabel('Date')
plt.ylabel('Chlorophyll Concentration')
plt.legend(loc='best')
plt.xticks(rotation='vertical')
plt.ylim(0, 1)
plt.grid()
plt.show()
