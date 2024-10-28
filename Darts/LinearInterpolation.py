import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values
from sklearn.metrics import mean_absolute_percentage_error

# Load data from CSV and filter the 'Est2' column
data = pd.read_csv('Chlor_A data.csv', index_col=0, parse_dates=True)

# Filter data from 2013-08-14 to 2014-01-01
data = data.loc['2013-08-14':'2014-01-01']

# Keep the original data intact for comparison
original_values = data['Est2'].copy()

# Set values from 2013-11-01 to 2013-12-30 to NaN in a copy of the DataFrame
data.loc['2013-11-01':'2013-12-30', 'Est2'] = np.nan

# Extract values from the Est2 column
values = data['Est2'].values  

# Convert the values to a TimeSeries object
GapDataset = TimeSeries.from_values(values)

# Fill missing values using linear interpolation
FilledDataset = fill_missing_values(GapDataset)

# Calculate MAPE for the gap section
# Get the true values for the gap section (original values)
true_values = original_values.loc['2013-11-01':'2013-12-30'].values

# Get the interpolated values for the same gap section
predicted_values = FilledDataset.values()[(data.index >= '2013-11-01') & (data.index <= '2013-12-30')]

# Use a mask to filter valid indices in the gap section
mask = ~np.isnan(true_values)

# Filter both arrays using the mask
true_values_filtered = true_values[mask]
predicted_values_filtered = predicted_values[mask]  # Corresponding filtered predicted values

# Calculate MAPE if we have valid data
if len(true_values_filtered) > 0 and len(predicted_values_filtered) > 0:
    mape = mean_absolute_percentage_error(true_values_filtered, predicted_values_filtered)
else:
    mape = None

# Prepare for plotting
plt.figure(figsize=(12, 6))

# Plot original non-NaN data
plt.plot(original_values.index, original_values, label="Original Data", color='blue', linewidth=2)

# Plot interpolated data
plt.plot(data.index, FilledDataset.values(), label="Filled Data Set (Interpolated)", color='orange', linestyle='--')

# Add MAPE annotation
if mape is not None:
    plt.text(
        0.5, 0.9, f"MAPE: {mape * 100:.2f}%", 
        transform=plt.gca().transAxes, 
        fontsize=12, 
        color='red',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5')
    )
else:
    plt.text(
        0.5, 0.9, "No valid data for MAPE calculation", 
        transform=plt.gca().transAxes, 
        fontsize=12, 
        color='red',
        bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.5')
    )

plt.title("Chlorophyll Data with Linear Interpolation")
plt.xlabel("Date")
plt.ylabel("Chlorophyll Concentration")
plt.legend()
plt.grid()
plt.show()
