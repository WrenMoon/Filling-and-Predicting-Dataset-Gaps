import pandas as pd
import numpy as np

# Load data from CSV and select the 'Est2' column
data = pd.read_csv('Chlor_A data.csv', index_col=0, parse_dates=True)

# Define the date range and create NaNs
start_date = '2013-08-14'
end_date = '2014-01-01'
gap_start = '2013-11-01'
gap_end = '2013-12-30'

# Select the relevant data
values = data['Est2'].loc[start_date:end_date]

# Create a mask for the gap
mask = (values.index >= gap_start) & (values.index <= gap_end)
gapped_values = values.copy()
gapped_values[mask] = np.nan

# Initialize a dictionary to hold filled values
filled_values = {}

# Linear Interpolation
filled_values['Linear Interpolation'] = gapped_values.interpolate()

# Mean Imputation
mean_value = values.mean()
filled_values['Mean Imputation'] = gapped_values.fillna(mean_value)

# Spline Interpolation
filled_values['Cubic Spline'] = gapped_values.interpolate(method='spline', order=3)  # Cubic Spline

# Polynomial Interpolation
filled_values['Polynomial'] = gapped_values.interpolate(method='polynomial', order=3)  # Polynomial of degree 3

# Cubic Interpolation
filled_values['Cubic'] = gapped_values.interpolate(method='cubic')

# Save filled values to a DataFrame
filled_df = pd.DataFrame(filled_values)

# Save the filled DataFrame to a CSV file
filled_df.to_csv('Filled_Chlorophyll_Data.csv')

print("Filled values calculated and saved to 'Filled_Chlorophyll_Data.csv'.")
