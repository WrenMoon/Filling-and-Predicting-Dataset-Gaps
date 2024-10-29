import os
import pandas as pd
import numpy as np

# Load data from CSV and select the MainEst column
data = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
dates = pd.read_csv('Data/Dates.csv')

# Define the date range and create NaNs
start_date = pd.to_datetime(dates.loc[0, 'start_date'])
end_date = pd.to_datetime(dates.loc[0, 'end_date'])
gap_start = pd.to_datetime(dates.loc[0, 'gap_start'])
gap_end = pd.to_datetime(dates.loc[0, 'gap_end'])

print(start_date, end_date, gap_start, gap_end)

# Select the relevant data
values = data['Est5'].loc[start_date:end_date]

# Create a mask for the gap
mask = (values.index >= gap_start) & (values.index <= gap_end)
gapped_values = values.copy()
gapped_values[mask] = np.nan

print(mask)

# Initialize a dictionary to hold filled values
filled_values = {}

# Linear Interpolation
filled_values['Linear Interpolation'] = gapped_values.interpolate()

# Mean Imputation
mean_value = values.mean()
# filled_values['Mean Imputation'] = gapped_values.fillna(mean_value)

# Spline Interpolation
filled_values['Cubic Spline'] = gapped_values.interpolate(method='spline', order=3)

# Polynomial Interpolation
filled_values['Polynomial'] = gapped_values.interpolate(method='polynomial', order=3)

# Cubic Interpolation
filled_values['Cubic'] = gapped_values.interpolate(method='cubic')

# Save filled values to a DataFrame
filled_df = pd.DataFrame(filled_values)

# Load or create the existing filled DataFrame from CSV
filled_csv_path = 'Data/Filled_Chlorophyll_Data.csv'
if os.path.exists(filled_csv_path) and os.path.getsize(filled_csv_path) > 0:
    # Load the CSV only if it is not empty
    existing_filled_df = pd.read_csv(filled_csv_path, index_col=0, parse_dates=True)
    
    # Align indices in case of any discrepancies
    filled_df = filled_df.reindex(existing_filled_df.index, fill_value=np.nan)
else:
    # Create an empty DataFrame with the same index as filled_df if file doesn't exist or is empty
    existing_filled_df = pd.DataFrame(index=filled_df.index)

# Update only the columns that need to be filled
for column in filled_df.columns:
    existing_filled_df[column] = filled_df[column]

# Save the updated DataFrame back to the CSV file
existing_filled_df.to_csv(filled_csv_path)

print("Filled values calculated and updated in 'Filled_Chlorophyll_Data.csv'.")
