import os
import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy data
csv_file_path = 'Data/accuracy.csv'
df = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)

# Prepare data for plotting, dropping NaN values
gap_lengths = df['gap_length']

# Filter the data for RMSE, ensuring NaNs are dropped
rmse_3_point = df['3 Point Prediction (RMSE)']
rmse_9_point = df['9 Point Prediction (RMSE)']

# Drop NaN values and create a unified DataFrame for valid RMSE
valid_indices = df.index[df[['3 Point Prediction (RMSE)', '9 Point Prediction (RMSE)']].notna().all(axis=1)]

# Create a new DataFrame with valid data
valid_df = df.loc[valid_indices]

# Group by gap length and calculate mean for RMSE
averaged_data = valid_df.groupby('gap_length').agg(
    RMSE_3=('3 Point Prediction (RMSE)', 'mean'),
    RMSE_9=('9 Point Prediction (RMSE)', 'mean')
).reset_index()

# Sort the averaged data by gap length
averaged_data = averaged_data.sort_values('gap_length')

# Create the bar chart
bar_width = 0.15
x = range(len(averaged_data))

# Create the figure
plt.figure(figsize=(20, 8))

# Bar Chart for RMSE
plt.bar(x, averaged_data['RMSE_3'], width=bar_width, label='3 Input Model RMSE', color='cyan', alpha=0.6)
plt.bar([p + bar_width for p in x], averaged_data['RMSE_9'], width=bar_width, label='9 Input Model RMSE', color='red', alpha=0.6)

# Customize the plot
plt.xlabel('Gap Length')
plt.ylabel('RMSE')
plt.xticks([p + bar_width / 2 for p in x], averaged_data['gap_length'])
plt.legend()
plt.grid()

# Save the plot
plt.savefig('plots/rmse_comparison.png')
plt.savefig('plots/rmse_comparison.ps')
plt.show()
