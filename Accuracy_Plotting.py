import os
import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy data
df = pd.read_csv('Data/accuracy.csv', index_col=0, parse_dates=True).dropna()

# Prepare data for plotting, dropping NaN values
gap_lengths = df['gap_length']

# Filter the data for RMSE, ensuring NaNs are dropped
rmse_3_point = df['3 Point Prediction (RMSE)']
rmse_9_point = df['9 Point Prediction (RMSE)']
rmse_linear = df['Linear Interpolation (RMSE)']
rmse_timeless = df['Timeless Prediction (RMSE)']

# Filter the data for MAPE, ensuring NaNs are dropped
mape_3_point = df['3 Point Prediction (MAPE)']
mape_9_point = df['9 Point Prediction (MAPE)']
mape_linear = df['Linear Interpolation (MAPE)']
mape_timeless = df['Timeless Prediction (MAPE)']

# Drop NaN values and create a unified DataFrame for valid RMSE and MAPE
valid_indices = df.index[df[['3 Point Prediction (RMSE)', '9 Point Prediction (RMSE)', 
                              '3 Point Prediction (MAPE)', '9 Point Prediction (MAPE)']].notna().all(axis=1)]

# Create a new DataFrame with valid data
valid_df = df.loc[valid_indices]

# Group by gap length and calculate mean for RMSE and MAPE
averaged_data = valid_df.groupby('gap_length').agg(
    RMSE_3=('3 Point Prediction (RMSE)', 'mean'),
    RMSE_9=('9 Point Prediction (RMSE)', 'mean'),
    RMSE_LINEAR=('Linear Interpolation (RMSE)', 'mean'),
    RMSE_TIMELESS=('Timeless Prediction (RMSE)', 'mean'),
    MAPE_3=('3 Point Prediction (MAPE)', 'mean'),
    MAPE_9=('9 Point Prediction (MAPE)', 'mean'),
    MAPE_LINEAR=('Linear Interpolation (MAPE)', 'mean'),
    MAPE_TIMELESS=('Timeless Prediction (MAPE)', 'mean')
).reset_index()

# Sort the averaged data by gap length
averaged_data = averaged_data.sort_values('gap_length')

# Create the first plot for RMSE
bar_width = 0.15
x = range(len(averaged_data))

plt.figure(figsize=(12, 6))

# Bar Chart for RMSE
plt.bar(x, averaged_data['RMSE_LINEAR'], width=bar_width, label='Linear Interpolation RMSE', color='magenta', alpha=0.6)
plt.bar([p + bar_width for p in x], averaged_data['RMSE_3'], width=bar_width, label='3 Input Model RMSE', color='cyan', alpha=0.6)
plt.bar([p + bar_width*2 for p in x], averaged_data['RMSE_9'], width=bar_width, label='9 Input Model RMSE', color='red', alpha=0.6)
plt.bar([p + bar_width*3 for p in x], averaged_data['RMSE_TIMELESS'], width=bar_width, label='Timeless Prediction RMSE', color='blue', alpha=0.6)

# Customize the RMSE plot
plt.xlabel('\nGap Length')
plt.ylabel('RMSE\n')
plt.xticks([p + bar_width * 1.5 for p in x], averaged_data['gap_length'])
plt.legend(prop={'size': 6})
plt.grid()

# Save the RMSE plot
plt.savefig('plots/rmse_comparison.png')
plt.savefig('plots/rmse_comparison.eps')
plt.show()

# Create the second plot for MAPE
plt.figure(figsize=(12, 6))

# Bar Chart for MAPE
plt.bar(x, averaged_data['MAPE_LINEAR'], width=bar_width, label='Linear Interpolation MAPE', color='purple', alpha=0.6)
plt.bar([p + bar_width for p in x], averaged_data['MAPE_3'], width=bar_width, label='3 Input Model MAPE', color='orange', alpha=0.6)
plt.bar([p + bar_width*2 for p in x], averaged_data['MAPE_9'], width=bar_width, label='9 Input Model MAPE', color='green', alpha=0.6)
plt.bar([p + bar_width*3 for p in x], averaged_data['MAPE_TIMELESS'], width=bar_width, label='Timeless Prediction MAPE', color='brown', alpha=0.6)

# Customize the MAPE plot
plt.xlabel('\nGap Length')
plt.ylabel('MAPE\n')
plt.xticks([p + bar_width * 1.5 for p in x], averaged_data['gap_length'])
plt.legend(prop={'size': 6})
plt.grid()

# Save the MAPE plot
plt.savefig('plots/mape_comparison.png')
plt.savefig('plots/mape_comparison.eps')
plt.show()
