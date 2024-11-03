import os
import pandas as pd
import matplotlib.pyplot as plt

# Load accuracy data
df = pd.read_csv('Data/accuracy.csv', index_col=0, parse_dates=True).dropna()

# Filter to only include 9 Point Prediction RMSE and MAPE data
df = df[['gap_length', '9 Point Prediction (RMSE)', '9 Point Prediction (MAPE)']]

# Reset the index to get a sequential index for dividing into blocks
df = df.reset_index(drop=True)

# Divide data into blocks of 500 rows and label each block
df['block'] = (df.index // 250) + 1

# Group by each 500-row block and calculate overall average RMSE and MAPE for the block
averaged_data = df.groupby('block').agg(
    RMSE_9=('9 Point Prediction (RMSE)', 'mean'),
    MAPE_9=('9 Point Prediction (MAPE)', 'mean')
).reset_index()

# Prepare data for plotting
bar_width = 0.35
x = range(len(averaged_data))

# Plot for RMSE
plt.figure(figsize=(12, 6))
plt.bar(x, averaged_data['RMSE_9'], width=bar_width, label='9 Point Model RMSE', color='red', alpha=0.6)

# Customize the RMSE plot
plt.xlabel('\nBlock (500 rows each)')
plt.ylabel('Average 9 Point Prediction RMSE\n')
plt.xticks(x, averaged_data['block'], rotation=90)
plt.legend(prop={'size': 6})
plt.grid()

# Save the RMSE plot
plt.tight_layout()
plt.savefig('plots/rmse_9_point_comparison.png')
plt.savefig('plots/rmse_9_point_comparison.eps')
plt.show()

# Plot for MAPE
plt.figure(figsize=(12, 6))
plt.bar(x, averaged_data['MAPE_9'], width=bar_width, label='9 Point Model MAPE', color='green', alpha=0.6)

# Customize the MAPE plot
plt.xlabel('\nBlock (500 rows each)')
plt.ylabel('Average 9 Point Prediction MAPE\n')
plt.xticks(x, averaged_data['block'], rotation=90)
plt.legend(prop={'size': 6})
plt.grid()

# Save the MAPE plot
plt.tight_layout()
plt.savefig('plots/mape_9_point_comparison.png')
plt.savefig('plots/mape_9_point_comparison.eps')
plt.show()
