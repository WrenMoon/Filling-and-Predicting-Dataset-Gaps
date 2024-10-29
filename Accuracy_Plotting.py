import os
import pandas as pd
import matplotlib.pyplot as plt

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the accuracy data
accuracy_df = pd.read_csv('Data/accuracy.csv', index_col=0, parse_dates=True)

# Drop rows with NaN values in MAPE or RMSE
accuracy_df = accuracy_df.dropna(subset=['3 Point Prediction (MAPE)', '3 Point Prediction (RMSE)', 
                                          '9 Point Prediction (MAPE)', '9 Point Prediction (RMSE)'])

# Create scatter plots
plt.figure(figsize=(14, 12))

# Plot for MAPE
plt.subplot(2, 1, 1)
plt.scatter(accuracy_df['gap_length'], accuracy_df['3 Point Prediction (MAPE)'], color='blue', label='3 Point Prediction MAPE', alpha=0.6)
plt.scatter(accuracy_df['gap_length'], accuracy_df['9 Point Prediction (MAPE)'], color='orange', label='9 Point Prediction MAPE', alpha=0.6)
plt.plot(accuracy_df['gap_length'], accuracy_df['3 Point Prediction (MAPE)'], color='blue', alpha=0.3)
plt.plot(accuracy_df['gap_length'], accuracy_df['9 Point Prediction (MAPE)'], color='orange', alpha=0.3)

plt.title('MAPE vs Gap Length')
plt.xlabel('Gap Length (days)')
plt.ylabel('MAPE')
plt.legend()
plt.grid()

# Plot for RMSE
plt.subplot(2, 1, 2)
plt.scatter(accuracy_df['gap_length'], accuracy_df['3 Point Prediction (RMSE)'], color='blue', label='3 Point Prediction RMSE', alpha=0.6)
plt.scatter(accuracy_df['gap_length'], accuracy_df['9 Point Prediction (RMSE)'], color='orange', label='9 Point Prediction RMSE', alpha=0.6)
plt.plot(accuracy_df['gap_length'], accuracy_df['3 Point Prediction (RMSE)'], color='blue', alpha=0.3)
plt.plot(accuracy_df['gap_length'], accuracy_df['9 Point Prediction (RMSE)'], color='orange', alpha=0.3)

plt.title('RMSE vs Gap Length')
plt.xlabel('Gap Length (days)')
plt.ylabel('RMSE')
plt.legend()
plt.grid()

# Save the plots
plt.tight_layout()
plt.savefig('plots/Accuracy_Plots.png')
plt.close()
