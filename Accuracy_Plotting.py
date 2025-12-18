import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load accuracy data
df = pd.read_csv('Data/accuracy.csv', index_col=0, parse_dates=True).dropna()

# Get all unique methods from the columns
# Columns are like: '3 Point Prediction (MAPE)', '3 Point Prediction (RMSE)', etc.
# Extract unique method names
all_columns = df.columns.tolist()
methods = set()

for col in all_columns:
    if col.endswith('(MAPE)'):
        method_name = col.replace(' (MAPE)', '')
        methods.add(method_name)
    elif col.endswith('(RMSE)'):
        method_name = col.replace(' (RMSE)', '')
        methods.add(method_name)

# Remove 'gap_length' if it somehow got in there
methods.discard('gap_length')
methods = sorted(list(methods))  # Sort for consistent ordering

print(f"Found methods: {methods}")

# ------------------------------------------------------------------
# Group by gap length and calculate mean for both MAPE and RMSE
# ------------------------------------------------------------------
# Build aggregation dictionary with correct syntax
agg_dict = {}
for method in methods:
    mape_col = f'{method} (MAPE)'
    rmse_col = f'{method} (RMSE)'
    
    if mape_col in df.columns:
        agg_dict[mape_col] = 'mean'
    if rmse_col in df.columns:
        agg_dict[rmse_col] = 'mean'

averaged_data = df.groupby('gap_length').agg(agg_dict).reset_index()

# Sort by gap length
averaged_data = averaged_data.sort_values('gap_length')

print(f"\nAveraged data shape: {averaged_data.shape}")
print(f"Gap lengths: {averaged_data['gap_length'].tolist()}")

# ------------------------------------------------------------------
# Create two subplots: one for RMSE, one for MAPE
# ------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Define colors for each method (cycle through if more methods than colors)
colors = ['magenta', 'cyan', 'red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
method_colors = {method: colors[i % len(colors)] for i, method in enumerate(methods)}

bar_width = 0.8 / len(methods)  # Adjust bar width based on number of methods
x = np.arange(len(averaged_data))

# ------------------------------------------------------------------
# Plot 1: RMSE comparison
# ------------------------------------------------------------------
for i, method in enumerate(methods):
    rmse_col = f'{method} (RMSE)'
    if rmse_col in averaged_data.columns:
        offset = (i - len(methods)/2 + 0.5) * bar_width
        ax1.bar(
            x + offset,
            averaged_data[rmse_col],
            width=bar_width,
            label=method,
            color=method_colors[method],
            alpha=0.7
        )

ax1.set_xlabel('Gap Length (days)', fontsize=10)
ax1.set_ylabel('RMSE', fontsize=10)
ax1.set_title('Root Mean Squared Error Comparison by Gap Length', fontsize=12, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(averaged_data['gap_length'])
ax1.legend(loc='best', fontsize=8)
ax1.grid(axis='y', alpha=0.3)

# ------------------------------------------------------------------
# Plot 2: MAPE comparison
# ------------------------------------------------------------------
for i, method in enumerate(methods):
    mape_col = f'{method} (MAPE)'
    if mape_col in averaged_data.columns:
        offset = (i - len(methods)/2 + 0.5) * bar_width
        ax2.bar(
            x + offset,
            averaged_data[mape_col],
            width=bar_width,
            label=method,
            color=method_colors[method],
            alpha=0.7
        )

ax2.set_xlabel('Gap Length (days)', fontsize=10)
ax2.set_ylabel('MAPE', fontsize=10)
ax2.set_title('Mean Absolute Percentage Error Comparison by Gap Length', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(averaged_data['gap_length'])
ax2.legend(loc='best', fontsize=8)
ax2.grid(axis='y', alpha=0.3)

# ------------------------------------------------------------------
# Adjust layout and save
# ------------------------------------------------------------------
plt.tight_layout()

# Save the plots
plt.savefig('plots/accuracy_comparison_all_methods.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/accuracy_comparison_all_methods.eps', bbox_inches='tight')

print("\nSaved plots to:")
print("  - plots/accuracy_comparison_all_methods.png")
print("  - plots/accuracy_comparison_all_methods.eps")

plt.show()