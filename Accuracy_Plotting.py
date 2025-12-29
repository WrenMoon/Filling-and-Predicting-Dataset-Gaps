import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# CONFIGURATION: Choose which methods to plot
# ------------------------------------------------------------------
# Set to None to plot ALL methods found in the data
# Or provide a list of method names you want to include
METHODS_TO_PLOT = None  # Plot all methods

# Example: Only plot specific methods
# METHODS_TO_PLOT = [
#     '9 Point Prediction',
#     '3 Point Prediction',
#     'Linear Interpolation',
#     'Spatial Linear Interpolation'
# ]

# ------------------------------------------------------------------
# Load accuracy data
# ------------------------------------------------------------------
df = pd.read_csv('Data/accuracy.csv', index_col=0, parse_dates=True).dropna()

# Get all unique methods from the columns
all_columns = df.columns.tolist()
method_set = set()

for col in all_columns:
    if col.endswith('(MAPE)'):
        method_name = col.replace(' (MAPE)', '')
        method_set.add(method_name)
    elif col.endswith('(RMSE)'):
        method_name = col.replace(' (RMSE)', '')
        method_set.add(method_name)

# Remove 'gap_length' if it somehow got in there
method_set.discard('gap_length')

all_methods = sorted(method_set)

print(f"All methods found in data: {all_methods}")

# ------------------------------------------------------------------
# Define custom order for methods
# ------------------------------------------------------------------
desired_order = [
    '9 Point Prediction',
    '3 Point Prediction',
    'Linear Interpolation',
    'Spatial Linear Interpolation',
    'Mean Imputation',
    'Polynomial',
    'Cubic Spline',
    'Cubic'
]

# Filter based on METHODS_TO_PLOT configuration
if METHODS_TO_PLOT is not None:
    # Only include methods that are both in METHODS_TO_PLOT and exist in data
    methods_to_use = [m for m in METHODS_TO_PLOT if m in all_methods]
    
    # Warn about any requested methods that don't exist
    missing = [m for m in METHODS_TO_PLOT if m not in all_methods]
    if missing:
        print(f"\nWarning: These requested methods were not found in the data: {missing}")
    
    # Order them according to desired_order, then add any extras
    methods = [m for m in desired_order if m in methods_to_use]
    remaining = [m for m in methods_to_use if m not in desired_order]
    methods.extend(remaining)
else:
    # Plot all methods found in data
    methods = [m for m in desired_order if m in all_methods]
    remaining = [m for m in all_methods if m not in desired_order]
    methods.extend(remaining)

if not methods:
    print("\nError: No methods to plot!")
    exit(1)

print(f"\nMethods to plot (in order): {methods}")

# ------------------------------------------------------------------
# Group by gap length and calculate mean for both MAPE and RMSE
# ------------------------------------------------------------------
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

# Define colors for each method
color_map = {
    '9 Point Prediction': 'red',
    '3 Point Prediction': 'cyan',
    'Linear Interpolation': 'magenta',
    'Spatial Linear Interpolation': 'black',
    'Mean Imputation': 'green',
    'Polynomial': 'blue',
    'Cubic Spline': 'orange',
    'Cubic': 'orange',
}

# Fallback colors for any additional methods
fallback_colors = ['purple', 'brown', 'pink', 'gray', 'olive', 'navy', 'teal', 'maroon']
method_colors = {}
fallback_idx = 0

for method in methods:
    if method in color_map:
        method_colors[method] = color_map[method]
    else:
        method_colors[method] = fallback_colors[fallback_idx % len(fallback_colors)]
        fallback_idx += 1

bar_width = 0.8 / max(len(methods), 1)
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

os.makedirs('plots', exist_ok=True)
plt.savefig('plots/accuracy_comparison_all_methods.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/accuracy_comparison_all_methods.eps', bbox_inches='tight')

print("\nSaved plots to:")
print("  - plots/accuracy_comparison_all_methods.png")
print("  - plots/accuracy_comparison_all_methods.eps")

plt.show()