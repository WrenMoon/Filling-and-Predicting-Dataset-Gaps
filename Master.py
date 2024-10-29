import pandas as pd
import subprocess
import os
from datetime import timedelta
import random

# Load the dates configuration
dates_df = pd.read_csv('Data/Dates.csv')

# Define start and end dates
start_date = pd.to_datetime(dates_df.loc[0, 'start_date'])
end_date = pd.to_datetime(dates_df.loc[0, 'end_date'])

# Define the gap lengths (7 to 10 days)
gap_durations = [timedelta(days=x) for x in range(15, 20)]  # Gaps of 7 to 10 days

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Counter for plots generated
plots_generated = 0
max_plots = 30

# Check if filter is true
apply_filter = dates_df.loc[0, 'filter'] == 'true'

# Calculate possible gap start offsets
max_gap_start_offset = (end_date - start_date).days

# Run the scripts for different gaps
while plots_generated < max_plots:
    # Randomly choose a gap duration
    gap = random.choice(gap_durations)

    # Randomly select a gap start offset ensuring it fits within the range
    gap_start_offset = random.randint(1, max_gap_start_offset - gap.days)
    gap_start = start_date + timedelta(days=gap_start_offset)
    gap_end = gap_start + gap

    # Update dates.csv
    dates_df.loc[0, 'gap_start'] = gap_start.strftime('%Y-%m-%d')
    dates_df.loc[0, 'gap_end'] = gap_end.strftime('%Y-%m-%d')
    dates_df.to_csv('Data/Dates.csv', index=False)

    # Run MethodComparison and Plotting scripts
    subprocess.run(['python', 'MethodComparision.py'])
    subprocess.run(['python', 'Plotting.py'])

    # Increment plot counter if filter is false
    if not apply_filter:
        plots_generated += 1
    else:
        # Load filled DataFrame to check MAPE after running Plotting.py
        filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)
        with open('Data/plot_mape_status.txt', 'r') as file:
            plot_successful = file.read().strip() == "success"
        if plot_successful:
            plots_generated += 1

print("Completed generating plots for 10 specified gaps.")
