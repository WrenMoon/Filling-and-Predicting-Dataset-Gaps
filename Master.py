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
gap_durations = [timedelta(days=x) for x in range(14, 22)]  # Gaps of 7 to 10 days

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Counter for plots generated
plots_generated = 0
max_plots = 1000

# Check if filter is true

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

    plots_generated += 1
   

print("Completed generating plots for 10 specified gaps.")
subprocess.run(['python', 'Accuracy_Plotting'])
