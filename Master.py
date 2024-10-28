import pandas as pd
import subprocess
import os
from datetime import timedelta

# Load the dates configuration
dates_df = pd.read_csv('Data/Dates.csv')

# Define start and end dates
start_date = pd.to_datetime(dates_df.loc[0, 'start_date'])
end_date = pd.to_datetime(dates_df.loc[0, 'end_date'])

# Generate varying gap dates
gap_duration = [timedelta(days=x) for x in range(1, 31)]  # Example: gaps of 1 to 30 days
num_gaps = len(gap_duration)

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Run the scripts for different gaps
for gap in gap_duration:
    for gap_start_offset in range(1, (end_date - start_date).days - gap.days):
        gap_start = start_date + timedelta(days=gap_start_offset)
        gap_end = gap_start + gap

        # Update dates.csv
        dates_df.loc[0, 'gap_start'] = gap_start.strftime('%Y-%m-%d')
        dates_df.loc[0, 'gap_end'] = gap_end.strftime('%Y-%m-%d')
        dates_df.to_csv('Data/Dates.csv', index=False)

        # Run each script
        subprocess.run(['python', 'MethodComparision.py'])
        subprocess.run(['python', 'KerasGapFilling.py'])
        subprocess.run(['python', 'Plotting.py'])

print("Completed generating plots for all specified gaps.")
