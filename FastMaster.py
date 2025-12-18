import pandas as pd
import random
import subprocess
from datetime import timedelta

# Configuration
max_runs = 1000  # Number of runs to complete
gap_durations = [timedelta(days=i) for i in range(14, 22)]  # Gap lengths from 14 to 21 days

# Load dates
dates_df = pd.read_csv('Data/Dates.csv')
start_date = pd.to_datetime(dates_df.loc[0, 'start_date'])
end_date = pd.to_datetime(dates_df.loc[0, 'end_date'])

# Calculate the maximum offset for gap start
max_gap_start_offset = (end_date - start_date).days - max(gap_durations).days

# Clear or create accuracy.csv
with open('Data/accuracy.csv', 'w') as f:
    f.write('')

print(f"Starting fast accuracy calculation for {max_runs} gaps...")
print(f"Gap durations: {[g.days for g in gap_durations]} days")
print("-" * 60)

successful_runs = 0
total_attempts = 0

while successful_runs < max_runs:
    total_attempts += 1
    
    # Randomly choose a gap duration
    gap = random.choice(gap_durations)
    
    # Randomly select a gap start offset
    gap_start_offset = random.randint(1, max_gap_start_offset - gap.days)
    gap_start = start_date + timedelta(days=gap_start_offset)
    gap_end = gap_start + gap
    
    # Update dates.csv
    dates_df.loc[0, 'gap_start'] = gap_start.strftime('%Y-%m-%d')
    dates_df.loc[0, 'gap_end'] = gap_end.strftime('%Y-%m-%d')
    dates_df.to_csv('Data/Dates.csv', index=False)
    
    # Run FastMethodComparison
    result = subprocess.run(
        ['python', 'FastMethodComparison.py'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        successful_runs += 1
        print(f"[{successful_runs}/{max_runs}] {result.stdout.strip()}")
    else:
        print(f"[Attempt {total_attempts}] Failed.")
        if result.stdout.strip():
            print("  STDOUT:", result.stdout.strip())
        if result.stderr.strip():
            print("  STDERR:", result.stderr.strip())

print("-" * 60)
print(f"Completed! {successful_runs} runs out of {total_attempts} attempts.")
print(f"Success rate: {100 * successful_runs / total_attempts:.1f}%")
print("\nRunning Accuracy_Plotting.py to generate summary plots...")

subprocess.run(['python', 'Accuracy_Plotting.py'])

print("\nDone! Check 'Data/accuracy.csv' and 'plots/' folder for results.")