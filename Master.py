import pandas as pd
import subprocess
import os
from datetime import timedelta
import random

# Load the dates configuration
dates_df = pd.read_csv('Data/Dates.csv')
max_plots  = 250

# Define start and end dates
start_date = pd.to_datetime(dates_df.loc[0, 'start_date'])
end_date = pd.to_datetime(dates_df.loc[0, 'end_date'])

with open("progress.txt", "w") as file:
    pass

# Define the gap lengths (14 to 21 days)
gap_durations = [timedelta(days=x) for x in range(14, 22)]  # Gaps of 14 to 21 days

# Create 'plots' directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Load the configurations for the 9-point model from Nodes.csv
node_configs = pd.read_csv('Data/NodeConfigs.csv')

# Iterate over each configuration of layers from Nodes.csv
for idx, row in node_configs.iterrows():
    # Retrieve the list of nodes from the current row
    cur_config = [int(x) for x in row.dropna().tolist()]
    config_name = f"Config_{'_'.join(str(int(x)) if x == int(x) else str(x) for x in cur_config)}"
    
    with open("progress.txt", "a") as file:
        file.write("\n\nChecking configuration: " + config_name+ f"\n      Plots Generated: 0 / {max_plots}\n")

    pd.DataFrame([cur_config]).to_csv('Data/Nodes.csv', index=False, header=False)


    # Run KerasGapFilling once for the current configuration
    subprocess.run(['python', 'KerasGapFilling.py'])

    # Reset plot counter for this configuration
    plots_generated = 0

    try: 
        initial_entry_count = pd.read_csv('Data/accuracy.csv').shape[0]
    except pd.errors.EmptyDataError:
        initial_entry_count = 0

    # Run until we generate 50 valid entries in accuracy.csv for this configuration
    while plots_generated < max_plots:
        # Randomly choose a gap duration and start offset
        gap = random.choice(gap_durations)
        gap_start_offset = random.randint(1, (end_date - start_date).days - gap.days)
        gap_start = start_date + timedelta(days=gap_start_offset)
        gap_end = gap_start + gap

        # Update gap start and end in Dates.csv
        dates_df.loc[0, 'gap_start'] = gap_start.strftime('%Y-%m-%d')
        dates_df.loc[0, 'gap_end'] = gap_end.strftime('%Y-%m-%d')
        dates_df.to_csv('Data/Dates.csv', index=False)

        # Run MethodComparison and Plotting scripts
        # subprocess.run(['python', 'MethodComparision.py'])
        subprocess.run(['python', 'Plotting.py'])

        # Check for new entries in accuracy.csv
        accuracy_df = pd.read_csv('Data/accuracy.csv')
        current_entry_count = accuracy_df.shape[0]
        new_entries = current_entry_count - initial_entry_count

        plots_generated = new_entries

        # Read the current contents of the file
        with open('progress.txt', "r") as file:
            lines = file.readlines()

        # Modify the last line if the file is not empty
        if lines:  # Check if the file is not empty
            lines[-1] = f"      Plots Generated: {plots_generated} / {max_plots}\n"  # Replace the last line

        # Write the modified contents back to the file
        with open('progress.txt', "w") as file:
            file.writelines(lines)
            
        # If max_plots new entries for this config are reached, save results and exit loop
        # if new_entries >= max_plots:
        #     # Extract just the last max_plots entries for this configuration
        #     latest_entries_df = accuracy_df.tail(max_plots)

        #     # Filter for 9-point model results, including gap_length, MAPE, and RMSE
        #     neural_accuracy = latest_entries_df[['gap_length', '9 Point Prediction (MAPE)', '9 Point Prediction (RMSE)']]

        #     # Rename columns to include configuration details in the desired format
        #     neural_accuracy.columns = [f'{config_name} ({col.split(" (")[1]})' if ' (' in col else f'{config_name} ({col})' for col in ['gap_length', '9 Point Prediction (MAPE)', '9 Point Prediction (RMSE)']]

        #     try: 
        #         past_neural_accuracy = pd.read_csv('Data/NeuralAccuracy.csv')
        #     except pd.errors.EmptyDataError:
        #         past_neural_accuracy = pd.DataFrame

        #     if not past_neural_accuracy.empty:

        #         combined_df = pd.concat([past_neural_accuracy, neural_accuracy], axis=1)
        #         combined_df.to_csv('Data/NeuralAccuracy.csv', mode='w', header=True)
            
        #     else:

        #         neural_accuracy.to_csv('Data/NeuralAccuracy.csv', mode='w', header=True)


            # Save to NeuralAccuracy.csv
            # if os.path.exists('Data/NeuralAccuracy.csv'):
            #     neural_accuracy.to_csv('Data/NeuralAccuracy.csv', mode='a', header=True)
            # else:
            #     neural_accuracy.to_csv('Data/NeuralAccuracy.csv', mode='w', header=True)
            # break

