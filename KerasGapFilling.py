import pandas as pd
import numpy as np
from numpy import float32
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load data
FilteredCSV = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True).dropna()
dates = pd.read_csv('Data/Dates.csv')

# Extract start and end dates
start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']
gap_start = dates.loc[0, 'gap_start']
gap_end = dates.loc[0, 'gap_end']

# Define station columns
Est1 = 'Est1'
Est2 = 'Est2'
Est3 = 'Est3'

# Filter data between start_date and end_date
FilteredCSV = FilteredCSV.loc[start_date:end_date]

# Set values in Est2 to NaN for the gap period
FilteredCSV.loc[gap_start:gap_end, Est2] = np.nan

# Add a week column
FilteredCSV['week'] = FilteredCSV.index.isocalendar().week

# Define features and target
features = FilteredCSV[[Est1, Est3, 'week']].astype(float32)
target = FilteredCSV[Est2].astype(float32)

# Drop NaN values from both features and target in a synchronized manner
Filtered = features.join(target.rename('target')).dropna()
X_train = Filtered[[Est1, Est3, 'week']].values
y_train = Filtered['target'].values

# Scale training data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# Build and train the model
model = Sequential([
    Dense(300, activation='linear', input_shape=(3,)),
    Dense(700, activation='linear'),
    Dense(200, activation='linear'),
    Dense(50, activation='linear'),
    Dense(1, activation='linear')
])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Train the model only on non-null data
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Prepare the complete data for predictions (still within start_date and end_date)
X_complete = features.values  # Use features again for complete prediction
X_complete = scaler.transform(X_complete)
y_complete = model.predict(X_complete)

# Create a new column for completed Est2 predictions
FilteredCSV['Est2_Completed'] = y_complete.flatten()  # Store predictions directly

# Ensure no NaNs in y_true and y_pred for MAPE calculation
y_true = FilteredCSV[Est2].values  # True values (including NaNs)
y_pred = FilteredCSV['Est2_Completed'].values.flatten()  # Predictions

# Filter out NaNs from both y_true and y_pred
mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
y_true_filtered = y_true[mask]
y_pred_filtered = y_pred[mask]

# Calculate MAPE on filtered data
if len(y_true_filtered) > 0 and len(y_pred_filtered) > 0:
    mape = mean_absolute_percentage_error(y_true_filtered, y_pred_filtered)
else:
    mape = None

# Add predictions and MAPE to filled DataFrame
filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)
filled_df['Neural Network Prediction'] = FilteredCSV['Est2_Completed']

# Save updated filled values with predictions to a new CSV file
filled_df.to_csv('Data/Filled_Chlorophyll_Data.csv')

# Print MAPE result
if mape is not None:
    print(f"MAPE for the predictions: {mape * 100:.2f}%")
else:
    print("No valid data for MAPE calculation.")
