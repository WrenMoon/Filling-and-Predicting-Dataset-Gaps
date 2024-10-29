import pandas as pd
import numpy as np
from numpy import float32
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras import layers

# Load data
FilteredCSV = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True).dropna()
FullCSV = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
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
FullCSV['week'] = FullCSV.index.isocalendar().week

# Define features and target
features = FilteredCSV[[Est2, 'week', Est1, Est3]].astype(float32)
target = FilteredCSV[Est2].astype(float32)

# Drop NaN values from both features and target in a synchronized manner for training
Filtered = features.join(target.rename('target')).dropna()
X_train = Filtered[['week', Est1, Est3]].values
y_train = Filtered['target'].values

# Scale training data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# Build and train the model
model = keras.Sequential([
    layers.Input(shape=(3,)),  
    layers.Dense(128, activation='relu', kernel_regularizer='l2'),   # L2 regularization
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu', kernel_regularizer='l2'),
    layers.Dense(1)  # Output layer for regression
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])

# Train the model only on non-null data
model.fit(X_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, verbose=2)

# Prepare the complete data for predictions (including NaN entries)
X_complete = FullCSV[['week', Est1, Est3]].astype(float32).values  # Use features again for complete prediction
X_complete = scaler.transform(X_complete)

# Predict all values including where Est2 was originally NaN
y_complete = model.predict(X_complete)

# Create a new column for completed Est2 predictions
FullCSV['Est2_Completed'] = y_complete.flatten()

# Optional: Adjust predictions to account for observed bias
observed_bias = 0.015
FullCSV['Est2_Completed'] += observed_bias  # Adjust predictions

# Ensure no NaNs in y_true and y_pred for MAPE calculation
y_true = FullCSV[Est2].values  # True values (including NaNs)
y_pred = FullCSV['Est2_Completed'].values.flatten()  # Predictions

# Filter out NaNs from both y_true and y_pred for MAPE calculation
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
filled_df['Neural Network Prediction'] = FullCSV['Est2_Completed']

# Save updated filled values with predictions to a new CSV file
filled_df.to_csv('Data/Filled_Chlorophyll_Data.csv')

# Print MAPE result
if mape is not None:
    print(f"MAPE for the predictions: {mape * 100:.2f}%")
else:
    print("No valid data for MAPE calculation.")
