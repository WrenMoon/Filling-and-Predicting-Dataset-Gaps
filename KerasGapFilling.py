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
FilteredCSV = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
FullCSV = pd.read_csv('Data/Chlor_A data.csv', index_col=0, parse_dates=True)
dates = pd.read_csv('Data/Dates.csv')

# Extract start and end dates
start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']
gap_start = dates.loc[0, 'gap_start']
gap_end = dates.loc[0, 'gap_end']
MainEst = dates.loc[0, 'primary_data_point']

# Define station columns
# Est_columns = ['Est1', 'Est2', 'Est3', 'Est4', MainEst, 'Est6', 'Est7', 'Est8', 'Est9']
Est_columns = ['Est2', MainEst, 'Est8']

# Filter data between start_date and end_date
FilteredCSV = FilteredCSV.loc[start_date:end_date]

# Set values in Est5 to NaN for the gap period
FilteredCSV.loc[gap_start:gap_end, MainEst] = np.nan

# Add a week column
FilteredCSV['week'] = FilteredCSV.index.isocalendar().week
FullCSV['week'] = FullCSV.index.isocalendar().week

# Define a function to fill missing values based on nearby points
def fill_missing(data_row, target=MainEst):
    if pd.isna(data_row[target]):
        return data_row  # Target missing, discard during training
    
    # Count non-missing surrounding points
    surrounding_points = data_row.drop(target)
    available_points = surrounding_points.notna().sum()
    
    # If too few points, discard row
    if available_points < len(Est_columns)/2:
        return None

    # Fill missing points by averaging based on distance
    for col in surrounding_points.index:
        if pd.isna(data_row[col]):
            data_row[col] = surrounding_points.mean()  # Simple mean for simplicity
            
    return data_row

# Apply the fill function and drop rows with missing target or too many missing points

if (len(Est_columns) > 3):
    FilledTrainData = FilteredCSV.apply(fill_missing, axis=1).dropna()
else:
    FilledTrainData = FilteredCSV.dropna()
# Define features and target for training
features = FilledTrainData[['week'] + [col for col in Est_columns if col != MainEst]].astype(float32)
target = FilledTrainData[MainEst].astype(float32)
X_train = features.values
y_train = target.values

print((Est_columns))
# print(features)

# Scale training data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# Build and train the model
model = keras.Sequential([
    layers.Input(shape=(len(Est_columns),)),  # Dynamically set input shape
    layers.Dense(256, activation='relu', kernel_regularizer='l2'), 
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu', kernel_regularizer='l2'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=2)

# Prepare data for predictions with missing handling
def fill_for_prediction(row, target=MainEst):
    if not row.drop(target).isna().all():
        # Use mean of available points or one point if it's the only one available
        available_points = row.drop(target).dropna()
        if available_points.empty:
            return None
        row.fillna(available_points.mean(), inplace=True)
    return row

FullCSV[Est_columns] = FullCSV[Est_columns].apply(fill_for_prediction, axis=1)
X_complete = FullCSV[['week'] + [col for col in Est_columns if col != MainEst]].astype(float32)
X_complete = scaler.transform(X_complete)

# Predict all values including where Est5 was originally NaN
y_complete = model.predict(X_complete)

# Create a new column for completed Est5 predictions
FullCSV['PredictedData'] = y_complete.flatten()

# Optional: Adjust predictions to account for observed bias
observed_bias = 0.01
FullCSV['PredictedData'] -= observed_bias  # Adjust predictions

# Calculate MAPE only on valid predictions
y_true = FullCSV[MainEst].values  # True values (including NaNs)
y_pred = FullCSV['PredictedData'].values.flatten()  # Predictions
mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
y_true_filtered = y_true[mask]
y_pred_filtered = y_pred[mask]
if len(y_true_filtered) > 0 and len(y_pred_filtered) > 0:
    mape = mean_absolute_percentage_error(y_true_filtered, y_pred_filtered)
else:
    mape = None

# Save updated predictions to a new CSV file
filled_df = pd.read_csv('Data/Filled_Chlorophyll_Data.csv', index_col=0, parse_dates=True)
filled_df['3 Point Prediction'] = FullCSV['PredictedData']
filled_df.to_csv('Data/Filled_Chlorophyll_Data.csv')

# Print MAPE result
if mape is not None:
    print(f"MAPE for the predictions: {mape * 100:.2f}%")
else:
    print("No valid data for MAPE calculation.")
