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
layer_nodes = pd.read_csv('Data/Nodes.csv')
dates = pd.read_csv('Data/Dates.csv')

# Extract start and end dates
start_date = dates.loc[0, 'start_date']
end_date = dates.loc[0, 'end_date']
gap_start = dates.loc[0, 'gap_start']
gap_end = dates.loc[0, 'gap_end']

# Define station columns
Est_columns = ['Est1', 'Est2', 'Est3', 'Est4', 'Est5', 'Est6', 'Est7', 'Est8', 'Est9']
# Est_columns = ['Est2', 'Est5', 'Est8']

# Filter data between start_date and end_date
FilteredCSV = FilteredCSV.loc[start_date:end_date]

# Define a function to fill missing values based on nearby points
def fill_missing(data_row, target='Est5'):
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
if len(Est_columns) > 3:
    FilledTrainData = FilteredCSV.apply(fill_missing, axis=1).dropna()
else:
    FilledTrainData = FilteredCSV.dropna()

# Define features and target for training, excluding 'day' column
features = FilledTrainData[[col for col in Est_columns if col != 'Est5']].astype(float32)
target = FilledTrainData['Est5'].astype(float32)
X_train = features.values
y_train = target.values

# Scale training data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

model = keras.Sequential()
model.add(layers.Input(shape=(len(Est_columns)-1,)))  # Dynamically set input shape, excluding Est5

for nodes in layer_nodes.columns:
    model.add(layers.Dense(nodes, activation='relu', kernel_regularizer='l2'))
    model.add(layers.Dropout(0.2))

# Add the final output layer with a single node
model.add(layers.Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_percentage_error'])
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, verbose=2)

# Prepare data for predictions with missing handling
def fill_for_prediction(row, target='Est5'):
    if not row.drop(target).isna().all():
        # Use mean of available points or one point if it's the only one available
        available_points = row.drop(target).dropna()
        if available_points.empty:
            return None
        row.fillna(available_points.mean(), inplace=True)
    return row

FullCSV[Est_columns] = FullCSV[Est_columns].apply(fill_for_prediction, axis=1)
X_complete = FullCSV[[col for col in Est_columns if col != 'Est5']].astype(float32)
X_complete = scaler.transform(X_complete)

# Predict all values including where Est5 was originally NaN
y_complete = model.predict(X_complete)

# Create a new column for completed Est5 predictions
FullCSV['PredictedData'] = y_complete.flatten()

# Optional: Adjust predictions to account for observed bias
observed_bias = 0.01
FullCSV['PredictedData'] -= observed_bias  # Adjust predictions

# Calculate MAPE only on valid predictions
y_true = FullCSV['Est5'].values  # True values (including NaNs)
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
filled_df['Timeless Prediction'] = FullCSV['PredictedData']
filled_df.to_csv('Data/Filled_Chlorophyll_Data.csv')

# Print MAPE result
if mape is not None:
    print(f"MAPE for the predictions: {mape * 100:.2f}%")
else:
    print("No valid data for MAPE calculation.")







# It is better to use numbers of nodes in layers in deacreasing powers of 2
    
# 1. Memory Efficiency
# Binary Compatibility: Computers work with binary data, and powers of 2 align well with binary storage and memory allocation. This can make data transfer and allocation more efficient in memory, particularly on GPU architectures that are highly optimized for binary computation.
# Cache Optimization: Powers of 2 can align with memory cache sizes, helping the model access and store data more efficiently, especially in environments with limited memory.
    
# 2. Training Stability
# Balanced Gradient Flow: Gradually decreasing layer sizes can help with the stability of gradient flow, reducing the risk of issues like vanishing gradients in deep networks. By reducing the number of neurons, each layer becomes a form of dimensionality reduction, condensing learned features into more meaningful, abstract representations.
# Reducing Overfitting: Smaller layers can act as implicit regularization, particularly in later layers, helping the network to focus on key features and avoid overfitting on complex patterns that may not generalize well.
    
# 3. Heuristic and Practical Experience
# Empirical Success: Powers of 2 have shown to work well in practice due to compatibility with both hardware and algorithmic flow. Many libraries, including GPU-based frameworks like CUDA, have optimizations for power-of-2 sizes, making it a practical default choice.
# Simplicity and Reproducibility: Using powers of 2 helps maintain uniformity across different network architectures, making network design simpler and results more reproducible, especially when experimenting with variations on similar tasks.
    
# 4. Hyperparameter Tuning
# Intuitive Layer Sizing: Reducing each layer’s size by half is a simple approach that often maintains model capacity without overloading it. If every layer is too large, the network may have too much capacity and could overfit or become too slow. Reducing layer sizes gradually gives the network a controlled capacity reduction as it progresses toward the final output.
# Using powers of 2 isn’t required, and networks can work with arbitrary layer sizes, but this heuristic provides a good balance of performance, memory usage, and efficiency.