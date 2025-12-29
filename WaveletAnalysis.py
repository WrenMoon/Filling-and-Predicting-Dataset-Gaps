import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from scipy.fft import fft, fftfreq
import pywt
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================================================
# OUTPUT DIRECTORY
# ============================================================================
output_dir = 'Wavelet Analysis'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("FFT and Wavelet Analysis: Comparing Gap-Filling Methods")
print("Methods: Linear Interpolation, 3-Point ANN, 9-Point ANN")
print("="*80)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================
print("\n[1] Loading data...")
data = pd.read_csv('Data/Chlor_A Data.csv', index_col=0, parse_dates=True)
data = data.sort_index()

# Extract the specific date range for Est5
start_date = '2015-08-23'
end_date = '2016-06-28'
target_data = data.loc[start_date:end_date, 'Est5'].copy()

print(f"Date range: {start_date} to {end_date}")
print(f"Total days: {len(target_data)}")
print(f"Missing values in original data: {target_data.isna().sum()}")

# Remove any existing NaNs from the selected period
original_series = target_data.dropna()
original_dates = original_series.index
original_values = original_series.values

print(f"Days after removing NaNs: {len(original_values)}")
print(f"Mean value: {original_values.mean():.4f}")
print(f"Std value: {original_values.std():.4f}")

# ============================================================================
# 2. INTRODUCE ARTIFICIAL GAP
# ============================================================================
print("\n[2] Introducing artificial gap...")
gap_start_idx = 120
gap_length = 50
gap_end_idx = gap_start_idx + gap_length

# Create series with gap
gapped_values = original_values.copy()
true_gap_values = original_values[gap_start_idx:gap_end_idx].copy()  # Save for comparison
gapped_values[gap_start_idx:gap_end_idx] = np.nan

print(f"Gap introduced: indices {gap_start_idx} to {gap_end_idx-1} ({gap_length} days)")
print(f"Gap dates: {original_dates[gap_start_idx]} to {original_dates[gap_end_idx-1]}")

# ============================================================================
# 3. APPLY GAP-FILLING METHODS
# ============================================================================
print("\n[3] Applying gap-filling methods...")

filled_data = {}

# Method 1: Linear Interpolation
print("  - Linear Interpolation...")
series_with_gap = pd.Series(gapped_values, index=original_dates)
filled_linear = series_with_gap.interpolate(method='linear').values
filled_data['Linear Interpolation'] = filled_linear
print(f"    ✓ Linear Interpolation applied")
print(f"    Gap region mean: {filled_linear[gap_start_idx:gap_end_idx].mean():.6f}")

# Method 2: 3-Point ANN Model
print("  - 3-Point ANN Model...")
try:
    # Check for model files with correct paths
    model_3p_path = 'Training/models/gapfill_3point_model.keras'
    scaler_3p_path = 'Training/models/gapfill_3point_scaler.pkl'
    
    if not os.path.exists(model_3p_path):
        raise FileNotFoundError(f"Model not found: {model_3p_path}")
    
    # Load model and scaler
    model_3p = keras.models.load_model(model_3p_path)
    with open(scaler_3p_path, 'rb') as f:
        scaler_3p = pickle.load(f)
    
    # Prepare data for 3-point model
    full_data = data.loc[start_date:end_date, ['Est2', 'Est5', 'Est8']].copy()
    
    # Create features
    full_data['week'] = full_data.index.isocalendar().week
    full_data['dayofyear'] = full_data.index.dayofyear
    full_data['sin_doy'] = np.sin(2 * np.pi * full_data['dayofyear'] / 365.25)
    full_data['cos_doy'] = np.cos(2 * np.pi * full_data['dayofyear'] / 365.25)
    full_data['month'] = full_data.index.month
    
    # Create filled version with gap
    filled_3p = gapped_values.copy()
    
    # For gap region, compute spatial mean from neighbors and predict correction
    for i in range(gap_start_idx, gap_end_idx):
        row = full_data.iloc[i]
        
        # Spatial mean from neighbors (Est2 and Est8)
        spatial_mean = np.nanmean([row['Est2'], row['Est8']])
        
        # Deviations
        est2_dev = row['Est2'] - spatial_mean
        est8_dev = row['Est8'] - spatial_mean
        
        # Features for model
        features = np.array([[
            row['week'], row['month'], row['sin_doy'], row['cos_doy'],
            spatial_mean, est2_dev, est8_dev
        ]], dtype=np.float32)
        
        features_scaled = scaler_3p.transform(features)
        correction = model_3p.predict(features_scaled, verbose=0)[0, 0]
        
        # Predicted value = spatial_mean + correction
        filled_3p[i] = spatial_mean + correction
    
    filled_data['3 Point Prediction'] = filled_3p
    print(f"    ✓ 3-Point model applied successfully")
    print(f"    Gap region mean: {filled_3p[gap_start_idx:gap_end_idx].mean():.6f}")
    
except Exception as e:
    print(f"    ✗ Error with 3-Point model: {e}")
    print(f"    Using Linear Interpolation as fallback")
    filled_data['3 Point Prediction'] = filled_linear.copy()

# Method 3: 9-Point ANN Model
print("  - 9-Point ANN Model...")
try:
    # Check for model files with correct paths
    model_9p_path = 'Training/models/gapfill_9point_model.keras'
    scaler_9p_path = 'Training/models/gapfill_9point_scaler.pkl'
    
    if not os.path.exists(model_9p_path):
        raise FileNotFoundError(f"Model not found: {model_9p_path}")
    
    # Load model and scaler
    model_9p = keras.models.load_model(model_9p_path)
    with open(scaler_9p_path, 'rb') as f:
        scaler_9p = pickle.load(f)
    
    # Prepare data for 9-point model
    all_stations = ['Est1', 'Est2', 'Est3', 'Est4', 'Est5', 'Est6', 'Est7', 'Est8', 'Est9']
    full_data_9p = data.loc[start_date:end_date, all_stations].copy()
    
    # Create features
    full_data_9p['week'] = full_data_9p.index.isocalendar().week
    full_data_9p['dayofyear'] = full_data_9p.index.dayofyear
    full_data_9p['sin_doy'] = np.sin(2 * np.pi * full_data_9p['dayofyear'] / 365.25)
    full_data_9p['cos_doy'] = np.cos(2 * np.pi * full_data_9p['dayofyear'] / 365.25)
    full_data_9p['month'] = full_data_9p.index.month
    
    # Create filled version with gap
    filled_9p = gapped_values.copy()
    
    # For gap region, compute spatial mean from neighbors and predict correction
    neighbor_stations = [s for s in all_stations if s != 'Est5']
    
    for i in range(gap_start_idx, gap_end_idx):
        row = full_data_9p.iloc[i]
        
        # Spatial mean from all neighbors
        neighbor_values = [row[s] for s in neighbor_stations]
        spatial_mean = np.nanmean(neighbor_values)
        
        # Deviations for each neighbor
        feature_list = [
            row['week'], row['month'], row['sin_doy'], row['cos_doy'],
            spatial_mean
        ]
        
        for station in neighbor_stations:
            dev = row[station] - spatial_mean
            feature_list.append(dev)
        
        features = np.array([feature_list], dtype=np.float32)
        features_scaled = scaler_9p.transform(features)
        correction = model_9p.predict(features_scaled, verbose=0)[0, 0]
        
        # Predicted value = spatial_mean + correction
        filled_9p[i] = spatial_mean + correction
    
    filled_data['9 Point Prediction'] = filled_9p
    print(f"    ✓ 9-Point model applied successfully")
    print(f"    Gap region mean: {filled_9p[gap_start_idx:gap_end_idx].mean():.6f}")
    
except Exception as e:
    print(f"    ✗ Error with 9-Point model: {e}")
    print(f"    Using Linear Interpolation as fallback")
    filled_data['9 Point Prediction'] = filled_linear.copy()

# Verify all methods produced different results
print("\n  Verification - Gap region statistics:")
print(f"    Original mean: {true_gap_values.mean():.6f}")
for method_name, filled_values in filled_data.items():
    gap_vals = filled_values[gap_start_idx:gap_end_idx]
    print(f"    {method_name}: mean={gap_vals.mean():.6f}, std={gap_vals.std():.6f}")

# ============================================================================
# 4. FFT ANALYSIS
# ============================================================================
print("\n[4] Performing FFT analysis...")

def compute_fft(signal_data):
    """Compute FFT and return frequencies and magnitudes"""
    n = len(signal_data)
    fft_values = fft(signal_data)
    fft_freq = fftfreq(n, d=1.0)  # d=1.0 for daily sampling
    
    # Only positive frequencies
    positive_freq_idx = fft_freq > 0
    frequencies = fft_freq[positive_freq_idx]
    magnitudes = np.abs(fft_values[positive_freq_idx])
    
    return frequencies, magnitudes

# Compute FFT for original and all filled methods
fft_results = {}
fft_results['Original'] = compute_fft(original_values)

for method_name, filled_values in filled_data.items():
    fft_results[method_name] = compute_fft(filled_values)
    print(f"  ✓ FFT computed for {method_name}")

# ============================================================================
# 5. WAVELET ANALYSIS
# ============================================================================
print("\n[5] Performing Wavelet analysis...")

def compute_dwt(signal_data, wavelet='db4', level=5):
    """Compute Discrete Wavelet Transform"""
    coeffs = pywt.wavedec(signal_data, wavelet, level=level)
    return coeffs

def compute_cwt(signal_data, wavelet='morl', scales=None):
    """Compute Continuous Wavelet Transform"""
    if scales is None:
        scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet)
    return coefficients, frequencies, scales

# DWT Analysis
dwt_results = {}
dwt_results['Original'] = compute_dwt(original_values)

for method_name, filled_values in filled_data.items():
    dwt_results[method_name] = compute_dwt(filled_values)
    print(f"  ✓ DWT computed for {method_name}")

# CWT Analysis
cwt_results = {}
scales = np.arange(1, 64)
cwt_results['Original'] = compute_cwt(original_values, scales=scales)

for method_name, filled_values in filled_data.items():
    cwt_results[method_name] = compute_cwt(filled_values, scales=scales)
    print(f"  ✓ CWT computed for {method_name}")

# ============================================================================
# 6. COMPUTE METRICS (including wavelet comparison metrics)
# ============================================================================
print("\n[6] Computing Gibbs effect and wavelet-based metrics...")

def flatten_dwt_coeffs(coeffs):
    """Flatten list of DWT coefficient arrays into a single 1D vector."""
    return np.concatenate([c.ravel() for c in coeffs])

metrics = {}

orig_dwt = dwt_results['Original']
orig_dwt_flat = flatten_dwt_coeffs(orig_dwt)
orig_cwt, orig_cwt_freqs, orig_cwt_scales = cwt_results['Original']

for method_name, filled_values in filled_data.items():
    # ----------------------------------------------------------------------
    # Existing metrics
    # ----------------------------------------------------------------------
    # RMSE in gap region
    gap_rmse = np.sqrt(np.mean(
        (filled_values[gap_start_idx:gap_end_idx] -
         original_values[gap_start_idx:gap_end_idx]) ** 2
    ))
    
    # High-frequency energy (frequencies > 0.1 cycles/day)
    freqs, mags = fft_results[method_name]
    high_freq_mask = freqs > 0.1
    high_freq_energy = np.sum(mags[high_freq_mask] ** 2)
    
    # DWT detail coefficients energy
    dwt_coeffs = dwt_results[method_name]
    detail_energy = sum([np.sum(c ** 2) for c in dwt_coeffs[1:]])  # Skip approximation
    
    # Peak overshoot near gap boundaries
    boundary_window = 5
    left_boundary = filled_values[gap_start_idx - boundary_window:gap_start_idx + boundary_window]
    right_boundary = filled_values[gap_end_idx - boundary_window:gap_end_idx + boundary_window]
    peak_overshoot = max(
        np.max(np.abs(np.diff(left_boundary))),
        np.max(np.abs(np.diff(right_boundary)))
    )

    # ----------------------------------------------------------------------
    # New: Wavelet-based comparison metrics (DWT & CWT)
    # ----------------------------------------------------------------------
    # DWT-based metrics
    method_dwt = dwt_results[method_name]
    method_dwt_flat = flatten_dwt_coeffs(method_dwt)

    diff_dwt = method_dwt_flat - orig_dwt_flat
    wavelet_l2 = np.linalg.norm(diff_dwt)
    wavelet_rmse = np.sqrt(np.mean(diff_dwt ** 2))

    # Correlation (shape similarity)
    if np.std(orig_dwt_flat) == 0 or np.std(method_dwt_flat) == 0:
        wavelet_corr = np.nan
    else:
        wavelet_corr = np.corrcoef(orig_dwt_flat, method_dwt_flat)[0, 1]

    # Cosine similarity (directional match)
    denom = (np.linalg.norm(orig_dwt_flat) * np.linalg.norm(method_dwt_flat) + 1e-12)
    wavelet_cos_sim = float(np.dot(orig_dwt_flat, method_dwt_flat) / denom)

    # Energy ratio per DWT scale (per level, including approximation as level 0)
    energy_ratio_per_scale = []
    for lvl in range(len(orig_dwt)):
        e_orig = np.sum(orig_dwt[lvl] ** 2)
        e_meth = np.sum(method_dwt[lvl] ** 2)
        ratio = e_meth / (e_orig + 1e-12)
        energy_ratio_per_scale.append(ratio)

    # CWT-based wavelet coherence (time series)
    method_cwt, _, _ = cwt_results[method_name]  # shape: (n_scales, n_time)
    # Cross-spectrum averaged over time for each scale
    cross_spectrum = orig_cwt * np.conjugate(method_cwt)
    Sxy = np.mean(cross_spectrum, axis=1)
    Sxx = np.mean(np.abs(orig_cwt) ** 2, axis=1)
    Syy = np.mean(np.abs(method_cwt) ** 2, axis=1)
    wavelet_coherence_per_scale = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-12)
    mean_wavelet_coherence = float(np.mean(wavelet_coherence_per_scale))

    metrics[method_name] = {
        # Existing metrics
        'Gap RMSE': gap_rmse,
        'High-Freq Energy': high_freq_energy,
        'Detail Energy': detail_energy,
        'Peak Overshoot': peak_overshoot,
        # New wavelet metrics
        'Wavelet L2 (DWT)': wavelet_l2,
        'Wavelet RMSE (DWT)': wavelet_rmse,
        'Wavelet Corr (DWT)': wavelet_corr,
        'Wavelet Cosine Sim (DWT)': wavelet_cos_sim,
        # Stored as list; will appear as string in CSV
        'Energy Ratio per Scale (DWT)': energy_ratio_per_scale,
        'Mean Wavelet Coherence (CWT)': mean_wavelet_coherence
        # If you also want per-scale coherence, you could add:
        # 'Wavelet Coherence per Scale (CWT)': wavelet_coherence_per_scale
    }
    
    print(f"  {method_name}:")
    print(f"    Gap RMSE: {gap_rmse:.6f}")
    print(f"    High-Freq Energy: {high_freq_energy:.2f}")
    print(f"    Detail Energy: {detail_energy:.2f}")
    print(f"    Peak Overshoot: {peak_overshoot:.6f}")
    print(f"    Wavelet L2 (DWT): {wavelet_l2:.6f}")
    print(f"    Wavelet RMSE (DWT): {wavelet_rmse:.6f}")
    print(f"    Wavelet Corr (DWT): {wavelet_corr:.6f}")
    print(f"    Wavelet Cosine Sim (DWT): {wavelet_cos_sim:.6f}")
    print(f"    Mean Wavelet Coherence (CWT): {mean_wavelet_coherence:.6f}")

# ============================================================================
# 7. VISUALIZATION
# ============================================================================
print("\n[7] Creating visualizations...")

# Plot 1: Time Series Comparison
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

# Original
axes[0].plot(original_dates, original_values, 'b-', linewidth=1.5, label='Original')
axes[0].axvspan(original_dates[gap_start_idx], original_dates[gap_end_idx-1], 
                alpha=0.2, color='red', label='Gap Region')
axes[0].set_ylabel('Chlorophyll-a')
axes[0].set_title('Original Data (No Gap)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Linear Interpolation
axes[1].plot(original_dates, original_values, 'b--', alpha=0.3, label='Original')
axes[1].plot(original_dates, filled_data['Linear Interpolation'], 'g-', linewidth=1.5, label='Linear Interpolation')
axes[1].axvspan(original_dates[gap_start_idx], original_dates[gap_end_idx-1], 
                alpha=0.2, color='red')
axes[1].set_ylabel('Chlorophyll-a')
axes[1].set_title('Linear Interpolation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3-Point ANN
axes[2].plot(original_dates, original_values, 'b--', alpha=0.3, label='Original')
axes[2].plot(original_dates, filled_data['3 Point Prediction'], 'orange', linewidth=1.5, label='3-Point ANN')
axes[2].axvspan(original_dates[gap_start_idx], original_dates[gap_end_idx-1], 
                alpha=0.2, color='red')
axes[2].set_ylabel('Chlorophyll-a')
axes[2].set_title('3-Point ANN Prediction')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# 9-Point ANN
axes[3].plot(original_dates, original_values, 'b--', alpha=0.3, label='Original')
axes[3].plot(original_dates, filled_data['9 Point Prediction'], 'purple', linewidth=1.5, label='9-Point ANN')
axes[3].axvspan(original_dates[gap_start_idx], original_dates[gap_end_idx-1], 
                alpha=0.2, color='red')
axes[3].set_ylabel('Chlorophyll-a')
axes[3].set_title('9-Point ANN Prediction')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Time_Series_Comparison.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {os.path.join(output_dir, 'Time_Series_Comparison.png')}")

# Plot 2: Zoomed Gap Region
fig, ax = plt.subplots(figsize=(12, 6))
zoom_start = max(0, gap_start_idx - 10)
zoom_end = min(len(original_values), gap_end_idx + 10)
zoom_indices = range(zoom_start, zoom_end)
zoom_dates = original_dates[zoom_start:zoom_end]

ax.plot(zoom_dates, original_values[zoom_start:zoom_end], 'b-', linewidth=2, 
        marker='o', markersize=4, label='Original')
ax.plot(zoom_dates, filled_data['Linear Interpolation'][zoom_start:zoom_end], 'g--', 
        linewidth=2, marker='s', markersize=4, label='Linear Interpolation')
ax.plot(zoom_dates, filled_data['3 Point Prediction'][zoom_start:zoom_end], 'orange', 
        linewidth=2, marker='^', markersize=4, label='3-Point ANN')
ax.plot(zoom_dates, filled_data['9 Point Prediction'][zoom_start:zoom_end], 'purple', 
        linewidth=2, marker='d', markersize=4, label='9-Point ANN')

ax.axvspan(original_dates[gap_start_idx], original_dates[gap_end_idx-1], 
           alpha=0.2, color='red', label='Gap Region')
ax.set_xlabel('Date')
ax.set_ylabel('Chlorophyll-a')
ax.set_title('Zoomed View: Gap Region Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Gap_Region_Zoom.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {os.path.join(output_dir, 'Gap_Region_Zoom.png')}")

# Plot 3: FFT Magnitude Spectra
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

methods_to_plot = ['Original', 'Linear Interpolation', '3 Point Prediction', '9 Point Prediction']
colors = ['blue', 'green', 'orange', 'purple']

for idx, (method, color) in enumerate(zip(methods_to_plot, colors)):
    ax = axes[idx // 2, idx % 2]
    freqs, mags = fft_results[method]
    ax.semilogy(freqs, mags, color=color, linewidth=1.5)
    ax.set_xlabel('Frequency (cycles/day)')
    ax.set_ylabel('Magnitude')
    ax.set_title(f'FFT Spectrum: {method}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.5])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'FFT_Spectra.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {os.path.join(output_dir, 'FFT_Spectra.png')}")

# Plot 4: FFT Magnitude Error (compared to original)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

orig_freqs, orig_mags = fft_results['Original']
methods_compare = ['Linear Interpolation', '3 Point Prediction', '9 Point Prediction']
colors_compare = ['green', 'orange', 'purple']

for idx, (method, color) in enumerate(zip(methods_compare, colors_compare)):
    freqs, mags = fft_results[method]
    mag_error = np.abs(mags - orig_mags)
    
    axes[idx].semilogy(freqs, mag_error, color=color, linewidth=1.5)
    axes[idx].set_xlabel('Frequency (cycles/day)')
    axes[idx].set_ylabel('Magnitude Error')
    axes[idx].set_title(f'FFT Error: {method}')
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_xlim([0, 0.5])

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'FFT_Errors.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {os.path.join(output_dir, 'FFT_Errors.png')}")

# Plot 5: Continuous Wavelet Transform
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (method, color) in enumerate(zip(methods_to_plot, colors)):
    ax = axes[idx // 2, idx % 2]
    coeffs, freqs, scales = cwt_results[method]
    
    im = ax.imshow(np.abs(coeffs), extent=[0, len(original_values), scales[-1], scales[0]], 
                   cmap='jet', aspect='auto', interpolation='bilinear')
    ax.axvline(gap_start_idx, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(gap_end_idx, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Scale')
    ax.set_title(f'CWT: {method}')
    plt.colorbar(im, ax=ax, label='Magnitude')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'CWT_Analysis.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {os.path.join(output_dir, 'CWT_Analysis.png')}")

# Plot 6: Metrics Comparison (for the original 4 scalar metrics)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metric_names = ['Gap RMSE', 'High-Freq Energy', 'Detail Energy', 'Peak Overshoot']
methods_list = list(metrics.keys())

for idx, metric_name in enumerate(metric_names):
    ax = axes[idx // 2, idx % 2]
    values = [metrics[method][metric_name] for method in methods_list]
    bars = ax.bar(methods_list, values, color=['green', 'orange', 'purple'])
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} Comparison')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Metrics_Comparison.png'), dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {os.path.join(output_dir, 'Metrics_Comparison.png')}")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================
print("\n[8] Saving results to CSV...")

results_df = pd.DataFrame(metrics).T
results_df.to_csv(os.path.join(output_dir, 'FFT_Wavelet_Analysis_Results.csv'))
print(f"  ✓ Saved: {os.path.join(output_dir, 'FFT_Wavelet_Analysis_Results.csv')}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files (in 'Wavelet Analysis' directory):")
print("  1. Time_Series_Comparison.png - Time series plots for all methods")
print("  2. Gap_Region_Zoom.png - Zoomed view of gap region")
print("  3. FFT_Spectra.png - FFT magnitude spectra")
print("  4. FFT_Errors.png - FFT magnitude errors compared to original")
print("  5. CWT_Analysis.png - Continuous Wavelet Transform visualizations")
print("  6. Metrics_Comparison.png - Gibbs effect metrics comparison (time/Fourier)")
print("  7. FFT_Wavelet_Analysis_Results.csv - Numerical results (incl. wavelet metrics)")
print("\nMetrics Summary:")
print(results_df)