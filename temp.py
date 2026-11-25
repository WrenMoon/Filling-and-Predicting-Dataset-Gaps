import os
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq
from scipy import interpolate as sp_interpolate

import pywt

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# =============================================================================
# USER SETTINGS (edit these)
# =============================================================================
START_DATE = "2015-08-23"
END_DATE   = "2016-06-28"

# Gap settings (index within the cleaned Est5 series after dropping NaNs)
GAP_START_IDX = 30
GAP_LENGTH    = 30
GAP_END_IDX   = GAP_START_IDX + GAP_LENGTH  # exclusive

# Paths
DATA_PATH      = "Data/Chlor_A Data.csv"
MODEL_3P_PATH  = "Training/models/gapfill_3point_model.keras"
SCALER_3P_PATH = "Training/models/gapfill_3point_scaler.pkl"
MODEL_9P_PATH  = "Training/models/gapfill_9point_model.keras"
SCALER_9P_PATH = "Training/models/gapfill_9point_scaler.pkl"

# Output
OUTPUT_DIR = "Wavelet Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# HEADER
# =============================================================================
print("=" * 80)
print("FFT and Wavelet Analysis: Comparing Gap-Filling Methods")
print("Methods: Linear Interpolation, Cubic Interpolation, 3-Point ANN, 9-Point ANN")
print("=" * 80)

# =============================================================================
# 1) LOAD AND PREPARE DATA
# =============================================================================
print("\n[1] Loading data...")
data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()

target = data.loc[START_DATE:END_DATE, "Est5"].copy()
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Total days in window (including NaNs): {len(target)}")
print(f"Missing values in Est5 inside window: {target.isna().sum()}")

# Clean Est5 (this defines the working index for all analysis)
original_series = target.dropna()
original_dates = original_series.index
original_values = original_series.values.astype(float)

print(f"Total days after dropping NaNs in Est5: {len(original_values)}")
print(f"Mean value: {original_values.mean():.4f}")
print(f"Std  value: {original_values.std():.4f}")

if GAP_END_IDX > len(original_values):
    raise ValueError(
        f"GAP_END_IDX={GAP_END_IDX} exceeds data length={len(original_values)}. "
        "Adjust GAP_START_IDX and GAP_LENGTH."
    )

print(f"\nGap indices: {GAP_START_IDX} to {GAP_END_IDX - 1} ({GAP_LENGTH} days)")
print(f"Gap dates: {original_dates[GAP_START_IDX]} to {original_dates[GAP_END_IDX - 1]}")

# =============================================================================
# 2) INTRODUCE ARTIFICIAL GAP
# =============================================================================
print("\n[2] Introducing artificial gap...")
gapped_values = original_values.copy()
true_gap_values = original_values[GAP_START_IDX:GAP_END_IDX].copy()
gapped_values[GAP_START_IDX:GAP_END_IDX] = np.nan

series_with_gap = pd.Series(gapped_values, index=original_dates)

# =============================================================================
# 3) APPLY GAP-FILLING METHODS
# =============================================================================
print("\n[3] Applying gap-filling methods...")

filled_data = {}

def gap_rmse(filled: np.ndarray, original: np.ndarray, s: int, e: int) -> float:
    """RMSE vs original in the gap region [s:e)."""
    return float(np.sqrt(np.mean((filled[s:e] - original[s:e]) ** 2)))

# --- Method 1: Linear interpolation
print("  - Linear Interpolation...")
filled_linear = series_with_gap.interpolate(method="linear", limit_direction="both").values
filled_data["Linear Interpolation"] = filled_linear
print("    ✓ Done")

# --- Method 2: Cubic interpolation (NEW)
print("  - Cubic Interpolation...")
try:
    # Pandas cubic uses SciPy interpolation on the index; should work for internal gaps.
    filled_cubic = series_with_gap.interpolate(method="cubic", limit_direction="both").values
    if np.isnan(filled_cubic[GAP_START_IDX:GAP_END_IDX]).any():
        raise ValueError("Cubic interpolation produced NaNs in the gap.")
    print("    ✓ Done (pandas cubic)")
except Exception as e:
    # Robust fallback using SciPy interp1d on numeric positions
    print(f"    ⚠ Pandas cubic failed ({e}). Trying SciPy interp1d(kind='cubic') fallback...")
    x = np.arange(len(series_with_gap), dtype=float)
    mask = ~np.isnan(series_with_gap.values)
    x_known = x[mask]
    y_known = series_with_gap.values[mask].astype(float)

    # Need at least 4 points for cubic
    if len(x_known) < 4:
        print("    ✗ Not enough known points for cubic fallback. Falling back to linear.")
        filled_cubic = filled_linear.copy()
    else:
        f = sp_interpolate.interp1d(
            x_known,
            y_known,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        filled_cubic = f(x).astype(float)
        # If extrapolation did something weird, still okay; gap is internal for your settings.
        print("    ✓ Done (SciPy cubic fallback)")

filled_data["Cubic Interpolation"] = filled_cubic

# --- Method 3: 3-Point ANN model
print("  - 3-Point ANN Model...")
try:
    if not os.path.exists(MODEL_3P_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_3P_PATH}")
    if not os.path.exists(SCALER_3P_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_3P_PATH}")

    model_3p = keras.models.load_model(MODEL_3P_PATH)
    with open(SCALER_3P_PATH, "rb") as f:
        scaler_3p: StandardScaler = pickle.load(f)

    # Align full_data_3p to the cleaned Est5 index (important if Est5 had NaNs)
    full_data_3p = data.loc[START_DATE:END_DATE, ["Est2", "Est5", "Est8"]].copy()
    full_data_3p = full_data_3p.loc[original_dates]

    full_data_3p["week"] = full_data_3p.index.isocalendar().week.astype(int)
    full_data_3p["dayofyear"] = full_data_3p.index.dayofyear.astype(int)
    full_data_3p["sin_doy"] = np.sin(2 * np.pi * full_data_3p["dayofyear"] / 365.25)
    full_data_3p["cos_doy"] = np.cos(2 * np.pi * full_data_3p["dayofyear"] / 365.25)
    full_data_3p["month"] = full_data_3p.index.month.astype(int)

    filled_3p = gapped_values.copy()
    for i in range(GAP_START_IDX, GAP_END_IDX):
        row = full_data_3p.iloc[i]

        spatial_mean = np.nanmean([row["Est2"], row["Est8"]])
        est2_dev = row["Est2"] - spatial_mean
        est8_dev = row["Est8"] - spatial_mean

        features = np.array([[
            row["week"], row["month"], row["sin_doy"], row["cos_doy"],
            spatial_mean, est2_dev, est8_dev
        ]], dtype=np.float32)

        features_scaled = scaler_3p.transform(features)
        correction = model_3p.predict(features_scaled, verbose=0)[0, 0]
        filled_3p[i] = spatial_mean + correction

    print("    ✓ Done")
except Exception as e:
    print(f"    ✗ Error with 3-Point model: {e}")
    print("    → Falling back to linear interpolation for 3-point series.")
    filled_3p = filled_linear.copy()

filled_data["3 Point Prediction"] = filled_3p

# --- Method 4: 9-Point ANN model
print("  - 9-Point ANN Model...")
try:
    if not os.path.exists(MODEL_9P_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_9P_PATH}")
    if not os.path.exists(SCALER_9P_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_9P_PATH}")

    model_9p = keras.models.load_model(MODEL_9P_PATH)
    with open(SCALER_9P_PATH, "rb") as f:
        scaler_9p: StandardScaler = pickle.load(f)

    all_stations = ["Est1", "Est2", "Est3", "Est4", "Est5", "Est6", "Est7", "Est8", "Est9"]
    full_data_9p = data.loc[START_DATE:END_DATE, all_stations].copy()
    full_data_9p = full_data_9p.loc[original_dates]

    full_data_9p["week"] = full_data_9p.index.isocalendar().week.astype(int)
    full_data_9p["dayofyear"] = full_data_9p.index.dayofyear.astype(int)
    full_data_9p["sin_doy"] = np.sin(2 * np.pi * full_data_9p["dayofyear"] / 365.25)
    full_data_9p["cos_doy"] = np.cos(2 * np.pi * full_data_9p["dayofyear"] / 365.25)
    full_data_9p["month"] = full_data_9p.index.month.astype(int)

    neighbor_stations = [s for s in all_stations if s != "Est5"]
    filled_9p = gapped_values.copy()

    for i in range(GAP_START_IDX, GAP_END_IDX):
        row = full_data_9p.iloc[i]

        neighbor_vals = [row[s] for s in neighbor_stations]
        spatial_mean = np.nanmean(neighbor_vals)

        feature_list = [row["week"], row["month"], row["sin_doy"], row["cos_doy"], spatial_mean]
        for station in neighbor_stations:
            feature_list.append(row[station] - spatial_mean)

        features = np.array([feature_list], dtype=np.float32)
        features_scaled = scaler_9p.transform(features)
        correction = model_9p.predict(features_scaled, verbose=0)[0, 0]
        filled_9p[i] = spatial_mean + correction

    print("    ✓ Done")
except Exception as e:
    print(f"    ✗ Error with 9-Point model: {e}")
    print("    → Falling back to linear interpolation for 9-point series.")
    filled_9p = filled_linear.copy()

filled_data["9 Point Prediction"] = filled_9p

# --- Verification + RMSE summary (NEW: RMSE explicitly for each method)
print("\n  Verification - Gap region statistics + RMSE vs original (gap only):")
print(f"    Original gap mean={true_gap_values.mean():.6f}, std={true_gap_values.std():.6f}")
rmse_table = {}
for method_name, filled_values in filled_data.items():
    gap_vals = filled_values[GAP_START_IDX:GAP_END_IDX]
    rmse_val = gap_rmse(filled_values, original_values, GAP_START_IDX, GAP_END_IDX)
    rmse_table[method_name] = rmse_val
    print(
        f"    {method_name}: mean={gap_vals.mean():.6f}, std={gap_vals.std():.6f}, "
        f"RMSE={rmse_val:.6f}"
    )

# =============================================================================
# 3b) SAVE GAP-FILL COMPARISON CSV (from your second script, now integrated)
# =============================================================================
print("\n[3b] Saving comparison CSV (Original + all filled series)...")
comparison_df = pd.DataFrame(
    {
        "Original_Est5": original_values,
        "Linear_Filled": filled_linear,
        "Cubic_Filled": filled_cubic,
        "ThreePoint_Filled": filled_3p,
        "NinePoint_Filled": filled_9p,
    },
    index=original_dates,
)
comparison_csv_path = os.path.join(OUTPUT_DIR, "GapFill_Est5_Comparison.csv")
comparison_df.to_csv(comparison_csv_path)
print(f"  ✓ Saved: {comparison_csv_path}")

# =============================================================================
# 4) FFT ANALYSIS
# =============================================================================
print("\n[4] Performing FFT analysis...")

def compute_fft(signal_data: np.ndarray):
    n = len(signal_data)
    fft_values = fft(signal_data)
    fft_freq = fftfreq(n, d=1.0)  # daily sampling

    positive = fft_freq > 0
    return fft_freq[positive], np.abs(fft_values[positive])

fft_results = {"Original": compute_fft(original_values)}
for method_name, filled_values in filled_data.items():
    fft_results[method_name] = compute_fft(filled_values)
    print(f"  ✓ FFT computed for {method_name}")

# =============================================================================
# 5) WAVELET ANALYSIS (DWT + CWT)
# =============================================================================
print("\n[5] Performing Wavelet analysis...")

def compute_dwt(signal_data, wavelet="db4", level=5):
    return pywt.wavedec(signal_data, wavelet, level=level)

def compute_cwt(signal_data, wavelet="morl", scales=None):
    if scales is None:
        scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(signal_data, scales, wavelet)
    return coefficients, frequencies, scales

dwt_results = {"Original": compute_dwt(original_values)}
for method_name, filled_values in filled_data.items():
    dwt_results[method_name] = compute_dwt(filled_values)
    print(f"  ✓ DWT computed for {method_name}")

scales = np.arange(1, 64)
cwt_results = {"Original": compute_cwt(original_values, scales=scales)}
for method_name, filled_values in filled_data.items():
    cwt_results[method_name] = compute_cwt(filled_values, scales=scales)
    print(f"  ✓ CWT computed for {method_name}")

# =============================================================================
# 6) METRICS (incl. RMSE per method vs original in gap region) + wavelet metrics
# =============================================================================
print("\n[6] Computing metrics (including RMSE in gap region + wavelet metrics)...")

def flatten_dwt_coeffs(coeffs):
    return np.concatenate([c.ravel() for c in coeffs])

metrics = {}

orig_dwt = dwt_results["Original"]
orig_dwt_flat = flatten_dwt_coeffs(orig_dwt)
orig_cwt, orig_cwt_freqs, orig_cwt_scales = cwt_results["Original"]

for method_name, filled_values in filled_data.items():
    # --- RMSE in gap region (explicit feature b)
    gap_rmse_val = gap_rmse(filled_values, original_values, GAP_START_IDX, GAP_END_IDX)

    # --- High-frequency energy (freqs > 0.1 cycles/day)
    freqs, mags = fft_results[method_name]
    high_freq_energy = float(np.sum((mags[freqs > 0.1]) ** 2))

    # --- DWT detail energy
    dwt_coeffs = dwt_results[method_name]
    detail_energy = float(sum(np.sum(c ** 2) for c in dwt_coeffs[1:]))

    # --- Peak overshoot near gap boundaries
    boundary_window = 5
    left = filled_values[GAP_START_IDX - boundary_window:GAP_START_IDX + boundary_window]
    right = filled_values[GAP_END_IDX - boundary_window:GAP_END_IDX + boundary_window]
    peak_overshoot = float(max(np.max(np.abs(np.diff(left))), np.max(np.abs(np.diff(right)))))

    # --- Wavelet comparison metrics (DWT)
    method_dwt_flat = flatten_dwt_coeffs(dwt_coeffs)
    diff_dwt = method_dwt_flat - orig_dwt_flat
    wavelet_l2 = float(np.linalg.norm(diff_dwt))
    wavelet_rmse = float(np.sqrt(np.mean(diff_dwt ** 2)))

    if np.std(orig_dwt_flat) == 0 or np.std(method_dwt_flat) == 0:
        wavelet_corr = np.nan
    else:
        wavelet_corr = float(np.corrcoef(orig_dwt_flat, method_dwt_flat)[0, 1])

    denom = (np.linalg.norm(orig_dwt_flat) * np.linalg.norm(method_dwt_flat) + 1e-12)
    wavelet_cos_sim = float(np.dot(orig_dwt_flat, method_dwt_flat) / denom)

    energy_ratio_per_scale = []
    for lvl in range(len(orig_dwt)):
        e_orig = float(np.sum(orig_dwt[lvl] ** 2))
        e_meth = float(np.sum(dwt_coeffs[lvl] ** 2))
        energy_ratio_per_scale.append(e_meth / (e_orig + 1e-12))

    # --- Wavelet "coherence-like" per scale (CWT), averaged over time
    method_cwt, _, _ = cwt_results[method_name]
    cross = orig_cwt * np.conjugate(method_cwt)
    Sxy = np.mean(cross, axis=1)
    Sxx = np.mean(np.abs(orig_cwt) ** 2, axis=1)
    Syy = np.mean(np.abs(method_cwt) ** 2, axis=1)
    wavelet_coherence_per_scale = (np.abs(Sxy) ** 2) / (Sxx * Syy + 1e-12)
    mean_wavelet_coherence = float(np.mean(wavelet_coherence_per_scale))

    metrics[method_name] = {
        "Gap RMSE": gap_rmse_val,
        "High-Freq Energy": high_freq_energy,
        "Detail Energy": detail_energy,
        "Peak Overshoot": peak_overshoot,
        "Wavelet L2 (DWT)": wavelet_l2,
        "Wavelet RMSE (DWT)": wavelet_rmse,
        "Wavelet Corr (DWT)": wavelet_corr,
        "Wavelet Cosine Sim (DWT)": wavelet_cos_sim,
        "Energy Ratio per Scale (DWT)": energy_ratio_per_scale,  # list
        "Mean Wavelet Coherence (CWT)": mean_wavelet_coherence,
    }

    print(f"  {method_name}: Gap RMSE={gap_rmse_val:.6f}")

# =============================================================================
# 7) VISUALIZATIONS (updated to include cubic)
# =============================================================================
print("\n[7] Creating visualizations...")

# ---- Plot 1: Time series panels (Original + each method)
n_panels = 1 + len(filled_data)
fig, axes = plt.subplots(n_panels, 1, figsize=(15, 3.2 * n_panels), sharex=True)
if n_panels == 1:
    axes = [axes]

# Original panel
axes[0].plot(original_dates, original_values, "b-", linewidth=1.5, label="Original")
axes[0].axvspan(original_dates[GAP_START_IDX], original_dates[GAP_END_IDX - 1],
                alpha=0.2, color="red", label="Gap Region")
axes[0].set_ylabel("Chlorophyll-a")
axes[0].set_title("Original Data (No Gap)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Method panels
for i, (method_name, filled_values) in enumerate(filled_data.items(), start=1):
    axes[i].plot(original_dates, original_values, "b--", alpha=0.3, label="Original")
    axes[i].plot(original_dates, filled_values, linewidth=1.7, label=method_name)
    axes[i].axvspan(original_dates[GAP_START_IDX], original_dates[GAP_END_IDX - 1],
                    alpha=0.2, color="red")
    axes[i].set_ylabel("Chlorophyll-a")
    axes[i].set_title(method_name)
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
ts_path = os.path.join(OUTPUT_DIR, "Time_Series_Comparison.png")
plt.savefig(ts_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {ts_path}")

# ---- Plot 2: Zoomed gap region
fig, ax = plt.subplots(figsize=(12, 6))
zoom_start = max(0, GAP_START_IDX - 10)
zoom_end = min(len(original_values), GAP_END_IDX + 10)
zoom_dates = original_dates[zoom_start:zoom_end]

ax.plot(zoom_dates, original_values[zoom_start:zoom_end], "b-", linewidth=2,
        marker="o", markersize=4, label="Original")

for method_name, filled_values in filled_data.items():
    ax.plot(zoom_dates, filled_values[zoom_start:zoom_end], linewidth=2, label=method_name)

ax.axvspan(original_dates[GAP_START_IDX], original_dates[GAP_END_IDX - 1],
           alpha=0.2, color="red", label="Gap Region")
ax.set_xlabel("Date")
ax.set_ylabel("Chlorophyll-a")
ax.set_title("Zoomed View: Gap Region Comparison")
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

gap_zoom_path = os.path.join(OUTPUT_DIR, "Gap_Region_Zoom.png")
plt.savefig(gap_zoom_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {gap_zoom_path}")

# ---- Plot 3: FFT spectra (Original + each method)
methods_fft = ["Original"] + list(filled_data.keys())
n = len(methods_fft)
cols = 2
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
axes = np.array(axes).ravel()

for idx, method in enumerate(methods_fft):
    ax = axes[idx]
    freqs, mags = fft_results[method]
    ax.semilogy(freqs, mags, linewidth=1.5)
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Magnitude")
    ax.set_title(f"FFT Spectrum: {method}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.5])

# Hide unused axes
for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
fft_path = os.path.join(OUTPUT_DIR, "FFT_Spectra.png")
plt.savefig(fft_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {fft_path}")

# ---- Plot 4: FFT magnitude error (each method vs original)
orig_freqs, orig_mags = fft_results["Original"]
methods_compare = list(filled_data.keys())
n = len(methods_compare)
cols = 2
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, 4.5 * rows))
axes = np.array(axes).ravel()

for idx, method in enumerate(methods_compare):
    ax = axes[idx]
    freqs, mags = fft_results[method]
    mag_error = np.abs(mags - orig_mags)
    ax.semilogy(freqs, mag_error, linewidth=1.5)
    ax.set_xlabel("Frequency (cycles/day)")
    ax.set_ylabel("Magnitude Error")
    ax.set_title(f"FFT Error vs Original: {method}")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.5])

for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
fft_err_path = os.path.join(OUTPUT_DIR, "FFT_Errors.png")
plt.savefig(fft_err_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {fft_err_path}")

# ---- Plot 5: CWT analysis (Original + each method)
methods_cwt = ["Original"] + list(filled_data.keys())
n = len(methods_cwt)
cols = 2
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(16, 5.5 * rows))
axes = np.array(axes).ravel()

for idx, method in enumerate(methods_cwt):
    ax = axes[idx]
    coeffs, freqs, scales = cwt_results[method]
    im = ax.imshow(
        np.abs(coeffs),
        extent=[0, len(original_values), scales[-1], scales[0]],
        cmap="jet",
        aspect="auto",
        interpolation="bilinear",
    )
    ax.axvline(GAP_START_IDX, color="white", linestyle="--", linewidth=2, alpha=0.7)
    ax.axvline(GAP_END_IDX,   color="white", linestyle="--", linewidth=2, alpha=0.7)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Scale")
    ax.set_title(f"CWT: {method}")
    plt.colorbar(im, ax=ax, label="Magnitude")

for j in range(n, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
cwt_path = os.path.join(OUTPUT_DIR, "CWT_Analysis.png")
plt.savefig(cwt_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {cwt_path}")

# ---- Plot 6: Metrics comparison (scalar metrics)
metric_names = ["Gap RMSE", "High-Freq Energy", "Detail Energy", "Peak Overshoot"]
methods_list = list(metrics.keys())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = np.array(axes).ravel()

for idx, metric_name in enumerate(metric_names):
    ax = axes[idx]
    values = [metrics[m][metric_name] for m in methods_list]
    bars = ax.bar(methods_list, values)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Comparison")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                f"{height:.4f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
metrics_path = os.path.join(OUTPUT_DIR, "Metrics_Comparison.png")
plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
print(f"  ✓ Saved: {metrics_path}")

# =============================================================================
# 8) SAVE METRICS RESULTS
# =============================================================================
print("\n[8] Saving results to CSV...")

results_df = pd.DataFrame(metrics).T
results_csv_path = os.path.join(OUTPUT_DIR, "FFT_Wavelet_Analysis_Results.csv")
results_df.to_csv(results_csv_path)
print(f"  ✓ Saved: {results_csv_path}")

rmse_only_df = pd.DataFrame({"Gap_RMSE": rmse_table}).T
rmse_only_path = os.path.join(OUTPUT_DIR, "Gap_RMSE_Summary.csv")
pd.DataFrame({"Gap_RMSE": rmse_table}).to_csv(rmse_only_path)
print(f"  ✓ Saved: {rmse_only_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print("\nGenerated files (in 'Wavelet Analysis' directory):")
print("  1. Time_Series_Comparison.png")
print("  2. Gap_Region_Zoom.png")
print("  3. FFT_Spectra.png")
print("  4. FFT_Errors.png")
print("  5. CWT_Analysis.png")
print("  6. Metrics_Comparison.png")
print("  7. GapFill_Est5_Comparison.csv (Original + Linear/Cubic/3p/9p filled)")
print("  8. FFT_Wavelet_Analysis_Results.csv (all metrics)")
print("  9. Gap_RMSE_Summary.csv (RMSE per method in gap region)")
print("\nRMSE Summary (gap region only):")
print(pd.Series(rmse_table).sort_values())