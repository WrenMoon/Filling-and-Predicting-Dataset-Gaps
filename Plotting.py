import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta
from scipy.interpolate import interp1d
from sklearn.metrics import root_mean_squared_error
import matplotlib.dates as mdates

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
OUT_DIR = "outputs"
DATA_DIR = "Data"
os.makedirs(OUT_DIR, exist_ok=True)

# Files
DATES_CSV = os.path.join(DATA_DIR, "Dates.csv")
FILLED_CSV = os.path.join(DATA_DIR, "Filled_Chlorophyll_Data.csv")
ORIG_CSV = os.path.join(DATA_DIR, "Chlor_A data.csv")

# Column in original data
ORIG_COL = "Est5"

# Plot window padding around gap
PADDING_DAYS = 50

# Spectral analysis settings
FFT_RESAMPLE_FREQ = "1D"     # resample to daily for uniform spacing
FFT_INTERP_METHOD = "time"   # interpolate missing after resample
WINDOW = "hann"              # 'hann' or None

# ------------------------------------------------------------------
# LOAD INPUTS
# ------------------------------------------------------------------
dates = pd.read_csv(DATES_CSV)

filled_df = pd.read_csv(FILLED_CSV, index_col=0, parse_dates=True)

original_data = pd.read_csv(ORIG_CSV, index_col=0, parse_dates=True)
values_full = original_data[ORIG_COL].copy()

start_date = pd.to_datetime(dates.loc[0, "start_date"])
end_date = pd.to_datetime(dates.loc[0, "end_date"])

# NOTE: keep your existing ±1 day behavior (as in your code)
gap_start = pd.to_datetime(dates.loc[0, "gap_start"]) - timedelta(days=1)
gap_end = pd.to_datetime(dates.loc[0, "gap_end"]) + timedelta(days=1)

# Slice the analysis window
values = values_full.loc[start_date:end_date].copy()

# Plot window
plot_start = max(gap_start - timedelta(days=PADDING_DAYS), start_date)
plot_end = min(gap_end + timedelta(days=PADDING_DAYS), end_date)

# ------------------------------------------------------------------
# CREATE ARTIFICIAL GAP
# ------------------------------------------------------------------
mask = (values.index >= gap_start) & (values.index <= gap_end)
gap_index = values.index[mask]

values_with_gap = values.copy()
values_with_gap.loc[mask] = np.nan

# ------------------------------------------------------------------
# GAP FILL METHODS (3pt, 9pt, linear, cubic)
# ------------------------------------------------------------------
methods = {}

# Linear
methods["Linear"] = values_with_gap.interpolate(method="linear")

# Cubic (may fail if insufficient points)
try:
    methods["Cubic"] = values_with_gap.interpolate(method="cubic")
except Exception:
    # fallback to linear if cubic fails
    methods["Cubic"] = values_with_gap.interpolate(method="linear")

# Neural net columns (if present)
if "3 Point Prediction" in filled_df.columns:
    # Align to our index and window
    methods["3pt_NN"] = filled_df["3 Point Prediction"].reindex(values.index)
else:
    methods["3pt_NN"] = pd.Series(index=values.index, dtype=float)

if "9 Point Prediction" in filled_df.columns:
    methods["9pt_NN"] = filled_df["9 Point Prediction"].reindex(values.index)
else:
    methods["9pt_NN"] = pd.Series(index=values.index, dtype=float)

# Ensure all are Series with same index
for k in list(methods.keys()):
    methods[k] = pd.Series(methods[k], index=values.index)

# ------------------------------------------------------------------
# METRICS (RMSE over the gap only)
# ------------------------------------------------------------------
rmse_results = {}

y_true_gap = values.loc[gap_index]

for name, pred in methods.items():
    y_pred_gap = pred.loc[gap_index]

    pair = pd.concat([y_true_gap.rename("true"), y_pred_gap.rename("pred")], axis=1)
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()

    if len(pair) == 0:
        rmse_results[name] = np.nan
        continue

    rmse_results[name] = root_mean_squared_error(
        pair["true"].to_numpy(dtype=float),
        pair["pred"].to_numpy(dtype=float),
    )

rmse_df = (
    pd.DataFrame({"Method": list(rmse_results.keys()), "RMSE": list(rmse_results.values())})
    .sort_values("RMSE", na_position="last")
    .reset_index(drop=True)
)

# ------------------------------------------------------------------
# NAMING CONVENTION
# ------------------------------------------------------------------
gap_tag = f"{gap_start.strftime('%Y%m%d')}_{gap_end.strftime('%Y%m%d')}"
prefix_metrics = f"metrics_rmse_{gap_tag}"
prefix_plot = f"plot_gapfill_{gap_tag}"
prefix_data = f"data_gapfill_{gap_tag}"
prefix_spec = f"spectral_{gap_tag}"

# ------------------------------------------------------------------
# PLOT: original + all methods (over plot window)
# ------------------------------------------------------------------
plot_values = values.loc[plot_start:plot_end]

fig, ax = plt.subplots(figsize=(9, 3.2), dpi=600)

ax.plot(plot_values.index, plot_values, label="Original", color="blue", linewidth=0.8)
ax.scatter(plot_values.index, plot_values, color="blue", s=2, alpha=0.7)

# Plot each method only in the gap (for clarity) but you can change to full if you want
for name in ["Linear", "Cubic", "3pt_NN", "9pt_NN"]:
    s = methods[name].loc[gap_start:gap_end]
    rmse = rmse_results.get(name, np.nan)
    rmse_str = "NA" if not np.isfinite(rmse) else f"{rmse:.4f}"

    label = f"{name} (RMSE: {rmse_str})"
    style = "--"
    lw = 1.6

    # Slight styling tweak
    if name == "9pt_NN":
        ax.plot(s.index, s, label=label, color="black", linestyle=style, linewidth=lw)
    else:
        ax.plot(s.index, s, label=label, linestyle=style, linewidth=lw)

# Gap markers
ax.axvline(gap_start, color="blue", linestyle="--", linewidth=1)
ax.axvline(gap_end, color="blue", linestyle="--", linewidth=1)

ax.set_xlim(plot_start, plot_end)
ax.set_ylim(0, 2)
ax.set_yticks(np.arange(0, 2.01, 0.25))
ax.set_ylabel("Chlorophyll-a (mg m$^{-3}$)", fontsize=9)

ax.grid(True, linestyle="--", alpha=0.4)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
ax.tick_params(axis="x", which="major", labelsize=8)

ax.legend(loc="upper left", fontsize=8, frameon=True)
fig.tight_layout()

plot_png = os.path.join(OUT_DIR, f"{prefix_plot}.png")
plot_eps = os.path.join(OUT_DIR, f"{prefix_plot}.eps")
fig.savefig(plot_png, dpi=600, bbox_inches="tight")
fig.savefig(plot_eps, dpi=600, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# CSV: original + (2 columns per method)
#   - <method>_full: original everywhere except gap replaced by method
#   - <method>_gap_only: only gap values, NaN elsewhere
# ------------------------------------------------------------------
export_idx = plot_values.index
export_df = pd.DataFrame(index=export_idx)
export_df["Original"] = plot_values

gap_mask_export = (export_df.index >= gap_start) & (export_df.index <= gap_end)

for name in ["Linear", "Cubic", "3pt_NN", "9pt_NN"]:
    method_series_window = methods[name].reindex(export_idx)

    # full: original data but gap replaced by method
    full_col = export_df["Original"].copy()
    full_col.loc[gap_mask_export] = method_series_window.loc[gap_mask_export]

    # gap-only: only gap filled values
    gap_only_col = pd.Series(np.nan, index=export_idx, dtype=float)
    gap_only_col.loc[gap_mask_export] = method_series_window.loc[gap_mask_export]

    export_df[f"{name}_full"] = full_col
    export_df[f"{name}_gap_only"] = gap_only_col

data_csv = os.path.join(OUT_DIR, f"{prefix_data}.csv")
export_df.to_csv(data_csv)

# Also save RMSE table
rmse_csv = os.path.join(OUT_DIR, f"{prefix_metrics}.csv")
rmse_df.to_csv(rmse_csv, index=False)

# ------------------------------------------------------------------
# SPECTRAL ANALYSIS (FFT) on:
#   - Original (with gap removed -> resample -> interpolate)
#   - Each method FULL series (original with gap filled by that method)
# Outputs:
#   - spectral_<gapTag>.csv (frequency + amplitude columns)
#   - spectral_<gapTag>.png (overlay plot)
# ------------------------------------------------------------------
def _prepare_uniform_series(s: pd.Series) -> pd.Series:
    """Resample to uniform daily grid and interpolate for FFT stability."""
    s = s.sort_index()
    s_u = s.resample(FFT_RESAMPLE_FREQ).mean()
    # Fill missing for FFT
    s_u = s_u.interpolate(method=FFT_INTERP_METHOD).ffill().bfill()
    return s_u

def _fft_amplitude(s_uniform: pd.Series) -> pd.DataFrame:
    """Return one-sided amplitude spectrum for a uniformly spaced series."""
    x = s_uniform.to_numpy(dtype=float)
    n = len(x)

    # Detrend (remove mean)
    x = x - np.nanmean(x)

    # Windowing
    if WINDOW is None:
        w = np.ones(n)
        cg = 1.0
    elif WINDOW == "hann":
        w = np.hanning(n)
        cg = np.mean(w) if np.mean(w) != 0 else 1.0  # coherent gain
    else:
        raise ValueError("WINDOW must be None or 'hann'")

    xw = x * w

    # Sampling interval in days (since we resampled daily)
    dt_days = pd.to_timedelta(pd.tseries.frequencies.to_offset(FFT_RESAMPLE_FREQ)).total_seconds() / 86400.0

    # FFT
    fft_vals = np.fft.rfft(xw)
    freqs = np.fft.rfftfreq(n, d=dt_days)  # cycles per day

    # One-sided amplitude (scaled, window-corrected)
    amp = (2.0 / n) * np.abs(fft_vals) / cg
    amp[0] = amp[0] / 2.0  # DC component not doubled

    out = pd.DataFrame({"freq_cpd": freqs, "amplitude": amp})
    # Add period in days (avoid div0)
    out["period_days"] = np.where(out["freq_cpd"] > 0, 1.0 / out["freq_cpd"], np.inf)
    return out

# Build series to analyze
series_for_fft = {}

# Original with gap still present (then interpolated after resample)
series_for_fft["Original_gap"] = values_with_gap.loc[plot_start:plot_end]

# Method FULL series in the window (original with gap replaced)
for name in ["Linear", "Cubic", "3pt_NN", "9pt_NN"]:
    s_win = export_df[f"{name}_full"]  # already windowed index
    series_for_fft[name] = s_win

# Compute spectra
spec_df = None
spec_cache = {}

for name, s in series_for_fft.items():
    s_u = _prepare_uniform_series(s)
    spec = _fft_amplitude(s_u)
    spec_cache[name] = spec

    if spec_df is None:
        spec_df = spec[["freq_cpd", "period_days"]].copy()
    spec_df[f"amp_{name}"] = spec["amplitude"].to_numpy()

# Save spectral CSV
spec_csv = os.path.join(OUT_DIR, f"{prefix_spec}.csv")
spec_df.to_csv(spec_csv, index=False)

# Plot spectra (amplitude vs period in days on log x for readability)
fig2, ax2 = plt.subplots(figsize=(9, 3.2), dpi=600)

# Avoid the inf period (DC); plot from index 1 onward
plot_spec = spec_df.iloc[1:].copy()

for name in series_for_fft.keys():
    ax2.plot(
        plot_spec["period_days"],
        plot_spec[f"amp_{name}"],
        label=name,
        linewidth=1.2,
    )

ax2.set_xscale("log")
ax2.set_xlabel("Period (days, log scale)")
ax2.set_ylabel("Amplitude (a.u.)")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.legend(fontsize=8, frameon=True)
fig2.tight_layout()

spec_png = os.path.join(OUT_DIR, f"{prefix_spec}.png")
spec_eps = os.path.join(OUT_DIR, f"{prefix_spec}.eps")
fig2.savefig(spec_png, dpi=600, bbox_inches="tight")
fig2.savefig(spec_eps, dpi=600, bbox_inches="tight")
plt.show()

print("Saved:")
print(" ", data_csv)
print(" ", rmse_csv)
print(" ", plot_png)
print(" ", plot_eps)
print(" ", spec_csv)
print(" ", spec_png)
print(" ", spec_eps)