import os
import glob
import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
import warnings
from datetime import datetime

# SciPy for filtering and stats
from scipy import signal
from scipy.signal import butter, lfilter, welch, find_peaks
from scipy.stats import skew, kurtosis

# Suppress runtime warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------- UI / CONFIGURATION -----------------
def get_user_inputs():
    root = tk.Tk()
    root.withdraw()

    dataset_idx = simpledialog.askstring("Dataset", "1=CareWear, 2=GalaxyPPG, 3=Zenodo")
    dataset_map = {"1": "CareWear", "2": "GalaxyPPG", "3": "Zenodo"}
    dataset_name = dataset_map.get(dataset_idx, "Dataset")

    device_idx = simpledialog.askstring("Device", "1=Galaxy Watch, 2=Empatica E4, 3=Polar H10, 4=Other")
    device_map = {"1": "GalaxyWatch", "2": "EmpaticaE4", "3": "PolarH10", "4": "Other"}
    device_name = device_map.get(device_idx, "UnknownDevice")

    overlap_pct = simpledialog.askinteger("Overlap", "Overlap % (0-99):", initialvalue=50)
    
    data_folder = filedialog.askdirectory(title="Select Input Accelerometer Data Folder")
    
    root.destroy()
    return dataset_name, device_name, overlap_pct, data_folder

# ----------------- FILTERING MODULE -----------------


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    # Fs must be reasonably high to apply a 5Hz highcut filter
    if fs <= 2 * highcut:
        return data  # Skip bandpass if Nyquist rate is lower than highcut
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def process_filter_code(data_array, Fs, low=0.1, high=20.0):
    # 1) DETREND (removes DC / Gravity)
    detrended_signal = signal.detrend(data_array)
    # 2) BAND PASS 0.1-20 Hz (Targets both gross motor activity and high-freq stress tremors)
    band_pass_filtered = butter_bandpass_filter(detrended_signal, low, high, Fs, order=5)
    
    return np.array(band_pass_filtered)

def apply_filters(df, Fs):
    df = df.copy()
    
    # 1. First extract components as raw arrays
    x_vals = df['x'].values
    y_vals = df['y'].values
    z_vals = df['z'].values
    # True Signal Vector Magnitude (Euclidean Norm)
    max_acc_vals = np.sqrt(x_vals**2 + y_vals**2 + z_vals**2)
    
    # 2. Process filters (requires detrend, bandpass)
    df["Filtered_x"] = process_filter_code(x_vals, Fs)
    df["Filtered_y"] = process_filter_code(y_vals, Fs)
    df["Filtered_z"] = process_filter_code(z_vals, Fs)
    df["Filtered_max_acc"] = process_filter_code(max_acc_vals, Fs)
    
    return df

# ----------------- PSD & STAT FEATURES -----------------
def calculate_psd_features(axis_data, Fs, axis_name):
    # welch requires nperseg <= len(axis_data)
    nperseg = min(256, len(axis_data))
    if len(axis_data) < nperseg:
        nperseg = len(axis_data)
        
    freqs, psd = welch(axis_data, fs=Fs, nperseg=nperseg) 
    psd = np.abs(psd)
    
    peak_indices, _ = find_peaks(psd)
    peaks_freq = freqs[peak_indices]
    peaks_psd = psd[peak_indices]
    peak_diffs_freq = np.diff(peaks_freq)

    feature_max = np.max(psd) if len(psd) > 0 else np.nan
    feature_min = np.min(psd) if len(psd) > 0 else np.nan
    feature_skewness = skew(psd) if len(psd) > 0 else np.nan
    feature_kurtosis = kurtosis(psd) if len(psd) > 0 else np.nan
    feature_mean = np.mean(psd) if len(psd) > 0 else np.nan
    feature_sum_freq_diff = np.sum(peak_diffs_freq) if len(peak_diffs_freq) > 0 else 0
    feature_average_power = np.mean(peaks_psd) if len(peaks_psd) > 0 else 0
    feature_num_peaks = len(peaks_psd)
    
    # --- NEW: Spectral Entropy (Predictability vs Chaos) ---
    if len(psd) > 0 and np.sum(psd) > 0:
        psd_norm = psd / np.sum(psd)
        # Avoid log(0)
        feature_spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + np.finfo(float).eps))
        # Normalize between 0 and 1
        feature_spectral_entropy /= np.log2(len(psd_norm))
    else:
        feature_spectral_entropy = np.nan
        
    # --- NEW: Frequency Band Ratios (PA vs Fidgeting) ---
    pa_mask = (freqs >= 0.5) & (freqs <= 3.0)
    stress_mask = (freqs > 3.0) & (freqs <= 20.0)
    
    feature_pa_power = np.sum(psd[pa_mask]) if np.any(pa_mask) else 0.0
    feature_stress_power = np.sum(psd[stress_mask]) if np.any(stress_mask) else 0.0
    feature_stress_pa_ratio = feature_stress_power / feature_pa_power if feature_pa_power > 0 else 0.0

    return {
        f"{axis_name}_max_power_psd": feature_max,
        f"{axis_name}_min_power_psd": feature_min,
        f"{axis_name}_skewness_power_psd": feature_skewness,
        f"{axis_name}_kurtosis_power_psd": feature_kurtosis,
        f"{axis_name}_mean_power_psd": feature_mean,
        f"{axis_name}_sum_freq_diff_power_psd": feature_sum_freq_diff,
        f"{axis_name}_average_power_psd": feature_average_power,
        f"{axis_name}_num_peaks_power_psd": feature_num_peaks,
        f"{axis_name}_spectral_entropy": feature_spectral_entropy,
        f"{axis_name}_pa_band_power": feature_pa_power,
        f"{axis_name}_stress_band_power": feature_stress_power,
        f"{axis_name}_stress_pa_power_ratio": feature_stress_pa_ratio
    }

def calculate_stat_features(axis_data, axis_name):
    if len(axis_data) == 0:
        return {}
        
    mean_val = np.nanmean(axis_data)
    
    # Calculate Zero-Crossing Rate and Mean-Crossing Rate
    zero_crossings = np.where(np.diff(np.sign(axis_data)))[0]
    mean_crossings = np.where(np.diff(np.sign(axis_data - mean_val)))[0]
    
    zcr = len(zero_crossings) / len(axis_data) if len(axis_data) > 0 else 0
    mcr = len(mean_crossings) / len(axis_data) if len(axis_data) > 0 else 0

    return {
        f"{axis_name}_mean": mean_val,
        f"{axis_name}_std": np.nanstd(axis_data),
        f"{axis_name}_max": np.nanmax(axis_data),
        f"{axis_name}_min": np.nanmin(axis_data),
        f"{axis_name}_skew": skew(axis_data, nan_policy='omit'),
        f"{axis_name}_kurtosis": kurtosis(axis_data, nan_policy='omit'),
         # Interquartile Range
        f"{axis_name}_iqr": np.percentile(axis_data, 75) - np.percentile(axis_data, 25) if not np.isnan(axis_data).all() else np.nan,
        f"{axis_name}_zcr": zcr,
        f"{axis_name}_mcr": mcr
    }

def extract_acc_features(window_df, Fs):
    # First apply filter over the window
    window_df = apply_filters(window_df, Fs)
    
    features = {}
    axes_to_process = {
        'Filtered_x': window_df['Filtered_x'].values,
        'Filtered_y': window_df['Filtered_y'].values,
        'Filtered_z': window_df['Filtered_z'].values,
        'Filtered_max_acc': window_df['Filtered_max_acc'].values
    }
    
    for axis_name, axis_data in axes_to_process.items():
        # Clean data of purely NaNs
        axis_data = axis_data[~np.isnan(axis_data)]
        if len(axis_data) < 2:
            return {} # Invalid window for required PSD & stat functions
            
        features.update(calculate_psd_features(axis_data, Fs, axis_name))
        features.update(calculate_stat_features(axis_data, axis_name))

    # --- NEW: Cross-axis Pearson Correlation ---
    try:
        x_vals = window_df['Filtered_x'].values
        y_vals = window_df['Filtered_y'].values
        z_vals = window_df['Filtered_z'].values
        
        # Ensure we have enough data and not all NaNs
        if len(x_vals) > 1 and not (np.isnan(x_vals).all() or np.isnan(y_vals).all() or np.isnan(z_vals).all()):
            corr_xy = np.corrcoef(x_vals, y_vals)[0, 1]
            corr_xz = np.corrcoef(x_vals, z_vals)[0, 1]
            corr_yz = np.corrcoef(y_vals, z_vals)[0, 1]
            
            features['corr_xy'] = corr_xy if not np.isnan(corr_xy) else 0.0
            features['corr_xz'] = corr_xz if not np.isnan(corr_xz) else 0.0
            features['corr_yz'] = corr_yz if not np.isnan(corr_yz) else 0.0
        else:
            features['corr_xy'] = 0.0
            features['corr_xz'] = 0.0
            features['corr_yz'] = 0.0
    except Exception:
        # Fallback if corrcoef fails
        features['corr_xy'] = 0.0
        features['corr_xz'] = 0.0
        features['corr_yz'] = 0.0

    return features

# ----------------- HELPERS -----------------
def get_sampling_rate(df):
    if 'Timestamp_pd' not in df.columns or len(df) < 2:
        return 50.0 # Acc is usually ~50-100Hz
    diffs = df['Timestamp_pd'].diff().dt.total_seconds().dropna()
    median_interval = diffs.median()
    return 1.0 / median_interval if median_interval > 0 else 50.0

def extract_metadata_from_filename(file_name):
    p_id = "Unknown"
    act_id = "Unknown"
    
    # Matches patterns like `activity_id_1.0_acc_10_merged_labels.csv`
    act_match = re.search(r"activity_id_([\d\.]+)", file_name)
    if act_match: act_id = act_match.group(1)
        
    p_match = re.search(r"_acc_(\d+)", file_name)
    if p_match: p_id = p_match.group(1)
        
    return p_id, act_id

def generate_activity_summary(df, output_path):
    if 'Participant' in df.columns and 'Activity_Int' in df.columns:
        summary = df.groupby(['Participant', 'Activity_Int']).size().unstack(fill_value=0)
        summary.to_csv(output_path)
        print(f"Activity summary table saved to: {output_path}")

# ----------------- MAIN PIPELINE -----------------
def main():
    dataset_name, device_name, overlap_pct, data_folder = get_user_inputs()
    if not data_folder: return

    files = glob.glob(os.path.join(data_folder, "*.csv"))
    print(f"--- Processing {len(files)} Accelerometer CSV files ---")
    
    window_sizes = [2, 5, 10, 30, 60]
    all_features_rows = {ws: [] for ws in window_sizes}

    for file_path in files:
        file_name = os.path.basename(file_path)
        if file_name.startswith("._") or "label_mapping" in file_name:
            continue
            
        try:
            df = pd.read_csv(file_path, low_memory=False)
            
            # ----------- DYNAMIC COLUMN MAPPING -----------
            actual_cols = {c.lower(): c for c in df.columns}
            x_candidates = ["x", "accel_x", "acc_x", "accx", "acceleration_x", "axis1"]
            y_candidates = ["y", "accel_y", "acc_y", "accy", "acceleration_y", "axis2"]
            z_candidates = ["z", "accel_z", "acc_z", "accz", "acceleration_z", "axis3"]
            time_candidates = ["timestamp_pd", "timestamp", "time", "unix_timestamp", "unix_timesamp", "datetime"]
            act_candidates = ["activity_int_merged", "activity_int", "activity_id", "label", "activity"]

            x_col = next((actual_cols[cand] for cand in x_candidates if cand in actual_cols), None)
            y_col = next((actual_cols[cand] for cand in y_candidates if cand in actual_cols), None)
            z_col = next((actual_cols[cand] for cand in z_candidates if cand in actual_cols), None)
            time_col = next((actual_cols[cand] for cand in time_candidates if cand in actual_cols), None)
            act_col = next((actual_cols[cand] for cand in act_candidates if cand in actual_cols), None)
            
            # Identify columns
            if not all([x_col, y_col, z_col, time_col]):
                print(f"  [SKIP] {file_name} missing required columns. Found: x={x_col}, y={y_col}, z={z_col}, time={time_col}")
                continue
            
            # Standardize column names internally so the rest of the pipeline works seamlessly
            df = df.rename(columns={x_col: 'x', y_col: 'y', z_col: 'z', time_col: 'Timestamp_pd'})
            if act_col: 
                df = df.rename(columns={act_col: 'activity_int'})
            # ----------------------------------------------
            
            # Timestamp Conversion
            if pd.api.types.is_numeric_dtype(df['Timestamp_pd']):
                max_ts = df['Timestamp_pd'].max()
                if max_ts > 1e16: # likely nanoseconds (19 digits)
                    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], unit='ns', errors='coerce')
                elif max_ts > 1e14: # likely microseconds (16 digits)
                    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], unit='us', errors='coerce')
                elif max_ts > 1e11: # likely milliseconds (13 digits)
                    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], unit='ms', errors='coerce')
                else: # likely seconds
                    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], unit='s', errors='coerce')
            else:
                df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], errors='coerce')

            df = df.dropna(subset=['Timestamp_pd', 'x', 'y', 'z']).reset_index(drop=True)
            if len(df) < 2: continue

            # Numeric conversion
            for col in ['x', 'y', 'z']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # --- UNIT CONVERSION PIPELINE (Standardize to g) ---
            # Galaxy Watch: m/s^2 -> g
            if device_name == "GalaxyWatch":
                for col in ['x', 'y', 'z']:
                    df[col] = df[col] / 9.81
            # Empatica E4: 1/64 g -> g
            elif device_name == "EmpaticaE4":
                for col in ['x', 'y', 'z']:
                    df[col] = df[col] / 64.0
            # Polar H10: mg -> g
            elif device_name == "PolarH10":
                for col in ['x', 'y', 'z']:
                    df[col] = df[col] / 1000.0
            # ---------------------------------------------------

            # Second dropna in case strings were coerced to NaNs
            df = df.dropna(subset=['x', 'y', 'z']).reset_index(drop=True)
            if len(df) < 2: continue

            # Compute FS based on timestamp median deltas
            fs = get_sampling_rate(df)
            participant_id, activity_id = extract_metadata_from_filename(file_name)

            for window_sec in window_sizes:
                window_size_samples = int(window_sec * fs)
                # Ensure step samples is valid based on user requested overlap percent
                step_samples = max(1, int(window_size_samples * (1 - overlap_pct / 100)))
                
                windows_processed = 0
                # Windowing logic Loop (Same as HR Pipeline)
                for start in range(0, len(df) - window_size_samples + 1, step_samples):
                    window_df = df.iloc[start : start + window_size_samples]
                    
                    # Fetch features for this window
                    features = extract_acc_features(window_df, fs)
                    if not features: continue # Window discarded due to empty array

                    # Majority Activity Identification
                    act_int = 0
                    if 'activity_int' in window_df.columns:
                        mode_res = window_df['activity_int'].mode()
                        if not mode_res.empty: act_int = mode_res.iloc[0]

                    features.update({
                        "FileName": file_name,
                        "Participant": participant_id,
                        "ActivityID_FileName": activity_id,
                        "Activity_Int": act_int,
                        "Device": device_name,
                        "Fs": round(fs, 2),
                        "StartTime": window_df['Timestamp_pd'].iloc[0]
                    })
                    all_features_rows[window_sec].append(features)
                    windows_processed += 1

                if windows_processed > 0:
                    print(f"  [OK] {file_name} ({window_sec}s): {windows_processed} windows.")

        except Exception as e:
            print(f"  [ERROR] {file_name}: {e}")

    # After full folder process, aggregate results
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = os.path.join(data_folder, f"extracted_features_{current_time}")
    os.makedirs(output_folder, exist_ok=True)
    
    saved_any = False
    for window_sec, rows in all_features_rows.items():
        if rows:
            output_df = pd.DataFrame(rows)
            output_csv_path = os.path.join(output_folder, f"{dataset_name}_{device_name}_AccFeatures_{window_sec}s_{overlap_pct}.csv")
            output_df.to_csv(output_csv_path, index=False)
            
            summary_csv_path = os.path.join(output_folder, f"{dataset_name}_{device_name}_AccActivity_Summary_{window_sec}s_{overlap_pct}.csv")
            generate_activity_summary(output_df, summary_csv_path)
            
            print(f"SUCCESS: {window_sec}s Acc Features saved to {output_csv_path}")
            saved_any = True

    if not saved_any:
        print("\nFATAL: No valid windows found in any files.")

if __name__ == "__main__":
    main()
