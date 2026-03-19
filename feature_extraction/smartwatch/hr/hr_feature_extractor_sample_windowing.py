import os
import glob
import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from scipy.stats import skew, kurtosis
import warnings

# Suppress runtime warnings for cleaner terminal output
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------- UI / CONFIGURATION -----------------
def get_user_inputs():
    root = tk.Tk()
    root.withdraw()

    dataset_idx = simpledialog.askstring("Dataset", "1=CareWear, 2=GalaxyPPG, 3=Zenodo")
    dataset_map = {"1": "CareWear", "2": "GalaxyPPG", "3": "Zenodo"}
    dataset_name = dataset_map.get(dataset_idx, "Dataset")

    window_sec = simpledialog.askinteger("Window", "Seconds (e.g. 5):", initialvalue=5)
    overlap_pct = simpledialog.askinteger("Overlap", "Overlap % (0-99):", initialvalue=50)
    
    data_folder = filedialog.askdirectory(title="Select Input Data Folder")
    
    root.destroy()
    return dataset_name, window_sec, overlap_pct, data_folder

# ----------------- SUMMARY FUNCTION -----------------
def generate_activity_summary(df, output_path):
    if 'Participant' in df.columns and 'Activity_Int' in df.columns:
        summary = df.groupby(['Participant', 'Activity_Int']).size().unstack(fill_value=0)
        summary.to_csv(output_path)
        print(f"Activity summary table saved to: {output_path}")

# ----------------- FEATURE MATH -----------------
def calculate_hr_zones(hr):
    bins = [0, 40, 60, 80, 100, 120, 140, 160, 180, 200, np.inf]
    labels = [
        "HR Range: 0–40 bpm",
        "HR Range: 40–60 bpm",
        "HR Range: 60–80 bpm",
        "HR Range: 80–100 bpm",
        "HR Range: 100–120 bpm",
        "HR Range: 120–140 bpm",
        "HR Range: 140–160 bpm",
        "HR Range: 160–180 bpm",
        "HR Range: 180–200 bpm",
        "HR Range: >200 bpm",
    ]
    categorized = pd.cut(hr, bins=bins, labels=labels, include_lowest=True, right=False)
    return categorized.value_counts().reindex(labels, fill_value=0).to_dict()

def extract_features(hr_series):
    hr_series = np.array(hr_series, dtype=np.float64)
    valid_mask = (hr_series > 30) & (hr_series < 220) & (~np.isnan(hr_series))
    hr_series_valid = hr_series[valid_mask]
    
    if len(hr_series_valid) < 2:
        nan_features = {k: np.nan for k in [
            'hr_mean', 'hr_median', 'hr_std', 'hr_min', 'hr_max', 'hr_iqr',
            'hr_skew', 'hr_kurtosis', 'hr_rmssd', 'hr_pnn50', 'hr_range',
            'hr_slope', 'hr_start', 'hr_end'
        ]}
        empty_zones = calculate_hr_zones([])
        for k in empty_zones.keys():
            nan_features[k] = np.nan
        return nan_features

    hr_diff = np.diff(hr_series_valid)
    rmssd = np.sqrt(np.mean(hr_diff ** 2)) if len(hr_diff) > 0 else np.nan
    nn50 = np.sum(np.abs(hr_diff) > 3) 
    pnn50 = nn50 / len(hr_diff) if len(hr_diff) > 0 else np.nan
    
    # Handle constant data to avoid Skew/Kurtosis warnings
    if np.all(hr_series_valid == hr_series_valid[0]):
        hr_skew = 0.0
        hr_kurtosis = 0.0
    else:
        hr_skew = skew(hr_series_valid, nan_policy='omit')
        hr_kurtosis = kurtosis(hr_series_valid, nan_policy='omit')

    x = np.arange(len(hr_series_valid))
    try:
        slope = np.polyfit(x, hr_series_valid, 1)[0]
    except:
        slope = np.nan

    features = {
        'hr_mean': np.nanmean(hr_series_valid),
        'hr_median': np.nanmedian(hr_series_valid),
        'hr_std': np.nanstd(hr_series_valid),
        'hr_min': np.nanmin(hr_series_valid),
        'hr_max': np.nanmax(hr_series_valid),
        'hr_iqr': np.percentile(hr_series_valid, 75) - np.percentile(hr_series_valid, 25),
        'hr_skew': hr_skew,
        'hr_kurtosis': hr_kurtosis,
        'hr_rmssd': rmssd,
        'hr_pnn50': pnn50, 
        'hr_range': np.nanmax(hr_series_valid) - np.nanmin(hr_series_valid), 
        'hr_slope': slope,
        'hr_start': hr_series_valid[0], 
        'hr_end': hr_series_valid[-1]
    }
    
    features.update(calculate_hr_zones(hr_series_valid))
    
    return features

# ----------------- HELPERS -----------------
def get_sampling_rate(df):
    if 'Timestamp_pd' not in df.columns or len(df) < 2:
        return 1.0 
    diffs = df['Timestamp_pd'].diff().dt.total_seconds().dropna()
    median_interval = diffs.median()
    return 1.0 / median_interval if median_interval > 0 else 1.0

def extract_metadata_from_filename(file_name):
    p_id = "Unknown"
    match = re.search(r"(?:_hr_|P|pid_)(\d+)", file_name)
    if match: p_id = match.group(1)
    return p_id

# ----------------- MAIN PIPELINE -----------------
def main():
    dataset_name, window_sec, overlap_pct, data_folder = get_user_inputs()
    if not data_folder: return

    files = glob.glob(os.path.join(data_folder, "*.csv"))
    print(f"--- Processing {len(files)} CSV files ---")
    
    all_features_rows = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        if file_name.startswith("._") or  ("-1.0" in file_name) or "label_mapping" in file_name:
            print("Skipping -1.0 features")
            continue
            
        try:
            df = pd.read_csv(file_path, low_memory=False)
            actual_cols = {c.lower(): c for c in df.columns}
            
            hr_candidates = ["heartrate", "hr", "bpm", "heart_rate", "heart_rate (bpm)", "value"]
            time_candidates = ["timestamp_pd", "timestamp", "time", "unix_timestamp", "datetime"]
            act_candidates = ["activity_int_merged", "activity_id", "activity_int", "label", "activity"]

            hr_col = next((actual_cols[cand] for cand in hr_candidates if cand in actual_cols), None)
            time_col = next((actual_cols[cand] for cand in time_candidates if cand in actual_cols), None)
            act_col = next((actual_cols[cand] for cand in act_candidates if cand in actual_cols), None)
            
            if not hr_col or not time_col: continue
            
            df['HeartRate'] = pd.to_numeric(df[hr_col], errors='coerce')
            
            if pd.api.types.is_numeric_dtype(df[time_col]):
                is_ms = df[time_col].max() > 1e12
                df['Timestamp_pd'] = pd.to_datetime(df[time_col], unit='ms' if is_ms else 's', errors='coerce')
            else:
                df['Timestamp_pd'] = pd.to_datetime(df[time_col], errors='coerce')

            df = df.dropna(subset=['Timestamp_pd', 'HeartRate']).reset_index(drop=True)
            if len(df) < 2: continue

            fs = get_sampling_rate(df)
            window_size_samples = int(window_sec * fs)
            step_samples = max(1, int(window_size_samples * (1 - overlap_pct / 100)))
            participant_id = extract_metadata_from_filename(file_name)

            windows_processed = 0
            for start in range(0, len(df) - window_size_samples + 1, step_samples):
                window_df = df.iloc[start : start + window_size_samples]
                features = extract_features(window_df['HeartRate'].values)
                if np.isnan(features['hr_mean']): continue

                act_int = 0
                if act_col:
                    mode_res = window_df[act_col].mode()
                    if not mode_res.empty: act_int = mode_res.iloc[0]

                features.update({
                    "FileName": file_name,
                    "Participant": participant_id,
                    "Activity_Int": act_int,
                    "Fs": round(fs, 2),
                    "StartTime": window_df['Timestamp_pd'].iloc[0]
                })
                all_features_rows.append(features)
                windows_processed += 1

            if windows_processed > 0:
                print(f"  [OK] {file_name}: {windows_processed} windows.")

        except Exception as e:
            print(f"  [ERROR] {file_name}: {e}")

    if all_features_rows:
        output_df = pd.DataFrame(all_features_rows)
        output_folder = os.path.join(data_folder, "extracted_features")
        os.makedirs(output_folder, exist_ok=True)
        
        output_df.to_csv(os.path.join(output_folder, f"{dataset_name}_Features_{window_sec}s_{overlap_pct}.csv"), index=False)
        generate_activity_summary(output_df, os.path.join(output_folder, f"{dataset_name}_Activity_Summary_{window_sec}s_{overlap_pct}.csv"))
        
        print(f"\nSUCCESS: Features saved to {output_folder}")
    else:
        print("\nFATAL: No valid windows found.")

if __name__ == "__main__":
    main()