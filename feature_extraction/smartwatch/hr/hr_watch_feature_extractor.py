import os
import glob
import pandas as pd
import numpy as np
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from scipy.stats import skew, kurtosis
from pathlib import Path

# ----------------- UI / CONFIGURATION -----------------
def get_user_inputs():
    root = tk.Tk()
    root.withdraw()

    dataset_idx = simpledialog.askstring(
        "Dataset Selection",
        "1 = CareWear\n2 = GalaxyPPG\n3 = Zenodo"
    )
    dataset_map = {"1": "CareWear", "2": "GalaxyPPG", "3": "Zenodo"}
    dataset_name = dataset_map.get(dataset_idx, "Dataset")

    window_seconds = simpledialog.askinteger("Window Size", "Window size in seconds (e.g., 60):", initialvalue=60)
    overlap_percent = simpledialog.askinteger("Overlap", "Overlap percentage (0-99):", initialvalue=50)
    
    data_folder = filedialog.askdirectory(title="Select the folder containing merged CSV files")
    root.destroy()
    
    return dataset_name, window_seconds, overlap_percent, data_folder

# ----------------- NEW: HARD OUTLIER REMOVAL -----------------

def remove_hard_outliers(df, column="HeartRate"):
    """
    Strictly removes values outside the physiological range for human HR.
    This prevents sensor 'spikes' from affecting the resampled averages.
    """
    if column in df.columns:
        # Values below 30 BPM or above 220 BPM are physiologically improbable 
        # for these datasets and usually represent sensor contact loss.
        df.loc[(df[column] < 30) | (df[column] > 220), column] = np.nan
    return df

# ----------------- CORE FEATURE MATH -----------------

def extract_features(hr_series, timestamps=None):
    hr_series = np.array(hr_series, dtype=np.float64)
    
    # 1. SCIENTIFIC VALIDATION: Check for data gaps
    if timestamps is not None and len(timestamps) > 1:
        max_gap = pd.Series(timestamps).diff().dt.total_seconds().max()
        if max_gap > 2.0:
            return {k: np.nan for k in ['hr_mean','hr_median','hr_std','hr_min','hr_max','hr_iqr',
                                      'hr_skew','hr_kurtosis','hr_rmssd','hr_pnn50','hr_range',
                                      'hr_slope','hr_start','hr_end']}

    # 2. DATA DENSITY CHECK: Reject if less than 80% of window contains real data
    valid_mask = ~np.isnan(hr_series)
    hr_series_valid = hr_series[valid_mask]
    
    if len(hr_series_valid) < (len(hr_series) * 0.8):
        return {k: np.nan for k in ['hr_mean','hr_median','hr_std','hr_min','hr_max','hr_iqr',
                                  'hr_skew','hr_kurtosis','hr_rmssd','hr_pnn50','hr_range',
                                  'hr_slope','hr_start','hr_end']}

    x = np.arange(len(hr_series))[valid_mask]
    slope = np.polyfit(x, hr_series_valid, 1)[0] if len(hr_series_valid) > 1 else np.nan
    hr_diff = np.diff(hr_series_valid)
    
    rmssd = np.sqrt(np.mean(hr_diff ** 2)) if len(hr_diff) > 0 else np.nan
    nn50 = np.sum(np.abs(hr_diff) > (50/1000*60)) 
    pnn50 = nn50 / len(hr_diff) if len(hr_diff) > 0 else np.nan

    return {
        'hr_mean': np.nanmean(hr_series_valid),
        'hr_median': np.nanmedian(hr_series_valid),
        'hr_std': np.nanstd(hr_series_valid),
        'hr_min': np.nanmin(hr_series_valid),
        'hr_max': np.nanmax(hr_series_valid),
        'hr_iqr': np.percentile(hr_series_valid, 75) - np.percentile(hr_series_valid, 25),
        'hr_skew': skew(hr_series_valid, nan_policy='omit') if np.nanstd(hr_series_valid) > 1e-6 else 0,
        'hr_kurtosis': kurtosis(hr_series_valid, nan_policy='omit') if np.nanstd(hr_series_valid) > 1e-6 else 0,
        'hr_rmssd': rmssd,
        'hr_pnn50': pnn50, 
        'hr_range': np.nanmax(hr_series_valid) - np.nanmin(hr_series_valid), 
        'hr_slope': slope,
        'hr_start': hr_series_valid[0], 
        'hr_end': hr_series_valid[-1]
    }

# ----------------- DATA STANDARDIZATION -----------------

def standardize_hr_column(df):
    for col in ["HeartRate", "hr", "bpm", "Heart Rate"]:
        if col in df.columns:
            df["HeartRate"] = df[col]
            return df
    return df

def normalize_timestamp(df):
    for col in ["Timestamp_pd", "Timestamp", "datetime", "timestamp"]:
        if col in df.columns:
            df["Timestamp_pd"] = pd.to_datetime(df[col], errors="coerce")
            return df
    return df

def extract_metadata(file_name):
    p_id = "Unknown"
    p_patterns = [r"_hr_(\d+)_", r"heart_rate_(\d+)", r"P(\d+)", r"pid_(\d+)"]
    for p in p_patterns:
        match = re.search(p, file_name)
        if match: 
            p_id = int(match.group(1))
            break
            
    act_id = np.nan
    act_match = re.search(r"activity_id_(-?\d+\.?\d*)", file_name)
    if act_match:
        act_id = float(act_match.group(1))
        
    return p_id, act_id

# ----------------- PROCESSING PIPELINE -----------------

def re_sample_hr(df, activity_col):
    df = df.dropna(subset=['Timestamp_pd']).copy()
    df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')
    
    # APPLY HARD REMOVAL BEFORE AGGREGATION
    df = remove_hard_outliers(df, column="HeartRate")
    
    agg_map = {'HeartRate': 'mean'}
    if activity_col and activity_col in df.columns:
        agg_map[activity_col] = 'first'
    
    df = df.groupby('Timestamp_pd').agg(agg_map)
    
    # Resample to 1Hz
    df_hr = df[['HeartRate']].resample('1s').mean()
    df_hr['HeartRate'] = df_hr['HeartRate'].interpolate(method='linear', limit=2)
    
    if activity_col and activity_col in df.columns:
        df_labels = df[[activity_col]].resample('1s').ffill()
        df_resampled = pd.concat([df_hr, df_labels], axis=1)
    else:
        df_resampled = df_hr
    
    return df_resampled.reset_index()

def main():
    dataset_name, window_seconds, overlap_percent, data_folder = get_user_inputs()
    files = [f for f in glob.glob(os.path.join(data_folder, "*.csv")) if not os.path.basename(f).startswith("._")]
    
    all_features_rows = []

    for file_path in files:
        file_name = os.path.basename(file_path)
        if "-1.0" in file_name or "label_mapping" in file_name:
            continue

        try:
            print(f"Processing: {file_name}")
            df = pd.read_csv(file_path, low_memory=False)
            df = standardize_hr_column(df)
            df = normalize_timestamp(df)
            
            participant, file_activity = extract_metadata(file_name)
            activity_col = next((c for c in ["activity_int", "manual_labels_activity", "activity", "label"] if c in df.columns), None)
            
            df_resampled = re_sample_hr(df, activity_col)
            
            step = int(window_seconds * (1 - overlap_percent / 100))
            if step < 1: step = 1
            
            start_idx = 0
            window_id = 0
            
            while start_idx + window_seconds <= len(df_resampled):
                window_df = df_resampled.iloc[start_idx : start_idx + window_seconds]
                features = extract_features(window_df['HeartRate'].values, window_df['Timestamp_pd'])
                
                if not np.isnan(features['hr_mean']):
                    if not np.isnan(file_activity):
                        activity = file_activity
                    elif activity_col:
                        mode_val = window_df[activity_col].mode()
                        activity = mode_val.iloc[0] if not mode_val.empty else np.nan
                    else:
                        activity = "Overall"

                    features.update({
                        "FileName": file_name,
                        "Participant": participant,
                        "Activity": activity,
                        "WindowNumber": window_id,
                        "WindowStart": window_df["Timestamp_pd"].iloc[0],
                        "WindowEnd": window_df["Timestamp_pd"].iloc[-1],
                    })

                    all_features_rows.append(features)
                    window_id += 1
                
                start_idx += step
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    if all_features_rows:
        output_df = pd.DataFrame(all_features_rows)
        output_folder = os.path.join(data_folder, "extracted_features")
        os.makedirs(output_folder, exist_ok=True)
        
        output_path = os.path.join(output_folder, f"{dataset_name}_ML_features_{window_seconds}s.csv")
        output_df.to_csv(output_path, index=False)
        print(f"\nSUCCESS: Results saved to {output_folder}")
    else:
        print("No features passed the scientific quality threshold.")

if __name__ == "__main__":
    main()