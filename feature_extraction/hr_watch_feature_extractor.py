import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# --- Feature extraction functions ---
# Statistical Features (per window):

# Mean HR
# Median HR
# Standard Deviation (SDNN)
# Minimum HR
# Maximum HR
# Interquartile Range (IQR)
# Skewness
# Kurtosis

# Heart Rate Variability (HRV) Features:
# RMSSD (Root Mean Square of Successive Differences)
# pNN50 (Proportion of intervals differing by more than 50 ms)
# HR Slope (trend over window)
# HR Range (max-min)
# Temporal Features:
# HR at start/end of window
# Window duration (if variable)

# resample data at 1 Hz.
def re_sample_hr(df):
    # Parse the timestamp column if not already parsed
    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'])

    # Aggregate duplicate timestamps: mean for HR, first for activity
    df = df.groupby('Timestamp_pd').agg({
        'HeartRate': 'mean',
        'activity': 'first'  # or use a custom function if you want the mode
    })

    # Now index is unique, set it as index
    # (If not already, but groupby sets it as index)
    # df = df.set_index('Timestamp_pd')  # Not needed, already set

    # Resample to 1 Hz (1 second intervals)
    df_resampled = df.resample('1S').mean()

    # Interpolate missing HR values (linear interpolation is standard)
    df_resampled['HeartRate'] = df_resampled['HeartRate'].interpolate(method='linear')

    # Forward-fill activity labels
    df_resampled['activity'] = df['activity'].resample('1S').ffill()

    # Reset index if you want Timestamp as a column again
    df_resampled = df_resampled.reset_index()
    return df_resampled

# Save or use df_resampled as your 1 Hz, regularly sampled data
# print(df_resampled.head())
def extract_features(hr_series):
    # Remove NaN and zero values for slope calculation
    valid_idx = (~np.isnan(hr_series)) & (hr_series != 0)
    hr_series_valid = hr_series[valid_idx]
    x = np.arange(len(hr_series))[valid_idx]

    # Calculate slope only if enough valid points
    if len(hr_series_valid) > 1:
        slope = np.polyfit(x, hr_series_valid, 1)[0]
    else:
        slope = np.nan

    # Calculate diff, ignoring NaNs
    hr_series_no_nan = hr_series[~np.isnan(hr_series)]
    diff = np.diff(hr_series_no_nan)
    rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else np.nan
    nn50 = np.sum(np.abs(diff) > 0.83)
    pnn50 = nn50 / len(diff) if len(diff) > 0 else np.nan

    # If the window is all NaN or all zero, hr_series_no_nan may be empty!
    if len(hr_series_no_nan) == 0:
        mean = median = std = min_ = max_ = iqr = skewness = kurt = hr_range = hr_start = hr_end = np.nan
    else:
        mean = np.nanmean(hr_series)
        median = np.nanmedian(hr_series)
        std = np.nanstd(hr_series)
        min_ = np.nanmin(hr_series)
        max_ = np.nanmax(hr_series)
        iqr = np.nanpercentile(hr_series, 75) - np.nanpercentile(hr_series, 25)
        skewness = skew(hr_series, nan_policy='omit')
        kurt = kurtosis(hr_series, nan_policy='omit')
        hr_range = max_ - min_
        hr_start = hr_series_no_nan[0]
        hr_end = hr_series_no_nan[-1]

    return {
        'hr_mean': mean,
        'hr_median': median,
        'hr_std': std,
        'hr_min': min_,
        'hr_max': max_,
        'hr_iqr': iqr,
        'hr_skew': skewness,
        'hr_kurtosis': kurt,
        'hr_rmssd': rmssd,
        'hr_pnn50': pnn50,
        'hr_range': hr_range,
        'hr_slope': slope,
        'hr_start': hr_start,
        'hr_end': hr_end
    }

# --- Main pipeline ---

def process_all_files(input_folder, output_csv, window_size=60):
    all_features = []
    for file in glob.glob(os.path.join(input_folder, "resampled_heart_rate_P*.csv")):
        print("Participant ID ====================",file.split("/")[-1],"====================")
        df = pd.read_csv(file)
        df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')
        # resampled_df = re_sample_hr(df)
        # resampled_df.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024-25/resampled_HR/resampled_" + file.split("/")[-1])
        participant_id = os.path.splitext(os.path.basename(file))[0]
        hr_values = df['HeartRate'].values

        # Sliding window feature extraction
        for start in range(0, len(hr_values) - window_size + 1, window_size):
            window_hr = hr_values[start:start+window_size]
            window_activity = df['activity'].values[start:start+window_size]

            # Find the most common activity in the window
            if len(window_activity) > 0:
                # Use pandas mode for tie-breaking (returns sorted, pick first)
                activity_mode = pd.Series(window_activity).mode()
                activity = activity_mode.iloc[0] if not activity_mode.empty else np.nan
            else:
                activity = np.nan

            features = extract_features(window_hr)
            features['participant'] = participant_id
            features['window_start_idx'] = start
            features['window_end_idx'] = start + window_size - 1
            features['activity'] = activity   # <--- Add activity here
            all_features.append(features)
    # Save to CSV
    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_csv, index=False)
    print(f"Feature table saved to {output_csv}")

# --- Run pipeline ---
if __name__ == "__main__":
    process_all_files(input_folder="/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024-25/resampled_HR", output_csv="/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024-25/feature_set/"+"all_participants_features.csv", window_size=60)
