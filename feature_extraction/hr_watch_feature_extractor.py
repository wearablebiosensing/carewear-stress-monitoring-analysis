import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

# ----------------- Resample HR to 1 Hz -----------------
def re_sample_hr(df):
    # Ensure Timestamp_pd is datetime
    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], errors='coerce')
    df = df.dropna(subset=['Timestamp_pd'])

    # Remove HR outliers (<1 or >300)
    df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')
    df.loc[(df['HeartRate'] <= 0) | (df['HeartRate'] > 300), 'HeartRate'] = np.nan

    # Set Timestamp as index
    df = df.set_index('Timestamp_pd')

    # Aggregate duplicate timestamps: mean for numeric, first for non-numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    agg_dict = {col: 'mean' for col in numeric_cols}
    agg_dict.update({col: 'first' for col in non_numeric_cols})

    df = df.groupby(df.index).agg(agg_dict)

    # Now resample at 1 Hz
    df_numeric = df[numeric_cols].resample('1s').mean()
    df_non_numeric = df[non_numeric_cols].resample('1s').ffill()
    df_resampled = pd.concat([df_numeric, df_non_numeric], axis=1)

    # Interpolate HR if needed
    if 'HeartRate' in df_resampled.columns:
        df_resampled['HeartRate'] = df_resampled['HeartRate'].interpolate(method='linear')

    return df_resampled.reset_index()


# ----------------- Feature Extraction -----------------

def extract_features(hr_series):
    hr_series = np.array(hr_series, dtype=np.float64)
    
    # Remove invalid HR values
    valid_mask = (~np.isnan(hr_series)) & (hr_series > 0) & (hr_series <= 300)
    hr_series_valid = hr_series[valid_mask]
    x = np.arange(len(hr_series))[valid_mask]

    # Slope
    slope = np.polyfit(x, hr_series_valid, 1)[0] if len(hr_series_valid) > 1 else np.nan

    # Successive differences
    hr_diff = np.diff(hr_series_valid)
    rmssd = np.sqrt(np.mean(hr_diff ** 2)) if len(hr_diff) > 0 else np.nan
    nn50 = np.sum(np.abs(hr_diff) > 50/1000*60)  # convert 50ms to bpm equivalent?
    pnn50 = nn50 / len(hr_diff) if len(hr_diff) > 0 else np.nan

    if len(hr_series_valid) == 0:
        return dict.fromkeys(['hr_mean','hr_median','hr_std','hr_min','hr_max','hr_iqr',
                              'hr_skew','hr_kurtosis','hr_rmssd','hr_pnn50','hr_range',
                              'hr_slope','hr_start','hr_end'], np.nan)

    # Basic stats
    mean = np.nanmean(hr_series_valid)
    median = np.nanmedian(hr_series_valid)
    std = np.nanstd(hr_series_valid)
    min_ = np.nanmin(hr_series_valid)
    max_ = np.nanmax(hr_series_valid)
    iqr = np.nanpercentile(hr_series_valid, 75) - np.nanpercentile(hr_series_valid, 25)
    hr_range = max_ - min_
    hr_start = hr_series_valid[0]
    hr_end = hr_series_valid[-1]

    # Skewness and kurtosis: handle nearly-constant windows
    if std < 1e-6:  # nearly constant
        skewness = 0.0
        kurt = 0.0
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skewness = skew(hr_series_valid, nan_policy='omit')
            kurt = kurtosis(hr_series_valid, nan_policy='omit')

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

# ----------------- Main Pipeline -----------------
def process_all_files(input_folder, output_csv, window_size=60):
    all_features = []

    for file_path in glob.glob(os.path.join(input_folder, "*.csv")):
        participant_id = os.path.splitext(os.path.basename(file_path))[0]
        print(f"Processing {participant_id}")

        df = pd.read_csv(file_path,low_memory=False)

        # Resample HR to 1 Hz and remove invalid HR
        df_resampled = re_sample_hr(df)
        hr_values = df_resampled['HeartRate'].values
        activity_values = df_resampled['activity_int'].values

        # Sliding window feature extraction
        for start in range(0, len(hr_values) - window_size + 1, window_size):
            window_hr = hr_values[start:start + window_size]
            window_activity = activity_values[start:start + window_size]

            activity_mode = pd.Series(window_activity).mode()
            activity = activity_mode.iloc[0] if not activity_mode.empty else np.nan

            features = extract_features(window_hr)
            print("participant_id ==================================== ",participant_id)
            features.update({
                'participant': participant_id.split("_")[5],  # now correctly gets participant number
                'window_start_idx': start,
                'window_end_idx': start + window_size - 1,
                'activity_int': activity
            })

            all_features.append(features)

    features_df = pd.DataFrame(all_features)
    features_df.to_csv(output_csv, index=False)
    print(f"Saved all features to {output_csv}")

# ----------------- Run -----------------
if __name__ == "__main__":
    process_all_files(
        input_folder="/Volumes/CW_2024/hr_chunks",
        output_csv="/Volumes/CW_2024/Features/all_hr_features_winsize_60.csv",
        window_size=60
    )
