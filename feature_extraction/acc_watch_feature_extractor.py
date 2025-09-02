import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.fft import fft, fftfreq

# --- Accelerometer Feature Extraction Functions ---

def clean_activity_labels(df):
    """
    Cleans up inconsistent activity labels by mapping messy strings to a single, clean label.
    This function is crucial for handling typos and partial labels in the raw data.

    Args:
        df (pd.DataFrame): Input DataFrame with an 'activity' column.

    Returns:
        pd.DataFrame: DataFrame with a cleaned 'activity' column.
    """
    if 'activity' not in df.columns:
        return df

    # Normalize labels to lowercase and strip whitespace
    df['activity'] = df['activity'].str.lower().str.strip()

    # Define a mapping of messy labels to a single, clean label
    # This dictionary should be expanded as more typos are identified
    mapping = {
        'stationary_bike_legs': 'stationary_bike_legs',
        'stationary_biketationary_bike_legs': 'stationary_bike_legs',
        'st_legs': 'stationary_bike_legs',
        'sstationary_bike_legs': 'stationary_bike_legs',
        'stationaregs': 'stationary_bike_legs',
        'sy_bike_legs': 'stationary_bike_legs',
        'statiotationary_bike_legs': 'stationary_bike_legs',
        'stationary_bike_lege_legs': 'stationary_bike_legs',
        'statioke_legs': 'stationary_bike_legs',
        'stationary_nary_bike_legs': 'stationary_bike_legs',
        'sbike_legs': 'stationary_bike_legs',
        'statiy_bike_legs': 'stationary_bike_legs',

        'stationary_bike_hand': 'stationary_bike_hand',
        'stationary_bike_hantationary_bike_hand': 'stationary_bike_hand',
        'statationary_bike_hand': 'stationary_bike_hand',
        'statiand': 'stationary_bike_hand',
        'stationar_hand': 'stationary_bike_hand',
        'stationary_by_bike_hand': 'stationary_bike_hand',
        'stationary_bikeke_hand': 'stationary_bike_hand',
        'statind': 'stationary_bike_hand',
        'stationary_bike_hand_hand': 'stationary_bike_hand',
        
        'give_speech': 'give_speech',
        'prepare_speech': 'prepare_speech',
        'mental_math': 'mental_math',
        'rest_1': 'rest',
        'rest_2': 'rest',
        'rest_3': 'rest',
        'rest': 'rest',
    }

    # Use a lambda function with a dictionary lookup to apply the mapping
    def map_activity(label):
        # Find the best match, preferring exact matches, then substrings
        for key, value in mapping.items():
            if key in str(label):
                return value
        return label # Return the original label if no match is found

    df['activity'] = df['activity'].apply(map_activity)

    return df


def re_sample_accel(df, target_sampling_rate=30):
    """
    Resamples tri-axial accelerometer data to a specified target sampling rate.
    This function handles duplicate timestamps by averaging and interpolates
    missing data, using only the 'date_time' column for time conversion.

    Args:
        df (pd.DataFrame): Input DataFrame with 'x', 'y', 'z' and 'date_time' column.
        target_sampling_rate (int): The desired sampling rate in Hz (e.g., 20 or 30 for fidgeting).
    Returns:
        pd.DataFrame: Resampled DataFrame with a 'date_time' index.
    """
    # Strip whitespace from all column names for robustness
    df.columns = df.columns.str.strip()

    # Use only 'date_time' for time conversion
    if 'date_time' not in df.columns:
        raise ValueError("The 'date_time' column is required but was not found.")
    
    # Convert 'date_time' to datetime, coercing errors to NaT
    df['datetime'] = pd.to_datetime(df['date_time'], errors='coerce', format='mixed', utc=True)
    
    # Drop rows with invalid or missing timestamps. This handles invalid 'date_time' entries.
    df = df.dropna(subset=['datetime'])
    if df.empty:
        print("Warning: DataFrame became empty after cleaning invalid time entries. Returning empty DataFrame.")
        return pd.DataFrame() # Return an empty DataFrame

    df = df.set_index('datetime') # Use the new 'datetime' column as the index

    # Define aggregation dictionary for groupby and resample
    agg_dict = {col: 'mean' for col in ['x', 'y', 'z'] if col in df.columns}
    if 'activity' in df.columns:
        agg_dict['activity'] = 'first' # Take the first activity label for the resampled interval

    # Use resample().agg() to handle both resampling and the correct aggregations in one step.
    df_resampled = df.resample(f'{1000 // target_sampling_rate}ms').agg(agg_dict)

    # Interpolate missing accelerometer values (linear interpolation is standard)
    for col in ['x', 'y', 'z']:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].interpolate(method='linear')
        else:
            # If a required column was completely missing, add it as NaN
            df_resampled[col] = np.nan
    
    # Forward-fill activity labels to propagate the last known activity
    if 'activity' in df_resampled.columns:
        df_resampled['activity'] = df_resampled['activity'].ffill()

    df_resampled = df_resampled.reset_index()
    return df_resampled

def calculate_time_domain_features(series):
    """
    Calculates common time-domain statistical features for a given data series.

    Args:
        series (pd.Series): A time-series of accelerometer data (e.g., x, y, z, or VM).

    Returns:
        dict: A dictionary of calculated time-domain features.
    """
    series_no_nan = series.dropna()
    if series_no_nan.empty:
        # Return NaN for all features if the series is empty after dropping NaNs
        return {
            'mean': np.nan, 'median': np.nan, 'std': np.nan, 'min': np.nan,
            'max': np.nan, 'range': np.nan, 'iqr': np.nan, 'skew': np.nan,
            'kurtosis': np.nan, 'zcr': np.nan
        }

    mean = np.mean(series_no_nan)
    median = np.median(series_no_nan)
    std = np.std(series_no_nan)
    min_val = np.min(series_no_nan)
    max_val = np.max(series_no_nan)
    rng = max_val - min_val
    iqr = np.percentile(series_no_nan, 75) - np.percentile(series_no_nan, 25)
    skewness = skew(series_no_nan)
    kurt = kurtosis(series_no_nan)

    # Zero-crossing rate: Number of times the signal changes sign
    # Detrending (mean subtraction) makes ZCR more meaningful for acceleration
    detrended_series = series_no_nan - mean
    zero_crossings = np.where(np.diff(np.sign(detrended_series)))[0]
    zcr = len(zero_crossings) / len(series_no_nan) if len(series_no_nan) > 0 else np.nan

    return {
        'mean': mean, 'median': median, 'std': std, 'min': min_val,
        'max': max_val, 'range': rng, 'iqr': iqr, 'skew': skewness,
        'kurtosis': kurt, 'zcr': zcr
    }

def calculate_frequency_domain_features(series, sampling_rate, window_length_seconds):
    """
    Calculates frequency-domain features, including dominant frequency, spectral centroid,
    total energy from FFT, and energy within predefined frequency bands, using Welch's method for PSD.

    Args:
        series (pd.Series): A time-series of accelerometer data for a window.
        sampling_rate (int): The sampling rate of the series in Hz.
        window_length_seconds (int): The duration of the window in seconds.

    Returns:
        dict: A dictionary of calculated frequency-domain features.
    """
    series_no_nan = series.dropna()
    if len(series_no_nan) < 2: # Need at least 2 points for FFT and Welch's
        return {
            'dom_freq': np.nan, 'spec_centroid': np.nan, 'total_energy_fft': np.nan,
            'band_energy_0_0_5hz': np.nan, 'band_energy_0_5_2_5hz': np.nan,
            'band_energy_2_5_5hz': np.nan, 'band_energy_5_plus_hz': np.nan
        }

    N = len(series_no_nan)
    yf = fft(series_no_nan)
    xf = fftfreq(N, 1 / sampling_rate) # Frequencies corresponding to FFT components

    # Focus on positive frequencies and their corresponding power spectrum
    positive_frequencies = xf[:N // 2]
    power_spectrum = (np.abs(yf[:N // 2]))**2 # Power is magnitude squared

    if np.sum(power_spectrum) == 0: # Avoid division by zero
        return {
            'dom_freq': np.nan, 'spec_centroid': np.nan, 'total_energy_fft': np.nan,
            'band_energy_0_0_5hz': np.nan, 'band_energy_0_5_2_5hz': np.nan,
            'band_energy_2_5_5hz': np.nan, 'band_energy_5_plus_hz': np.nan
        }

    # Dominant Frequency (excluding the DC component at 0 Hz)
    dominant_frequency = np.nan
    if len(power_spectrum[1:]) > 0: # Ensure there are non-DC components
        dominant_frequency_idx = np.argmax(power_spectrum[1:]) + 1 # +1 to get correct index in positive_frequencies
        dominant_frequency = positive_frequencies[dominant_frequency_idx]

    # Spectral Centroid (weighted average of frequencies)
    spectral_centroid = np.sum(positive_frequencies * power_spectrum) / np.sum(power_spectrum)

    # Total Energy from FFT (proportional to total signal energy)
    total_energy_fft = np.sum(power_spectrum) / N # Average power

    # Band Energies using Welch's method for Power Spectral Density (PSD)
    nperseg_val = int(window_length_seconds * sampling_rate)
    if nperseg_val > N: # Ensure nperseg is not greater than the number of samples
        nperseg_val = N
    if nperseg_val < 4: # Welch's requires nperseg >= 4, use N if it's less
         nperseg_val = N
         if N < 4: # If window itself is too short for Welch, return nan
             return {
                'dom_freq': np.nan, 'spec_centroid': np.nan, 'total_energy_fft': np.nan,
                'band_energy_0_0_5hz': np.nan, 'band_energy_0_5_2_5hz': np.nan,
                'band_energy_2_5_5hz': np.nan, 'band_energy_5_plus_hz': np.nan
            }

    # `noverlap` is typically half of `nperseg` for Welch's.
    f, Pxx = welch(series_no_nan, fs=sampling_rate, nperseg=nperseg_val, noverlap=nperseg_val // 2)

    band_energies = {}
    # Define frequency bands relevant for human movement and fidgeting, given a 20Hz sampling rate
    # (Nyquist frequency is 10 Hz for 20Hz sampling)
    bands = {
        '0_0_5hz': (0, 0.5),      # Very slow movements, postural shifts
        '0_5_2_5hz': (0.5, 2.5),  # Typical human activity, walking
        '2_5_5hz': (2.5, 5),      # Faster movements, some fidgeting
        '5_plus_hz': (5, sampling_rate / 2) # High frequency fidgeting, tremors (up to Nyquist)
    }

    for band_name, (low_freq, high_freq) in bands.items():
        # Find indices within the frequency array (f) that fall into the current band
        idx_band = np.logical_and(f >= low_freq, f < high_freq)
        # Sum the Power Spectral Density (Pxx) values within this band
        band_power = np.sum(Pxx[idx_band])
        band_energies[f'band_energy_{band_name.replace(".", "_")}'] = band_power

    return {
        'dom_freq': dominant_frequency,
        'spec_centroid': spectral_centroid,
        'total_energy_fft': total_energy_fft,
        **band_energies # Unpack the band energies dictionary into the main features dictionary
    }

def extract_accel_features(window_df, sampling_rate, window_length_seconds):
    """
    Extracts comprehensive features from a window of tri-axial accelerometer data,
    including time-domain statistics, vector magnitude features, and frequency-domain metrics.

    Args:
        window_df (pd.DataFrame): DataFrame containing accelerometer data for a single window.
                                  Expected columns: 'x', 'y', 'z'.
        sampling_rate (int): The sampling rate of the data within the window (after resampling).
        window_length_seconds (int): The duration of the window in seconds.

    Returns:
        dict: A dictionary of all extracted features for the window.
    """
    features = {}
    # Accelerometer columns based on user's input format
    accel_cols = ['x', 'y', 'z']

    # Ensure accelerometer columns are present in the window_df. If not, add them as NaN.
    for col in accel_cols:
        if col not in window_df.columns:
            window_df[col] = np.nan
            print(f"Warning: Column '{col}' not found in window_df. Filling with NaN.")

    # Calculate Vector Magnitude (VM) only for rows where all X, Y, Z components are present
    has_all_accel_data = window_df[accel_cols].notna().all(axis=1)
    window_df['VM'] = np.where(has_all_accel_data,
                               np.sqrt(window_df['x']**2 + window_df['y']**2 + window_df['z']**2),
                               np.nan)

    # Extract features for each axis (x, y, z) and the Vector Magnitude (VM)
    for axis in accel_cols + ['VM']:
        series = window_df[axis]
        if series.dropna().empty: # If the series is entirely NaN after dropping NaNs
             # Fill all features for this axis/VM with NaN
            for stat in ['mean', 'median', 'std', 'min', 'max', 'range', 'iqr', 'skew', 'kurtosis', 'zcr',
                         'dom_freq', 'spec_centroid', 'total_energy_fft',
                         'band_energy_0_0_5hz', 'band_energy_0_5_2_5hz',
                         'band_energy_2_5_5hz', 'band_energy_5_plus_hz']:
                features[f'accel_{axis}_{stat}'] = np.nan
            continue # Move to the next axis/VM

        # Time-domain features
        time_feats = calculate_time_domain_features(series)
        for key, value in time_feats.items():
            features[f'accel_{axis}_{key}'] = value

        # Frequency-domain features
        # Calculate only if there's enough data for meaningful frequency analysis
        if len(series.dropna()) >= 2: # Need at least 2 non-NaN points
            freq_feats = calculate_frequency_domain_features(series, sampling_rate, window_length_seconds)
            for key, value in freq_feats.items():
                features[f'accel_{axis}_{key}'] = value
        else: # Fill with NaN if not enough data for frequency features
            features[f'accel_{axis}_dom_freq'] = np.nan
            features[f'accel_{axis}_spec_centroid'] = np.nan
            features[f'accel_{axis}_total_energy_fft'] = np.nan
            features[f'accel_{axis}_band_energy_0_0_5hz'] = np.nan
            features[f'accel_{axis}_band_energy_0_5_2_5hz'] = np.nan
            features[f'accel_{axis}_band_energy_2_5_5hz'] = np.nan
            features[f'accel_{axis}_band_energy_5_plus_hz'] = np.nan

    # Global time-domain features for the window: Signal Magnitude Area (SMA) and Total Energy
    # Only calculate if there is at least some valid accelerometer data in the window
    if not window_df.empty and has_all_accel_data.any():
        # SMA (Signal Magnitude Area): Sum of absolute values of components over the window, normalized by length
        features['accel_sma'] = window_df[accel_cols].abs().sum(axis=1).sum() / len(window_df)
        # Total Energy: Sum of squared values across all three axes for each row, then summed over the window
        features['accel_total_energy'] = (window_df[accel_cols]**2).sum(axis=1).sum()
    else:
        features['accel_sma'] = np.nan
        features['accel_total_energy'] = np.nan

    return features

# --- Main Pipeline for Processing Accelerometer Files ---

def process_all_accel_files(input_folder, output_csv, window_size_seconds=60, target_sampling_rate=20, activity_to_process=None):
    """
    Processes all raw accelerometer data files in a specified folder.
    It resamples each file, extracts features using a sliding window, and
    saves all extracted features into a single CSV file.

    Args:
        input_folder (str): Path to the folder containing raw accelerometer CSV files.
                            Assumes files have 'x', 'y', 'z', and 'date_time' columns.
        output_csv (str): Path to the output CSV file where all extracted features will be saved.
        window_size_seconds (int): The size of the sliding window in seconds for feature extraction.
        target_sampling_rate (int): The target sampling rate in Hz to which the data will be resampled.
                                    A higher rate (e.g., 20Hz) is recommended for fidgeting analysis.
        activity_to_process (str, optional): A specific, clean activity label to process. If None,
                                             all data is processed.
    """
    header_written = False
    
    # Iterate through all CSV files in the input folder that contain "acc"
    for file_path in glob.glob(os.path.join(input_folder, "*acc*.csv")):
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")

        try:
            # Use chunksize to read the file in smaller parts to save memory.
            chunk_size = 100000
            df_chunks = pd.read_csv(file_path, low_memory=False, chunksize=chunk_size)

            for chunk_df in df_chunks:
                print(f"Processing chunk of size {len(chunk_df)}...")
                
                # Strip whitespace from all column names
                chunk_df.columns = chunk_df.columns.str.strip()

                # Explicitly convert the accelerometer columns to numeric, coercing errors to NaN.
                accel_cols = ['x', 'y', 'z']
                for col in accel_cols:
                    if col in chunk_df.columns:
                        chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                    else:
                        chunk_df[col] = np.nan
                
                # Ensure 'date_time' column is present
                if 'date_time' not in chunk_df.columns:
                    print(f"Error: 'date_time' column not found in {file_name}. Skipping this file.")
                    break # Break out of the chunk loop and move to the next file

                # Resample the current chunk
                resampled_chunk = re_sample_accel(chunk_df, target_sampling_rate=target_sampling_rate)

                # Clean the activity labels
                if 'activity' in resampled_chunk.columns:
                    resampled_chunk = clean_activity_labels(resampled_chunk)

                # Filter the chunk based on the specified activity, if provided
                if activity_to_process:
                    resampled_chunk = resampled_chunk[resampled_chunk['activity'] == activity_to_process]
                    if resampled_chunk.empty:
                        print(f"Warning: No '{activity_to_process}' activity found in this chunk. Skipping.")
                        continue
                
                if resampled_chunk.empty:
                    print(f"Skipping empty chunk from {file_name}.")
                    continue
                
                # Extract participant ID from filename
                participant_id = os.path.splitext(file_name)[0]

                # Calculate the number of samples in each window after resampling
                window_size_samples = window_size_seconds * target_sampling_rate
                num_windows = (len(resampled_chunk) - window_size_samples + 1) // window_size_samples

                if num_windows <= 0:
                    print(f"Warning: Not enough data in current chunk from {file_name} for one window. Skipping.")
                    continue

                # Initialize a list to hold features for the current chunk
                chunk_features = []
                # Iterate through the resampled data using a sliding window
                for i in range(num_windows):
                    start_idx = i * window_size_samples
                    end_idx = start_idx + window_size_samples
                    window_df = resampled_chunk.iloc[start_idx:end_idx].copy()

                    activity = np.nan
                    if 'activity' in window_df.columns and not window_df['activity'].dropna().empty:
                        activity_mode = window_df['activity'].mode()
                        activity = activity_mode.iloc[0] if not activity_mode.empty else np.nan

                    features = extract_accel_features(window_df, target_sampling_rate, window_size_seconds)
                    features['participant'] = participant_id
                    features['window_start_idx'] = start_idx
                    features['window_end_idx'] = end_idx - 1
                    features['window_start_time'] = window_df['datetime'].min() if 'datetime' in window_df.columns and not window_df['datetime'].empty else pd.NaT
                    features['window_end_time'] = window_df['datetime'].max() if 'datetime' in window_df.columns and not window_df['datetime'].empty else pd.NaT
                    features['activity'] = activity
                    chunk_features.append(features)
                
                # After processing all windows in the current chunk, save the results
                if chunk_features:
                    features_df = pd.DataFrame(chunk_features)
                    # Define column order
                    first_cols = ['participant', 'window_start_idx', 'window_end_idx',
                                  'window_start_time', 'window_end_time', 'activity']
                    other_cols = [col for col in features_df.columns if col not in first_cols]
                    features_df = features_df[first_cols + other_cols]

                    # Write header only for the first write, then append
                    if not header_written:
                        features_df.to_csv(output_csv, mode='w', index=False)
                        header_written = True
                    else:
                        features_df.to_csv(output_csv, mode='a', header=False, index=False)
                
                print(f"Features for chunk saved to {output_csv}")

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
            continue

    if not header_written:
        print("No features were extracted. Please check input files, column names, and ensure sufficient data length.")
    else:
        print(f"\nAll features extracted and saved to '{output_csv}'.")

# --- Example Usage (Run Pipeline) ---
if __name__ == "__main__":
    # Specify the directory containing your raw data files
    input_folder_path = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File"

    # Specify the target activity. Use 'None' to process all activities.
    # The clean labels are: 'stationary_bike_legs', 'stationary_bike_hand', 'mental_math', 'give_speech', 'prepare_speech', 'rest'
    activity_to_process = 'stationary_bike_legs' 
    
    # Define the output file name, including the activity name if you are filtering
    if activity_to_process:
        output_csv_path = f"/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/feature_set/{activity_to_process}_accel_features_win_size_10.csv"
    else:
        output_csv_path = f"/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/feature_set/all_participants_accel_features_win_size_10.csv"

    # Set the window size and sampling rate
    window_size_seconds = 10
    target_sampling_rate = 20

    process_all_accel_files(input_folder=input_folder_path,
                            output_csv=output_csv_path,
                            window_size_seconds=window_size_seconds,
                            target_sampling_rate=target_sampling_rate,
                            activity_to_process=activity_to_process)
    
    print(f"\nFeatures extracted for activity '{activity_to_process}' and saved to '{output_csv_path}'.")
