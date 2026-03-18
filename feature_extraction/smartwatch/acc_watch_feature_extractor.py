import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from scipy.fft import fft, fftfreq
import gc

# ----------------- Label Cleaning -----------------
def clean_activity_labels(df):
    """Normalize messy activity labels and enforce pandas StringDtype."""
    if 'activity' not in df.columns:
        return df
    df['activity'] = df['activity'].astype("string")
    df['activity'] = df['activity'].replace("nan", pd.NA)
    df['activity'] = df['activity'].str.lower().str.strip()

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

    def map_activity(label):
        if pd.isna(label):
            return pd.NA
        for key, value in mapping.items():
            if key in str(label):
                return value
        return label

    df['activity'] = df['activity'].apply(map_activity)
    return df


# ----------------- Resampling -----------------
def re_sample_accel(df, target_sampling_rate=30):
    """Resample accelerometer data to target Hz using Timestamp_pd as index."""
    df.columns = df.columns.str.strip()

    if 'Timestamp_pd' not in df.columns:
        raise ValueError("Missing 'Timestamp_pd' column.")

    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], errors='coerce', utc=True)
    df = df.dropna(subset=['Timestamp_pd'])
    if df.empty:
        return pd.DataFrame()

    df = df.set_index('Timestamp_pd')

    agg_dict = {col: 'mean' for col in ['x', 'y', 'z'] if col in df.columns}
    if 'activity' in df.columns:
        agg_dict['activity'] = 'first'
    if 'activity_int' in df.columns:
        agg_dict['activity_int'] = 'first'

    df_resampled = df.resample(f'{1000 // target_sampling_rate}ms').agg(agg_dict)

    for col in ['x', 'y', 'z']:
        if col in df_resampled.columns:
            df_resampled[col] = df_resampled[col].interpolate(method='linear')
        else:
            df_resampled[col] = np.nan

    if 'activity' in df_resampled.columns:
        df_resampled['activity'] = df_resampled['activity'].ffill()
    if 'activity_int' in df_resampled.columns:
        df_resampled['activity_int'] = df_resampled['activity_int'].ffill()

    return df_resampled.reset_index()


# ----------------- Feature Extraction -----------------
def calculate_time_domain_features(series):
    series = series.dropna()
    if series.empty:
        return {k: np.nan for k in ['mean','median','std','min','max','range','iqr','skew','kurtosis','zcr']}

    mean = series.mean()
    median = series.median()
    std = series.std()
    min_val, max_val = series.min(), series.max()
    rng = max_val - min_val
    iqr = np.percentile(series,75) - np.percentile(series,25)
    skewness = skew(series)
    kurt_val = kurtosis(series)
    detrended = series - mean
    zero_crossings = np.where(np.diff(np.sign(detrended)))[0]
    zcr = len(zero_crossings)/len(series) if len(series)>0 else np.nan

    return {'mean': mean,'median': median,'std': std,'min': min_val,'max': max_val,
            'range': rng,'iqr': iqr,'skew': skewness,'kurtosis': kurt_val,'zcr': zcr}


def calculate_frequency_domain_features(series, sampling_rate, window_length_seconds):
    series = series.dropna()
    if len(series) < 2:
        return {k: np.nan for k in [
            'dom_freq','spec_centroid','total_energy_fft',
            'band_energy_0_0_5hz','band_energy_0_5_2_5hz',
            'band_energy_2_5_5hz','band_energy_5_plus_hz']}

    N = len(series)
    yf = fft(series)
    xf = fftfreq(N, 1/sampling_rate)
    pos_freqs, power = xf[:N//2], (np.abs(yf[:N//2]))**2
    if power.sum() == 0:
        return {k: np.nan for k in [
            'dom_freq','spec_centroid','total_energy_fft',
            'band_energy_0_0_5hz','band_energy_0_5_2_5hz',
            'band_energy_2_5_5hz','band_energy_5_plus_hz']}

    dom_freq = pos_freqs[np.argmax(power[1:])+1] if len(power)>1 else np.nan
    spec_centroid = np.sum(pos_freqs*power)/np.sum(power)
    total_energy_fft = power.sum()/N

    nperseg_val = min(max(int(window_length_seconds*sampling_rate),4), N)
    f, Pxx = welch(series, fs=sampling_rate, nperseg=nperseg_val, noverlap=nperseg_val//2)
    bands = {'0_0_5hz':(0,0.5),'0_5_2_5hz':(0.5,2.5),'2_5_5hz':(2.5,5),'5_plus_hz':(5,sampling_rate/2)}
    band_energies = {f'band_energy_{k}': Pxx[(f>=lo)&(f<hi)].sum() for k,(lo,hi) in bands.items()}

    return {'dom_freq':dom_freq,'spec_centroid':spec_centroid,'total_energy_fft':total_energy_fft,**band_energies}


def extract_accel_features(window_df, sampling_rate, window_length_seconds):
    features = {}
    for axis in ['x','y','z']:
        if axis not in window_df: window_df[axis] = np.nan

    window_df['VM'] = np.sqrt(window_df['x']**2 + window_df['y']**2 + window_df['z']**2)

    for axis in ['x','y','z','VM']:
        series = window_df[axis]
        if series.dropna().empty:
            for k in ['mean','median','std','min','max','range','iqr','skew','kurtosis','zcr',
                      'dom_freq','spec_centroid','total_energy_fft',
                      'band_energy_0_0_5hz','band_energy_0_5_2_5hz',
                      'band_energy_2_5_5hz','band_energy_5_plus_hz']:
                features[f'accel_{axis}_{k}'] = np.nan
            continue
        features.update({f'accel_{axis}_{k}':v for k,v in calculate_time_domain_features(series).items()})
        features.update({f'accel_{axis}_{k}':v for k,v in calculate_frequency_domain_features(series,sampling_rate,window_length_seconds).items()})

    if not window_df[['x','y','z']].isna().all().all():
        features['accel_sma'] = window_df[['x','y','z']].abs().sum(axis=1).sum()/len(window_df)
        features['accel_total_energy'] = (window_df[['x','y','z']]**2).sum(axis=1).sum()
    else:
        features['accel_sma'],features['accel_total_energy'] = np.nan,np.nan

    return features


# ----------------- Main Loop (No Chunking) -----------------
def process_all_accel_files(input_folder, output_csv, window_size_seconds=10, target_sampling_rate=20):
    header_written = False

    for file_path in glob.glob(os.path.join(input_folder,"*acc*.csv")):
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")

        try:
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()

            for col in ['x','y','z']:
                if col in df: df[col] = pd.to_numeric(df[col], errors='coerce')
                else: df[col] = np.nan

            if 'Timestamp_pd' not in df:
                print(f"⚠️ Skipping {file_name}, missing Timestamp_pd")
                continue

            resampled = re_sample_accel(df, target_sampling_rate)
            if resampled.empty: continue
            resampled = clean_activity_labels(resampled)
            # os.path.splitext(file_name)[0] = activity_id_8.0_acc_1_merged_labels
            participant_id = os.path.splitext(file_name)[0].split("_")[4]
            window_size_samples = window_size_seconds*target_sampling_rate
            num_windows = (len(resampled)-window_size_samples+1)//window_size_samples
            if num_windows <=0: continue

            all_features = []
            for i in range(num_windows):
                start,end = i*window_size_samples,(i+1)*window_size_samples
                window_df = resampled.iloc[start:end].copy()

                activity = window_df['activity'].mode().iloc[0] if 'activity' in window_df and not window_df['activity'].dropna().empty else np.nan
                activity_int = window_df['activity_int'].mode().iloc[0] if 'activity_int' in window_df and not window_df['activity_int'].dropna().empty else np.nan

                feats = extract_accel_features(window_df,target_sampling_rate,window_size_seconds)
                feats.update({
                    'participant':participant_id,
                    'source_file':file_name,
                    'window_start_time':window_df['Timestamp_pd'].min(),
                    'window_end_time':window_df['Timestamp_pd'].max(),
                    'activity':activity,
                    'activity_int':activity_int
                })
                all_features.append(feats)

            if all_features:
                features_df = pd.DataFrame(all_features)
                if not header_written:
                    features_df.to_csv(output_csv, mode='w', index=False)
                    header_written = True
                else:
                    features_df.to_csv(output_csv, mode='a', header=False, index=False)

            del df,resampled,all_features,features_df
            gc.collect()

        except Exception as e:
            print(f"❌ Error {file_name}: {e}")
            continue

    print(f"\n✅ All features saved to {output_csv}" if header_written else "⚠️ No features extracted.")


# ----------------- Example Run -----------------
if __name__=="__main__":
    input_folder_path = "/Volumes/CW_2024/acc_chunks"
    output_csv_path = "/Volumes/CW_2024/Features/all_accel_features_window_size_60.csv"

    process_all_accel_files(
        input_folder=input_folder_path,
        output_csv=output_csv_path,
        window_size_seconds=60,
        target_sampling_rate=30
    )
