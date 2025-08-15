import os
import time
import pandas as pd
import numpy as np
import neurokit2 as nk
from scipy.stats import kurtosis, skew

################################################################################################
# Input Raw ECG signal in numpy array format, along with known sample rate
# Returns: follwoing SQI metrics - avgQRS_mean,avgQRS_time,kurtosis_val,kurtosis_time,
#                                  skewness_val,entropy_val,entropy_time,skewness_time
###############################################################################################

def get_ecg_sqi(ecg_signal,sampling_rate):
    # Processing pipeline
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
    ecg_cleaned = signals["ECG_Clean"].values
    rpeaks = info["ECG_R_Peaks"]

    # --- Timed Computations ---
    # avgQRS_mean
    start_time = time.time()
    sqi_avg = nk.ecg_quality(ecg_cleaned, rpeaks, sampling_rate=sampling_rate, method="averageQRS")
    avgQRS_mean = np.mean(sqi_avg)
    avgQRS_time = time.time() - start_time

    # kurtosis
    start_time = time.time()
    kurtosis_val = kurtosis(ecg_cleaned)
    kurtosis_time = time.time() - start_time

    # skewness
    start_time = time.time()
    skewness_val = skew(ecg_cleaned)
    skewness_time = time.time() - start_time

    # sample entropy
    start_time = time.time()
    entropy_val = nk.entropy_sample(ecg_cleaned)
    entropy_time = time.time() - start_time
    return avgQRS_mean,avgQRS_time,kurtosis_val,kurtosis_time,skewness_val,entropy_val,entropy_time,skewness_time

def batch_ecg_quality(input_folder, output_csv, signal_column="ECG_X", sampling_rate=125):
    results = []

    for filename in os.listdir(input_folder):
        if "biopac.csv" in filename:
            participant_id = filename.split("_")[0]
            file_path = os.path.join(input_folder, filename)

            try:
                df = pd.read_csv(file_path)
                if signal_column not in df.columns:
                    print(f"Column '{signal_column}' not found in {filename}, skipping.")
                    continue

                # Remove unwanted "samples" rows
                df = df[~df.astype(str).apply(lambda row: row.str.contains("samples", case=False, na=False)).any(axis=1)]

                # Convert ECG signal column to numeric, coercing errors
                ecg_signal = pd.to_numeric(df[signal_column], errors='coerce').dropna().values
                avgQRS_mean,avgQRS_time,kurtosis_val,kurtosis_time,skewness_val,entropy_val,entropy_time,skewness_time = get_ecg_sqi(ecg_signal,sampling_rate)
                results.append({
                    "participant_id": int(participant_id[1:]),  # Strip 'P' and convert to int
                    "avgQRS_mean": avgQRS_mean,
                    "avgQRS_time_sec": avgQRS_time,
                    "kurtosis": kurtosis_val,
                    "kurtosis_time_sec": kurtosis_time,
                    "skewness": skewness_val,
                    "skewness_time_sec": skewness_time,
                    "sample_entropy": entropy_val,
                    "sample_entropy_time_sec": entropy_time
                })

            except Exception as e:
                print(f"Failed processing {filename}: {e}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"ECG SQI metrics saved to {output_csv}")


# Example usage:
input_folder = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/merged_lables"
output_folder = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/other/SQI_biopac_ECG"
batch_ecg_quality(input_folder, output_folder + "/ecg_sqi_results.csv", signal_column="ECG_X", sampling_rate=125)
