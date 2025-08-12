import pandas as pd
import numpy as np
import neurokit2 as nk

def get_ecg_hr_neurokit(df):
    df = df[~df.astype(str).apply(lambda row: row.str.contains("samples", case=False, na=False)).any(axis=1)]

    df["ECG_X"] = df["ECG_X"].astype(float)
    
    fs = 125  # Sampling frequency of ECG in Hz
    ecg_signal = df["ECG_X"].values

    # Process ECG signal with NeuroKit
    signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs)
    # # Visualise the processing
    # nk.ecg_plot(signals, info)

    # Get the R-peaks (sample indices)
    rpeaks = info["ECG_R_Peaks"]
    
    # Calculate RR intervals in seconds
    rr_intervals = np.diff(rpeaks) / fs
    rr_diff_ms = rr_intervals * 1000

    print("RR intervals (ms):", rr_diff_ms)
    num_rr_intervals_short = np.sum(rr_diff_ms < 250)
    print(f"Short RR intervals (<250ms): {num_rr_intervals_short}")
    print(f"Minimum RR interval: {rr_diff_ms.min():.2f} ms")

    # Instantaneous HR in BPM
    inst_hr = 60 / rr_intervals
    hr_times = rpeaks[1:] / fs  # Corresponding time stamps (in seconds)

    # # Create HR Series and set index as time in seconds
    hr_series = pd.Series(inst_hr, index=pd.to_timedelta(hr_times, unit='s'))

    # Resample to 1Hz and interpolate missing values
    hr_1hz = hr_series.resample('1S').mean().interpolate()

    print("hr_1hz:")
    print(hr_1hz)
    return hr_series, rpeaks, num_rr_intervals_short
df = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File/P2_prepare_biopac.csv")
get_ecg_hr_neurokit(df)