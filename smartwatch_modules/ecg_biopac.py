import heartpy as hp
import pandas as pd
import numpy as np 

OUTPUT_FOLDER = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/ecg_biopac_hr"
filename = "P16_biopac.csv"
df = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File/"+filename)

def get_ecg_hr(df):
    fs = 125
    ecg = df["ECG_X"].values
    working_data, measures = hp.process(ecg, fs)
    print("working_data")
    print(working_data)
    peak_indices = working_data['peaklist']

    peak_times = np.array(peak_indices) / fs
    rr_intervals = np.diff(peak_times)
    inst_hr = 60 / rr_intervals
    hr_times = peak_times[1:]

    hr_series = pd.Series(inst_hr, index=hr_times)

    # FIX: Convert index to TimedeltaIndex
    hr_series.index = pd.to_timedelta(hr_series.index, unit='s')

    # Now resample to 1 Hz
    hr_1hz = hr_series.resample('1S').mean().interpolate()
    # hr_1hz.to_csv(OUTPUT_FOLDER + "/ecg_hr_" +filename, header=['HR_BPM'])
    print("hr_1hz:====")
    print(hr_1hz)
    return hr_series,peak_indices
