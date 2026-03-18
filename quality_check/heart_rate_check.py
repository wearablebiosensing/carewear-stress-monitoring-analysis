import numpy as np
import pandas as pd
import os
import glob
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from datetime import timedelta


print("\n==============================")
print("Universal Heart Rate Pipeline")
print("==============================\n")


# ==========================================================
# SIGNAL QUALITY
# ==========================================================

def detect_outliers(hr_series, min_bpm=30, max_bpm=220):

    total_outliers = ((hr_series < min_bpm) | (hr_series > max_bpm)).sum()

    percentage_outliers = (
        total_outliers / len(hr_series) * 100 if len(hr_series) > 0 else 0
    )

    return total_outliers, percentage_outliers

def estimate_sampling_rate(df):
    
    diffs = df["Timestamp"].diff().dt.total_seconds().dropna()

    if len(diffs) == 0:
        return 1

    median_interval = diffs.median()

    if median_interval <= 0:
        return 1

    Fs = 1 / median_interval

    return Fs

def detect_flatlines(hr_series, window=10):

    return hr_series.rolling(window).std() == 0


def calculate_sample_rate_consistency(timestamps):

    if len(timestamps) < 2:
        return 0

    timestamps_ns = timestamps.values.astype("datetime64[ns]").view("int64")

    diffs = np.diff(timestamps_ns) / 1e9

    median_interval = np.median(diffs)

    if median_interval <= 0:
        return 1

    std_dev = np.std(diffs)

    dev_ratio = std_dev / median_interval

    return min(dev_ratio, 1.0)


def calculate_signal_quality(hr_series, timestamps):

    total = len(hr_series)

    outliers, percentage_outliers = detect_outliers(hr_series)

    flatline_mask = detect_flatlines(hr_series)

    flatline_count = flatline_mask.sum()

    flatline_percent = flatline_count / total * 100 if total > 0 else 0

    diffs = timestamps.diff().dt.total_seconds().fillna(0)

    flatline_duration = diffs[flatline_mask].sum()

    consistency_penalty = calculate_sample_rate_consistency(timestamps)

    if total > 1:
        duration = (timestamps.max() - timestamps.min()).total_seconds()

        sample_rate_hz = total / duration if duration > 0 else 0

    else:

        sample_rate_hz = 0

    point_quality = 1 - (outliers + flatline_count) / total if total > 0 else 0

    consistency_factor = 1 - consistency_penalty

    quality_score = point_quality * consistency_factor

    return {
        "percentage_outliers": percentage_outliers,
        "outliers": outliers,
        "flatlines": flatline_count,
        "flatline_percent": flatline_percent,
        "flatline_duration_sec": flatline_duration,
        "sample_rate_hz": sample_rate_hz,
        "sample_rate_penalty": consistency_penalty,
        "sample_rate_consistency": consistency_factor,
        "quality_score": max(0, min(1, quality_score)),
    }


# ==========================================================
# CLEAN HR
# ==========================================================

def clean_hr_signal(hr):

    hr = pd.to_numeric(hr, errors="coerce")

    hr = hr.interpolate()

    hr = hr.ffill().bfill()

    return hr.round().astype(int)


# ==========================================================
# HR ZONES
# ==========================================================

def calculate_hr_zones(hr):

    bins = [0,40,60,80,100,120,140,160,180,200,np.inf]

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


# ==========================================================
# TIMESTAMP NORMALIZATION
# ==========================================================

def normalize_timestamp(df):

    for col in [
        "Timestamp",
        "datetime",
        "timestamp",
        "watch_timestamp",
        "internal_ts",
    ]:

        if col in df.columns:

            df["Timestamp"] = pd.to_datetime(df[col], errors="coerce")

            return df

    raise ValueError("No timestamp column detected")


# ==========================================================
# HR COLUMN STANDARDIZATION
# ==========================================================

def standardize_hr_column(df):

    for col in ["HeartRate", "hr", "bpm"]:

        if col in df.columns:

            df["HeartRate"] = df[col]

            return df

    raise ValueError("No HR column found")


# ==========================================================
# PARTICIPANT EXTRACTION
# ==========================================================

def extract_participant_id(file_name):
    patterns = [
        r"_hr_(\d+)",           # Updated CareWear pattern (matches _hr_5_)
        r"heart_rate_(\d+)",    # Original CareWear/Generic
        r"P(\d+)",              # GalaxyPPG
        r"pid_(\d+)",           # Zenodo
    ]

    for p in patterns:
        match = re.search(p, file_name)
        if match:
            return int(match.group(1))

    return None

# ==========================================================
# SAMPLE-BASED WINDOWING
# ==========================================================
def sample_based_windowing(df, participant, activity, file_name,
                           window_seconds, overlap_percent):

    rows = []

    df = df.sort_values("Timestamp").reset_index(drop=True)

    Fs = estimate_sampling_rate(df)

    samples_per_window = int(Fs * window_seconds)

    if samples_per_window < 5:
        samples_per_window = 5

    step = int(samples_per_window * (1 - overlap_percent / 100))

    start = 0
    window_id = 0

    while start + samples_per_window <= len(df):

        window_df = df.iloc[start:start+samples_per_window]

        hr = clean_hr_signal(window_df["HeartRate"])

        report = calculate_signal_quality(hr, window_df["Timestamp"])

        zones = calculate_hr_zones(hr)

        row = {
            "FileName": file_name,
            "Participant": participant,
            "Activity": activity,
            "WindowNumber": window_id,
            "WindowStart": window_df["Timestamp"].iloc[0],
            "WindowEnd": window_df["Timestamp"].iloc[-1],
            "Total Samples": len(window_df),
        }

        row.update(report)
        row.update(zones)

        rows.append(row)

        window_id += 1

        start += step

    return rows
# ==========================================================
# MAIN
# ==========================================================

root = tk.Tk()
root.withdraw()

dataset = simpledialog.askstring(
    "Dataset",
    "1 = CareWear\n2 = GalaxyPPG\n3 = Zenodo"
)

dataset_map = {"1":"CareWear","2":"GalaxyPPG","3":"Zenodo"}

dataset_name = dataset_map.get(dataset,"Dataset")

window_seconds = simpledialog.askinteger("Window","Seconds",initialvalue=2)

overlap = simpledialog.askinteger("Overlap","%",initialvalue=50)

data_folder = filedialog.askdirectory(title="Select dataset folder")

root.destroy()

files = glob.glob(os.path.join(data_folder,"*.csv"))

print("Files detected:",len(files))

all_rows = []

for file_path in files:

    file_name = os.path.basename(file_path)

    if "label_mapping" in file_name:
        continue

    try:

        df = pd.read_csv(file_path)

        df = standardize_hr_column(df)

        df = normalize_timestamp(df)

        df = df.dropna(subset=["Timestamp","HeartRate"])

        participant = extract_participant_id(file_name)

        if participant is None:
            print("Participant not found:",file_name)
            continue

        activity_col = None

        for c in [
            "manual_labels_activity",
            "activity",
            "label",
            "segment",
            "int_session",
        ]:
            if c in df.columns:
                activity_col = c
                break

        if activity_col:

            for act,g in df.groupby(activity_col):

                rows = sample_based_windowing(
                    g,
                    participant,
                    act,
                    file_name,
                    window_seconds,
                    overlap,
                )

                all_rows.extend(rows)

        else:

            rows = sample_based_windowing(
                df,
                participant,
                "Overall",
                file_name,
                window_seconds,
                overlap,
            )
            all_rows.extend(rows)

        print("Processed:",file_name)

    except Exception as e:

        print("Error:",file_name,e)


# ==========================================================
# SAVE
# ==========================================================

output_folder = os.path.join(data_folder,"quality_reports")

os.makedirs(output_folder,exist_ok=True)

features = pd.DataFrame(all_rows)

if len(features) == 0:

    print("No data generated")

    exit()

features.sort_values(["Participant","Activity"],inplace=True)

output_file = f"{dataset_name}_heart_rate_quality_features_{window_seconds}s_{overlap}overlap.csv"

features.to_csv(os.path.join(output_folder,output_file),index=False)

print("\n==============================")

print("Rows generated:",len(features))

print("Unique participants:",features["Participant"].nunique())

print("Saved file:",output_file)

print("==============================")