import numpy as np
import pandas as pd
import os
import glob
import re
import tkinter as tk
from tkinter import filedialog, simpledialog
from datetime import datetime, timedelta

# ==========================================================
# ---------- ORIGINAL FUNCTIONS (UNCHANGED) ----------
# ==========================================================

def detect_data_gaps(df, time_col='Timestamp', threshold_sec=2.0):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(by=time_col).reset_index(drop=True)
    df['TimeDiff'] = df[time_col].diff().dt.total_seconds()
    gaps = df[df['TimeDiff'] > threshold_sec]

    gap_info = []
    for idx, row in gaps.iterrows():
        start_time = df.loc[idx - 1, time_col]
        end_time = row[time_col]
        duration = row['TimeDiff']
        gap_info.append({
            'Gap Start': start_time,
            'Gap End': end_time,
            'Gap Duration (seconds)': duration
        })

    return pd.DataFrame(gap_info)


def calculate_sample_rate_consistency(timestamps):
    if len(timestamps) < 2:
        return 0
    timestamps_ns = timestamps.values.astype('datetime64[ns]').view('int64')
    time_diffs = np.diff(timestamps_ns) / 1e9
    median_interval = np.median(time_diffs)
    if median_interval == 0:
        return 1
    std_dev = np.std(time_diffs)
    dev_ratio = std_dev / median_interval
    return min(dev_ratio, 1.0)


def detect_outliers(hr_series, min_bpm=30, max_bpm=220):
    total_outliers = ((hr_series < min_bpm) | (hr_series > max_bpm)).sum()
    percentage_outliers = (total_outliers / hr_series.shape[0]) * 100
    return total_outliers, percentage_outliers


def detect_flatlines(hr_series, window=10):
    return (hr_series.rolling(window).std() == 0).sum()


def calculate_signal_quality(hr_series, timestamps):
    total = len(hr_series)
    outliers, percentage_outliers = detect_outliers(hr_series)
    flatlines = detect_flatlines(hr_series)
    consistency_penalty = calculate_sample_rate_consistency(timestamps)

    point_quality = 1 - (outliers + flatlines) / total
    consistency_factor = 1 - consistency_penalty
    quality_score = point_quality * consistency_factor

    return {
        "percentage_outliers": percentage_outliers,
        "outliers": outliers,
        "flatlines": flatlines,
        "sample_rate_penalty": consistency_penalty,
        "quality_score": max(0, min(1, quality_score))
    }


def clean_hr_signal(hr_series):
    hr_series = pd.to_numeric(hr_series, errors='coerce')
    hr_series = hr_series.interpolate(method='linear')
    hr_series = hr_series.ffill().bfill()
    hr_series = hr_series.round().astype(int)
    return hr_series


def calculate_hr_zones(hr_series):
    bins = [0, 40, 60, 80, 100, 120, 140, 160, 180, 200, np.inf]

    range_labels = [
        "HR Range: 0–40 bpm",
        "HR Range: 40–60 bpm",
        "HR Range: 60–80 bpm",
        "HR Range: 80–100 bpm",
        "HR Range: 100–120 bpm",
        "HR Range: 120–140 bpm",
        "HR Range: 140–160 bpm",
        "HR Range: 160–180 bpm",
        "HR Range: 180–200 bpm",
        "HR Range: >200 bpm"
    ]

    hr_categorized = pd.cut(
        hr_series,
        bins=bins,
        labels=range_labels,
        include_lowest=True,
        right=False
    )

    zone_counts = hr_categorized.value_counts().reindex(range_labels, fill_value=0)
    return zone_counts.to_dict()


# ==========================================================
# ---------- PARTICIPANT ID PARSER ----------
# ==========================================================

def extract_carewear_participant_id(file_name):
    match = re.search(r'heart_rate_(\d+)_', file_name)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract participant ID from {file_name}")


# ==========================================================
# ---------- DATASET SELECTION ----------
# ==========================================================

root = tk.Tk()
root.withdraw()

dataset_choice = simpledialog.askstring(
    "Dataset Selection",
    "Type dataset name:\n\n1 = CareWear\n2 = GalaxyPPG"
)

if dataset_choice not in ["1", "2"]:
    raise Exception("Invalid selection. Exiting.")

data_folder = filedialog.askdirectory(title="Select Dataset Root Folder")
root.destroy()

if not data_folder:
    raise Exception("No folder selected. Exiting script.")


# ==========================================================
# ---------- FILE PARSING ----------
# ==========================================================

if dataset_choice == "1":
    file_pattern = os.path.join(data_folder, "heart_rate*.csv")
    file_list = glob.glob(file_pattern)
    dataset_type = "CareWear"
else:
    # Only search for CSVs inside GalaxyWatch subfolders
    file_list = glob.glob(os.path.join(data_folder, "P*/GalaxyWatch/*.csv"), recursive=True)
    dataset_type = "GalaxyPPG"


# ==========================================================
# ---------- PROCESSING ----------
# ==========================================================

expected_lables_list = [
    'rest1', 'rest3', "stationary_Bike1",
    'stationary_Bike2', 'prepare speech',
    'rest2', 'give speech', 'mental math'
]

all_features = []

for file_path in file_list:
    try:
        file_name = os.path.basename(file_path)

        if dataset_type == "CareWear":
            participant_id = extract_carewear_participant_id(file_name)
        else:
            # ---------- GALAXYPPG PARTICIPANT PARSING ----------
            # Extract participant ID from the parent folder of GalaxyWatch
            participant_folder = os.path.dirname(os.path.dirname(file_path))  # .../P01/GalaxyWatch
            participant_basename = os.path.basename(participant_folder)
            match = re.match(r'P0*(\d+)', participant_basename, re.IGNORECASE)
            if match:
                participant_id = int(match.group(1))
            else:
                # Skip this file if participant folder does not match P<number>
                print(f"Skipping file {file_path} because participant folder not found")
                continue

        df_raw = pd.read_csv(file_path)

        # -------- GalaxyPPG Handling --------
        if dataset_type == "GalaxyPPG":
            hr_cols = [c for c in df_raw.columns if "hr" in c.lower()]
            if not hr_cols:
                continue

            df_raw["HeartRate"] = df_raw[hr_cols[0]]

            if "Timestamp" not in df_raw.columns:
                start_time = datetime.now()
                df_raw["Timestamp"] = [
                    start_time + timedelta(seconds=i)
                    for i in range(len(df_raw))
                ]

        df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Timestamp'])

        # ======================================================
        # ---------------- CAREWEAR FULL LOGIC -----------------
        # ======================================================
        if dataset_type == "CareWear":

            df = df_raw[df_raw['manual_labels_activity'].isin(expected_lables_list)]

            hr_clean = clean_hr_signal(df["HeartRate"])
            report = calculate_signal_quality(hr_clean, df["Timestamp"])
            hr_zone_counts = calculate_hr_zones(hr_clean)

            feature_row = {
                "FileName": file_name,
                "Participant": participant_id,
                "Activity": "Overall",
                "Total Samples": len(df),
                "Outliers": report["outliers"],
                "Percentage Outliers": report["percentage_outliers"],
                "Flatlines": report["flatlines"],
                "Sample Rate Penalty": report["sample_rate_penalty"],
                "Quality Score": report["quality_score"]
            }

            feature_row.update(hr_zone_counts)
            all_features.append(feature_row)

            for activity in expected_lables_list:
                df_activity = df[df['manual_labels_activity'] == activity]
                if df_activity.empty:
                    continue

                hr_clean_a = clean_hr_signal(df_activity["HeartRate"])
                report_a = calculate_signal_quality(hr_clean_a, df_activity["Timestamp"])
                hr_zone_counts_a = calculate_hr_zones(hr_clean_a)

                feature_row_a = {
                    "FileName": file_name,
                    "Participant": participant_id,
                    "Activity": activity,
                    "Total Samples": len(df_activity),
                    "Outliers": report_a["outliers"],
                    "Percentage Outliers": report_a["percentage_outliers"],
                    "Flatlines": report_a["flatlines"],
                    "Sample Rate Penalty": report_a["sample_rate_penalty"],
                    "Quality Score": report_a["quality_score"]
                }

                feature_row_a.update(hr_zone_counts_a)
                all_features.append(feature_row_a)

        # ======================================================
        # ---------------- GALAXYPPG ---------------------------
        # ======================================================
        else:

            df = df_raw.copy()
            hr_clean = clean_hr_signal(df["HeartRate"])
            report = calculate_signal_quality(hr_clean, df["Timestamp"])
            hr_zone_counts = calculate_hr_zones(hr_clean)

            feature_row = {
                "FileName": file_name,
                "Participant": participant_id,
                "Activity": "Overall",
                "Total Samples": len(df),
                "Outliers": report["outliers"],
                "Percentage Outliers": report["percentage_outliers"],
                "Flatlines": report["flatlines"],
                "Sample Rate Penalty": report["sample_rate_penalty"],
                "Quality Score": report["quality_score"]
            }

            feature_row.update(hr_zone_counts)
            all_features.append(feature_row)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


# ==========================================================
# ---------- SAVE OUTPUT ----------
# ==========================================================

output_folder = os.path.join(data_folder, "quality_reports")
os.makedirs(output_folder, exist_ok=True)

features_df = pd.DataFrame(all_features)
features_df.sort_values(["Participant", "Activity"], inplace=True)

features_df.to_csv(
    os.path.join(output_folder, "heart_rate_quality_features.csv"),
    index=False
)

print(f"\nProcessing complete for {dataset_type}.")
print("Feature reports saved successfully.")