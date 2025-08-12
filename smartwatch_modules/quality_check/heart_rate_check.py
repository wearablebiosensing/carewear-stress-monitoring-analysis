import numpy as np
import pandas as pd
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog
from datetime import datetime, timedelta

# ---------- Functions ----------

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
    print("===========================Timegaps Info ========================")
    print(gap_info)
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

def detect_flatlines(hr_series, window=5):
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
    hr_series = hr_series.fillna(method='bfill').fillna(method='ffill')
    hr_series = hr_series.round().astype(int)
    return hr_series

# ---------- File Selection ----------
root = tk.Tk()
root.withdraw()
data_folder = filedialog.askdirectory(title="Select your Concat_File folder")
root.destroy()
if not data_folder:
    raise Exception("No folder selected. Exiting script.")

# ---------- File Load ----------
file_pattern = os.path.join(data_folder, "heart_rate*.csv")
file_list = glob.glob(file_pattern)
expected_lables_list = ['rest1', 'rest3', "stationary_Bike1", 'stationary_Bike2', 'prepare speech', 'rest2', 'give speech', 'mental math']

quality_scores = []
outlier_flatline_scores = []
gaps_df_strict_list = []
activity_quality = []
activity_flatlines = []
activity_outliers = []

# ---------- Data Processing ----------
for file_path in file_list:
    try:
        participant_id = os.path.basename(file_path).split("_")[2].replace(".csv", "")
        df_raw = pd.read_csv(file_path)
        df_raw['Timestamp'] = pd.to_datetime(df_raw['Timestamp'], errors='coerce')
        df_raw = df_raw.dropna(subset=['Timestamp'])

        df = df_raw[df_raw['manual_labels_activity'].isin(expected_lables_list)]

        # Time gap detection
        gaps_df_strict = detect_data_gaps(df, time_col='Timestamp', threshold_sec=5.0)
        gaps_df_strict["pid"] = participant_id
        gaps_df_strict_list.append(gaps_df_strict)

        # Overall metrics
        hr_clean = clean_hr_signal(df["HeartRate"])
        report = calculate_signal_quality(hr_clean, df["Timestamp"])
        quality_scores.append({
            "Participant": participant_id,
            "Quality Score": report["quality_score"]
        })
        outlier_flatline_scores.append({
            "Participant": participant_id,
            "Outliers": report["outliers"],
            "Flatlines": report["flatlines"],
            "percentage_outliers": report["percentage_outliers"]
        })

        # Activity-wise metrics
        for activity in expected_lables_list:
            df_activity = df[df['manual_labels_activity'] == activity]
            if df_activity.empty:
                continue
            hr_clean_a = clean_hr_signal(df_activity["HeartRate"])
            report_a = calculate_signal_quality(hr_clean_a, df_activity["Timestamp"])
            activity_quality.append({"Participant": participant_id, "Activity": activity, "Quality Score": report_a["quality_score"]})
            activity_flatlines.append({"Participant": participant_id, "Activity": activity, "Flatlines": report_a["flatlines"]})
            activity_outliers.append({"Participant": participant_id, "Activity": activity, "percentage_outliers": report_a["percentage_outliers"]})

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# ---------- DataFrames ----------
quality_df = pd.DataFrame(quality_scores).sort_values("Participant")
outlier_flatline_df = pd.DataFrame(outlier_flatline_scores).sort_values("Participant")

df_q = pd.DataFrame(activity_quality)
df_f = pd.DataFrame(activity_flatlines)
df_o = pd.DataFrame(activity_outliers)

# ---------- Plotting ----------
fig = make_subplots(rows=3, cols=2, shared_xaxes=True, vertical_spacing=0.13,
    subplot_titles=("Overall: Quality Score", "Per Activity: Quality Score",
                    "Overall: % Outliers", "Per Activity: % Outliers",
                    "Overall: Flatlines", "Per Activity: Flatlines"))

# --- Left Column ---
fig.add_trace(go.Bar(x=quality_df["Participant"], y=quality_df["Quality Score"],
    name="Overall Quality", marker_color='darkgreen', text=quality_df["Quality Score"].round(2),
    textposition='outside'), row=1, col=1)

fig.add_trace(go.Bar(x=outlier_flatline_df["Participant"], y=outlier_flatline_df["percentage_outliers"],
    name="Overall Outliers %", marker_color='indianred', text=outlier_flatline_df["percentage_outliers"].round(2),
    textposition='outside'), row=2, col=1)

fig.add_trace(go.Bar(x=outlier_flatline_df["Participant"], y=outlier_flatline_df["Flatlines"],
    name="Overall Flatlines", marker_color='mediumblue', text=outlier_flatline_df["Flatlines"],
    textposition='outside'), row=3, col=1)

# --- Right Column (Activity-Based) ---
for activity in expected_lables_list:
    df_act_q = df_q[df_q["Activity"] == activity]
    fig.add_trace(go.Bar(x=df_act_q["Participant"], y=df_act_q["Quality Score"], name=activity), row=1, col=2)

    df_act_o = df_o[df_o["Activity"] == activity]
    fig.add_trace(go.Bar(x=df_act_o["Participant"], y=df_act_o["percentage_outliers"], name=activity, showlegend=False), row=2, col=2)

    df_act_f = df_f[df_f["Activity"] == activity]
    fig.add_trace(go.Bar(x=df_act_f["Participant"], y=df_act_f["Flatlines"], name=activity, showlegend=False), row=3, col=2)

# ---------- Final Layout ----------
fig.update_layout(
    height=1800,
    width=2200,
    barmode="group",
    showlegend=True,
    title_text="Heart Rate Signal Quality Metrics (Overall and Activity-Based)",
    plot_bgcolor="lightgray",
    title_font=dict(size=22),
    uniformtext_minsize=8,
    uniformtext_mode='hide'
)

fig.update_yaxes(title="Quality Score (0–1)", row=1, col=1)
fig.update_yaxes(title="Quality Score (0–1)", row=1, col=2)
fig.update_yaxes(title="Percentage", row=2, col=1)
fig.update_yaxes(title="Percentage", row=2, col=2)
fig.update_yaxes(title="Flatlines Count", row=3, col=1)
fig.update_yaxes(title="Flatlines Count", row=3, col=2)

fig.update_xaxes(title="Participant ID", row=3, col=1)
fig.update_xaxes(title="Participant ID", row=3, col=2)

fig.show()
