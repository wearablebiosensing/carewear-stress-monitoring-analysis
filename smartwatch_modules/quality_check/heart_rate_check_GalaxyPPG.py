import numpy as np
import pandas as pd
import os
import glob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog

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

    # quality_score = point_quality + consistency_factor
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

# GUI for folder selection
root = tk.Tk()
root.withdraw()
data_folder = filedialog.askdirectory(title="Select the root folder containing participant folders (e.g., P01, P02, ...)")
root.destroy()

if not data_folder:
    raise Exception("No folder selected. Exiting script.")

# Find all HR.csv files in GalaxyWatch subfolders of each participant
file_pattern = os.path.join(data_folder, "P*", "GalaxyWatch", "HR.csv")
file_list = glob.glob(file_pattern)

quality_scores = []
outlier_flatline_scores = []

for file_path in file_list:
    try:
        # Extract participant ID (e.g., "P02")
        participant_id = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        df = pd.read_csv(file_path)
    
        print("=========================="+ "pid = " + participant_id +"======================")
        print(df.head())
        # Adjust these column names if your HR.csv uses different headers
        # Common headers: 'Timestamp' or 'timestamp', 'HeartRate' or 'hr'
        # Try to auto-detect
        timestamp_col = None
        hr_col = None
        for col in df.columns:
            if col.lower() in ['timestamp', 'time', 'datetime']:
                timestamp_col = col
            if col.lower() in ['heartrate', 'hr', 'bpm']:
                hr_col = col
        if not timestamp_col or not hr_col:
            raise ValueError(f"Could not find timestamp or heart rate columns in {file_path}")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors='coerce')
        df = df.dropna(subset=[timestamp_col])
        hr_clean = clean_hr_signal(df[hr_col])
        report = calculate_signal_quality(hr_clean, df[timestamp_col])
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
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Create DataFrame and plot
quality_df = pd.DataFrame(quality_scores).sort_values("Participant")
quality_df["Participant_Num"] = quality_df["Participant"].str.extract(r'P(\d+)').astype(int)
quality_df = quality_df.sort_values("Participant_Num")
outlier_flatline_df = pd.DataFrame(outlier_flatline_scores)
outlier_flatline_df["Participant_Num"] = outlier_flatline_df["Participant"].str.extract(r'P(\d+)').astype(int)
outlier_flatline_df = outlier_flatline_df.sort_values("Participant_Num")

# Reshape for grouped bar plot
melted_df = outlier_flatline_df.melt(id_vars=["Participant"], value_vars=["Outliers", "Flatlines"],
                                     var_name="Metric", value_name="Count")

# Plotting
trace1 = go.Bar(
    x=quality_df["Participant"],
    y=quality_df["Quality Score"],
    text=quality_df["Quality Score"].round(2),
    textposition='outside',
    marker_color='rgba(7, 55, 99, 1)',
    name="Quality Score"
)
outlier_trace = go.Bar(
    x=outlier_flatline_df["Participant"],
    y=outlier_flatline_df["percentage_outliers"],
    name="Outliers",
    marker_color='rgb(191,144,0,100)',
    text=outlier_flatline_df["percentage_outliers"],
    textposition='outside'
)
flatline_trace = go.Bar(
    x=outlier_flatline_df["Participant"],
    y=outlier_flatline_df["Flatlines"],
    name="Flatlines",
    marker_color='rgba(150,94,235,1)',
    text=outlier_flatline_df["Flatlines"],
    textposition='outside'
)

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.15,
    subplot_titles=("Galaxy PPG Heart Rate: Signal quality score (0-1)",
                    "Heart Rate: Total number of outliers (min_bpm=30, max_bpm=220)",
                    "Heart Rate: Total number of flat lines (window=5)")
)
fig.add_trace(trace1, row=1, col=1)
fig.add_trace(outlier_trace, row=2, col=1)
fig.add_trace(flatline_trace, row=3, col=1)

fig.update_layout(
    height=1800,
    width=1200,
    showlegend=True,
    barmode="group",
    uniformtext_minsize=8,
    uniformtext_mode='hide',
    title_font=dict(size=20),
    plot_bgcolor='lightgray',
)
fig.update_yaxes(range=[0, 1.05], title_font=dict(size=20), tickfont=dict(size=22), row=1, col=1, title="Quality Score (0–1)")
fig.update_xaxes(title="Participant ID", title_font=dict(size=20), tickfont=dict(size=22), showticklabels=True, row=1, col=1)
fig.update_yaxes(title="Percentage", title_font=dict(size=20), tickfont=dict(size=22), row=2, col=1)
fig.update_xaxes(title="Participant ID", title_font=dict(size=20), tickfont=dict(size=22), showticklabels=True, row=2, col=1)
fig.update_yaxes(title="Count", title_font=dict(size=20), tickfont=dict(size=22), row=3, col=1)
fig.update_xaxes(title="Participant ID", title_font=dict(size=20), tickfont=dict(size=22), row=3, col=1)

fig.show()
