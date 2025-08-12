import numpy as np
import pandas as pd
import os
import os
import glob
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_heart_rate_quality(df, timestamp_col='Timestamp', hr_col='HeartRate'):
    """
    Analyze heart rate data for timestamp gaps and abnormal values.
    
    Parameters:
    - df: pandas DataFrame with 'HeartRate' and 'Timestamp' columns
    - timestamp_col: name of timestamp column in datetime format
    - hr_col: name of heart rate column
    
    Returns:
    - df: original DataFrame with added 'validity' column: ['valid', 'gap', 'error']
    """
    # Ensure timestamp is datetime
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Sort by time
    df = df.sort_values(by=timestamp_col).reset_index(drop=True)
    
    # Initialize validity column
    df['validity'] = 'valid'

    # Compute time difference in seconds
    time_diff = df[timestamp_col].diff().dt.total_seconds()

    # Mark gaps (more than 1 second difference)
    df.loc[time_diff > 1.0, 'validity'] = 'gap'

    # Heart rate values
    hr_values = df[hr_col].values

    # Mark abnormal values
    for i in range(10, len(df) - 10):
        if df.loc[i, 'validity'] == 'gap':
            continue  # Skip already marked gaps
        neighborhood = np.concatenate([hr_values[i-10:i], hr_values[i+1:i+11]])
        mean_neighbors = np.mean(neighborhood)
        if abs(hr_values[i] - mean_neighbors) > 300:
            df.loc[i, 'validity'] = 'error'

    return df

# This function measures how consistent the time intervals are between 
# consecutive samples in your data (i.e., how regular your sample rate is).
# Returns:  A number between 0 and 1 that tells you how "jittery" your sample timing is. Lower is better!
def calculate_sample_rate_consistency(timestamps):
    """Calculate sample rate consistency using timestamp differences [1][5]"""
    if len(timestamps) < 2:
        return 0  # Not enough data
    # Convert to numpy datetime64 in nanoseconds
    timestamps_ns = timestamps.values.astype('datetime64[ns]').view('int64')
    time_diffs = np.diff(timestamps_ns) / 1e9  # Convert to seconds
    
    # Calculate consistency metrics [1][4]
    median_interval = np.median(time_diffs)
    if median_interval == 0:
        return 1  # Invalid data
    std_dev = np.std(time_diffs)
    print("std_dev: c",std_dev)
    dev_ratio = std_dev / median_interval
    print("dev_ratio: ",dev_ratio)
    print("consistency_penalty",min(dev_ratio, 1.0))

    return min(dev_ratio, 1.0)  # Cap at 100% penalty

def detect_outliers(hr_series, min_bpm=30, max_bpm=220):
    print("detect_outliers =",((hr_series < min_bpm) | (hr_series > max_bpm)).sum())
    return ((hr_series < min_bpm) | (hr_series > max_bpm)).sum()

def detect_flatlines(hr_series, window=5):
    print("flat lines: ",(hr_series.rolling(window).std() == 0).sum())
    return (hr_series.rolling(window).std() == 0).sum()

def calculate_signal_quality(hr_series,timestamps):
    total = len(hr_series)
    # dropouts = detect_dropouts(hr_series)
    outliers = detect_outliers(hr_series)
    flatlines = detect_flatlines(hr_series)
    """Calculate comprehensive quality score with sample rate consistency"""
    # Sample rate consistency check
    consistency_penalty = calculate_sample_rate_consistency(timestamps)
    # quality_score = 1 - (outliers + flatlines) / total
      # Calculate component scores [2][4]
    point_quality = 1 - (outliers + flatlines) / total
    consistency_factor = 1 - consistency_penalty
    
    # Weighted final score (adjust weights as needed)     quality_score = (point_quality * 0.5) + (consistency_factor * 0.5)

    quality_score = (point_quality ) + (consistency_factor)

    return {
        "outliers": outliers,
        "flatlines": flatlines,
        "sample_rate_penalty": consistency_penalty,
        "quality_score": max(0, min(1, quality_score))
    }

def clean_hr_signal(hr_series):
    # Convert everything to numeric (coerce non-numeric to NaN)
    hr_series = pd.to_numeric(hr_series, errors='coerce')

    # Replace 0s with NaN (dropouts)
    # hr_series = hr_series.replace(0, np.nan)

    # Interpolate to fill missing values
    hr_series = hr_series.interpolate(method='linear')
    # Backward and forward fill edge NaNs
    hr_series = hr_series.fillna(method='bfill').fillna(method='ffill')

    # Convert to int (optional, if you want integer heart rate values)
    hr_series = hr_series.round().astype(int)
    return hr_series

# Path to your data folder
# data_folder = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Concat_File"
# --- Add this before you set data_folder ---
root = tk.Tk()
root.withdraw()  # Hide the main window
data_folder = filedialog.askdirectory(title="Select your Concat_File folder")
root.destroy()

if not data_folder:
    raise Exception("No folder selected. Exiting script.")


file_pattern = os.path.join(data_folder, "heart_rate*.csv")
file_list = glob.glob(file_pattern)

# Store participant scores
quality_scores = []
# Store participant scores
quality_scores = []
outlier_flatline_scores = []

for file_path in file_list:
    try:
        participant_id = os.path.basename(file_path).split("_")[-1].replace(".csv", "")
        # Step 1: Read CSV normally (don't try to parse dates yet)
        df = pd.read_csv(file_path)

        # Step 2: Convert the 'Timestamp' column to datetime, coercing errors to NaT
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

        # Step 3: Drop rows with invalid timestamps (NaT)
        df = df.dropna(subset=['Timestamp'])
        # Step 3: Drop rows with invalid timestamps
        df = df.dropna(subset=['Timestamp'])
        hr_clean = clean_hr_signal(df["HeartRate"])
        report = calculate_signal_quality(hr_clean,df["Timestamp"])
        quality_scores.append({
            "Participant": participant_id,
            "Quality Score": report["quality_score"]
        })
        # Collect outliers and flatlines
        outlier_flatline_scores.append({
            "Participant": participant_id,
            "Outliers": report["outliers"],
            "Flatlines": report["flatlines"]
        })
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Create DataFrame and plot
quality_df = pd.DataFrame(quality_scores).sort_values("Participant")
# Sort by numeric value of participant ID
quality_df["Participant_Num"] = quality_df["Participant"].str.extract(r'P(\d+)').astype(int)
quality_df = quality_df.sort_values("Participant_Num")
outlier_flatline_df = pd.DataFrame(outlier_flatline_scores)
outlier_flatline_df["Participant_Num"] = outlier_flatline_df["Participant"].str.extract(r'P(\d+)').astype(int)
outlier_flatline_df = outlier_flatline_df.sort_values("Participant_Num")
# Reshape for grouped bar plot
melted_df = outlier_flatline_df.melt(id_vars=["Participant"], value_vars=["Outliers", "Flatlines"],
                                     var_name="Metric", value_name="Count")

df = pd.read_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/Concat_File/heart_rate_P3.csv")

# Drop rows where 'HeartRate' column contains "HeartRate", "Timestamp", or is non-numeric
df = df[~df['HeartRate'].astype(str).str.contains('HeartRate|Timestamp', na=False)]

# Convert HeartRate to float
df['HeartRate'] = pd.to_numeric(df['HeartRate'], errors='coerce')
df = df[df['HeartRate'].notna()]

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
df = df[df['Timestamp'].notna()]
df_gaps_marked = analyze_heart_rate_quality(df, timestamp_col='Timestamp', hr_col='HeartRate')
df_gaps_marked.to_csv("/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/heart_rate_gaps_P3.csv",index=0)
print("Original DF = ",df.shape)
print("Gaps DF = ",df_gaps_marked.shape)

fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            subplot_titles=("HR with gaps P3", " Raw HR")
        )
fig.add_trace(go.Scattergl(
            x=df_gaps_marked['Timestamp_pd'],
            y=df_gaps_marked['HeartRate'],
            mode='markers+lines',
            marker=dict(size=8, color='#1d8cad'),
            line=dict(dash='dash', color='#1d8cad'),
            name=" "
        ),
         row=1, col=1
        )
fig.add_trace(go.Scattergl(
            x=df['Timestamp_pd'],
            y=df['HeartRate'],
            mode='markers+lines',
            marker=dict(size=8, color='#1d8cad'),
            line=dict(dash='dash', color='#1d8cad'),
            name=""
        ),
         row=2, col=1
        )
fig.show()