# Appends experimenter data labels to
TAG = "TASK TIMELINE MERGE"
#!/usr/bin/env python3
import pandas as pd
import glob
import os
import plotly.express as px
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# --- Constants ---
# Use pathlib for more robust path handling, or keep os.path for now
INPUT_DIR = "/Volumes/CW_2024/Concat_File"
OUTPUT_DIR = "/Volumes/CW_2024/merged_lables"
MANUAL_LABELS_DIR = "/Volumes/CW_2024/Task_Time_Line_Manual"
BELT_LABELS_DIR = "/Volumes/CW_2024/Task_Time_Line_Belt"

# Define the mapping dictionary
activity_mapping = {
    'Stationary_Bike_Legs': 'stationary_Bike1',
    'Stationary_Bike_Hand': 'stationary_Bike2',
    'Prepare_Speech': 'prepare speech',
    'Rest_1': 'rest1',
    'Rest_2': 'rest2',
    'Rest_3': 'rest3',
    'Give_Speech': 'give speech',
    'Mental_Math': 'mental math',
    'activity': 'activity',
    'None': 'None',
    "Stationary_Bike_ke_Hand": "stationary_Bike2",
    "Stationary_BiHand": "stationary_Bike2",
    np.nan: np.nan
}

def map_activity_value(x):
    """Maps a single activity value using the predefined dictionary."""
    return activity_mapping.get(x, x)

def count_class_samples(df, class_column):
    """
    Returns the count of samples in each unique class in the specified column,
    excluding the class 'None' (as a string or actual None/NaN).
    """
    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found in DataFrame.")
    
    filtered = df[~df[class_column].isin(['None']) & df[class_column].notna()]
    return filtered[class_column].value_counts()

def add_activity_labels(df, labels_df, base_date, label_col_name, activity_col):
    """Assigns activity labels to a DataFrame based on start/end times from a labels DataFrame."""
    # Convert and align label times
    for col in ('start_time', 'end_time'):
        labels_df[col] = pd.to_datetime(labels_df[col], utc=True, errors='coerce')
        labels_df[col] = labels_df[col].dt.tz_localize(None)
        labels_df[col] = labels_df[col].apply(lambda ts: ts.replace(
            year=base_date.year,
            month=base_date.month,
            day=base_date.day
        ) if pd.notnull(ts) else ts)

    # Add column if not present
    if label_col_name not in df.columns:
        df[label_col_name] = "None"

    # Assign labels
    for _, row in labels_df.iterrows():
        start, stop, activity = row['start_time'], row['end_time'], row[activity_col]
        if pd.isna(start) or pd.isna(stop):
            continue
        mask = (df['Timestamp_pd'] >= start) & (df['Timestamp_pd'] <= stop)
        if not mask.any():
            i0 = (df['Timestamp_pd'] - start).abs().idxmin()
            i1 = (df['Timestamp_pd'] - stop).abs().idxmin()
            mask = (df.index >= i0) & (df.index <= i1)
        df.loc[mask, label_col_name] = activity

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Which file types would you like to process?")
    print("Options: biopac, heart_rate, acc, gyro")
    file_type = input("Enter the file type (e.g., 'heart_rate'): ").strip().lower()

    # List to store skipped files and their reasons
    skipped_files_log = []

    if file_type not in ['biopac', 'heart_rate', 'acc', 'gyro']:
        print(f"Invalid file type '{file_type}'. Please choose from the available options.")
    else:
        pattern = os.path.join(INPUT_DIR, f"{file_type}_P*.csv")
        file_paths_to_process = sorted(glob.glob(pattern))

        if not file_paths_to_process:
            print(f"No files found for type '{file_type}' in {INPUT_DIR}.")
        else:
            activity_records = []
            per_participant_activity_json = {}

            for file_path in file_paths_to_process:
                fname = os.path.basename(file_path)
                pid = fname.split('_P')[-1].split('.csv')[0]
                print(f"\n→ Processing P{pid} for {file_type} data")

                try:
                    df = pd.read_csv(file_path, low_memory=False)

                    # Check for the correct timestamp column name and handle conversions
                    if 'Timestamp_pd' in df.columns:
                        timestamp_col = 'Timestamp_pd'
                    elif 'date_time' in df.columns:
                        timestamp_col = 'date_time'
                        df.rename(columns={'date_time': 'Timestamp_pd'}, inplace=True)
                    else:
                        reason = f"Neither 'Timestamp_pd' nor 'date_time' column found."
                        print(f"⚠️ {fname}: {reason}. Skipping.")
                        skipped_files_log.append({'filename': fname, 'reason': reason})
                        continue
                    
                    # Convert the timestamp column to datetime objects, coercing errors to NaT
                    df['Timestamp_pd'] = pd.to_datetime(df['Timestamp_pd'], errors='coerce', utc=False)
                    
                    # Drop rows with invalid timestamps
                    initial_rows = len(df)
                    df.dropna(subset=['Timestamp_pd'], inplace=True)
                    if df.empty:
                        reason = f"All rows dropped due to invalid timestamps after parsing '{timestamp_col}'."
                        print(f"⚠️ {fname}: {reason}. Skipping.")
                        skipped_files_log.append({'filename': fname, 'reason': reason})
                        continue

                    base_date = df['Timestamp_pd'].min()
                    
                    # Map activity column right after loading
                    if 'activity' in df.columns:
                        df['activity'] = df['activity'].apply(map_activity_value)

                    # Add Belt labels
                    belt_labels_path = os.path.join(BELT_LABELS_DIR, f"belt_task_timeline_P{pid}.csv")
                    if os.path.exists(belt_labels_path):
                        belt_labels_df = pd.read_csv(belt_labels_path)
                        add_activity_labels(df, belt_labels_df, base_date, "Belt_Activity_Labels", activity_col='Belt_Activity_Labels')
                        print("Unique belt activity merged = ", df["Belt_Activity_Labels"].unique())
                    else:
                        print(f"[P{pid}] ⚠️ missing {os.path.basename(belt_labels_path)}")

                    # Add Manual labels
                    manual_labels_path = os.path.join(MANUAL_LABELS_DIR, f"manual_task_timeline_P{pid}.csv")
                    if os.path.exists(manual_labels_path):
                        manual_labels_df = pd.read_csv(manual_labels_path)
                        add_activity_labels(df, manual_labels_df, base_date, "manual_labels_activity", activity_col='manual_labels_activity')
                        print("Unique manual activity merged = ", df["manual_labels_activity"].unique())

                        num_samples_each_class = count_class_samples(df, "manual_labels_activity")
                        print("Number of samples in each class ====")
                        print(num_samples_each_class)
                        per_participant_activity_json[f"P{pid}"] = num_samples_each_class.to_dict()

                        for activity, count in num_samples_each_class.items():
                            activity_records.append({
                                "Participant": f"P{pid}",
                                "Activity": activity,
                                "Samples": count
                            })
                    else:
                        print(f"[P{pid}] ⚠️ missing {os.path.basename(manual_labels_path)}")
                        if 'activity' in df.columns:
                            df["manual_labels_activity"] = df["activity"]
                            num_samples_each_class = count_class_samples(df, "manual_labels_activity")
                            for activity, count in num_samples_each_class.items():
                                activity_records.append({
                                    "Participant": f"P{pid}",
                                    "Activity": activity,
                                    "Samples": count
                                })
                    
                    for col in ['activity', 'manual_labels_activity', 'Belt_Activity_Labels']:
                        if col in df.columns:
                            df[col] = df[col].apply(map_activity_value)

                    out_csv = os.path.join(OUTPUT_DIR, f"{file_type}_{pid}_merged_labels.csv")
                    df.to_csv(out_csv, index=False)
                    print(f"[P{pid}] saved → {out_csv}")
                
                except Exception as e:
                    reason = f"Error processing file: {e}"
                    print(f"❌ {fname}: {reason}. Skipping.")
                    skipped_files_log.append({'filename': fname, 'reason': reason})
                    continue

            if activity_records:
                df_plot = pd.DataFrame(activity_records)
                if not df_plot.empty:
                    df_plot = df_plot[df_plot['Participant'] != 'P10']
                    df_plot['Participant_num'] = df_plot['Participant'].str.replace('P', '').astype(int)
                    sorted_participants = df_plot[['Participant', 'Participant_num']].drop_duplicates().sort_values('Participant_num')['Participant'].tolist()
                    df_plot['Participant'] = pd.Categorical(df_plot['Participant'], categories=sorted_participants, ordered=True)
                    df_plot = df_plot.sort_values('Participant')

                    fig2 = px.bar(
                        df_plot,
                        x="Participant",
                        y="Samples",
                        color="Activity",
                        barmode="group",
                        category_orders={"Participant": sorted_participants},
                        title=f"Sample Count per Participant for {file_type.upper()} Data (Grouped by Activity)",
                        height=600,
                        width=1200
                    )
                    fig2.update_layout(
                        plot_bgcolor='white', title_font=dict(size=22), legend_title="Activity",
                        font=dict(size=16), xaxis_title="Participant ID", yaxis_title="Sample Count"
                    )
                    fig2.show()

                json_output_path = os.path.join(OUTPUT_DIR, f"{file_type}_samples_per_activity_per_participant.json")
                with open(json_output_path, 'w') as f:
                    json.dump(per_participant_activity_json, f, indent=4)
                print(f"✅ Saved activity sample counts JSON → {json_output_path}")

    # --- Output Skipped Files Log ---
    if skipped_files_log:
        print("\n--- Files Not Written to Output Folder ---")
        for entry in skipped_files_log:
            print(f"File: {entry['filename']}, Reason: {entry['reason']}")
        
        # Save the skipped files log to a CSV
        skipped_df = pd.DataFrame(skipped_files_log)
        skipped_csv_path = os.path.join(OUTPUT_DIR, f"{file_type}_skipped_files_log.csv")
        skipped_df.to_csv(skipped_csv_path, index=False)
        print(f"\n✅ Saved skipped files log → {skipped_csv_path}")
    else:
        print("\n🎉 All files were processed and written successfully!")