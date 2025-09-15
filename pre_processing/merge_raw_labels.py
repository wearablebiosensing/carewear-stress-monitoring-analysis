import pandas as pd
import glob
import os
import plotly.express as px
import json
import numpy as np
import argparse

# --- Constants ---
# NOTE: Please update these paths to your local machine
root = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024_25/"
INPUT_DIR = root + "Concat_File"
OUTPUT_DIR = root + "merged_lables"
MANUAL_LABELS_DIR = root + "other/Task_Time_Line_Manual"
BELT_LABELS_DIR = root + "Task_Time_Line_Belt"

# Define the mapping dictionary
activity_mapping = {
    "rest1": "rest1",
    "prepare speech": "prepare speech",
    "give speech": "give speech",
    "rest2": "rest2",
    "mental math": "mental math",
    "rest3": "rest3",
    "stationary_Bike1": "stationary_Bike1",
    "stationary_Bike2": "stationary_Bike2",
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

def add_activity_labels(df, labels_df, label_col_name, activity_col):
    """
    Assigns activity labels to a DataFrame based on start/end times from a labels DataFrame.
    This version is robust to bad time formats in the labels file.
    """
    if 'Timestamp_pd' not in df.columns or df['Timestamp_pd'].isna().all():
        raise ValueError("DataFrame has no valid 'Timestamp_pd' data to merge with.")

    base_date = df['Timestamp_pd'].min().date()

    # Clean up labels dataframe first
    labels_df.dropna(subset=['start_time', 'end_time', activity_col], inplace=True)

    for col in ('start_time', 'end_time'):
        # Convert time-only strings to datetime objects, coercing errors to NaT
        parsed_times = pd.to_datetime(labels_df[col], errors='coerce').dt.time
        
        # Combine with base_date, but only for valid times
        labels_df[col] = [
            pd.to_datetime(f"{base_date} {t}") if pd.notna(t) else pd.NaT
            for t in parsed_times
        ]

    # Drop any rows that failed to parse, and log the action
    original_rows = len(labels_df)
    labels_df.dropna(subset=['start_time', 'end_time'], inplace=True)
    if len(labels_df) < original_rows:
        print(f"  - WARNING: Skipped {original_rows - len(labels_df)} rows from labels file due to invalid time format.")

    if label_col_name not in df.columns:
        df[label_col_name] = "None"

    for _, row in labels_df.iterrows():
        start, stop, activity = row['start_time'], row['end_time'], row[activity_col]
        mask = (df['Timestamp_pd'] >= start) & (df['Timestamp_pd'] <= stop)
        df.loc[mask, label_col_name] = activity

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge activity labels with data files.")
    parser.add_argument("--file_type", required=True, choices=['biopac', 'heart_rate', 'acc', 'gyro'],
                        help="The type of data file to process (e.g., 'heart_rate').")
    parser.add_argument("--labels_dir", required=True, choices=['manual', 'belt'],
                        help="The type of labels to merge ('manual' or 'belt').")
    
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Processing data type: {args.file_type} with labels from: {args.labels_dir}")

    skipped_files_log = []
    pattern = os.path.join(INPUT_DIR, f"{args.file_type}_P*.csv")
    file_paths_to_process = sorted(glob.glob(pattern))

    if not file_paths_to_process:
        print(f"No files found for type '{args.file_type}' in {INPUT_DIR}.")
    else:
        activity_records = []
        per_participant_activity_json = {}

        for file_path in file_paths_to_process:
            fname = os.path.basename(file_path)
            pid = fname.split('_P')[-1].split('.csv')[0]
            print(f"\n→ Processing P{pid} for {args.file_type} data")

            try:
                df = pd.read_csv(file_path, low_memory=False)

                # --- TIMESTAMP PARSING AND CORRECTION LOGIC ---
                if 'Timestamp_pd' in df.columns:
                    source_col = 'Timestamp_pd'
                elif 'Timestamp' in df.columns:
                    source_col = 'Timestamp'
                elif 'date_time' in df.columns:
                    source_col = 'date_time'
                else:
                    reason = "No valid timestamp column found ('Timestamp_pd', 'Timestamp', or 'date_time')."
                    print(f"⚠️ {fname}: {reason}. Skipping.")
                    skipped_files_log.append({'filename': fname, 'reason': reason})
                    continue

                print(f"  - Using '{source_col}' as the source for timestamps.")
                df['Timestamp_pd'] = pd.to_datetime(df[source_col], errors='coerce')
                
                df.dropna(subset=['Timestamp_pd'], inplace=True)
                if df.empty:
                    reason = "All rows dropped due to invalid timestamps."
                    print(f"⚠️ {fname}: {reason}. Skipping.")
                    skipped_files_log.append({'filename': fname, 'reason': reason})
                    continue

                # **FIX**: Add 12 hours to timestamps incorrectly parsed as AM
                print("  - Correcting timestamps from AM to PM...")
                condition = df['Timestamp_pd'].dt.hour < 12
                df.loc[condition, 'Timestamp_pd'] += pd.Timedelta(hours=12)
                # --- END OF TIMESTAMP LOGIC ---
                
                if args.labels_dir == 'manual':
                    labels_path = os.path.join(MANUAL_LABELS_DIR, f"manual_task_timeline_P{pid}.csv")
                    label_col_name = "manual_labels_activity"
                    activity_col_in_labels = 'manual_labels_activity'

                elif args.labels_dir == 'belt':
                    labels_path = os.path.join(BELT_LABELS_DIR, f"belt_task_timeline_P{pid}.csv")
                    label_col_name = "Belt_Activity_Labels"
                    activity_col_in_labels = 'Belt_Activity_Labels'

                if os.path.exists(labels_path):
                    labels_df = pd.read_csv(labels_path)
                    add_activity_labels(df, labels_df, label_col_name, activity_col=activity_col_in_labels)
                    print(f"  - Unique '{args.labels_dir}' activities merged: {df[label_col_name].unique()}")

                    num_samples_each_class = count_class_samples(df, label_col_name)
                    print("  - Number of samples in each class:")
                    print(num_samples_each_class)
                    per_participant_activity_json[f"P{pid}"] = num_samples_each_class.to_dict()

                    for activity, count in num_samples_each_class.items():
                        activity_records.append({
                            "Participant": f"P{pid}", "Activity": activity, "Samples": count
                        })
                else:
                    print(f"  - ⚠️ Missing {os.path.basename(labels_path)}. Skipping label merge for this participant.")
                    skipped_files_log.append({'filename': fname, 'reason': f"Missing labels file: {labels_path}"})
                    continue
                
                if label_col_name in df.columns:
                    df[label_col_name] = df[label_col_name].apply(map_activity_value)

                out_csv = os.path.join(OUTPUT_DIR, f"{args.file_type}_{pid}_merged_with_{args.labels_dir}_labels.csv")
                df.to_csv(out_csv, index=False)
                print(f"  - [P{pid}] saved → {os.path.basename(out_csv)}")
            
            except Exception as e:
                reason = f"Error processing file: {e}"
                print(f"❌ {fname}: {reason}. Skipping.")
                skipped_files_log.append({'filename': fname, 'reason': reason})
                continue
        # (The rest of the plotting and logging code remains the same)
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
                    title=f"Sample Count per Participant for {args.file_type.upper()} Data (Grouped by Activity)",
                    height=600,
                    width=1200
                )
                fig2.update_layout(
                    plot_bgcolor='white', title_font=dict(size=22), legend_title="Activity",
                    font=dict(size=16), xaxis_title="Participant ID", yaxis_title="Sample Count"
                )
                fig2.show()

            json_output_path = os.path.join(OUTPUT_DIR, f"{args.file_type}_samples_per_activity_per_participant.json")
            with open(json_output_path, 'w') as f:
                json.dump(per_participant_activity_json, f, indent=4)
            print(f"✅ Saved activity sample counts JSON → {json_output_path}")

    if skipped_files_log:
        print("\n--- Files Not Written to Output Folder ---")
        for entry in skipped_files_log:
            print(f"File: {entry['filename']}, Reason: {entry['reason']}")
        
        skipped_df = pd.DataFrame(skipped_files_log)
        skipped_csv_path = os.path.join(OUTPUT_DIR, f"{args.file_type}_{args.labels_dir}_skipped_files_log.csv")
        skipped_df.to_csv(skipped_csv_path, index=False)
        print(f"\n✅ Saved skipped files log → {skipped_csv_path}")
    else:
        print("\n🎉 All files were processed and written successfully!")