###############################################################
# Appends experimenter data labels to  
###############################################################
TAG = "TASK TIMELINE MERGE"
#!/usr/bin/env python3
import pandas as pd
import glob
import os
# Constants - 
INPUT_DIR = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Concat_File"
INPUT_DIR_merged = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Belt"
OUTPUT_DIR = "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/merged_lables"
error_pids = []
expected_lables_list = ['rest1', 'rest3', "stationary_Bike1",'stationary_Bike2', 'prepare speech', 'rest2'
 ,'give speech' ,'mental math']

def add_activity_labels(hr_df, labels_df, base_date, label_col_name, activity_col):
    """Assigns activity labels to hr_df based on start/end times from labels_df."""
    # Convert and align label times
    for col in ('start_time', 'end_time'):
        labels_df[col] = pd.to_datetime(labels_df[col], format='mixed',utc=True)
        labels_df[col] = labels_df[col].dt.tz_localize(None)
        labels_df[col] = labels_df[col].apply(lambda ts: ts.replace(
            year=base_date.year,
            month=base_date.month,
            day=base_date.day
        ) if pd.notnull(ts) else ts)

    # Add column if not present
    if label_col_name not in hr_df.columns:
        hr_df[label_col_name] = "None"

    # Assign labels
    for _, row in labels_df.iterrows():
        start, stop, activity = row['start_time'], row['end_time'], row[activity_col]
        if pd.isna(start) or pd.isna(stop):
            continue
        mask = (hr_df['Timestamp_pd'] >= start) & (hr_df['Timestamp_pd'] <= stop)
        if not mask.any():
            i0 = (hr_df['Timestamp_pd'] - start).abs().idxmin()
            i1 = (hr_df['Timestamp_pd'] - stop ).abs().idxmin()
            mask = (hr_df.index >= i0) & (hr_df.index <= i1)
        hr_df.loc[mask, label_col_name] = activity

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pattern = os.path.join(INPUT_DIR, "heart_rate_P*.csv")
    for hr_path in sorted(glob.glob(pattern)):
        fname = os.path.basename(hr_path)
        pid = fname.split('_P')[-1].split('.csv')[0]
        print(f"\n→ Processing P{pid}")
        hr_df = pd.read_csv(hr_path)
        hr_df['Timestamp_pd'] = pd.to_datetime(hr_df['Timestamp_pd'], utc=False)
        base_date = hr_df['Timestamp_pd'].min()

        # Add Belt labels
        belt_labels_path = os.path.join(
            "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Belt",
            f"belt_task_timeline_P{pid}.csv")
        if os.path.exists(belt_labels_path):
            belt_labels_df = pd.read_csv(belt_labels_path)
            add_activity_labels(hr_df, belt_labels_df, base_date, "Belt_Activity_Labels", activity_col='Belt_Activity_Labels')
            print("Unique belt activity merged = ",hr_df["Belt_Activity_Labels"].unique())

        else:
            print(f"[P{pid}] ⚠️ missing belt_task_timeline_P{pid}.csv")

        # Add Manual labels
        manual_labels_path = os.path.join(
            "/Users/shehjarsadhu/Desktop/UniversityOfRhodeIsland/Graduate/WBL/Project_Carehub_CareWear/DATASET/StudyData_Drive_2024/Task_Time_Line_Manual",
            f"manual_task_timeline_P{pid}.csv")
        if os.path.exists(manual_labels_path):
            manual_labels_df = pd.read_csv(manual_labels_path)
            add_activity_labels(hr_df, manual_labels_df, base_date, "manual_labels_activity", activity_col='manual_labels_activity')
            print("Unique watch activity merged = ",hr_df["manual_labels_activity"].unique())

        else:
            print(f"[P{pid}] ⚠️ missing manual_task_timeline_P{pid}.csv")

        # Save output
        out_csv = os.path.join(OUTPUT_DIR, f"heart_rate_{pid}_merged_labels.csv")
        hr_df.to_csv(out_csv, index=False)
        print(f"[P{pid}] saved → {out_csv}")

